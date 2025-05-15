import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn

import os
import inspect

import frame_utils
import random

def mesh_grid(B, H, W):
    # mesh grid
    x_base = torch.arange(0, W).repeat(B, H, 1)  # BHW
    y_base = torch.arange(0, H).repeat(B, W, 1).transpose(1, 2)  # BHW

    base_grid = torch.stack([x_base, y_base], 1)  # B2HW
    return base_grid


def norm_grid(v_grid):
    _, _, H, W = v_grid.size()

    # scale grid to [-1,1]
    v_grid_norm = torch.zeros_like(v_grid)
    v_grid_norm[:, 0, :, :] = 2.0 * v_grid[:, 0, :, :] / (W - 1) - 1.0
    v_grid_norm[:, 1, :, :] = 2.0 * v_grid[:, 1, :, :] / (H - 1) - 1.0
    return v_grid_norm.permute(0, 2, 3, 1)  # BHW2


def get_corresponding_map(data):
    """

    :param data: unnormalized coordinates Bx2xHxW
    :return: Bx1xHxW
    """
    B, _, H, W = data.size()

    # x = data[:, 0, :, :].view(B, -1).clamp(0, W - 1)  # BxN (N=H*W)
    # y = data[:, 1, :, :].view(B, -1).clamp(0, H - 1)

    x = data[:, 0, :, :].view(B, -1)  # BxN (N=H*W)
    y = data[:, 1, :, :].view(B, -1)

    # invalid = (x < 0) | (x > W - 1) | (y < 0) | (y > H - 1)   # BxN
    # invalid = invalid.repeat([1, 4])

    x1 = torch.floor(x)
    x_floor = x1.clamp(0, W - 1)
    y1 = torch.floor(y)
    y_floor = y1.clamp(0, H - 1)
    x0 = x1 + 1
    x_ceil = x0.clamp(0, W - 1)
    y0 = y1 + 1
    y_ceil = y0.clamp(0, H - 1)

    x_ceil_out = x0 != x_ceil
    y_ceil_out = y0 != y_ceil
    x_floor_out = x1 != x_floor
    y_floor_out = y1 != y_floor
    invalid = torch.cat([x_ceil_out | y_ceil_out,
                         x_ceil_out | y_floor_out,
                         x_floor_out | y_ceil_out,
                         x_floor_out | y_floor_out], dim=1)

    # encode coordinates, since the scatter function can only index along one axis
    corresponding_map = torch.zeros(B, H * W).type_as(data)
    indices = torch.cat([x_ceil + y_ceil * W,
                         x_ceil + y_floor * W,
                         x_floor + y_ceil * W,
                         x_floor + y_floor * W], 1).long()  # BxN   (N=4*H*W)
    values = torch.cat([(1 - torch.abs(x - x_ceil)) * (1 - torch.abs(y - y_ceil)),
                        (1 - torch.abs(x - x_ceil)) * (1 - torch.abs(y - y_floor)),
                        (1 - torch.abs(x - x_floor)) * (1 - torch.abs(y - y_ceil)),
                        (1 - torch.abs(x - x_floor)) * (1 - torch.abs(y - y_floor))],
                       1)
    # values = torch.ones_like(values)

    values[invalid] = 0

    corresponding_map.scatter_add_(1, indices, values)
    # decode coordinates
    corresponding_map = corresponding_map.view(B, H, W)

    return corresponding_map.unsqueeze(1)


def flow_warp(x, flow12, pad='border', mode='bilinear'):
    B, _, H, W = x.size()

    base_grid = mesh_grid(B, H, W).type_as(x)  # B2HW

    v_grid = norm_grid(base_grid + flow12)  # BHW2
    if 'align_corners' in inspect.getfullargspec(torch.nn.functional.grid_sample).args:
        im1_recons = nn.functional.grid_sample(x, v_grid, mode=mode, padding_mode=pad, align_corners=True)
    else:
        im1_recons = nn.functional.grid_sample(x, v_grid, mode=mode, padding_mode=pad)
    return im1_recons


def get_occu_mask_bidirection(flow12, flow21, scale=0.01, bias=0.5):
    flow21_warped = flow_warp(flow21, flow12, pad='zeros')
    flow12_diff = flow12 + flow21_warped
    mag = (flow12 * flow12).sum(1, keepdim=True) + \
          (flow21_warped * flow21_warped).sum(1, keepdim=True)
    occ_thresh = scale * mag + bias
    occ = (flow12_diff * flow12_diff).sum(1, keepdim=True) > occ_thresh
    return occ.float()


def get_occu_mask_backward(flow21, th=0.2):
    B, _, H, W = flow21.size()
    base_grid = mesh_grid(B, H, W).type_as(flow21)  # B2HW

    corr_map = get_corresponding_map(base_grid + flow21)  # BHW
    occu_mask = corr_map.clamp(min=0., max=1.) < th
    return occu_mask.float()



def bilinear_interpolate(image, x, y):
    x0 = int(x)
    x1 = min(x0 + 1, image.shape[1] - 1)
    y0 = int(y)
    y1 = min(y0 + 1, image.shape[0] - 1)

    Ia = image[y0, x0]
    Ib = image[y0, x1]
    Ic = image[y1, x0]
    Id = image[y1, x1]

    wa = (x1 - x) * (y1 - y)
    wb = (x - x0) * (y1 - y)
    wc = (x1 - x) * (y - y0)
    wd = (x - x0) * (y - y0)

    return wa*Ia + wb*Ib + wc*Ic + wd*Id

def backward_warp(image, flow):
    h, w = flow.shape[:2]
    warped_image = np.zeros_like(image)

    for y in range(h):
        for x in range(w):
            flow_x, flow_y = flow[y, x]
            src_x = x + flow_x
            src_y = y + flow_y

            # Check if the source coordinates are within the image bounds
            if 0 <= src_x < w and 0 <= src_y < h:
                if not (flow_x == 0. and flow_y == 0.):
                    warped_image[y, x] = bilinear_interpolate(image, src_x, src_y)
            else:
                # Use bilinear interpolation for the border pixels
                if 0 <= src_x < w and 0 <= src_y < h:
                    warped_image[y, x] = bilinear_interpolate(image, src_x, src_y)

    return warped_image

def read_flow_file(file_path):
    flow = cv2.imread(file_path, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_COLOR)
    flow = flow[:, :, ::-1].astype(np.float32)
    flow, valid = flow[:, :, :2], flow[:, :, 2]
    flow = (flow - 2**15) / 64.0
    flow = torch.from_numpy(flow).permute(2, 0, 1)
    return flow, valid

def writeFlowKITTI(filename, uv, valid=None):
    uv = 64.0 * uv + 2**15
    if valid is None:
        valid = np.ones([uv.shape[0], uv.shape[1], 1])
    uv = np.concatenate([uv, valid], axis=-1).astype(np.uint16)
    cv2.imwrite(filename, uv[..., ::-1])


def compute_center(mask):
    """
    Compute the center of the mask.
    
    Args:
    - mask: Binary mask of the object.
    
    Returns:
    - center: (cx, cy) tuple representing the center of the mask.
    """
    y, x = np.where(mask == 1)  # Find the coordinates of the object pixels
    cx = np.mean(x)
    cy = np.mean(y)
    return cx, cy



def update_optical_flow_with_scaling_offsets(flow, mask1, mask2, m1s, scale_factor):
    """
    Update the optical flow values for a scaled object, using the offsets between
    mask1 and its scaled version, and mask2 and its scaled version.
    
    Args:
    - flow: Optical flow between img1 and img2 (PyTorch tensor of shape (2, h, w)).
    - mask1: Mask of the object in img1 (binary mask, unscaled, NumPy array (h, w)).
    - mask2: Mask of the object in img2 (binary mask, unscaled, NumPy array (h, w)).
    - m1s: The original scaled mask for mask1 (NumPy array).
    - scale_factor: The scaling factor applied to the object.
    
    Returns:
    - updated_flow: Optical flow adjusted for the new object positions (same shape as input flow).
    """
    
    orig_cx1, orig_cy1 = compute_center(mask1)
    orig_cx2, orig_cy2 = compute_center(mask2)
    
    new_h, new_w = int(mask1.shape[0] * scale_factor), int(mask1.shape[1] * scale_factor)
    
    scaled_mask1 = cv2.resize(mask1.astype(np.float32), (new_w, new_h), interpolation=cv2.INTER_NEAREST)
    scaled_mask1 = (scaled_mask1 > 0.5).astype(np.bool_)
    
    scaled_mask2 = cv2.resize(mask2.astype(np.float32), (new_w, new_h), interpolation=cv2.INTER_NEAREST)
    scaled_mask2 = (scaled_mask2 > 0.5).astype(np.bool_)
    
    new_cx1, new_cy1 = compute_center(scaled_mask1)
    new_cx2, new_cy2 = compute_center(scaled_mask2)
    

    updated_flow = np.copy(flow)
    
    y_scaled, x_scaled = np.where(m1s)
    
    for y, x in zip(y_scaled, x_scaled):
        if flow[0, y, x] != 0 and flow[1, y, x] != 0:  
            updated_flow[0, y, x] *= scale_factor
            updated_flow[0, y, x] *= scale_factor

            updated_flow[0, y, x] += (-orig_cx2 + orig_cx1) * -1*(1-scale_factor)  
            updated_flow[1, y, x] += (-orig_cy2 + orig_cy1) * -1*(1-scale_factor)  

    return updated_flow

def segment_and_scale(image_np, flow_torch, mask_np, scale_factor=1.5):

    mask_np = mask_np.astype(bool)
    

    flow_np = flow_torch.numpy()

   
    transparent_image = np.zeros_like(image_np)
    transparent_image[mask_np] = image_np[mask_np]


    transparent_flow = np.zeros_like(flow_np)
    transparent_flow[:, mask_np] = flow_np[:, mask_np]


    y_indices, x_indices = np.where(mask_np)
    y_min, y_max = y_indices.min(), y_indices.max()
    x_min, x_max = x_indices.min(), x_indices.max()

   
    cropped_image = transparent_image[y_min:y_max+1, x_min:x_max+1]
    cropped_flow = transparent_flow[:, y_min:y_max+1, x_min:x_max+1]

   
    new_height = int(cropped_image.shape[0] * scale_factor)
    new_width = int(cropped_image.shape[1] * scale_factor)

   
    scaled_image = cv2.resize(cropped_image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
    scaled_flow = np.stack([
        cv2.resize(cropped_flow[0], (new_width, new_height), interpolation=cv2.INTER_LINEAR),
        cv2.resize(cropped_flow[1], (new_width, new_height), interpolation=cv2.INTER_LINEAR)
    ])


    new_image = np.copy(image_np)
    new_flow = np.copy(flow_np)

 
    paste_y = max(0, y_min - (new_height - (y_max - y_min + 1)) // 2)
    paste_x = max(0, x_min - (new_width - (x_max - x_min + 1)) // 2)

    
    paste_y_max = min(paste_y + new_height, image_np.shape[0])
    paste_x_max = min(paste_x + new_width, image_np.shape[1])
    
    paste_h = paste_y_max - paste_y
    paste_w = paste_x_max - paste_x

   
    alpha_mask = (scaled_image[:paste_h, :paste_w] > 0).astype(float)
    new_image[paste_y:paste_y_max, paste_x:paste_x_max] = (
        new_image[paste_y:paste_y_max, paste_x:paste_x_max] * (1 - alpha_mask) +
        scaled_image[:paste_h, :paste_w] * alpha_mask
    )


    new_flow[:, paste_y:paste_y_max, paste_x:paste_x_max] = scaled_flow[:, :paste_h, :paste_w]
    
    return new_image, new_flow



def make_colorwheel():
    """
    Generates a color wheel for optical flow visualization as presented in:
        Baker et al. "A Database and Evaluation Methodology for Optical Flow" (ICCV, 2007)
        URL: http://vision.middlebury.edu/flow/flowEval-iccv07.pdf

    Code follows the original C++ source code of Daniel Scharstein.
    Code follows the the Matlab source code of Deqing Sun.

    Returns:
        np.ndarray: Color wheel
    """

    RY = 15
    YG = 6
    GC = 4
    CB = 11
    BM = 13
    MR = 6

    ncols = RY + YG + GC + CB + BM + MR
    colorwheel = np.zeros((ncols, 3))
    col = 0

    # RY
    colorwheel[0:RY, 0] = 255
    colorwheel[0:RY, 1] = np.floor(255*np.arange(0,RY)/RY)
    col = col+RY
    # YG
    colorwheel[col:col+YG, 0] = 255 - np.floor(255*np.arange(0,YG)/YG)
    colorwheel[col:col+YG, 1] = 255
    col = col+YG
    # GC
    colorwheel[col:col+GC, 1] = 255
    colorwheel[col:col+GC, 2] = np.floor(255*np.arange(0,GC)/GC)
    col = col+GC
    # CB
    colorwheel[col:col+CB, 1] = 255 - np.floor(255*np.arange(CB)/CB)
    colorwheel[col:col+CB, 2] = 255
    col = col+CB
    # BM
    colorwheel[col:col+BM, 2] = 255
    colorwheel[col:col+BM, 0] = np.floor(255*np.arange(0,BM)/BM)
    col = col+BM
    # MR
    colorwheel[col:col+MR, 2] = 255 - np.floor(255*np.arange(MR)/MR)
    colorwheel[col:col+MR, 0] = 255
    return colorwheel

def flow_uv_to_colors(u, v, convert_to_bgr=False):
    """
    Applies the flow color wheel to (possibly clipped) flow components u and v.

    According to the C++ source code of Daniel Scharstein
    According to the Matlab source code of Deqing Sun

    Args:
        u (np.ndarray): Input horizontal flow of shape [H,W]
        v (np.ndarray): Input vertical flow of shape [H,W]
        convert_to_bgr (bool, optional): Convert output image to BGR. Defaults to False.

    Returns:
        np.ndarray: Flow visualization image of shape [H,W,3]
    """
    flow_image = np.zeros((u.shape[0], u.shape[1], 3), np.uint8)
    colorwheel = make_colorwheel()  # shape [55x3]
    ncols = colorwheel.shape[0]
    rad = np.sqrt(np.square(u) + np.square(v))
    a = np.arctan2(-v, -u)/np.pi
    fk = (a+1) / 2*(ncols-1)
    k0 = np.floor(fk).astype(np.int32)
    k1 = k0 + 1
    k1[k1 == ncols] = 0
    f = fk - k0
    for i in range(colorwheel.shape[1]):
        tmp = colorwheel[:,i]
        col0 = tmp[k0] / 255.0
        col1 = tmp[k1] / 255.0
        col = (1-f)*col0 + f*col1
        idx = (rad <= 1)
        col[idx]  = 1 - rad[idx] * (1-col[idx])
        col[~idx] = col[~idx] * 0.75   # out of range
        # Note the 2-i => BGR instead of RGB
        ch_idx = 2-i if convert_to_bgr else i
        flow_image[:,:,ch_idx] = np.floor(255 * col)
    return flow_image

def flow_to_image(flow_uv, clip_flow=None, convert_to_bgr=False):
    """
    Expects a two dimensional flow image of shape.

    Args:
        flow_uv (np.ndarray): Flow UV image of shape [H,W,2]
        clip_flow (float, optional): Clip maximum of flow values. Defaults to None.
        convert_to_bgr (bool, optional): Convert output image to BGR. Defaults to False.

    Returns:
        np.ndarray: Flow visualization image of shape [H,W,3]
    """
    assert flow_uv.ndim == 3, 'input flow must have three dimensions'
    assert flow_uv.shape[2] == 2, 'input flow must have shape [H,W,2]'
    if clip_flow is not None:
        flow_uv = np.clip(flow_uv, 0, clip_flow)
    u = flow_uv[:,:,0]
    v = flow_uv[:,:,1]
    rad = np.sqrt(np.square(u) + np.square(v))
    rad_max = np.max(rad)
    epsilon = 1e-5
    u = u / (rad_max + epsilon)
    v = v / (rad_max + epsilon)
    return flow_uv_to_colors(u, v, convert_to_bgr)


def compute_center(mask):
    """
    Compute the center of the object in the binary mask.
    """
    y_indices, x_indices = np.where(mask)
    return np.mean(x_indices), np.mean(y_indices)



def segment_scale_two_images12(image1_np, image2_np, flow_torch, mask1_np, mask2_np, scale_factor=1.5):
    mask1_np = mask1_np.astype(bool)
    mask2_np = mask2_np.astype(bool)

    flow_np = flow_torch.numpy()

    transparent_image1 = np.zeros_like(image1_np)
    transparent_image1[mask1_np] = image1_np[mask1_np]

    transparent_flow = np.zeros_like(flow_np)
    transparent_flow[:, mask1_np] = flow_np[:, mask1_np]

    y_indices, x_indices = np.where(mask1_np)
    y_min, y_max = y_indices.min(), y_indices.max()
    x_min, x_max = x_indices.min(), x_indices.max()

    cropped_image1 = transparent_image1[y_min:y_max+1, x_min:x_max+1]
    cropped_flow = transparent_flow[:, y_min:y_max+1, x_min:x_max+1]
    cropped_mask1 = mask1_np[y_min:y_max+1, x_min:x_max+1].astype(np.float32)

    new_height = int(cropped_image1.shape[0] * scale_factor)
    new_width = int(cropped_image1.shape[1] * scale_factor)

    scaled_image1 = cv2.resize(cropped_image1, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
    scaled_flow = np.stack([
        cv2.resize(cropped_flow[0], (new_width, new_height), interpolation=cv2.INTER_LINEAR),
        cv2.resize(cropped_flow[1], (new_width, new_height), interpolation=cv2.INTER_LINEAR)
    ])
    scaled_mask1 = cv2.resize(cropped_mask1, (new_width, new_height), interpolation=cv2.INTER_LINEAR)

    
    occlusion_mask = (transparent_flow[0] == 0) & (transparent_flow[1] == 0)
    occlusion_mask_cropped = occlusion_mask[y_min:y_max+1, x_min:x_max+1]

    scaled_occlusion_mask = cv2.resize(occlusion_mask_cropped.astype(np.float32), (new_width, new_height), interpolation=cv2.INTER_NEAREST)

    transparent_image2 = np.zeros_like(image2_np)
    transparent_image2[mask2_np] = image2_np[mask2_np]


    y_indices2, x_indices2 = np.where(mask2_np)
    y_min2, y_max2 = y_indices2.min(), y_indices2.max()
    x_min2, x_max2 = x_indices2.min(), x_indices2.max()


    cropped_image2 = transparent_image2[y_min2:y_max2+1, x_min2:x_max2+1]
    cropped_mask2 = mask2_np[y_min2:y_max2+1, x_min2:x_max2+1].astype(np.float32)

    
    new_height2 = int(cropped_image2.shape[0] * scale_factor)
    new_width2 = int(cropped_image2.shape[1] * scale_factor)

    scaled_image2 = cv2.resize(cropped_image2, (new_width2, new_height2), interpolation=cv2.INTER_LINEAR)
    scaled_mask2 = cv2.resize(cropped_mask2, (new_width2, new_height2), interpolation=cv2.INTER_LINEAR)

 
    new_image1 = np.copy(image1_np)
    new_image2 = np.copy(image2_np)
    new_flow = np.copy(flow_np)

    paste_y1 = max(0, y_min - (new_height - (y_max - y_min + 1)) // 2)
    paste_x1 = max(0, x_min - (new_width - (x_max - x_min + 1)) // 2)

    paste_y_max1 = min(paste_y1 + new_height, image1_np.shape[0])
    paste_x_max1 = min(paste_x1 + new_width, image1_np.shape[1])
    
    paste_h1 = paste_y_max1 - paste_y1
    paste_w1 = paste_x_max1 - paste_x1

    for c in range(3):
        new_image1[paste_y1:paste_y_max1, paste_x1:paste_x_max1, c] = (
            new_image1[paste_y1:paste_y_max1, paste_x1:paste_x_max1, c] * (1 - scaled_mask1[:paste_h1, :paste_w1]) +
            scaled_image1[:paste_h1, :paste_w1, c] * scaled_mask1[:paste_h1, :paste_w1]
        )

    cx1, cy1 = compute_center(mask1_np)
    cx2, cy2 = compute_center(mask2_np)

    center_displacement_x = cx2 - cx1
    center_displacement_y = cy2 - cy1

    for h in range(scaled_flow.shape[1]):
        for w in range(scaled_flow.shape[2]):
            if scaled_mask1[h, w] > 0:  # Only update flow where the mask is valid
                scaled_flow[0, h, w] = scaled_flow[0, h, w]  # Modify this line if needed for flow adjustment
                scaled_flow[1, h, w] = scaled_flow[1, h, w]

    for h in range(paste_h1):
        for w in range(paste_w1):
            original_y = paste_y1 + h
            original_x = paste_x1 + w
            
            if 0 <= original_y < new_flow.shape[1] and 0 <= original_x < new_flow.shape[2]:
                if scaled_mask1[h, w] > 0:  # Only update flow where the mask is valid
                    if scaled_occlusion_mask[h, w] > 0:
                        new_flow[0, original_y, original_x] = 0
                        new_flow[1, original_y, original_x] = 0
                    else:
                        new_flow[0, original_y, original_x] = scaled_flow[0, h, w]
                        new_flow[1, original_y, original_x] = scaled_flow[1, h, w]

    new_scaled_mask1 = np.zeros_like(mask1_np, dtype=np.uint8)
    
    new_scaled_mask1[paste_y1:paste_y_max1, paste_x1:paste_x_max1] = (scaled_mask1[:paste_h1, :paste_w1] * 255).astype(np.uint8)

    paste_y2 = max(0, y_min2 - (new_height2 - (y_max2 - y_min2 + 1)) // 2)
    paste_x2 = max(0, x_min2 - (new_width2 - (x_max2 - x_min2 + 1)) // 2)

    paste_y_max2 = min(paste_y2 + new_height2, image2_np.shape[0])
    paste_x_max2 = min(paste_x2 + new_width2, image2_np.shape[1])
    
    paste_h2 = paste_y_max2 - paste_y2
    paste_w2 = paste_x_max2 - paste_x2

    for c in range(3):
        new_image2[paste_y2:paste_y_max2, paste_x2:paste_x_max2, c] = (
            new_image2[paste_y2:paste_y_max2, paste_x2:paste_x_max2, c] * (1 - scaled_mask2[:paste_h2, :paste_w2]) +
            scaled_image2[:paste_h2, :paste_w2, c] * scaled_mask2[:paste_h2, :paste_w2]
        )

    new_scaled_mask2 = np.zeros_like(mask2_np, dtype=np.uint8)

    new_scaled_mask2[paste_y2:paste_y_max2, paste_x2:paste_x_max2] = (scaled_mask2[:paste_h2, :paste_w2] * 255).astype(np.uint8)

    return new_image1, new_image2, new_flow, new_scaled_mask1, new_scaled_mask2


def upscale_valid_map(valid_map, mask1, scale_factor=1.5):
    mask1 = mask1.astype(bool)

    masked_valid_map = np.zeros_like(valid_map)
    masked_valid_map[mask1] = valid_map[mask1]

    y_indices, x_indices = np.where(mask1)
    y_min, y_max = y_indices.min(), y_indices.max()
    x_min, x_max = x_indices.min(), x_indices.max()

    cropped_valid_map = masked_valid_map[y_min:y_max+1, x_min:x_max+1]
    cropped_mask1 = mask1[y_min:y_max+1, x_min:x_max+1].astype(np.float32)

    new_height = int(cropped_valid_map.shape[0] * scale_factor)
    new_width = int(cropped_valid_map.shape[1] * scale_factor)

    scaled_valid_map = cv2.resize(cropped_valid_map, (new_width, new_height), interpolation=cv2.INTER_NEAREST)
    scaled_mask1 = cv2.resize(cropped_mask1, (new_width, new_height), interpolation=cv2.INTER_NEAREST)

    scaled_valid_map = (scaled_valid_map > 0).astype(np.uint8)
    new_valid_map = np.copy(valid_map)

    paste_y = max(0, y_min - (new_height - (y_max - y_min + 1)) // 2)
    paste_x = max(0, x_min - (new_width - (x_max - x_min + 1)) // 2)


    paste_y_max = min(paste_y + new_height, valid_map.shape[0])
    paste_x_max = min(paste_x + new_width, valid_map.shape[1])
    
    paste_h = paste_y_max - paste_y
    paste_w = paste_x_max - paste_x

    new_valid_map[paste_y:paste_y_max, paste_x:paste_x_max] = (
        new_valid_map[paste_y:paste_y_max, paste_x:paste_x_max] * (1 - scaled_mask1[:paste_h, :paste_w]) +
        scaled_valid_map[:paste_h, :paste_w] * scaled_mask1[:paste_h, :paste_w]
    )

    return new_valid_map

from scipy.ndimage import generic_filter

folder_a = "image_2"
folder_b = "mask 2/mask"
folder_c = "mask1"
folder_f = "flow_occ"

folder_output = "output_scale"
folder_output_flow = "output_scale_flow"
folder_output_mask = "output_scale_mask"


img1 = cv2.imread("000000_10.png")
img2 = cv2.imread("000000_11.png")
flow, valid = read_flow_file("000000_10_f.png")
seg1 = cv2.imread("000000_10_mask.png", 0)
seg2 = cv2.imread("000000_11_mask.png", 0)

binary_mask1 = seg1 == 255
binary_mask2 = seg2 == 255


numbers = [round(x * 0.1, 1) for x in range(11, 15)]
selected_number = random.choice(numbers)

img1_, img2_, op, mask_s, mask2 = segment_scale_two_images12(img1, img2, flow, binary_mask1, binary_mask2, selected_number)



valid = upscale_valid_map(valid, binary_mask1, selected_number)

op = update_optical_flow_with_scaling_offsets(op, binary_mask1,  binary_mask2, mask_s, selected_number)



cv2.imwrite("new/000000_10.png", img1_)
cv2.imwrite("new/000000_11.png", img2_)
cv2.imwrite("new/000000_10_m.png", mask_s)
cv2.imwrite("new/000000_11_m.png", mask2)


op = op.transpose(1, 2, 0)#.cpu().numpy()


writeFlowKITTI("new/000000_10_f.png", op, valid[:,:,np.newaxis])

