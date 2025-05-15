# Shape and Texture: What Influences Reliable Optical Flow Estimation?

## Abstract
Recent methods have made significant progress in optical flow estimation. However, the evaluation of these methods focuses mainly on improved accuracy in benchmarks and often overlooks the analysis of network robustness, which may be important in safety-critical scenarios such as autonomous driving.
In this paper, we propose a novel method for robustness evaluation by modifying data from original benchmarks. Unlike previous benchmarks that focus on complex scenes, we propose to modify the shape and texture of objects from the original images in order to analyze the sensitivity to these changes observed in the output. Our aim is to identify common failure cases of state-of-the-art (SOTA) methods to evaluate their robustness and understand their behaviors.
We show that: Optical flow methods are more sensitive to shape changes than to texture changes; and
Optical flow methods tend to “remember” objects seen during training and may “ignore” the motion of unseen objects.
Our experimental results and findings provide a more in-depth understanding of the behavior of recent optical flow methods.

![PDF Image](workflow.png)