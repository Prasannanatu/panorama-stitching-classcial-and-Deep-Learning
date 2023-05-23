# Panorama Stitching

## Outputs

### Classical Panorama Stitching Output

| Output 1 | Output 2 |
|----------|----------|
| ![Output 1](./Phase1/Outputs/mypano1.png) | ![Output 2](./Phase1/Outputs/mypano2.png) |


### Supervised and Self-Supervised Output

| Supervised Homography Net Loss | Self-supervised Homography Net Loss |
|--------------------------------|-------------------------------------|
| ![Supervised Loss](./Phase2/deep_output/supervisedlos.jpeg) | ![Unsupervised Loss](./Phase2/deep_output/unsupervisedlos.jpg) |

## Table of Contents
- [About The Project](#about-the-project)
- [Repository Structure](#repository-structure)
- [Technologies](#technologies)
- [Installation & Usage](#installation--usage)
- [Reference](#reference)

## About The Project
In Phase 1, the project utilizes traditional computer vision methods for stitching together multiple images. This includes Harris Corner Detection, Feature Descriptors, Feature Matching, the RANSAC algorithm, and warping and blending of images.

Phase 2 employs deep learning techniques for panorama stitching. This includes a Homography Net used for supervised learning, and Tensor Direct Linear Transform (DLT) for self-supervised learning.

## Repository Structure

```markdown
.
├── Phase1
│   ├── code
│   │   ├── Wrapper.py
│   ├── Outputs
│   │   ├── mypano1.png
│   │   ├── mypano2.png
|   ├──Data
│   │   ├── Test
│   │   ├── Train
 Phase2
│   ├── Checkopints
│   │   ├──supervised_unsupervised.zip
│   ├── deep_output
│   │   ├── supervisedlos.jpeg
│   │   ├── unsupervisedlos.jpg
│   ├── code
│   │   ├── main.py
│   │   ├── utils.py
├──README.md
└──Report.pdf

```
## Technologies Used

This project uses several technologies and methodologies to achieve panorama stitching:

- **Python**: The main programming language used for the implementation of this project.
- **OpenCV**: A powerful library used extensively for image processing tasks.
- **NumPy**: Essential for high-performance mathematical computations on multi-dimensional arrays and matrices.
- **PyTorch**: The deep learning framework used for implementing the Homography Net and Tensor DLT.
- **Homography Net**: A supervised learning method using a neural network to predict homography between pairs of images.
- **Tensor Direct Linear Transform (DLT)**: A self-supervised learning method that predicts homographies purely from the image data.
- **Spatial Transformer**: A differentiable module that performs an explicit geometric transformation on the input image, enabling neural networks to learn how to perform spatial transformations on images.



## Installation & Usage
To run the project locally, follow these steps:

1. Clone the repository:

```shell
git clone https://github.com/Prasannanatu/panorama-stitching-classcial-and-Deep-Learning.git
 ```
 
Install the required dependencies. You can use the provided requirements.txt file to install the necessary packages. Run the following command:


```bash
pip install -r requirements.txt
 ```

## References

The following sources have been instrumental in the development of this project:

1. DeTone, D., Malisiewicz, T., & Rabinovich, A. (2016). Deep Image Homography Estimation. arXiv preprint arXiv:1606.03798 [[Link]](https://arxiv.org/pdf/1606.03798.pdf)
2. Nguyen, T., Chen, S. W., Shivakumar, S. S., Taylor, C. J., & Kumar, V. (2018). Unsupervised Deep Homography: A Fast and Robust Homography Estimation Model. In Proceedings of the European Conference on Computer Vision (ECCV) (pp. 621-637).[[Link]](https://arxiv.org/abs/1709.03966)

3.RBE-549 Computer Vision course project page[[link]](https://rbe549.github.io/spring2023/proj/p1/).


## Tags

- Homography Estimation
- Deep Learning
- Computer Vision
- Unsupervised Learning
- Robust Estimation
- Image Registration
- Feature Extraction
- HomographyNet



<!-- -Classical Panorama Stitching Output:
| ![image1](./Phase1/Outputs/mypano1.png) | ![image2](./Phase1/Outputs/mypano2.png) |
|:--:|:---:|
| output_1 | output_2|


-Supervised and Self-Supervised Output:
| ![image1](./Phase2/deep_output/supervisedlos.jpeg) | ![image2](./Phase2/deep_output/unsupervisedlos.jpg) |
|:--:|:---:|
| Suervised Homography Net loss | Self-suervised Homography Net loss| -->



