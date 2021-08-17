# LANet: A Luminance Attentive Network with Scale Invariance for HDR Image Reconstruction

It is very challenging to reconstruct a high dynamic range (HDR) from a low dynamic range (LDR) image as an ill-posed problem. This paper proposes a luminance attentive network named LANet for HDR reconstruction from a single LDR image. Our method is based on two fundamental observations: (1) HDR images stored in relative luminance are scale-invariant, which means the HDR images will hold the same information when multiplied by any positive real number. Based on this observation, we propose a novel normalization method called " HDR calibration " for HDR images stored in relative luminance, calibrating HDR images into a similar luminance scale according to the LDR images. (2) The main difference between HDR images and LDR images is in under-/over-exposed areas, especially those highlighted. Following this observation, we propose a luminance attention module with a two-stream structure for LANet to pay more attention to the under-/over-exposed areas. In addition, we propose an extended network called panoLANet for HDR panorama reconstruction from an LDR panorama and build a dual net structure for panoLANet to solve the distortion problem caused by the equirectangular panorama. Extensive experiments show that our proposed approach LANet can reconstruct visually convincing HDR images and demonstrate its superiority over state-of-the-art approaches in terms of all metrics in inverse tone mapping. The image-based lighting application with our proposed panoLANet also demonstrates that our method can simulate natural scene lighting using only LDR panorama. 


## Overview 

This is the author's reference implementation of the single-image HDR reconstruction using TensorFlow described in:
"LANet: A Luminance Attentive Network with Scale Invariance for HDR Image Reconstruction"

The network architecture details are shown in "model.py" and the data processing is in "utils.py".


## Prerequisites

* Python3
* numpy 
* OpenCV >= 3.4
* Tensorflow == 1.13.1
* TensorLayer == 1.11.1


## Usage

### Pretrained model

The pretrained LANet checkpoints can be found in the checkpoints folder on [Google Drive](https://drive.google.com/drive/folders/1cM6hTfCrGplMFSyMmVNz_pC9gqBQAWoo?usp=sharing).
The pretrained panoLANet checkpoints can be found in the checkpoints folder on [Google Drive](https://drive.google.com/drive/folders/1Ex8LzDqwhTgts46ACR0umpKUT1DgYNKQ?usp=sharing).

### Inference

* Run your own images (using our trained LANet):
``` 
cd LANet
python ./src/main.py --phase test --gpu 0 --checkpoint_dir ./checkpoint_LANet/ --test_dir ./test/ --out_dir ./out/
```

* Run your own panoramas (using our trained panoLANet):
``` 
cd panoLANet
python ./src/main.py --phase test --gpu 0 --checkpoint_dir ./checkpoint_panoLANet/ --test_dir ./test/ --out_dir ./out/
```

Parameters and their description:

>```checkpoint_dir```: path to the trained models.<br/>
>```test_dir```: input images directory. This project provides a few sample images.<br/>
>```out_dir```: path to output directory.<br/>
<br/>

See main.py for more settable parameters. 

