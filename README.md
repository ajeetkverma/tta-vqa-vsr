# TTA-VQA-VSR
**_This work has been submitted._**

<h2>Real-World Video Quality Assessment via Test-Time Adaptation and its application in Real-World super-resolution

**<h2>We are working to prepare the release version of the code. It will be available soon...**
<p>
  <img width="640" height="460" src="https://www.jaggaer.com/app/uploads/2022/09/Jaggaer-Software-Releases-1048x640.jpg">
</p>



## Abstract

<div style="text-align: justify">
Recently, several real-world video super-resolution (RW-VSR) algorithms have been introduced in the literature. These methods aim to replicate the various degradations observed in real-world low-resolution videos, such as noise, motion blur, and compression artifacts.
Currently, the literature lacks domain-specific quality assessment methods that can effectively evaluate the perceptual quality of real-world super-resolved videos. To address this gap, we first construct a comprehensive real-world video dataset annotated with subjective quality scores. This dataset comprises 1,155 videos with diverse resolutions, content types, and contextual variations, providing a robust foundation for evaluating perceptual quality across real-world scenarios. Rather than relying on a fully
end-to-end deep learning model, which is inherently data dependent, we propose a Test-Time Adaptation (TTA) algorithm. This innovative method utilizes auxiliary tasks, such as group contrastive loss and rank loss as objective functions, to dynamically adapt the model during inference. By doing so, it eliminates the need for large-scale video quality assessment datasets, especially when dealing with new and unknown distortions in low-resolution real-world videos, making it a more robust and adaptable approach. The proposed rank loss assumes that real-world videos are inherently noisy, and that their cleaner versions exhibit improved perceptual quality. Additionally, we propose integrating the RW-VQA algorithm with existing RW-VSR algorithms. The performance of both the proposed RW-VQA and RW-VSR algorithms surpasses that of existing state-of-the-art methods. 
</div>


# Pretrained Model

Download [2X](https://drive.google.com/drive/u/0/folders/1-TM-IzzL9DqIetmdJmDNndBdGs4wmsSR) and [4X](https://drive.google.com/drive/u/0/folders/1eCJdxf_g-Zg7t02mdpztVxpImf-XodBe) pretrained models.

# Dataset

Download [RW-VSR-QA-Dataset](https://github.com/ajeetkverma/tta-vqa-vsr)


# Inference
1. Download the pre-trained weights to `checkpoints/`.
2. Run the following scrip:
   ```
   $bash test.sh
   ```
   Make sure to update ```test.yml``` file for all arguments.
3. Results will be stored in corresponding experiment ```result``` directory 

# Qualitative Results:
**Samples from K|Lens Dataset:**
## RealVQA-DDSR compared with [RealBasic-VSR](https://github.com/ajeetkverma/RealBasicVSR)

<p float="left">
  <img width="230" height="150" src="https://github.com/ajeetkverma/RealVQA-DDSR/blob/main/resource/slider1_rbvsr.png"> <a href="https://interacty.me/projects/68c7327276e9827f">
  <img width="230" height="150" src="https://github.com/ajeetkverma/RealVQA-DDSR/blob/main/resource/slider2_rbvsr.png"> <a href="https://interacty.me/projects/68c7327276e9827f">
  <img width="230" height="150" src="https://github.com/ajeetkverma/RealVQA-DDSR/blob/main/resource/slider3_rbvsr.png"> <a href="https://interacty.me/projects/68c7327276e9827f">
</p>
    
## RealVQA-DDSR compared with [SRWD](https://openaccess.thecvf.com/content/CVPR2023W/NTIRE/papers/Jeelani_Expanding_Synthetic_Real-World_Degradations_for_Blind_Video_Super_Resolution_CVPRW_2023_paper.pdf)

<p float="left">
  <img width="230" height="150" src="https://github.com/ajeetkverma/RealVQA-DDSR/blob/main/resource/slider1_srwd.png"> <a href="https://interacty.me/projects/ca92568308c4271d">
  <img width="230" height="150" src="https://github.com/ajeetkverma/RealVQA-DDSR/blob/main/resource/slider2_srwd.png"> <a href="https://interacty.me/projects/ca92568308c4271d">
  <img width="230" height="150" src="https://github.com/ajeetkverma/RealVQA-DDSR/blob/main/resource/slider3_srwd.png"> <a href="https://interacty.me/projects/ca92568308c4271d">
</p>

# Video Demonstration
**The videos have been compressed. Therefore, the results are inferior to that of the actual outputs.**

## Demo on RealVQA-DDSR vs [DAT](https://github.com/zhengchen1999/DAT)
https://github.com/ajeetkverma/RealVQA-DDSR/assets/33716142/38c9f8a3-4d90-42ec-b091-c11f641132e3

## Demo on RealVQA-DDSR vs [HAT](https://github.com/jiawangbai/HAT)
https://github.com/ajeetkverma/RealVQA-DDSR/assets/33716142/30a9d3e7-0a02-410e-a3de-a58381354ff4

## Demo on RealVQA-DDSR vs [RealESR](https://github.com/xinntao/Real-ESRGAN)
https://github.com/ajeetkverma/RealVQA-DDSR/assets/33716142/1d7de8d2-06b8-4a3d-b379-e15415412a7c 

## Demo on RealVQA-DDSR vs [SRWD](https://openaccess.thecvf.com/content/CVPR2023W/NTIRE/papers/Jeelani_Expanding_Synthetic_Real-World_Degradations_for_Blind_Video_Super_Resolution_CVPRW_2023_paper.pdf)
https://github.com/ajeetkverma/RealVQA-DDSR/assets/33716142/e51fe475-6bb4-47b0-a60d-a25ecf6f5775









## Visualisations from different datasets:

![Results Visualisation](https://github.com/ajeetkverma/RealVQA-DDSR/blob/main/resource/aj_ddsr_results.png)

# Citation
````

