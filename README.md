# RealVQA-DDSR
**_This work has been Accepted for publication in CVPR 2024._**

<h2>RealVQA-DDSR: Real World Video Quality Assessment assisted Joint Motion Deblurring, Denoising, and Super-Resolution

**<h2>The code will be released soon...**
<p>
  <img width="640" height="460" src="https://www.jaggaer.com/app/uploads/2022/09/Jaggaer-Software-Releases-1048x640.jpg">
</p>



## Abstract

<div style="text-align: justify">
Video super-resolution (VSR) involves enhancing the resolution of low-resolution (LR) videos to high-resolution (HR) ones. To address the problem of real-world VSR, most of the existing algorithms simulate real-world degradation on the synthetic training datasets and train an end-to-end deep learning algorithm. However, these methods overlook motion blur simulation, resulting in reduced performance when handling videos affected by motion blur. This paper focuses on the intricate challenge of motion-blurred VSR, aiming to simultaneously enhance resolution and address deblurring/denoising in HR videos. Our strategy involves embedding motion blur degradation within the simulation framework and integrating a unique loss function derived from the real-world VSR quality assessment (RW-VSR-QA) algorithm, combined with denoising loss. Also, to suppress the artifacts produced due to different loss functions, we propose a simple solution by integrating denoising loss in the training pipeline. Specifically, we introduce a new dataset comprising 1155 videos with subjective scores and propose a new quality assessment algorithm for RW-VSR employing standardized negative similarity loss. These videos are generated from 11 recently proposed video super-resolution algorithms. Moreover, we propose a new pipeline for the purpose of generating real-world LR videos encompassing prevalent distortions like noise, blur, downsampling, pixel binning, compression artifacts, and motion blur, alongside the RW-VSR-QA algorithm. Both the proposed RW-VSR and RW-VSR-QA algorithms exhibit superior performance compared to existing methods.
</div>


# Pretrained Model

Download [2X](https://drive.google.com/drive/u/0/folders/1-TM-IzzL9DqIetmdJmDNndBdGs4wmsSR) and [4X](https://drive.google.com/drive/u/0/folders/1eCJdxf_g-Zg7t02mdpztVxpImf-XodBe) pretrained models.


# Inference



# Qualitative Results:
**Samples from K|Lens Dataset:**
## RealVQA-DDSR compared with [RealBasic-VSR](https://github.com/ajeetkverma/RealBasicVSR)

<p float="left">
  <img width="230" height="150" src="https://github.com/ajeetkverma/RealVQA-DDSR/blob/main/resource/slider1_rbvsr.png"> <a href="https://interacty.me/projects/68c7327276e9827f">
  <img width="230" height="150" src="https://github.com/ajeetkverma/RealVQA-DDSR/blob/main/resource/slider2_rbvsr.png"> <a href="https://interacty.me/projects/68c7327276e9827f">
  <img width="230" height="150" src="https://github.com/ajeetkverma/RealVQA-DDSR/blob/main/resource/slider3_rbvsr.png"> <a href="https://interacty.me/projects/68c7327276e9827f">
</p>




**Samples from K|Lens Dataset:**
## RealVQA-DDSR compared with [SRWD](https://openaccess.thecvf.com/content/CVPR2023W/NTIRE/papers/Jeelani_Expanding_Synthetic_Real-World_Degradations_for_Blind_Video_Super_Resolution_CVPRW_2023_paper.pdf)

<p float="left">
  <img width="230" height="150" src="https://github.com/ajeetkverma/RealVQA-DDSR/blob/main/resource/slider1_srwd.png"> <a href="https://interacty.me/projects/ca92568308c4271d">
  <img width="230" height="150" src="https://github.com/ajeetkverma/RealVQA-DDSR/blob/main/resource/slider2_srwd.png"> <a href="https://interacty.me/projects/ca92568308c4271d">
  <img width="230" height="150" src="https://github.com/ajeetkverma/RealVQA-DDSR/blob/main/resource/slider3_srwd.png"> <a href="https://interacty.me/projects/ca92568308c4271d">
</p>

# Video Demonstration

| <img width="230" height="150" src="https://github.com/ajeetkverma/RealVQA-DDSR/blob/main/resource/slider1_srwd.png"> <a href="https://interacty.me/projects/ca92568308c4271d">  | <img width="230" height="150" src="https://github.com/ajeetkverma/RealVQA-DDSR/blob/main/resource/slider1_srwd.png"> <a href="https://interacty.me/projects/ca92568308c4271d"> |
| <img width="230" height="150" src="https://github.com/ajeetkverma/RealVQA-DDSR/blob/main/resource/slider1_srwd.png"> <a href="https://interacty.me/projects/ca92568308c4271d">  | <img width="230" height="150" src="https://github.com/ajeetkverma/RealVQA-DDSR/blob/main/resource/slider1_srwd.png"> <a href="https://interacty.me/projects/ca92568308c4271d"> |
| ------------- | ------------- |
|  |  |




[<img width="640" height="460" src="https://github.com/ajeetkverma/RealVQA-DDSR/assets/33716142/38c9f8a3-4d90-42ec-b091-c11f641132e3" width="50%">](https://github.com/ajeetkverma/RealVQA-DDSR/assets/33716142/38c9f8a3-4d90-42ec-b091-c11f641132e3)
[<img width="640" height="460" src="https://github.com/ajeetkverma/RealVQA-DDSR/assets/33716142/30a9d3e7-0a02-410e-a3de-a58381354ff4" width="50%">](https://github.com/ajeetkverma/RealVQA-DDSR/assets/33716142/30a9d3e7-0a02-410e-a3de-a58381354ff4)
[<img width="640" height="460" src="https://github.com/ajeetkverma/RealVQA-DDSR/assets/33716142/1d7de8d2-06b8-4a3d-b379-e15415412a7c" width="50%">](https://github.com/ajeetkverma/RealVQA-DDSR/assets/33716142/1d7de8d2-06b8-4a3d-b379-e15415412a7c) 
[<img width="640" height="460" src="https://github.com/ajeetkverma/RealVQA-DDSR/assets/33716142/e51fe475-6bb4-47b0-a60d-a25ecf6f5775" width="50%">](https://github.com/ajeetkverma/RealVQA-DDSR/assets/33716142/e51fe475-6bb4-47b0-a60d-a25ecf6f5775)








## Visualisations from different datasets:

![Results Visualisation](https://github.com/ajeetkverma/RealVQA-DDSR/blob/main/resource/aj_ddsr_results.png)

# Citation
````
@inproceedings{verma2024realvqaddsr,
  author = {Verma, Ajeet K. and Mishra, Ambuj and Ahmad, Faizan S. and Thakur, Sadbhawana and Jaiswal, Sunil and Jakhetiya, Vinit},
  title = {RealVQA-DDSR: Real World Video Quality Assessment assisted Joint Motion Deblurring, Denoising, and Super-Resolution},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={------},
  year={2024}
}
