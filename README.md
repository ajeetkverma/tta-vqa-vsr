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

Download [RW-VSR-QA-Dataset](https://ieee-dataport.org/documents/rw-vqa-dataset) from IEEE-Dataport.


# Inference
1. Download the pre-trained weights to `checkpoints/`.
2. Run the following scrip:
   ```
   $bash test.sh
   ```
   Make sure to update ```test.yml``` file for all arguments.
3. Results will be stored in corresponding experiment ```result``` directory 



# Citation
````

