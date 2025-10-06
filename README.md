# TTA-VQA-VSR
**_This work has been Accepted in IEEE Transactions on Artificial Intelligence._**

<h2>Real-World Video Quality Assessment via Test-Time Adaptation and its application in Real-World super-resolution

**<h2>We are working to prepare the release version of the code. It will be available soon...**



## Abstract

<div style="text-align: justify">
Recently, several real-world video super-resolution (RW-VSR) algorithms have addressed challenges such as noise, motion blur, and compression artifacts inherently present in low-resolution (LR) videos, which significantly impact the perceptual quality of super-resolved videos. However, there is still a considerable gap in developing domain-specific real-world video super-resolution quality assessment (RW-VSR-QA) methods that can effectively evaluate the perceptual quality of real-world super-resolved videos and seamlessly integrate into the RW-VSR pipeline to produce visually compelling high-resolution (HR) videos. To address this, we present:  
(a) A comprehensive real-world video dataset annotated with subjective quality scores, comprising 1155 videos with diverse resolutions and content variations. This dataset serves as a benchmark for developing RW-VSR-QA methods.  
(b) A test-time adaptation (TTA)-based algorithm for RW-VSR-QA that dynamically adapts models during inference using auxiliary tasks like group contrastive and quality-aware rank loss, improving robustness to unknown distortions without ground truth labels. Unlike traditional deep learning models, TTA flexibly handles new distortions in real-world videos. The quality-aware rank loss assumes cleaner videos have better perceptual quality, enabling the model to learn effectively from noise patterns.
(c) An adaptive perceptual quality loss function for RW-VSR, driven by the proposed RW-VSR-QA algorithm, which dynamically adjusts to video distortions and effectively guides RW-VSR algorithms to align with end-user expectations. 
</div>


# Pretrained Model

Download pretrained models. (Link will be available soon)

# Dataset
Write us on "vinit.jakhetiya@iitjammu.ac.in" to obtain the dataset.



# Inference
1. Download the pre-trained weights to `checkpoints/`.
2. Run the following scrip:
   ```
   $bash test.sh
   ```
   Make sure to update ```test.yml``` file for all arguments.
3. Results will be stored in corresponding experiment ```result``` directory 



# Citation
```
@article{verma2025real-world,
  title={Real-World Video Quality Assessment via Test-Time Adaptation and its application in Real-World super-resolution},
  author={Verma, Ajeet Kumar and Mishra, Ambuj and Jakhetiya, Vinit and Subudhi, Badri Narayan and Jaiswal, Sunil},
  journal={IEEE Transactions on Artificial Intelligence},
  volume={-},
  number={-},
  pages={}
  year={2025},
  publisher={IEEE}
}

```


