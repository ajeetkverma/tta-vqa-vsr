import torch
import random
import os.path as osp
import fastvqa.models as models
import fastvqa.datasets as datasets
from NAFNet.basicsr_nf.runner_aj import denoise_nf, deblur_nf
import argparse
import pandas as pd
# from skimage.util import random_noise
from PIL import Image
from scipy.stats import spearmanr, pearsonr
from scipy.stats.stats import kendalltau as kendallr
import numpy as np

import time
from tqdm import tqdm
import pickle
import math

# import wandb
import yaml
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch.nn as nn

from functools import reduce
from thop import profile
import copy



transform = transforms.Compose([
            transforms.CenterCrop(size=224),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                             std=(0.229, 0.224, 0.225))
        ])

class GroupContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(GroupContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, pairs):
        loss = 0
        for anchor, positive, label_a, label_b in pairs:
            distance = F.pairwise_distance(anchor.unsqueeze(0), positive.unsqueeze(0))
            if label_a == label_b:
                # Positive pair: minimize distance
                loss += torch.mean(distance)
            else:
                # Negative pair: maximize distance
                loss += torch.mean(F.relu(self.margin - distance))
        return loss 
    
def create_pairs(features, labels):
    """ Create pairs for contrastive loss """
    pairs = []
    num_samples = features.size(0)
    for i in range(num_samples):
        for j in range(i + 1, num_samples):
            pairs.append((features[i], features[j], labels[i], labels[j]))
    return pairs

def add_noise(tensor):
    sigma1 = 0.00005+ np.random.random() * 0.000001
    sigma2 = 0.00001+ np.random.random() * 0.000001  # noise
    noise_tesnsor=torch.stack([random_noise(tensor[i], mode='gaussian',var=sigma1) for i in range(tensor.size(0))])     
    return noise_tesnsor
def random_noise(tensor, mode='gaussian', var=0.01):
    if mode == 'gaussian':
        noise = torch.randn_like(tensor) * var  # Gaussian noise
    return tensor + noise


def compress(image):
    transform = transforms.Compose([
            transforms.CenterCrop(size=224),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                             std=(0.229, 0.224, 0.225))
        ])
    sigma1 = 40 + np.random.random() * 20  # 40-60
    sigma2 = 80 + np.random.random() * 10  # 80-90
    comprs_tensor = torch.stack([transform(image[i]) for i in range(image.size(0))]) 
    return comprs_tensor


def add_blur(tensor, kernel_size=5):

    sigma2 = 40 + np.random.random() * 20
    sigma1 = 5 + np.random.random() * 15
    # We use a kernel size of 5 for the blur (you can change this)
    blur = transforms.GaussianBlur(kernel_size=kernel_size, sigma=sigma1)
    tensor_blurred = torch.stack([blur(tensor[i]) for i in range(tensor.size(0))])  
    return tensor_blurred

def apply_random_distortions(tensor):
    # Randomly choose between the three distortions
    choice = random.choice([0, 1, 2])
    if choice == 0:
        tensor = add_noise(tensor)  # Apply Gaussian noise
    elif choice == 1:
        # tensor = compress(tensor)   # Apply compression
        tensor = tensor  
    elif choice == 2:
        tensor = add_blur(tensor)  # Apply Gaussian blur
    
    return tensor


def train_test_split(dataset_path, ann_file, ratio=0.8, seed=42):
    random.seed(seed)
    video_infos = []
    with open(ann_file, "r") as fin:
        for line in fin.readlines():
            line_split = line.strip().split(",")
            filename, _, _, label = line_split
            label = float(label)
            filename = osp.join(dataset_path, filename)
            video_infos.append(dict(filename=filename, label=label))
    random.shuffle(video_infos)
    return (
        video_infos[: int(ratio * len(video_infos))],
        video_infos[int(ratio * len(video_infos)) :],
    )


def rank_loss(y_pred, y):
    ranking_loss = torch.nn.functional.relu(
        (y_pred - y_pred.t()) * torch.sign((y.t() - y))
    )
    scale = 1 + torch.max(ranking_loss)
    return (
        torch.sum(ranking_loss) / y_pred.shape[0] / (y_pred.shape[0] - 1) / scale
    ).float()

def plcc_loss(y_pred, y):
    sigma_hat, m_hat = torch.std_mean(y_pred, unbiased=False)
    y_pred = (y_pred - m_hat) / (sigma_hat + 1e-8)
    sigma, m = torch.std_mean(y, unbiased=False)
    y = (y - m) / (sigma + 1e-8)
    loss0 = torch.nn.functional.mse_loss(y_pred, y) / 4
    rho = torch.mean(y_pred * y)
    loss1 = torch.nn.functional.mse_loss(rho * y_pred, y) / 4
    return ((loss0 + loss1) / 2).float()

def rescaled_l2_loss(y_pred, y):
    y_pred_rs = (y_pred - y_pred.mean()) / y_pred.std()
    y_rs = (y - y.mean()) / (y.std() + eps)
    return torch.nn.functional.mse_loss(y_pred_rs, y_rs)

def rplcc_loss(y_pred, y, eps=1e-8):
    ## Literally (1 - PLCC) / 2
    cov = torch.cov(y_pred, y)
    std = (torch.std(y_pred) + eps) * (torch.std(y) + eps)
    return (1 - cov / std) / 2

def self_similarity_loss(f, f_hat, f_hat_detach=False):
    if f_hat_detach:
        f_hat = f_hat.detach()
    return 1 - torch.nn.functional.cosine_similarity(f, f_hat, dim=1).mean()

def contrastive_similarity_loss(f, f_hat, f_hat_detach=False, eps=1e-8):
    if f_hat_detach:
        f_hat = f_hat.detach()
    intra_similarity = torch.nn.functional.cosine_similarity(f, f_hat, dim=1).mean()
    cross_similarity = torch.nn.functional.cosine_similarity(f, f_hat, dim=0).mean()
    return (1 - intra_similarity) / (1 - cross_similarity + eps)

def rescale(pr, gt=None):
    if gt is None:
        pr = (pr - np.mean(pr)) / np.std(pr)
    else:
        pr = ((pr - np.mean(pr)) / np.std(pr)) * np.std(gt) + np.mean(gt)
    return pr


sample_types=["resize", "fragments", "crop", "arp_resize", "arp_fragments"]


# def finetune_epoch(val_loaders, ft_loader, model_aj, model_ema_aj, optimizer, scheduler, device, epoch=3, 
#                    need_upsampled=False, need_feat=False, need_fused=False, need_separate_sup=False, key=None):
#     model_aj.train()
#     bests = {}
#     bests_n = {}
#     results = []
#     for i, data in enumerate(tqdm(ft_loader, desc=f"Adapting: ")):
#         for ep in range(epoch):
#             optimizer.zero_grad()
#             video = {}
#             video_c = {}
#             for key in sample_types:
#                 if key in data:
#                     video[key] = data[key].to(device)
#                     q_score, _ = model_aj(video, inference=True,
#                                     return_pooled_feats=True, 
#                                     reduce_scores=True)
                    
#                     if len(q_score) > 1:
#                         q_test = reduce(lambda x,y : x + y, q_score)
#                     else:
#                         q_test = q_score[0]
#                     q_test = q_test.mean((-3, -2, -1))
#                     # if (q_test > 0.6):
#                         # print("Adding noise!", data[key].shape)
#                     # video_c[key]= apply_random_distortions(data[key]).to(device)
#                     # else: 
#                     # else: 
#                     #     # print("Cleaning!", data[key].shape)
#                     if i%10 == 0:
#                         video_c[key] = deblur_nf(data[key]).to(device)
#                     else: video_c[key] = denoise_nf(data[key].to(device))
#                     # video_c[key] = denoise_nf(data[key].to(device))

#             # print(video['fragments'].shape, video_c['fragments'].shape)
            
            
#             if need_upsampled:
#                 up_video = {}
#                 for key in sample_types:
#                     if key+"_up" in data:
#                         up_video[key] = data[key+"_up"].to(device)
#             y = data["gt_label"].float().detach().to(device).unsqueeze(-1)
#             if need_feat:
#                 scores, feats = model_aj(video, inference=False,
#                                     return_pooled_feats=True, 
#                                     reduce_scores=True)
#                 scores_c, feats_c = model_aj(video_c, inference=False,
#                                     return_pooled_feats=True, 
#                                     reduce_scores=True) 
                      
                
#                 if (q_test > 0.6):
#                     flattened = torch.flatten(scores, start_dim=1)  # Shape: [12, 1*16*7*7]
#                     feature_vector = feats['fragments']
#                 else:
#                     flattened = torch.flatten(scores_c, start_dim=1)  # Shape: [12, 1*16*7*7]
#                     feature_vector = feats_c['fragments']

#                 reduced_sum = torch.sum(flattened, dim=1, keepdim=True)

#                 reduced_sum = reduced_sum.squeeze(1)
#                 sorted_ind = torch.argsort(reduced_sum, descending=True)
#                 sorted_fv = feature_vector[sorted_ind]
                
#                 top_features = sorted_fv[:3]
#                 bottom_features = sorted_fv[-3:]
#                 top_labels = torch.ones(top_features.size(0), device='cuda')  # Label 1 for top features
#                 bottom_labels = torch.zeros(bottom_features.size(0), device='cuda')  # Label 0 for bottom features
#                 top_pairs = create_pairs(top_features, top_labels)
#                 bottom_pairs = create_pairs(bottom_features, bottom_labels)
#                 all_pairs = top_pairs + bottom_pairs
#                 loss_fn = GroupContrastiveLoss(margin=1.0)

#                 # Compute loss
#                 gc_loss = loss_fn(all_pairs)
#                 loss = 0.4*gc_loss

#                 if len(scores) > 1:
#                     y_pred = reduce(lambda x,y:x+y, scores)
#                 else:
#                     y_pred = scores[0]
#                 y_pred = y_pred.mean((-3, -2, -1))
#                 ############################################
#                 if len(scores_c) > 1:
#                     y_pred_c = reduce(lambda x,y:x+y, scores_c)
#                 else:
#                     y_pred_c = scores_c[0]
#                 y_pred_c = y_pred_c.mean((-3, -2, -1))
#                 ##############################################

#             # else:
#             #     scores = model(video, inference=False,
#             #                           reduce_scores=False) 
#             #     if len(scores) > 1:
#             #         y_pred = reduce(lambda x,y:x+y, scores)
#             #     else:
#             #         y_pred = scores[0]
#             #     y_pred = y_pred.mean((-3, -2, -1))
#             #     ################################################
#             #     scores_c = model(video_c, inference=False,
#             #                           reduce_scores=False) 
#             #     if len(scores_c) > 1:
#             #         y_pred_c = reduce(lambda x,y:x+y, scores_c)
#             #     else:
#             #         y_pred_c = scores_c[0]
#             #     y_pred_c = y_pred_c.mean((-3, -2, -1))
#             #     ##################################################
#             # if need_upsampled:
#             #     if need_feat:
#             #         scores_up, feats_up = model_aj(up_video, inference=True, 
#             #                                     return_pooled_feats=True,
#             #                                     reduce_scores=False)
#             #         print(scores_up)
#             #         exit()
#             #         if len(scores) > 1:
#             #             y_pred_up = reduce(lambda x,y:x+y, scores_up)
#             #         else:
#             #             y_pred_up = scores_up[0]
#             #         y_pred_up = y_pred_up.mean((-3, -2, -1))
#             #     else:
#             #         y_pred_up = model_aj(up_video, inference=False).mean((-3, -2, -1))                                                           
#             # frame_inds = data["frame_inds"]
            

#             #adversarial contrastive loss
#             # print("Predicted: \n",y_pred)
#             # print("Predicted Clean: \n",y_pred_c)
#             # exit()

#             eps = 0.01
#             y_diff = y_pred_c - y_pred
#             # zeros_t = torch.zeros_like(y_diff, dtype=y_diff.dtype, device=y_diff.device, requires_grad=True)


#             # print(torch.sum(y_diff), torch.sum(zeros_t))
#             # exit()
#             # loss = abs(eps + torch.sum(y_diff))
#             if torch.sum(y_diff) > 0:
#                 # loss =  torch.sum(zeros_t)
#                 loss =  eps + torch.sum(y_diff)
#             else:
#                 loss =  eps - torch.sum(y_diff)

#             # if need_feat:
#             #     ## Self-Supervised Loss, Similarity between different sampling densities
#             #     for key in feats:
#             #         sim_loss = self_similarity_loss(feats[key], feats_up[key])
#             #         loss += 0.25 * sim_loss

#             loss.backward()
#             optimizer.step()
#             scheduler.step()
#         model_aj.eval()

#         result = dict()
#         video, video_up = {}, {}
#         for key in sample_types:
#             if key in data:
#                 video[key] = data[key].to(device)
#                 ## Reshape into clips
#                 b, c, t, h, w = video[key].shape
#                 # print(b, c, key, data["num_clips"][key], t, h, w)
#                 # exit()
#                 video[key] = video[key].reshape(b, c, data["num_clips"][key], t // data["num_clips"][key], h, w).permute(0,2,1,3,4,5).reshape(b * data["num_clips"][key], c, t // data["num_clips"][key], h, w) 
#             if key + "_up" in data:
#                 video_up[key] = data[key+"_up"].to(device)
#                 ## Reshape into clips
#                 b, c, t, h, w = video_up[key].shape
#                 video_up[key] = video_up[key].reshape(b, c, data["num_clips"], t // data["num_clips"], h, w).permute(0,2,1,3,4,5).reshape(b * data["num_clips"], c, t // data["num_clips"], h, w) 
        
#         with torch.no_grad():
#             result["pr_labels"] = model_aj(video).cpu().numpy()
#             if len(list(video_up.keys())) > 0:
#                 result["pr_labels_up"] = model_aj(video_up).cpu().numpy()

#         result["gt_label"] = data["gt_label"].item()
#         del video, video_up
#         # result['frame_inds'] = data['frame_inds']
#         # del data
#         results.append(result)

#         # if i==99: break
#     print(f"Adapted for {epoch} epochs.")


def finetune_epoch(val_loaders, ft_loader, model_aj, model_ema_aj, optimizer, scheduler, device, epoch=3, 
                   need_upsampled=False, need_feat=False, need_fused=False, need_separate_sup=False, key=None):
    
    model_aj.train()
    bests = {}
    bests_n = {}
    results = []

    total_times = []

    for i, data in enumerate(tqdm(ft_loader, desc=f"Adapting: ")):
        sample_start_time = time.time()

        # === Start adaptation timing ===
        torch.cuda.reset_peak_memory_stats()
        adapt_start_time = time.time()

        for ep in range(epoch):
            optimizer.zero_grad()

            video = {}
            video_c = {}
            for key in sample_types:
                if key in data:
                    video[key] = data[key].to(device)
                    q_score, _ = model_aj(video, inference=True,
                                          return_pooled_feats=True, 
                                          reduce_scores=True)
                    q_test = reduce(lambda x, y: x + y, q_score) if len(q_score) > 1 else q_score[0]
                    q_test = q_test.mean((-3, -2, -1))

                    if i % 10 == 0:
                        video_c[key] = deblur_nf(data[key]).to(device)
                    else:
                        video_c[key] = denoise_nf(data[key].to(device))

            if need_upsampled:
                up_video = {key: data[key+"_up"].to(device) for key in sample_types if key+"_up" in data}
            
            y = data["gt_label"].float().detach().to(device).unsqueeze(-1)

            # print(video['fragments'].shape)
            # import matplotlib.pyplot as plt
            # frames = video['fragments'][0]
            # num_frames_to_show = 8
            # plt.figure(figsize=(20, 5))

            # for i in range(num_frames_to_show):
            #     frame = frames[:, i]  # shape: [3, 224, 224]
            #     frame = frame.permute(1, 2, 0)  # [H, W, C] for matplotlib
            #     plt.subplot(1, num_frames_to_show, i + 1)
            #     plt.imshow(frame.detach().cpu())
            #     plt.axis("off")
            #     plt.title(f"Frame {i}")

            # plt.tight_layout()
            # plt.show()
            

            # exit()

            if need_feat:
                scores, feats = model_aj(video, inference=False,
                                         return_pooled_feats=True, 
                                         reduce_scores=True)
                scores_c, feats_c = model_aj(video_c, inference=False,
                                             return_pooled_feats=True, 
                                             reduce_scores=True)
                
                flattened = torch.flatten(scores if (q_test > 0.6) else scores_c, start_dim=1)
                feature_vector = feats['fragments'] if (q_test > 0.6) else feats_c['fragments']

                reduced_sum = torch.sum(flattened, dim=1, keepdim=True).squeeze(1)
                sorted_ind = torch.argsort(reduced_sum, descending=True)
                sorted_fv = feature_vector[sorted_ind]

                top_features = sorted_fv[:3]
                bottom_features = sorted_fv[-3:]
                top_labels = torch.ones(top_features.size(0), device='cuda')
                bottom_labels = torch.zeros(bottom_features.size(0), device='cuda')
                top_pairs = create_pairs(top_features, top_labels)
                bottom_pairs = create_pairs(bottom_features, bottom_labels)
                all_pairs = top_pairs + bottom_pairs
                loss_fn = GroupContrastiveLoss(margin=1.0)

                gc_loss = loss_fn(all_pairs)
                loss = 0.4 * gc_loss

                y_pred = reduce(lambda x, y: x + y, scores) if len(scores) > 1 else scores[0]
                y_pred = y_pred.mean((-3, -2, -1))

                y_pred_c = reduce(lambda x, y: x + y, scores_c) if len(scores_c) > 1 else scores_c[0]
                y_pred_c = y_pred_c.mean((-3, -2, -1))
            else:
                # Add default loss if needed
                pass

            eps = 0.01
            y_diff = y_pred_c - y_pred
            loss = eps + torch.sum(y_diff) if torch.sum(y_diff) > 0 else eps - torch.sum(y_diff)

            loss.backward()
            optimizer.step()
            scheduler.step()

        adapt_end_time = time.time()
        adapt_peak_memory = torch.cuda.max_memory_allocated(device) / (1024 ** 2)  # in MB

        model_aj.eval()

        # === Start inference timing ===
        torch.cuda.reset_peak_memory_stats()
        inference_start_time = time.time()

        result = dict()
        video, video_up = {}, {}

        for key in sample_types:
            if key in data:
                video[key] = data[key].to(device)

                b, c, t, h, w = video[key].shape
                video[key] = video[key].reshape(
                    b, c, data["num_clips"][key], t // data["num_clips"][key], h, w
                ).permute(0, 2, 1, 3, 4, 5).reshape(
                    b * data["num_clips"][key], c, t // data["num_clips"][key], h, w
                )

            if key + "_up" in data:
                video_up[key] = data[key + "_up"].to(device)
                b, c, t, h, w = video_up[key].shape
                video_up[key] = video_up[key].reshape(
                    b, c, data["num_clips"], t // data["num_clips"], h, w
                ).permute(0, 2, 1, 3, 4, 5).reshape(
                    b * data["num_clips"], c, t // data["num_clips"], h, w
                )

        with torch.no_grad():
            result["pr_labels"] = model_aj(video).cpu().numpy()
            if video_up:
                result["pr_labels_up"] = model_aj(video_up).cpu().numpy()

        result["gt_label"] = data["gt_label"].item()
        results.append(result)

        inference_end_time = time.time()
        inference_peak_memory = torch.cuda.max_memory_allocated(device) / (1024 ** 2)  # in MB

        sample_end_time = time.time()

        adaptation_time = adapt_end_time - adapt_start_time
        inference_time = inference_end_time - inference_start_time
        total_time = sample_end_time - sample_start_time

        print(f"[Sample {i}] Adaptation Time: {adaptation_time:.3f}s | "
              f"Inference Time: {inference_time:.3f}s | "
              f"Total Time: {total_time:.3f}s | "
              f"Adapt Peak Mem: {adapt_peak_memory:.1f} MB | "
              f"Infer Peak Mem: {inference_peak_memory:.1f} MB")

        total_times.append({
            'sample_index': i,
            'adaptation_time': adaptation_time,
            'inference_time': inference_time,
            'total_time': total_time,
            'adapt_peak_memory_MB': adapt_peak_memory,
            'inference_peak_memory_MB': inference_peak_memory
        })

    print(f"Adapted for {epoch} epochs.")

    gt_labels = [r["gt_label"] for r in results]
    pr_labels = [np.mean(r["pr_labels"][:]) for r in results]
    pr_labels = rescale(pr_labels, gt_labels)

    s = spearmanr(gt_labels, pr_labels)[0]
    p = pearsonr(gt_labels, pr_labels)[0]
    k = kendallr(gt_labels, pr_labels)[0]
    r = np.sqrt(((gt_labels - pr_labels) ** 2).mean())

    df = pd.DataFrame({'gt_labels': gt_labels, 'pr_labels': pr_labels})
    df.to_csv(f'results/fvqa_tta_predictions_dndb_{round(s*10000)}.csv', index=False)

    print(f" ada_SRCC: {s}\n ada_PLCC: {p}\n ada_KRCC: {k}\n ada_RMSE: {r}")


    
def profile_inference(inf_set, model, device):
    
    video = {}
    data = inf_set[0]
    for key in sample_types:
        if key in data:
            video[key] = data[key].to(device).unsqueeze(0)
    with torch.no_grad():
        flops, params = profile(model, (video, ))
    print(f"The FLOps of the Variant is {flops} == {flops/1e9:.1f}G, with Params {params} == {params/1e6:.2f}M.")


def reshape_video(data, key, device, num_clips):
    """Helper function to reshape video or video_up tensor for each num_clip."""
    video = data[key].to(device)
    b, c, t, h, w = video.shape

    # Reshape for each num_clip individually (based on num_clips)
    reshaped_video = video.reshape(b, c, num_clips, t // num_clips, h, w) \
                        .permute(0, 2, 1, 3, 4, 5) \
                        .reshape(b * num_clips, c, t // num_clips, h, w)
    return reshaped_video


def inference_set(inf_loader, model, device, save_model=False, suffix='s', save_name="divide"):


    # best_s, best_p, best_k, best_r = best_
 
    results = []
    for i, data in enumerate(tqdm(inf_loader, desc="Validating")):
        result = dict()
        video, video_up = {}, {}
        for key in sample_types:
            if key in data:
                video[key] = data[key].to(device)
                ## Reshape into clips
                b, c, t, h, w = video[key].shape
                video[key] = video[key].reshape(b, c, data["num_clips"][key], t // data["num_clips"][key], h, w).permute(0,2,1,3,4,5).reshape(b * data["num_clips"][key], c, t // data["num_clips"][key], h, w) 
            if key + "_up" in data:
                video_up[key] = data[key+"_up"].to(device)
                ## Reshape into clips
                b, c, t, h, w = video_up[key].shape
                video_up[key] = video_up[key].reshape(b, c, data["num_clips"], t // data["num_clips"], h, w).permute(0,2,1,3,4,5).reshape(b * data["num_clips"], c, t // data["num_clips"], h, w) 
            #.unsqueeze(0)
        with torch.no_grad():
            result["pr_labels"] = model(video).cpu().numpy()
            if len(list(video_up.keys())) > 0:
                result["pr_labels_up"] = model(video_up).cpu().numpy()
                
        result["gt_label"] = data["gt_label"].item()
        del video, video_up
        # result['frame_inds'] = data['frame_inds']
        # del data
        results.append(result)
        

    # results = []
    # for i, data in enumerate(tqdm(inf_loader, desc="Validating from inference.")):
    #     result = dict()
    #     for key in sample_types:
    #         video, video_up = {}, {}
    #         if key in data:
    #             print(data)
    #             exit()
    #             for i in range(data[key].shape[0]):
    #                 # video[key] = data[key]
    #                 video[key] = data[key][i:i+1, :, 0:1, :, :, :].to(device)
    #                 ## Reshape into clips
    #                 b, c, t, h, w = video[key].shape
    #                 video[key] = video[key].reshape(b, c, data["num_clips"][key], t // data["num_clips"][key], h, w).permute(0,2,1,3,4,5).reshape(b * data["num_clips"][key], c, t // data["num_clips"][key], h, w) 
    #         if key + "_up" in data:
    #             for i in range(data[key].shape[0]):
    #                 # print(f'{key}_up shape: {data[key+"_up"].shape}')
    #                 # exit()
    #                 # video_up[key] = data[key+"_up"].to(device)
    #                 video_up[key] = data[key+"_up"][i:i+1, :, 0:1, :, :, :].to(device)
    #                 ## Reshape into clips
    #                 b, c, t, h, w = video_up[key].shape
    #                 video_up[key] = video_up[key].reshape(b, c, data["num_clips"], t // data["num_clips"], h, w).permute(0,2,1,3,4,5).reshape(b * data["num_clips"], c, t // data["num_clips"], h, w) 
    #         #.unsqueeze(0)
    #         print(video)
    #         exit()
    #         with torch.no_grad():
    #             result["pr_labels"] = model(video).cpu().numpy()
    #             if len(list(video_up.keys())) > 0:
    #                 result["pr_labels_up"] = model(video_up).cpu().numpy()
                    
    #         result["gt_label"] = data["gt_label"].item()
    #         del video, video_up
    #         # result['frame_inds'] = data['frame_inds']
    #         # del data
    #         results.append(result)

    ## generate the demo video for video quality localization
    gt_labels = [r["gt_label"] for r in results]
    pr_labels = [np.mean(r["pr_labels"][:]) for r in results]
    pr_labels = rescale(pr_labels, gt_labels)


    s = spearmanr(gt_labels, pr_labels)[0]
    p = pearsonr(gt_labels, pr_labels)[0]
    k = kendallr(gt_labels, pr_labels)[0]
    r = np.sqrt(((gt_labels - pr_labels) ** 2).mean())

    print(f"val_{suffix}/SRCC-{suffix}: {s}\n, val_{suffix}/PLCC-{suffix}: {p}\n, val_{suffix}/KRCC-{suffix}: {k}\n, val_{suffix}/RMSE-{suffix}: {r}")
    print(gt_labels)
    print(pr_labels)    
    # wandb.log({f"val_{suffix}/SRCC-{suffix}": s, f"val_{suffix}/PLCC-{suffix}": p, f"val_{suffix}/KRCC-{suffix}": k, f"val_{suffix}/RMSE-{suffix}": r})
    
    
    # if "pr_labels_up" in results[0]:
    #     pr_labels_up = [np.mean(r["pr_labels_up"][:]) for r in results]
    #     pr_labels_up = rescale(pr_labels_up, gt_labels)

    #     ups = spearmanr(gt_labels, pr_labels_up)[0]
    #     upp = pearsonr(gt_labels, pr_labels_up)[0]
    #     upk = kendallr(gt_labels, pr_labels_up)[0]
    #     upr = np.sqrt(((gt_labels - pr_labels_up) ** 2).mean())

        # wandb.log({f"val_{suffix}/up-SRCC-{suffix}": ups, f"val_{suffix}/up-PLCC-{suffix}": upp, f"val_{suffix}/up-KRCC-{suffix}": upk, f"val_{suffix}/up-RMSE-{suffix}": upr})
        
    # del results, result #, video, video_up
    # torch.cuda.empty_cache()

    # if s + p > best_s + best_p and save_model:
    #     state_dict = model.state_dict()
    #     torch.save(
    #         {
    #             "state_dict": state_dict,
    #             "validation_results": best_,
    #         },
    #         f"pretrained_weights/{save_name}_{suffix}_{int(s*1000)}.pth",
    #     )

    # best_s, best_p, best_k, best_r = (
    #     max(best_s, s),
    #     max(best_p, p),
    #     max(best_k, k),
    #     min(best_r, r),
    # )

    # wandb.log(
    #     {
    #         f"val_{suffix}/best_SRCC-{suffix}": best_s,
    #         f"val_{suffix}/best_PLCC-{suffix}": best_p,
    #         f"val_{suffix}/best_KRCC-{suffix}": best_k,
    #         f"val_{suffix}/best_RMSE-{suffix}": best_r,
    #     }
    # )

    # print(
    #     f"For {len(inf_loader)} videos, \nthe accuracy of the model: [{suffix}] is as follows:\n  SROCC: {s:.4f} best: {best_s:.4f} \n  PLCC:  {p:.4f} best: {best_p:.4f}  \n  KROCC: {k:.4f} best: {best_k:.4f} \n  RMSE:  {r:.4f} best: {best_r:.4f}."
    # )

    return gt_labels, pr_labels
    # return best_s, best_p, best_k, best_r

    # torch.save(results, f'{args.save_dir}/results_{dataset.lower()}_s{32}*{32}_ens{args.famount}.pkl')


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-o", "--opt", type=str, default="./options/divide/mradd.yml", help="the option file"
    )

    args = parser.parse_args()
    with open(args.opt, "r") as f:
        opt = yaml.safe_load(f)
    # print(opt)
    
    ## adaptively choose the device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = getattr(models, opt["model"]["type"])(**opt["model"]["args"]).to(device)
    
    val_datasets = {}
    for key in opt["data"]:
        if key.startswith("val"):
            val_dataset = getattr(datasets, opt["data"][key]["type"])(opt["data"][key]["args"])
            print(len(val_dataset.video_infos))
            val_datasets[key] = val_dataset
            
    val_loaders_tta = {}
    for key, val_dataset in val_datasets.items():
        val_loaders_tta[key] = torch.utils.data.DataLoader(
            # val_dataset, batch_size=1, num_workers=opt["num_workers"], pin_memory=True,
            val_dataset, batch_size=opt["batch_size"], num_workers=opt["num_workers"], pin_memory=True,
        )
    
    val_loaders_inf = {}
    for key, val_dataset in val_datasets.items():
        val_loaders_inf[key] = torch.utils.data.DataLoader(
            # val_dataset, batch_size=1, num_workers=opt["num_workers"], pin_memory=True,
            val_dataset, batch_size=1, num_workers=opt["num_workers"], pin_memory=True,
        )

    if "load_path_aux" in opt:
        state_dict = torch.load(opt["load_path"], map_location=device)["state_dict"]
        aux_state_dict = torch.load(opt["load_path_aux"], map_location=device)["state_dict"]

        from collections import OrderedDict

        fusion_state_dict = OrderedDict()
        for k, v in state_dict.items():
            if "head" in k:
                continue
            if k.startswith("vqa_head"):
                ki = k.replace("vqa", "fragments")
            else:
                ki = k
            fusion_state_dict[ki] = v

        for k, v in aux_state_dict.items():
            if "head" in k:
                continue
            if k.startswith("frag"):
                continue
            if k.startswith("vqa_head"):
                ki = k.replace("vqa", "resize")
            else:
                ki = k
            fusion_state_dict[ki] = v
        state_dict = fusion_state_dict
        print(model.load_state_dict(state_dict,strict=False))
    elif "load_path" in opt:
        state_dict = torch.load(opt["load_path"], map_location=device)

        if "state_dict" in state_dict:
            ### migrate training weights from mmaction / F-adaptation / LSVQ-pretrain
            state_dict = state_dict["state_dict"]
            from collections import OrderedDict

            i_state_dict = OrderedDict()
            for key in state_dict.keys():
                if "cls" in key:
                    tkey = key.replace("cls", "vqa")
                elif "backbone" in key and "_backbone" not in key:
                    i_state_dict["fragments_"+key] = state_dict[key]
                    i_state_dict["resize_"+key] = state_dict[key]
                else:
                    i_state_dict[key] = state_dict[key]
        t_state_dict = model.state_dict()
        for key, value in t_state_dict.items():
            if key in i_state_dict and i_state_dict[key].shape != value.shape:
                i_state_dict.pop(key)
        
        model.load_state_dict(i_state_dict, strict=False)
        # print(model.load_state_dict(i_state_dict, strict=False))
        

    if opt["ema"]:
        from copy import deepcopy
        model_ema_aj = deepcopy(model)
    else:
        model_ema_aj = None

    # to access the FLOps and Params.
    profile_inference(val_dataset, model, device)   

    param_groups=[]

    for key, value in dict(model.named_children()).items():
        if "backbone" in key:
            param_groups += [{"params": value.parameters(), "lr": opt["optimizer"]["lr"] * opt["optimizer"]["backbone_lr_mult"]}]
        else:
            param_groups += [{"params": value.parameters(), "lr": opt["optimizer"]["lr"]}]

    optimizer = torch.optim.AdamW(lr=opt["optimizer"]["lr"], params=param_groups,
                                    weight_decay=opt["optimizer"]["wd"],
                                    )
    warmup_iter = 0
    for val_loader in val_loaders_tta.values():
        warmup_iter += int(opt["warmup_epochs"] * len(val_loader))
    max_iter = int((opt["num_epochs"] + opt["l_num_epochs"]) * len(val_loader))
    lr_lambda = (
        lambda cur_iter: cur_iter / warmup_iter
        if cur_iter <= warmup_iter
        else 0.5 * (1 + math.cos(math.pi * (cur_iter - warmup_iter) / max_iter))
    )

    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lr_lambda,
    )

    bests = {}
    bests_n = {}
    for key in val_loaders_inf:
        bests[key] = -1,-1,-1,1000
        bests_n[key] = -1,-1,-1,1000

    from copy import deepcopy

    model_aj = deepcopy(model)

    for key, value in dict(model_aj.named_children()).items():
        if "backbone" in key:
            for param in value.parameters():
                            param.requires_grad = False
            for par in value.named_children():
                if "norm" in par:
                    for param in value.parameters():
                            param.requires_grad = True

    # for epoch in range(opt["tta_epochs"]):
    epoch = opt["tta_epochs"]
    print(f"Adapting for {epoch} Epochs:")
    for key, val_loader in val_loaders_tta.items():
        finetune_epoch(
            val_loaders_tta, val_loader, model_aj, model_ema_aj, optimizer, scheduler, device, epoch,
            opt.get("need_upsampled", False), opt.get("need_feat", True), opt.get("need_fused", False), key
        )
    



if __name__ == "__main__":
    main()
