import torch
import cv2
import random
import os
import os.path as osp
import fastvqa.models as models
import fastvqa.datasets as datasets
import argparse
from scipy.stats import spearmanr, pearsonr
from scipy.stats.stats import kendalltau as kendallr
import numpy as np
from time import time
from tqdm import tqdm
from keras.models import load_model
import glob
import yaml
from thop import profile
from utils import *
import shutil
import torch.backends.cudnn as cudnn

def rescale(pr, gt=None):
    if gt is None:
        print("mean", np.mean(pr), "std", np.std(pr))
        pr = (pr - np.mean(pr)) / np.std(pr)
    else:
        print(np.mean(pr), np.std(pr), np.std(gt), np.mean(gt))
        pr = ((pr - np.mean(pr)) / np.std(pr)) * np.std(gt) + np.mean(gt)
    return pr

sample_types=["resize", "fragments", "crop", "arp_resize", "arp_fragments"]


def profile_inference(inf_set, model, device):
    video = {}
    data = inf_set[0]
    for key in sample_types:
        if key in data:
            video[key] = data[key].to(device)
            c, t, h, w = video[key].shape
            video[key] = video[key].reshape(1, c, data["num_clips"][key], t // data["num_clips"][key], h, w).permute(0,2,1,3,4,5).reshape( data["num_clips"][key], c, t // data["num_clips"][key], h, w) 
    with torch.no_grad():
        flops, params = profile(model, (video, ))
    print(f"The FLOps of the Variant is {flops/1e9:.1f}G, with Params {params/1e6:.2f}M.")


def save_frame_as_video(frame, output_path, frame_rate, num_frames):
    height, width, _ = frame.shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_path, fourcc, frame_rate, (width, height))
    for _ in range(num_frames):
        video_writer.write(frame)
    video_writer.release()


def process_video(video_data, dataset, key, model, device):
    """ Helper function to process video data """
    video_data['fragments'] = dataset[key]['fragments'].to(device).unsqueeze(0)
    b, c, t, h, w = video_data["fragments"].shape
    num_clips = dataset[key]["num_clips"]["fragments"]
    
    # Reshape and permute
    video_data['fragments'] = (
        video_data['fragments']
        .reshape(b, c, num_clips, t // num_clips, h, w)
        .permute(0, 2, 1, 3, 4, 5)
        .reshape(b * num_clips, c, t // num_clips, h, w)
    )
    
    # Run model and return the averaged features
    _, feats = model(video_data, return_pooled_feats=True, reduce_scores=False)
    
    # features = feats['fragments'].cpu().numpy()
    # # Convert features to a 2D matrix (4x768) and plot heatmap
    # sns.heatmap(features, annot=False, cmap="viridis")
    # plt.xlabel("Feature Index")
    # plt.ylabel("Embedding Index")
    # plt.title("Heatmap of Embeddings")
    # plt.show()


    # for i, embedding in enumerate(features):
    #     print(f"Embedding {i}: Mean={embedding.mean().item()}, Std={embedding.std().item()}, Min={embedding.min().item()}, Max={embedding.max().item()}")
    # return torch.mean(feats['fragments'], dim=0, keepdim=True)
    return (feats['fragments'].view(1,-1))


def tta_aj(data, opt, model, video, device):
    quality_dict = {"args": {
                            "phase": "test",
                            "anno_file": " ",
                            "data_prefix": " ",
                            "sample_types": {
                                "fragments": {
                                    "fragments_h": 7,
                                    "fragments_w": 7,
                                    "fsize_h": 32,
                                    "fsize_w": 32,
                                    "aligned": 32,
                                    "clip_len": 32,
                                    "frame_interval": 2,
                                    "num_clips": 4
                                }
                            }
                        }
                    }

    M, N = 160, 160
    frame_rate = 25
    num_frames = int(frame_rate * 3)

    basic_folder = 'Dataset/VQADataset/all_data'

    param_groups=[]
    for key, value in dict(model.named_children()).items():
        if "backbone" in key:
            param_groups += [{"params": value.parameters(), "lr": opt["optimizer"]["lr"] * opt["optimizer"]["backbone_lr_mult"]}]

    # optimizer = torch.optim.AdamW(lr=opt["optimizer"]["lr"], params=param_groups,
    #                                   weight_decay=opt["optimizer"]["wd"],
    #                                  )
    optimizer= torch.optim.Adam(param_groups, lr = 0.0001)
    # Get all frame files inside the current video folder
    video_name = os.path.splitext(data["name"][0])[0]
    img_folder = osp.join(basic_folder, video_name)
    lq_path, hq_path = os.path.join(basic_folder, 'lq'), os.path.join(basic_folder, 'hq')

    # Delete existing lq and hq folders if they exist
    if osp.exists(lq_path):
        shutil.rmtree(lq_path)
    if osp.exists(hq_path):
        shutil.rmtree(hq_path)

    # Create new directories
    os.makedirs(lq_path, exist_ok=True)
    os.makedirs(hq_path, exist_ok=True)
    
    test_files = glob.glob(osp.join(img_folder, '*.jpg'))
    test_files.sort()
            

    quality_dict["args"]["data_prefix"] = lq_path
    lq_dataset = getattr(datasets, "FusionDataset")(quality_dict["args"])
    quality_dict['args']['data_prefix'] = hq_path
    hq_dataset = getattr(datasets, "FusionDataset")(quality_dict["args"])

    for params in model.parameters():
        params.requires_grad = False

    for layer in model.fragments_backbone.modules():
        if isinstance(layer, nn.BatchNorm3d) or isinstance(layer, nn.BatchNorm2d) or isinstance(layer, nn.BatchNorm1d) or isinstance(layer, nn.LayerNorm):
            # layer.train()
            layer.requires_grad_(True)

    # model.eval()
    # model.fragments_backbone.train()

    model.train()
    
    loss_history = []
    for iteration in range(3):
        lq_video = {}
        hq_video = {}
        avg_lq_feats = []
        avg_hq_feats = []
        for key in range(3):
            # Process low-quality video
            avg_lq_feats.append((process_video(lq_video, lq_dataset, key, model, device)).requires_grad_(True))

            # Process high-quality video
            avg_hq_feats.append((process_video(hq_video, hq_dataset, key, model, device)).requires_grad_(True))

        avg_lq_feats = torch.squeeze(torch.stack(avg_lq_feats), dim=1)
        avg_hq_feats = torch.squeeze(torch.stack(avg_hq_feats), dim=1)  
        
        loss_fn = GroupContrastiveLoss(avg_hq_feats.shape[0], 0.1).cuda()

        loss = loss_fn(avg_lq_feats, avg_hq_feats)

        loss.backward()
        optimizer.step()
        loss_history.append(loss.detach().cpu())

    print(loss_history)
    labels = test_single_video(model, video)
    return labels


def test_single_video(model, video):
    model.eval()
    with torch.no_grad():
        labels = model(video, reduce_scores=False)
        labels = [np.mean(l.cpu().numpy()) for l in labels]
    return labels



def inference_set(inf_loader, model, device, best_, save_model=False, suffix='s', set_name="na"):
    print(f"Validating for {set_name}.")
    results = []

    best_s, best_p, best_k, best_r = best_
    
    keys = []
    model.eval()
    for i, data in enumerate(tqdm(inf_loader, desc="Validating")):
        result = dict()
        video = {}
        for key in sample_types:
            if key not in keys:
                keys.append(key)
            if key in data:
                video[key] = data[key].to(device)
                b, c, t, h, w = video[key].shape
                video[key] = video[key].reshape(b, c, data["num_clips"][key], t // data["num_clips"][key], h, w).permute(0,2,1,3,4,5).reshape(b * data["num_clips"][key], c, t // data["num_clips"][key], h, w) 
        with torch.no_grad():
            labels = model(video,reduce_scores=False)
            labels = [np.mean(l.cpu().numpy()) for l in labels]
            result["pr_labels"] = labels
        result["gt_label"] = data["gt_label"].item()
        result["name"] = data["name"]
        # result['frame_inds'] = data['frame_inds']
        # del data
        results.append(result)

    
    ## generate the demo video for video quality localization
    gt_labels = [r["gt_label"] for r in results]
    pr_labels = 0
    pr_dict = {}
    for i, key in zip(range(len(results[0]["pr_labels"])), keys):
        key_pr_labels = np.array([np.mean(r["pr_labels"][i]) for r in results])
        pr_dict[key] = key_pr_labels
        pr_labels += rescale(key_pr_labels)
        
    pr_labels = rescale(pr_labels, gt_labels)

    s = spearmanr(gt_labels, pr_labels)[0]
    p = pearsonr(gt_labels, pr_labels)[0]
    k = kendallr(gt_labels, pr_labels)[0]
    r = np.sqrt(((gt_labels - pr_labels) ** 2).mean())
    
    
    results = sorted(results, key=lambda x: x["pr_labels"])

    # try:
    #     wandb.log({f"val/SRCC-{suffix}": s, f"val/PLCC-{suffix}": p, f"val/KRCC-{suffix}": k, f"val/RMSE-{suffix}": r})
    # except:
    #     pass

    best_s, best_p, best_k, best_r = (
        max(best_s, s),
        max(best_p, p),
        max(best_k, k),
        min(best_r, r),
    )

    # try:
    #     wandb.log(
    #         {
    #             f"val/best_SRCC-{suffix}": best_s,
    #             f"val/best_PLCC-{suffix}": best_p,
    #             f"val/best_KRCC-{suffix}": best_k,
    #             f"val/best_RMSE-{suffix}": best_r,
    #         }
    #     )
    # except:
    #     pass
    # print(
    #     f"For {len(inf_loader)} videos, \nthe accuracy of the model: [{suffix}] is as follows:\n  SROCC: {s:.4f} best: {best_s:.4f} \n  PLCC:  {p:.4f} best: {best_p:.4f}  \n  KROCC: {k:.4f} best: {best_k:.4f} \n  RMSE:  {r:.4f} best: {best_r:.4f}."
    # )

    return best_s, best_p, best_k, best_r, pr_labels



def main():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-o", "--opt", type=str, default="./options/fast/fast-b.yml", help="the option file"
    )

    args = parser.parse_args()
    with open(args.opt, "r") as f:
        opt = yaml.safe_load(f)
    print(opt)

    ## adaptively choose the device

    device = "cuda" if torch.cuda.is_available() else "cpu"
    #device = "cpu"

    ## setting seed
    cudnn.benchmark = True
    torch.manual_seed(43)
    torch.cuda.manual_seed(43)
    np.random.seed(43)
    random.seed(43)

    ## defining model and loading checkpoint

    bests_ = []
    
    model = getattr(models, opt["model"]["type"])(**opt["model"]["args"]).to(device)

    state_dict = torch.load(opt["test_load_path"], map_location=device)["state_dict"]

    if "test_load_path_aux" in opt:
        aux_state_dict = torch.load(opt["test_load_path_aux"], map_location=device)["state_dict"]
        
        from collections import OrderedDict
        
        fusion_state_dict = OrderedDict()
        for k, v in state_dict.items():
            if k.startswith("vqa_head"):
                ki = k.replace("vqa", "fragments")
            else:
                ki = k
            fusion_state_dict[ki] = v
            
        for k, v in aux_state_dict.items():
            if k.startswith("frag"):
                continue
            if k.startswith("vqa_head"):
                ki = k.replace("vqa", "resize")
            else:
                ki = k
            fusion_state_dict[ki] = v
        
        state_dict = fusion_state_dict
        
    model.load_state_dict(state_dict, strict=True)
    for key in opt["data"].keys():
        
        if "val" not in key and "test" not in key:
            continue
        
        val_dataset = getattr(datasets, opt["data"][key]["type"])(opt["data"][key]["args"])

        val_loader =  torch.utils.data.DataLoader(
            val_dataset, batch_size=1, num_workers=opt["num_workers"], pin_memory=True,
        )

        profile_inference(val_dataset, model, device)

        # test the model
        print(len(val_loader))

        best_ = -1, -1, -1, 1000

        best_ = inference_set(
            val_loader,
            model,
            device, best_,
            set_name=key,
        )

        # best_ = inference_set(
        #     val_dataset,
        #     val_loader,
        #     device, opt, best_,
        #     set_name=key,
        # )

        print(
            f"""Testing result on: [{len(val_loader)}] videos:
            SROCC: {best_[0]:.4f}
            PLCC:  {best_[1]:.4f}
            KROCC: {best_[2]:.4f}
            RMSE:  {best_[3]:.4f}."""
        )
        
        with open("results/"+opt["name"]+"_Test_"+key+".txt", "w") as f:
            for label in best_[-1]:
                f.write(f"{label}\n")

        # run.finish()


if __name__ == "__main__":
    main()
