import torch
import torch.nn.functional as F
import numpy as np
import cv2
import os
import yaml
import argparse
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')
import fastvqa.models as models  
import fastvqa.datasets as datasets
from image2video import convertImage, convertVideo


device = "cuda" if torch.cuda.is_available() else "cpu"
sample_types = ["resize", "fragments", "crop", "arp_resize", "arp_fragments"]


def auto_infer_hw(patch_area):
    for i in range(int(patch_area ** 0.5), 0, -1):
        if patch_area % i == 0:
            return i, patch_area // i
    raise ValueError(f"Cannot infer shape from patch area {patch_area}")


class SwinAttentionExtractor:
    def __init__(self, model, target_block_idx=0, sample_type="fragments"):
        self.model = model
        self.attentions = []
        self.sample_type = sample_type
        self.target_block_idx = target_block_idx
        self.layer_info = {}
        self._register_hook()

    def _register_hook(self):
        stage_idx = 3
        block_idx = self.target_block_idx
        block = self.model.fragments_backbone.layers[stage_idx].blocks[block_idx]
        self.hook = block.attn.attn_drop.register_forward_hook(self._save_attention)

        self.layer_info = {
            "backbone": "fragments_backbone",
            "stage_index": stage_idx,
            "block_index": block_idx,
            "module": "attn.attn_drop"
        }
        print(f"[INFO] Hook registered on: {self.layer_info}")

    def _save_attention(self, module, input, output):
        self.attentions.append(output.detach().cpu())  # (B*num_windows, num_heads, N, N)

    def remove_hook(self):
        self.hook.remove()

    def extract_attention_map(self, input_tensor):
        self.attentions = []
        with torch.no_grad():
            self.model(input_tensor, inference=False, return_pooled_feats=True, reduce_scores=True)

        if not self.attentions:
            raise RuntimeError("No attention captured. Check the hook layer.")

        attn = self.attentions[0]  # (B*num_windows, num_heads, N, N)

        # Use a single head (e.g., head 0)
        single_head_attn = attn[:, 0]  # (B*num_windows, N, N)
        attn_map = single_head_attn.mean(-1)  # average over query positions: (B*num_windows, N)

        patch_area = attn_map.size(-1)
        patch_h, patch_w = auto_infer_hw(patch_area)
        print(f"[DEBUG] Auto-inferred patch size: {patch_h} x {patch_w}")

        spatial_maps = attn_map.view(-1, patch_h, patch_w)  # (T, H, W)

        # ✅ Global normalization across all frames
        all_maps_np = spatial_maps.numpy()
        global_min = np.min(all_maps_np)
        global_max = np.max(all_maps_np)
        normalized_maps_np = (all_maps_np - global_min) / (global_max - global_min + 1e-8)

        return torch.tensor(normalized_maps_np)


def overlay_heatmap(heatmap, frame):
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap) + 1e-8

    heatmap_resized = cv2.resize(heatmap, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_LINEAR)
    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)

    if frame.dtype != np.uint8:
        frame = cv2.normalize(frame, None, 0, 255, cv2.NORM_MINMAX)
        frame = frame.astype(np.uint8)

    overlay = cv2.addWeighted(heatmap_colored, 0.4, frame, 0.6, 0)
    return overlay


def load_model(opt):
    model = getattr(models, opt["model"]["type"])(**opt["model"]["args"]).to(device)
    state_dict = torch.load(opt["test_load_path"], map_location=device)["state_dict"]
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--opt", type=str, required=True, help="Path to YAML config")
    args = parser.parse_args()

    with open(args.opt, "r") as f:
        opt = yaml.safe_load(f)

    model = load_model(opt)

    path = '/home/ubuntu/Desktop/ajeet_backup/proposed_vqa/FasterVQA/Dataset/VQADataset/val/_BSRGANx2_klens_5510_filter.avi'
    video = convertVideo(path)   # video as path

    print(video.keys(), video['fragments'].shape)

    input_clip = {'fragments': video['fragments']}  # what the extractor expects

    # Initialize attention extractor
    try:
        attn_extractor = SwinAttentionExtractor(model, target_block_idx=0)
    except AttributeError:
        print("❌ Could not find attention layer. Please update target_block_idx manually.")
        exit()

    attention_maps = attn_extractor.extract_attention_map(input_clip)

    # Output directory
    output_dir = f"attn_outputs/{video['name']}_stage{attn_extractor.layer_info['stage_index']}_block{attn_extractor.layer_info['block_index']}"
    os.makedirs(output_dir, exist_ok=True)

    # Save metadata file
    with open(f"{output_dir}/attention_info.txt", "w") as f:
        for k, v in attn_extractor.layer_info.items():
            f.write(f"{k}: {v}\n")

    # Save attention overlays
    for t in range(attention_maps.shape[0]):
        frame_tensor = video['fragments'][0, :, t, :, :].cpu()
        frame = frame_tensor.permute(1, 2, 0).numpy()
        frame = cv2.normalize(frame, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        overlay = overlay_heatmap(attention_maps[t].cpu().numpy(), frame)
        cv2.imwrite(os.path.join(output_dir, f"frame_{t:03d}.jpg"), overlay)

    print(f"✅ Saved attention frames to: {output_dir}")


if __name__ == "__main__":
    main()
