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


device = "cuda" if torch.cuda.is_available() else "cpu"
sample_types = ["resize", "fragments", "crop", "arp_resize", "arp_fragments"]


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.hook_handles = []
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        self.hook_handles.append(self.target_layer.register_forward_hook(forward_hook))
        self.hook_handles.append(self.target_layer.register_backward_hook(backward_hook))

    def remove_hooks(self):
        for handle in self.hook_handles:
            handle.remove()

    def generate(self, input_tensor):
        self.model.zero_grad()
        output, feats = self.model(input_tensor, inference=False, return_pooled_feats=True, reduce_scores=True)
        if isinstance(output, list):
            output = output[0]

        feats['fragments'].mean().backward()

        pooled_grads = torch.mean(self.gradients, dim=[0, 2, 3, 4])
        activations = self.activations.squeeze(0)
        for i in range(activations.shape[0]):
            activations[i] *= pooled_grads[i]

        heatmap = torch.mean(activations, dim=0).cpu().numpy()
        heatmap = np.maximum(heatmap, 0)
        for t in range(heatmap.shape[0]):
            heatmap[t] /= np.max(heatmap[t]) + 1e-8

        return heatmap


def overlay_heatmap(heatmap, frame):
    if heatmap.ndim == 3:
        heatmap = heatmap[0]

    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap) + 1e-8

    heatmap_resized = cv2.resize(heatmap, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    heatmap_colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)

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

    val_datasets = {}
    for key in opt["data"]:
        if key.startswith("val"):
            val_dataset = getattr(datasets, opt["data"][key]["type"])(opt["data"][key]["args"])
            print(len(val_dataset.video_infos))
            val_datasets[key] = val_dataset

    val_loaders = {
        key: torch.utils.data.DataLoader(
            val_dataset,
            batch_size=1,
            num_workers=opt["num_workers"],
            pin_memory=True,
        ) for key, val_dataset in val_datasets.items()
    }

    for key, val_loader in val_loaders.items():
        for i, data in enumerate(tqdm(val_loader)):
            video = {}
            for sample_key in sample_types:
                if sample_key in data:
                    video[sample_key] = data[sample_key].to(device)
                    print(video[sample_key].shape)
                    b, c, t, h, w = video[sample_key].shape
                    video[sample_key] = video[sample_key].reshape(
                        b, c, data["num_clips"][sample_key], t // data["num_clips"][sample_key], h, w
                    ).permute(0, 2, 1, 3, 4, 5).reshape(
                        b * data["num_clips"][sample_key], c, t // data["num_clips"][sample_key], h, w
                    )

            input_clip = video

            try:
                # use early convolution layer if available
                target_layer = model.fragments_backbone.patch_embed.proj
            except AttributeError:
                print("❌ Could not find target layer. Please update target_layer manually.")
                return

            cam = GradCAM(model, target_layer)
            heatmap = cam.generate(input_clip)

            frame_tensor = input_clip['fragments'][0, :, 0, :, :].cpu()
            frame = frame_tensor.permute(1, 2, 0).numpy()
            frame = cv2.normalize(frame, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

            overlay = overlay_heatmap(heatmap, frame)
            os.makedirs("gradcam_outputs", exist_ok=True)
            output_path = f"gradcam_outputs/{data['name']}_gradcam_adapted.jpg"
            cv2.imwrite(output_path, overlay)

            print(f"✅ Grad-CAM saved to: {output_path}")


if __name__ == "__main__":
    main()