import torch
import os, cv2
import numpy as np
from PIL import Image
from torchvision import transforms

def convertImage(image_path):
    num_frames = 8
    image_size = 224
    sample_types = ['fragments']
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ])

    img = Image.open(image_path).convert('RGB')
    tensor_img = transform(img)  # [3, 224, 224]

    video_tensor = tensor_img.unsqueeze(1).repeat(1, num_frames, 1, 1)  # [3, 8, 224, 224]
    video_tensor = video_tensor.unsqueeze(0)  # [1, 3, 8, 224, 224]

    # Extract name from path
    video_name = os.path.splitext(os.path.basename(image_path))[0]

    return {
        'fragments': video_tensor.to(device),
        'name': video_name
    }



def convertVideo(video_path, num_frames=8, image_size=224):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor()  # [C, H, W]
    ])

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames < num_frames:
        raise ValueError(f"Video has only {total_frames} frames, but {num_frames} are required.")

    # --- Sample frames evenly ---
    frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    frames = []

    for i in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            break
        if i in frame_indices:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            tensor = transform(frame_rgb)  # [3, 224, 224]
            frames.append(tensor)

    cap.release()

    if len(frames) != num_frames:
        raise RuntimeError(f"Expected {num_frames} frames but got {len(frames)}")

    video_tensor = torch.stack(frames, dim=1)  # [3, T, 224, 224]
    video_tensor = video_tensor.unsqueeze(0).to(device)  # [1, 3, T, 224, 224]

    video_name = os.path.splitext(os.path.basename(video_path))[0]

    return {
        'fragments': video_tensor,
        'name': video_name
    }