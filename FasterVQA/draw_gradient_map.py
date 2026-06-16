import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import matplotlib.cm as cm

# --- Load attention overlay images ---
img1_path = "/home/ubuntu/Desktop/ajeet_backup/proposed_vqa/FasterVQA/attn_outputs/_BSRGANx2_klens_5510_filter/frame_001.jpg"  # e.g., original model
img2_path = "/home/ubuntu/Desktop/ajeet_backup/proposed_vqa/FasterVQA/attn_outputs_ada/_BSRGANx2_klens_5510_filter/frame_001.jpg"  # e.g., adapted model (FasterVQA)

img1 = Image.open(img1_path)
img2 = Image.open(img2_path)

# --- Gradient (1D horizontal bar) ---
gradient = np.linspace(0, 1, 256).reshape(1, -1)  # shape: [1, 256]

# --- Create main figure with custom layout ---
fig = plt.figure(figsize=(12, 4))

# Image axes
ax1 = fig.add_subplot(1, 3, 1)
ax1.imshow(img1)
ax1.set_title("Original Model", fontsize=12)
ax1.axis("off")

ax2 = fig.add_subplot(1, 3, 2)
ax2.imshow(img2)
ax2.set_title("Adapted Model", fontsize=12)
ax2.axis("off")

# Custom smaller axes for gradient bar
gradient_ax = fig.add_axes([0.72, 0.28, 0.015, 0.45])  # [left, bottom, width, height]
gradient_ax.imshow(gradient.T, aspect='auto', cmap=cm.jet)
gradient_ax.set_title("Attention\nIntensity", fontsize=10, pad=10)
gradient_ax.set_yticks([0, 63, 127, 191, 255])
gradient_ax.set_yticklabels(["Low", "", "Medium", "", "High"])
gradient_ax.tick_params(axis='y', labelsize=9)
gradient_ax.xaxis.set_visible(False)

# --- Save or display ---
plt.tight_layout()
plt.savefig("attention_map_compact_gradient.jpg", dpi=300)
plt.show()