import os
import sys
sys.path.insert(0, os.path.abspath("."))

import argparse
import torch
import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt

try:
    from captum.attr import IntegratedGradients, LayerGradCam
    HAS_CAPTUM = True
except ImportError:
    HAS_CAPTUM = False

from src.models.cnn import build_3d_cnn_model

def parse_args():
    parser = argparse.ArgumentParser(description="Generate 3D Attribution maps (Grad-CAM, Integrated Gradients) for model explainability.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to trained PyTorch model checkpoint.")
    parser.add_argument("--image_path", type=str, required=True, help="Path to 3D NIfTI input volume.")
    parser.add_argument("--method", type=str, default="gradcam", choices=["gradcam", "ig"], help="Attribution method.")
    parser.add_argument("--target_class", type=int, default=0, help="Target class index for attribution.")
    parser.add_argument("--out_dir", type=str, default="./results/explanations", help="Output directory for saliency maps.")
    return parser.parse_args()

def generate_attribution_map(model, input_tensor, target_class=0, method="gradcam"):
    """
    Generate 3D attribution map using Captum.
    input_tensor: (1, C, D, H, W) PyTorch tensor.
    """
    if not HAS_CAPTUM:
        raise ImportError("Captum is required for explainability methods. Please install captum.")

    model.eval()

    if method == "gradcam":
        # Identify final convolutional layer in MONAI DenseNet / ResNet
        if hasattr(model, 'features'):
            target_layer = model.features[-1]
        elif hasattr(model, 'layer4'):
            target_layer = model.layer4[-1]
        else:
            target_layer = list(model.children())[-2]

        lgc = LayerGradCam(model, target_layer)
        attr = lgc.attribute(input_tensor, target=target_class)
    elif method == "ig":
        ig = IntegratedGradients(model)
        attr = ig.attribute(input_tensor, target=target_class, n_steps=20)
    else:
        raise ValueError(f"Unsupported method {method}")

    return attr.detach().cpu().numpy()

def save_attribution_overlay(input_np, attr_np, output_png):
    """Plot axial slice overlay of input scan and attribution heatmap."""
    d, h, w = input_np.shape
    mid_slice_idx = d // 2

    img_slice = input_np[mid_slice_idx, :, :]
    attr_slice = attr_np[mid_slice_idx, :, :]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(img_slice, cmap="gray")
    axes[0].set_title("Input MRI Scan (Axial Slice)")
    axes[0].axis("off")

    axes[1].imshow(attr_slice, cmap="jet")
    axes[1].set_title("Attribution Heatmap")
    axes[1].axis("off")

    axes[2].imshow(img_slice, cmap="gray")
    axes[2].imshow(attr_slice, cmap="jet", alpha=0.5)
    axes[2].set_title("Explainability Overlay")
    axes[2].axis("off")

    os.makedirs(os.path.dirname(output_png), exist_ok=True)
    plt.savefig(output_png, bbox_inches="tight", dpi=300)
    plt.close()
    print(f"✅ Attribution overlay saved to {output_png}")

def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

    checkpoint = torch.load(args.model_path, map_location=device)
    model_name = checkpoint.get('model_name', 'densenet121')
    model = build_3d_cnn_model(model_name=model_name, in_channels=3, num_classes=3)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)

    img_sitk = sitk.ReadImage(args.image_path, sitk.sitkFloat32)
    img_np = sitk.GetArrayFromImage(img_sitk)

    # Prepare dummy 3-channel input tensor (1, 3, D, H, W)
    input_tensor = torch.from_numpy(img_np).unsqueeze(0).unsqueeze(0).repeat(1, 3, 1, 1, 1).to(device)

    attr = generate_attribution_map(model, input_tensor, target_class=args.target_class, method=args.method)
    attr_3d = np.squeeze(attr)
    if attr_3d.ndim == 4:
        attr_3d = np.mean(attr_3d, axis=0)

    out_png = os.path.join(args.out_dir, f"attribution_{args.method}_{os.path.basename(args.image_path)}.png")
    save_attribution_overlay(img_np, attr_3d, out_png)

if __name__ == "__main__":
    main()
