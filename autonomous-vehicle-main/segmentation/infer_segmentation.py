# infer_segmentation.py
import os
import numpy as np
from PIL import Image
import torch
import torchvision
import matplotlib.pyplot as plt

# Define color map (same order as in training)
COLOR_MAP = np.array([
    [128, 64, 128], [244, 35, 232], [70, 70, 70], [102, 102, 156],
    [190, 153, 153], [153, 153, 153], [250, 170, 30], [220, 220, 0],
    [107, 142, 35], [152, 251, 152], [70, 130, 180], [220, 20, 60],
    [255, 0, 0], [0, 0, 142], [0, 0, 70], [0, 60, 100],
    [0, 80, 100], [0, 0, 230], [119, 11, 32]
], dtype=np.uint8)

def colorize_mask(mask):
    """Convert class IDs to RGB colors"""
    h, w = mask.shape
    color_mask = np.zeros((h, w, 3), dtype=np.uint8)
    for cid in np.unique(mask):
        if cid < len(COLOR_MAP):
            color_mask[mask == cid] = COLOR_MAP[cid]
    return color_mask

def main():
    device = torch.device("cpu")
    model_path = r"E:\AutonomousVehicle\segmentation\checkpoints\deeplabv3_cityscapes_cpu.pth"
    test_image = r"E:\CityScape_Dataset\val\img\val1.png"  # change to any real image you want

    if not os.path.exists(model_path):
        print("❌ Model checkpoint not found:", model_path)
        return

    print("✅ Loading model...")
    model = torchvision.models.segmentation.deeplabv3_resnet50(pretrained=False, num_classes=19)
    model.classifier[4] = torch.nn.Conv2d(256, 19, kernel_size=1)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict, strict=False)
    print("✅ Model loaded successfully (ignored unmatched keys).")

    model.eval().to(device)

    print("✅ Reading input image...")
    img = Image.open(test_image).convert("RGB")
    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize((256, 512)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
    ])
    img_t = transform(img).unsqueeze(0).to(device)

    print("🔹 Running inference...")
    with torch.no_grad():
        output = model(img_t)["out"]
        pred = torch.argmax(output.squeeze(), dim=0).cpu().numpy()

    print("🎨 Colorizing mask...")
    color_mask = colorize_mask(pred)

    # Display and save result
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].imshow(img)
    axes[0].set_title("Original Image")
    axes[0].axis("off")

    axes[1].imshow(color_mask)
    axes[1].set_title("Predicted Segmentation")
    axes[1].axis("off")

    plt.tight_layout()
    save_path = os.path.join(os.path.dirname(model_path), "val1_pred.png")
    plt.savefig(save_path)
    plt.show()
    print(f"✅ Saved output to {save_path}")

if __name__ == "__main__":
    main()
