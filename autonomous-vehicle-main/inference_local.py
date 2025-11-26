import torch
import cv2
import numpy as np
from torchvision.models.segmentation import deeplabv3_resnet50
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from pathlib import Path

# === Paths ===
model_path = r"E:\AutonomousVehicle\models\model_final.pth"
image_dir  = r"E:\AutonomousVehicle\test_images"

# === Device ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# === Load model ===
NUM_CLASSES = 19
model = deeplabv3_resnet50(num_classes=NUM_CLASSES)
state_dict = torch.load(model_path, map_location=device)
model.load_state_dict(state_dict)
model.to(device).eval()
print("✅ Model loaded successfully")

# === Transform ===
transform = transforms.Compose([
    transforms.Resize((512, 1024)),
    transforms.ToTensor(),
])

# === Inference loop ===
for img_path in Path(image_dir).rglob("*.png"):
    img = Image.open(img_path).convert("RGB")
    img_tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(img_tensor)['out']
        pred = output.argmax(1).squeeze().cpu().numpy().astype(np.uint8)

    plt.figure(figsize=(14,5))
    plt.subplot(1,2,1)
    plt.imshow(img)
    plt.title("Input Image")
    plt.axis('off')

    plt.subplot(1,2,2)
    plt.imshow(pred, cmap='nipy_spectral')
    plt.title("Predicted Segmentation")
    plt.axis('off')
    plt.show()
