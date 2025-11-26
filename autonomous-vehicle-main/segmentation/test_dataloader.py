from dataset import CityscapesDataset
from torch.utils.data import DataLoader

print("Loading dataset...")
train_ds = CityscapesDataset(r"E:\CityScape_Dataset\train\img", r"E:\CityScape_Dataset\train\label_id", size=(256, 512))
print("Total images:", len(train_ds))

print("Creating DataLoader...")
train_dl = DataLoader(train_ds, batch_size=2, shuffle=True, num_workers=0)

print("Iterating through DataLoader...")
for i, (imgs, masks, names) in enumerate(train_dl):
    print(f"Batch {i+1}: imgs={imgs.shape}, masks={masks.shape}")
    if i == 1:
        break

print("✅ DataLoader works fine!")
