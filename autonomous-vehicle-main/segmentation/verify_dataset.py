from dataset import CityscapesDataset

train_images = r"E:\CityScape_Dataset\train\img"
train_masks  = r"E:\CityScape_Dataset\train\label_id"

ds = CityscapesDataset(train_images, train_masks)
print(f"Total images found: {len(ds)}")

# show one sample info
img, mask, name = ds[0]
print("Sample file:", name)
print("Image tensor shape:", img.shape)
print("Mask shape:", mask.shape)
print("Unique mask IDs:", list(set(mask.flatten().tolist()))[:20])
