import torch
from torch.utils.data import DataLoader
from models.yolo11_star_seg import StarYOLOSegModel
from utils.dataset import GloveSegDataset
from utils.loss import SegmentationLoss
from tqdm import tqdm

# Config
EPOCHS = 50
BATCH_SIZE = 4
IMG_SIZE = 640
LR = 1e-4
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Dataset
train_dataset = GloveSegDataset('data/glove/train/images', 'data/glove/train/masks', IMG_SIZE)
val_dataset = GloveSegDataset('data/glove/val/images', 'data/glove/val/masks', IMG_SIZE)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=1)

# Model
model = StarYOLOSegModel(num_classes=1).to(DEVICE)
criterion = SegmentationLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

# Training Loop
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for imgs, masks in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
        imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
        preds = model(imgs)
        loss = criterion(preds, masks)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch+1} Loss: {avg_loss:.4f}")
