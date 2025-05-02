from torch.utils.data import Dataset
from PIL import Image
import os
import torch
import torchvision.transforms as T

class GloveSegDataset(Dataset):
    def __init__(self, image_dir, mask_dir, image_size=640):
        self.images = sorted(os.listdir(image_dir))
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.image_size = image_size
        self.transforms = T.Compose([
            T.Resize((image_size, image_size)),
            T.ToTensor(),
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.images[idx])
        mask_path = os.path.join(self.mask_dir, self.images[idx].replace('.jpg', '.png'))

        image = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')  # Binary mask

        image = self.transforms(image)
        mask = self.transforms(mask)
        return image, mask
