import torch
from torch.utils.data import Dataset
from PIL import Image
from pathlib import Path
import pandas as pd

class DocumentDataset(Dataset):
    def __init__(self, csv_file, data_dir, transform=None):
        self.df = pd.read_csv(csv_file)
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.is_test = 'label' not in self.df.columns

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # Directly append the relative path from the CSV to the data directory
        img_path = self.data_dir / row['filename']
        
        # 1. Load image and force conversion to 3-channel RGB
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            image = Image.new("RGB", (224, 224), (255, 255, 255))

        # 2. Apply transformations
        if self.transform:
            image = self.transform(image)

        # 3. Return Tensor and Label (or ID for test set)
        if self.is_test:
            return image, row['id']
        else:
            label = torch.tensor(row['label'], dtype=torch.long)
            return image, label