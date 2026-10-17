import torch
from torch.utils.data import Dataset
import pandas as pd
from PIL import Image
from pathlib import Path

class DocumentDataset(Dataset):
    def __init__(self, csv_file, data_dir, transform=None):
        """
        Args:
            csv_file (str/Path): Path to the train.csv file.
            data_dir (str/Path): Root directory containing the images.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.df = pd.read_csv(csv_file)
        self.data_dir = Path(data_dir)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # Construct path (CSV 'filename' column already includes 'train/invoice/...' prefix)
        img_path = self.data_dir / str(row['filename'])
        
        # Convert Grayscale ('L') to 'RGB' to satisfy pre-trained CNN architectures
        image = Image.open(img_path).convert("RGB")
        label = torch.tensor(row['label'], dtype=torch.long)
        
        if self.transform:
            image = self.transform(image)
            
        return image, label