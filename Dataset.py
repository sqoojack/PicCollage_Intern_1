import os
import pandas as pd     # used to read and address csv file
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

class ScatterPlotDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        self.data = pd.read_csv(csv_file)   # read csv file
        self.img_dir = img_dir
        self.transform = transform
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        # Read the corresponding image and label based on the index
        img_id = self.data.iloc[idx]['id']      
        label = float(self.data.iloc[idx]['corr'])  # transfrom it to float type
        img_path = os.path.join(self.img_dir, f"{img_id}.png")
        image = Image.open(img_path).convert("RGB")     # open image and convert it to RGB
        if self.transform:  # transfrom image if self.transfrom exists
            image = self.transform(image)
        
        label_tensor = torch.tensor(label, dtype=torch.float32)   # convert label to tensor
        return image, label_tensor
    
image_transforms = transforms.Compose([     # Normalize image (mean ≈ 0, std ≈ 1) for better training stability and optimization
    transforms.ToTensor(),  
    transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])     
])