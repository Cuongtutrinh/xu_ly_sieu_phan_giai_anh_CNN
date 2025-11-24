import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
from model.madnet import MADNet
import matplotlib.pyplot as plt

class SRDataset(Dataset):
    def __init__(self, lr_dir, hr_dir):
        self.lr_dir = lr_dir
        self.hr_dir = hr_dir
        self.image_names = os.listdir(lr_dir)
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        lr_img = Image.open(os.path.join(self.lr_dir, self.image_names[idx]))
        hr_img = Image.open(os.path.join(self.hr_dir, self.image_names[idx]))
        return self.transform(lr_img), self.transform(hr_img)

def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset = SRDataset('datasets/train/LR', 'datasets/train/HR')
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=2)
    
    model = MADNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.L1Loss()

    for epoch in range(100):
        for lr, hr in dataloader:
            lr, hr = lr.to(device), hr.to(device)
            sr = model(lr)
            loss = criterion(sr, hr)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        print(f'Epoch {epoch+1}, Loss: {loss.item():.4f}')
        if (epoch+1) % 100 == 0:
            torch.save(model.state_dict(), f'madnet_epoch_{epoch+1}.pth')

if __name__ == '__main__':
    train()