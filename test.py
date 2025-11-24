import torch
from PIL import Image
from torchvision import transforms
from model.madnet import MADNet
import matplotlib.pyplot as plt

def test(image_path):
    model = MADNet()
    model.load_state_dict(torch.load('madnet_epoch_100.pth', map_location='cpu'))
    model.eval()

    img = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    lr = transform(img).unsqueeze(0)

    with torch.no_grad():
        sr = model(lr).clamp(-1, 1)

    # Hiển thị ảnh
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title('Low Resolution')
    plt.imshow(img)

    plt.subplot(1, 2, 2)
    plt.title('Super Resolved')
    plt.imshow(sr.squeeze().permute(1, 2, 0).numpy() * 0.5 + 0.5)
    plt.show()

test('datasets/train/LR/0021.png') 