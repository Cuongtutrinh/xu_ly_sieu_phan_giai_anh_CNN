from PIL import Image
import os

hr_dir = 'datasets/train/HR'
lr_dir = 'datasets/train/LR'
os.makedirs(lr_dir, exist_ok=True)

# Kích thước cố định
hr_size = (512, 512)  # HR: 512x512
lr_size = (128, 128)  # LR: 128x128 (1/4 của HR)

for img_name in os.listdir(hr_dir):
    # Đọc ảnh HR
    hr_img = Image.open(os.path.join(hr_dir, img_name))
    
    # Cắt ảnh HR thành 512x512 (lấy vùng trung tâm)
    left = (hr_img.width - hr_size[0]) // 2
    top = (hr_img.height - hr_size[1]) // 2
    hr_img = hr_img.crop((left, top, left + hr_size[0], top + hr_size[1]))
    
    # Tạo ảnh LR bằng cách thu nhỏ ảnh HR đã cắt
    lr_img = hr_img.resize(lr_size, Image.BICUBIC)
    
    # Lưu ảnh
    hr_img.save(os.path.join(hr_dir, img_name))  # Ghi đè ảnh HR
    lr_img.save(os.path.join(lr_dir, img_name))  # Lưu ảnh LR