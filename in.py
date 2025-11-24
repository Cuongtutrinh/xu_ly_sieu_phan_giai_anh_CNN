from PIL import Image
import os
lr_dir = 'D:/Hoc_lap_trinh/MADNet_Project/datasets/train/HR'
for img_name in os.listdir(lr_dir)[:5]: 
    img = Image.open(os.path.join(lr_dir, img_name))
    print(f"{img_name}: {img.size}") 