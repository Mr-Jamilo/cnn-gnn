import os
import pandas as pd
from PIL import Image
from torchvision.io import decode_image
from torch.utils.data import Dataset

class CustomImageDataset(Dataset):
    def __init__(self, label_file, img_dir, transform=None, target_transform=None):
        self.data = pd.read_csv(label_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, index):
        row = self.data.iloc[index]
        #print(row)
        img_name = str(row['ID']) + ".png"
        img_path = os.path.join(self.img_dir, img_name)
        #Image.open(img_path).show()
        tensor_img = decode_image(img_path)
        labels = [col for col, val in row.items() if val == 1 and col not in ['ID', 'Disease_Risk']]
        return tensor_img, labels


dataset = CustomImageDataset(label_file='dataset/labels.csv', img_dir='dataset',)
#print(dataset.__getitem__(3))
