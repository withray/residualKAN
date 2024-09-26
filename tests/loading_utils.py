from torch.utils.data import Dataset
from PIL import Image
import os
import torch
import xml.etree.ElementTree as ET

class ImageNetValDataset(Dataset):
    def __init__(self, img_dir, ann_dir, synset_to_class, transform = None):
        self.img_dir = img_dir
        self.ann_dir = ann_dir
        self.transform = transform
        self.img_names = sorted(os.listdir(img_dir))
        self.synset_to_class = synset_to_class
        self.imgs = []

        for img_name in self.img_names:
            img_path = os.path.join(self.img_dir, img_name)
            ann_path = os.path.join(self.ann_dir, os.path.splitext(img_name)[0] + ".xml")

            tree = ET.parse(ann_path)
            root = tree.getroot()
            object_elem = root.find("object")
            if object_elem is not None:
                name_elem = object_elem.find("name")
                if name_elem is not None:
                    synset = name_elem.text
                    label = self.synset_to_class[synset]
                    self.imgs.append((img_path, label))
                else:
                    print(f"'name' not found in {ann_path}")
            else:
                print(f"'object' not found in {ann_path}")

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img_path, label = self.imgs[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        return image, label
    
class Food101Dataset(Dataset):
    def __init__(self, root_dir, txt_file, transform = None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []
        
        self.label_to_index = {}
        with open(txt_file, "r") as f:
            for line in f:
                line = line.strip()
                class_name = line.split("/")[0]
                if class_name not in self.label_to_index:
                    self.label_to_index[class_name] = len(self.label_to_index)
                self.image_paths.append(os.path.join(self.root_dir, line + ".jpg"))
                self.labels.append(self.label_to_index[class_name])
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = torch.tensor(self.labels[idx], dtype = torch.long)
        image = Image.open(img_path).convert("RGB")
        
        if self.transform:
            image = self.transform(image)
        
        return image, label
    
def generate_synset_to_class_mapping(train_dir):
    synset_to_class = {}
    for idx, synset in enumerate(sorted(os.listdir(train_dir))):
        synset_to_class[synset] = idx
    return synset_to_class