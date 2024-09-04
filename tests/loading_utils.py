from torch.utils.data import Dataset
from PIL import Image
import os
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
            synset = root.find("object").find("name").text
            label = self.synset_to_class[synset]

            self.imgs.append((img_path, label))

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img_path, label = self.imgs[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        return image, label
    
def generate_synset_to_class_mapping(train_dir):
    synset_to_class = {}
    for idx, synset in enumerate(sorted(os.listdir(train_dir))):
        synset_to_class[synset] = idx
    return synset_to_class