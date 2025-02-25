from torch.utils.data import Dataset
import torchvision.models as models
import torch.nn as nn
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
    
def generate_synset_to_class_mapping(train_dir):
    synset_to_class = {}
    for idx, synset in enumerate(sorted(os.listdir(train_dir))):
        synset_to_class[synset] = idx
    return synset_to_class

def load_model(model_name, num_classes, pretrained = False, device = "cpu"):
    model_class = getattr(models, model_name, None)
    if model_class is None:
        raise ValueError(f"Model '{model_name}' not found.")
    
    model = model_class(weights = "DEFAULT" if pretrained else None)
    if hasattr(model, "fc"):
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif hasattr(model, "classifier"):
        if isinstance(model.classifier, nn.Sequential):
            model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
        else:
            model.classifier = nn.Linear(model.classifier.in_features, num_classes)
    elif hasattr(model, "head"):
        model.head = nn.Linear(model.head.in_features, num_classes)
    else:
        raise ValueError(f"Unknown model '{model_name}' classification layer.")
    model = model.to(device)
    return model