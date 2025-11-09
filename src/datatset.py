import os
from collections import defaultdict
import random
from PIL import Image
from torch.utils.data import Dataset

class DeepFashionDataset(Dataset):
    def __init__(
        self, split='train', transform=None,
        paths_file=None, labels_file=None, 
        bbox_file=None, classes_file=None, img_folder=None,
        include_labels=None, downsample=False, max_samples_per_class=500
    ):
        with open(paths_file) as f:
            self.img_paths = [line.strip() for line in f]
        with open(labels_file) as f:
            self.labels = [int(line.strip()) for line in f]
        with open(bbox_file) as f:
            self.bboxes = [self._parse_coords(line) for line in f]
        
        with open(classes_file) as f:
            lines = f.readlines()[2:]
            self.classes = [
                line.strip().split()[0] for line in lines
            ]

        if downsample:
            inds = self._downsample(self.labels, max_samples_per_class)
            self.img_paths = [self.img_paths[i] for i in inds]
            self.labels = [self.labels[i] for i in inds]
            self.bboxes = [self.bboxes[i] for i in inds]

        self.img_folder = img_folder
        self.transform = transform
        

    def _downsample(self, labels, max_samples_per_class):
        class_to_indices = defaultdict(list)

        for idx, label in enumerate(labels):
            class_to_indices[label].append(idx)

        selected_inds = []
        for label, inds in class_to_indices.items():
            if len(inds) > max_samples_per_class:
                inds = random.sample(inds, max_samples_per_class)
            selected_inds += inds

        return selected_inds
    

    def _parse_coords(self, line):
        bbox = line.strip().split(' ')
        return [int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])]


    def __len__(self):
        return len(self.img_paths)
    
    
    def __getitem__(self, idx):
        img = Image.open(os.path.join(self.img_folder, self.img_paths[idx]))
        label = self.labels[idx]
        bbox = self.bboxes[idx]
        img = img.crop(bbox)
        if self.transform:
            img = self.transform(img)
        return img, label
    