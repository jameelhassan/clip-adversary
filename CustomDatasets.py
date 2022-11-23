from torchvision import datasets
from torchvision import transforms
import torch
from PIL import Image
import numpy as np

class Cifar10_preprocess2(datasets.CIFAR10):
    def __init__(self, root, train=True, transform_corr=None, transform=None, target_transform=None, download=False, index=0):
        super().__init__(root, train, transform, target_transform, download)
        self.index = index
        self.transform_corr = transform_corr
        self.transform = transform

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)
        img_ = img.copy()
        if self.transform_corr is not None:  # Not handling exceptions!
            img_transformed = self.transform_corr(img_)    # image transformed
            text_class_idx = self.transform_corr.transforms[0].index    # Get index of text corruption class for sample
        if self.transform is not None:
            img = self.transform(img)
            # img = transforms.Resize((224, 224))(img) # image resized
            # img_tensor = transforms.ToTensor()(img)  # tensor image
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img_transformed, img, text_class_idx, target


class Cifar100_preprocess2(datasets.CIFAR100):
    def __init__(self, root, train=True, transform_corr=None, transform=None, target_transform=None, download=False, index=0):
        super().__init__(root, train, transform, target_transform, download)
        self.index = index
        self.transform_corr = transform_corr
        self.transform = transform

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)
        img_ = img.copy()
        if self.transform_corr is not None:  # Not handling exceptions!
            img_transformed = self.transform_corr(img_)    # image transformed
            text_class_idx = self.transform_corr.transforms[0].index    # Get index of text corruption class for sample
        if self.transform is not None:
            img = self.transform(img)
            # img = transforms.Resize((224, 224))(img) # image resized
            # img_tensor = transforms.ToTensor()(img)  # tensor image
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img_transformed, img, text_class_idx, target


class Caltech101_preprocess2(datasets.Caltech101):
    def __init__(self, root, train=True, transform_corr=None, transform=None, target_transform=None, download=False, index=0):
        super().__init__(root, train, transform, target_transform, download)
        self.index = index
        self.transform_corr = transform_corr
        self.transform = transform

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)
        img_ = img.copy()
        if self.transform_corr is not None:  # Not handling exceptions!
            img_transformed = self.transform_corr(img_)    # image transformed
            text_class_idx = self.transform_corr.transforms[0].index    # Get index of text corruption class for sample
        if self.transform is not None:
            img = self.transform(img)
            # img = transforms.Resize((224, 224))(img) # image resized
            # img_tensor = transforms.ToTensor()(img)  # tensor image
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img_transformed, img, text_class_idx, target