import pickle
import numpy as np
import os
import torch
from torch import nn
import torchvision
import torchvision.transforms as transforms
import clip
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
import json
from PIL import Image, ImageFont, ImageDraw
import random
from our_models import *
from losses import *
from utils import *
from torchvision.utils import save_image
from torchvision.utils import make_grid
from CustomDatasets import Cifar10_preprocess2, Cifar100_preprocess2, MyCaltech101
from time import time
import datetime


class AddText(object):
    """
    Add a randomly chosen class as text on the image
    """
    def __init__(self, classes, fontsize=5, index=None):
        self.classes = classes
        self.index = index
        self.fontsize = fontsize
        self.random_choice = True if self.index is None else False
        self.font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', self.fontsize)

    def __call__(self, sample):
        image = sample
        self.index = np.random.choice(range(10)) if self.random_choice else self.index
        text_class = self.classes[self.index]
        img_tf = ImageDraw.Draw(image.convert("RGB")) #.convert("RGB") for Caltech
        
        #Setting possible positions and colours of text and choosing one in random 
        text_locs = [(np.round(i * image.size[0]), np.round(j * image.size[1])) for (i,j) in [(0.25, 0.25), (0.25, 0.6), (0.75, 0.25), (0.6, 0.6)]]
        text_pos = random.choice(text_locs)
        text_cols = [(255,0,0), (0,255,0), (0,0,255), (0,0,0), (255,255,255)]
        text_col = random.choice(text_cols) #(0,0,0) for Black
        img_tf.text(text_pos, text_class, text_col, font=self.font)

        return image

    
def validate(model):   
    model.eval()
    predictions = np.zeros(len(data_classes))
    top1 = 0.
    top5 = 0.
    attack_top1 = 0.
    attack_top5 = 0.
    n = 0.

    with tqdm(test_loader, unit="batch") as tepoch:
        for batch_idx, (img_corr, data, text_corr_idx, target) in enumerate(tepoch):
            img_corr, data, target, text_corr_idx = img_corr.to(device), data.to(device), target.to(device), text_corr_idx.to(device)
            
            with torch.no_grad():
                structured_noise = model(img_corr)      # Structured noise from generator with input as corrupted image
                adversary = structured_noise + data     # Add original image to structured noise with input as original image
                adversary = torch.min(torch.max(adversary, data - eps), data + eps)
                # adversary = torch.clamp(adversary, 0.0, 1.0)
                batch_images = make_grid(adversary, nrow=10, normalize=True)
                if (batch_idx == 0):
                    save_image(batch_images, f"./Adversary_images/{clipname}/{DATASET}/attacks/eps{str(eps)}.png", normalize=False)

            text_descriptions = [f"This is a photo of a {cl}" for cl in data_classes]
            text_tokens = clip.tokenize(text_descriptions).cuda()
            with torch.no_grad():
                text_features = featurizer.encode_text(text_tokens)
                text_features /= text_features.norm(dim=-1, keepdim=True)
                image_features = featurizer.encode_image(adversary)
                image_features /= image_features.norm(dim=-1, keepdim=True)
                logits = 100. * image_features @ text_features.T

            preds = torch.argmax(logits, dim=1)
            for p in preds.cpu():
                predictions[p] += 1

            #Calculate accuracy
            acc1, acc5 = accuracy(logits, target, topk=(1, 5)) # should decrease 

            #calculate targetted attack accuracy
            attack_acc1, attack_acc5 = accuracy(logits, text_corr_idx, topk=(1, 5)) # should increase 
            top1 += acc1
            top5 += acc5
            attack_top1 += attack_acc1
            attack_top5 += attack_acc5
            n += data.size(0)
    
    top1 = (top1 / n) * 100
    top5 = (top5 / n) * 100
    attack_top1 = (attack_top1 / n) * 100
    attack_top5 = (attack_top5 / n) * 100
    return top1, top5,attack_top1,attack_top5, predictions


def zeroshot(model):
    """
    Zero-shot classification using CLIP
    """
    model.eval()
    predictions = np.zeros(len(data_classes),)
    top1 = 0.
    top5 = 0.
    n = 0.

    with tqdm(zeroshot_loader, unit="batch") as tepoch:
        for batch_idx, (data, target) in enumerate(tepoch):
            data, target = data.to(device), target.to(device)
            # batch_images = make_grid(data, nrow=10, normalize=True)
            # if (batch_idx == 0):
            #     save_image(batch_images, f"./original_img/{clipname}/{DATASET}/epoch{ep}.png", normalize=False)

            text_descriptions = [f"This is a photo of a {cl}" for cl in data_classes]
            text_tokens = clip.tokenize(text_descriptions).cuda()
            with torch.no_grad():
                text_features = featurizer.encode_text(text_tokens)
                text_features /= text_features.norm(dim=-1, keepdim=True)
                image_features = featurizer.encode_image(data)
                image_features /= image_features.norm(dim=-1, keepdim=True)
                logits = 100. * image_features @ text_features.T

            preds = torch.argmax(logits, dim=1)
            for p in preds.cpu():
                predictions[p] += 1

            #Calculate accuracy
            acc1, acc5 = accuracy(logits, target, topk=(1, 5))
            top1 += acc1
            top5 += acc5
            n += data.size(0)
    
    top1 = (top1 / n) * 100
    top5 = (top5 / n) * 100
    return top1, top5, predictions
    


if __name__ == "__main__":
    MODEL_TAG = 'CLIP_CIFAR10-corr7'
    DATASET = 'Caltech101'
    checkpoint = "./checkpoints/ViT-B-16/CIFAR10/min_corr7_0.001_cosineTrue_encCLIP_eps01_2022-11-25 16:42:22.356050_chk_ep19.pth"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # clip_models = clip.available_models()[0:1] + clip.available_models()[6:7]
    clipname = 'ViT-B/16'
    featurizer, preprocess = clip.load(clipname)
    featurizer = featurizer.float().to(device)
    print("Loaded clip model")
    fontsize = 5 * round(224/32)
    idx = None # Index of class to be added as text
    eps = 0.1 # Epsilon for projection

    generator = GeneratorResnet_CLIP().to(device)
    generator.load_state_dict(torch.load(f"{checkpoint}"))
    generator.eval()
    
    batch_size = 12
    print("Loaded generator model")

    if DATASET == 'CIFAR10':
        data_classes = get_cifar10_classes('./data/cifar10/batches.meta')
        print(data_classes)
        preprocess_corrupt = transforms.Compose([AddText(data_classes, fontsize=fontsize, index=idx), preprocess])

        trainset = Cifar10_preprocess2(root='./data/cifar10', train=True, download=False, transform_corr=preprocess_corrupt, transform=preprocess)
        train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)

        testset = Cifar10_preprocess2(root='./data/cifar10', train=False, download=False, transform_corr=preprocess_corrupt, transform=preprocess)
        test_loader = torch.utils.data.DataLoader(dataset=testset, batch_size=batch_size, shuffle=False, num_workers=2)

        zeroshot_set = torchvision.datasets.CIFAR10(root='./data/cifar10', train=False, download=False, transform=preprocess)
        zeroshot_loader = torch.utils.data.DataLoader(dataset=zeroshot_set, batch_size=batch_size, shuffle=False, num_workers=2)

    elif DATASET == 'CIFAR100':
        data_classes = get_cifar100_classes('./data/cifar100/meta')
        print(data_classes)
        preprocess_corrupt = transforms.Compose([AddText(data_classes, fontsize=fontsize, index=idx), preprocess])

        testset = Cifar100_preprocess2(root='./data/cifar100', train=False, download=False, transform_corr=preprocess_corrupt, transform=preprocess)
        test_loader = torch.utils.data.DataLoader(dataset=testset, batch_size=batch_size, shuffle=False, num_workers=2)

        zeroshot_set = torchvision.datasets.CIFAR100(root='./data/cifar100', train=False, download=False, transform=preprocess)
        zeroshot_loader = torch.utils.data.DataLoader(dataset=zeroshot_set, batch_size=batch_size, shuffle=False, num_workers=2)

    elif DATASET == 'Caltech101':
        zeroshot_caltech_set = torchvision.datasets.ImageFolder(root='./data/caltech-101/101_ObjectCategories', transform=preprocess)
        data_classes = zeroshot_caltech_set.classes
        print(data_classes)
        _, zeroshot_caltech_set = torch.utils.data.random_split(zeroshot_caltech_set, lengths=[round(0.8*len(zeroshot_caltech_set)), round(0.2*len(zeroshot_caltech_set))])
        zeroshot_loader = torch.utils.data.DataLoader(dataset=zeroshot_caltech_set, batch_size=batch_size, shuffle=False, num_workers=2)

        preprocess_corrupt = transforms.Compose([AddText(data_classes, fontsize=fontsize, index=idx), preprocess])

        caltech_dataset = MyCaltech101(root='./data/caltech-101/101_ObjectCategories', transform_corr=preprocess_corrupt, transform=preprocess)
        caltech_train, caltech_test = torch.utils.data.random_split(caltech_dataset, lengths=[round(0.8*len(caltech_dataset)), round(0.2*len(caltech_dataset))])
        
        test_loader = torch.utils.data.DataLoader(caltech_test, batch_size=batch_size, shuffle=False, num_workers=2)
        

    clipname = clipname.replace('/', '-')
    if not os.path.exists(f"./Adversary_images/{clipname}/{DATASET}/attacks/"):
        os.makedirs(f"./Adversary_images/{clipname}/{DATASET}/attacks/")

    if not os.path.exists(f"./results/attacks/{clipname}/{DATASET}/"):
        os.makedirs(f"./results/attacks/{clipname}/{DATASET}/")


    # print(f"####### Zero Shot performance #######")
    # top1, top5, predictions = zeroshot(generator)
    # print(f"Top1: {top1:.2f} Top5: {top5:.2f}")

    # with open(f"./results/attacks/{clipname}/{DATASET}/{MODEL_TAG}.txt", "w") as f:
    #     f.write(f"####### Zero Shot performance #######\n")
    #     f.write(f"Top1: {top1:.2f} Top5: {top5:.2f}\n")


    print(f"####### Running validation for {DATASET} dataset #######")
    start = time()
    top1, top5, attack_top1, attack_top5, predictions = validate(generator)
    end = time()
    hours, rem = divmod(end - start, 3600)
    mins, secs = divmod(rem, 60)

    print(f"Top1: {top1:.2f} Top5: {top5:.2f}")
    print(f"Attack Top1: {attack_top1:.2f} Attack Top5: {attack_top5:.2f}")
    print(f"Total time taken for inference: {hours:.0f}:{mins:.0f}:{secs:.0f}")

    with open(f'./results/attacks/{clipname}/{DATASET}/{MODEL_TAG}.txt', 'a') as f:
        f.write(f"Top1: {top1:.2f} Top5: {top5:.2f}\n")
        f.write(f"Attack Top1: {attack_top1:.2f} Attack Top5: {attack_top5:.2f}\n")
        f.write(f"Total time taken for inference: {hours:.0f}:{mins:.0f}:{secs:.0f}")