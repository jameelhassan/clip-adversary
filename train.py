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
from losses import ContrastiveLoss
from utils import *
from torchvision.utils import save_image
from torchvision.utils import make_grid
from dataset_cifar import Cifar10_preprocess2

class AddText(object):
    """
    Add a randomly chosen class as text on the image
    """
    def __init__(self, classes, fontsize=5, index=0, random_choice=False):
        self.classes = classes
        self.index = random.choice(range(len(classes))) if index None else index # Randomly get index of class to corrupt if not specified
        self.fontsize = fontsize
        self.random_choice = random_choice
        self.font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', self.fontsize)

    def __call__(self, sample):
        image = sample
        text_class = np.random.choice(self.classes) if self.random_choice else self.classes[self.index]
        img_tf = ImageDraw.Draw(image)
        
        #Setting possible positions and colours of text and choosing one in random 
        text_locs = [(np.round(i * image.size[0]), np.round(j * image.size[1])) for (i,j) in [(0.25, 0.25), (0.25, 0.6), (0.75, 0.25), (0.6, 0.6)]]
        text_pos = random.choice(text_locs)
        text_cols = [(255,0,0), (0,255,0), (0,0,255), (0,0,0), (255,255,255)]
        text_col = random.choice(text_cols) #(0,0,0) for Black
        img_tf.text(text_pos, text_class, text_col, font=self.font)

        return image


def train(model):
    model.train()
    train_loss = 0

    with tqdm(train_loader, unit="batch") as tepoch:
        for batch_idx, (img_corr, data, text_corr_idx, target) in enumerate(tepoch):
            img_corr, data, target = img_corr.to(device), data.to(device), target.to(device)
            optimizer.zero_grad()
            corrupt_text = cifar_classes[text_corr_idx] # Get the class of corrupt label added to image

            text_corrupt = clip.tokenize([f"This is a photo of a {corrupt_text}"]).to(device)
            structured_noise = model(img_corr)      # Structured noise from generator with input as corrupted image
            adversary = structured_noise + data     # Add original image to structured noise with input as original image

            # projection
            adversary = torch.min(torch.max(adversary, data - eps), data + eps)
            # adversary = torch.clamp(adversary, 0.0, 1.0)
            # print(output.shape)

            z = featurizer.encode_image(data)
            z_hat = featurizer.encode_image(adversary)
            t_neg = featurizer.encode_text(text_corrupt)
            # t_neg = t_neg.expand #if needed
            # loss = F.mse_loss(z_hat, t_neg) #MSE loss on the corrupted image and text embeddings

            loss = criterion(z_hat, z, t_neg)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

    return train_loss


def validate(model):   
    model.eval()
    predictions = np.zeros(10,)
    top1 = 0.
    top5 = 0.
    n = 0.

    with tqdm(test_loader, unit="batch") as tepoch:
        for batch_idx, (img_corr, data, text_corr_idx, target) in enumerate(tepoch):
            img_corr, data, target = img_corr.to(device), data.to(device), target.to(device)
            optimizer.zero_grad()

            with torch.no_grad():
                structured_noise = model(img_corr)      # Structured noise from generator with input as corrupted image
                adversary = structured_noise + data     # Add original image to structured noise with input as original image
                adversary = torch.min(torch.max(adversary, data - eps), data + eps)
                # adversary = torch.clamp(adversary, 0.0, 1.0)
                batch_images = make_grid(adversary, nrow=10, normalize=True)
                if (batch_idx == 0):
                    save_image(batch_images, f"./Adversary_images/{clipname}/epoch{ep}_eps{str(eps)}.png", normalize=False)

            text_descriptions = [f"This is a photo of a {cl}" for cl in cifar_classes]
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
            acc1, acc5 = accuracy(logits, target, topk=(1, 5))
            top1 += acc1
            top5 += acc5
            n += data.size(0)
    
    top1 = (top1 / n) * 100
    top5 = (top5 / n) * 100
    return top1, top5, predictions


def zeroshot(model):
    """
    Zero-shot classification using CLIP
    """
    model.eval()
    predictions = np.zeros(10,)
    top1 = 0.
    top5 = 0.
    n = 0.

    with tqdm(zeroshot_loader, unit="batch") as tepoch:
        for batch_idx, (data, target) in enumerate(tepoch):
            data, target = data.to(device), target.to(device)
            batch_images = make_grid(data, nrow=10, normalize=True)
            if (batch_idx == 0):
                save_image(batch_images, f"./original_img/{clipname}/epoch{ep}.png", normalize=False)

            text_descriptions = [f"This is a photo of a {cl}" for cl in cifar_classes]
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
    

if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # clip_models = clip.available_models()[0:1] + clip.available_models()[6:7]
    clipname = 'ViT-B/16'
    featurizer, preprocess = clip.load(clipname)
    featurizer = featurizer.float().to(device)
    print("Loaded clip model")
    fontsize = 5
    idx = None # Index of class to be added as text
    eps = 0.05 # Epsilon for projection
    learning_rate = 1e-4

    model = GeneratorResnet().to(device)
    model.to(device)
    criterion = ContrastiveLoss()
    optimizer = torch.optim.AdamW(list(model.parameters()), lr=learning_rate)
    batch_size = 24
    epochs = 10
    print("Loaded generator model")

    cifar_classes = get_cifar10_classes('./data/cifar10/batches.meta')
    print(cifar_classes)
    preprocess_corrupt = transforms.Compose([AddText(cifar_classes, fontsize=fontsize, index=idx), preprocess])

    trainset = Cifar10_preprocess2(root='./data/cifar10', train=True, download=False, transform_corr=preprocess_corrupt, transform=preprocess)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)

    testset = Cifar10_preprocess2(root='./data/cifar10', train=False, download=False, transform_corr=preprocess_corrupt, transform=preprocess)
    test_loader = torch.utils.data.DataLoader(dataset=testset, batch_size=batch_size, shuffle=False, num_workers=2)

    zeroshot_set = torchvision.datasets.CIFAR10(root='./data/cifar10', train=False, download=False, transform=preprocess)
    zeroshot_loader = torch.utils.data.DataLoader(dataset=zeroshot_set, batch_size=batch_size, shuffle=False, num_workers=2)

    clipname = clipname.replace('/', '-')
    if not os.path.exists(f"./Adversary_images/{clipname}/"):
        os.makedirs(f"./Adversary_images/{clipname}/")
        os.makedirs(f"./original_img/{clipname}")

    if not os.path.exists(f'./checkpoints/{clipname}/'):
        os.makedirs(f'./checkpoints/{clipname}/')

    
    for ep in range(epochs):
        if ep == 0:
            print(f"####### Zero Shot CLIP performance #########")
            top1, top5, predictions = zeroshot(model)
            # print(f"Epoch {ep} - Top1: {top1:.2f} Top5: {top5:.2f}")
            # print(f"Predictions: {predictions}")

            with open(f'checkpoints/{clipname}/chk_fs{fontsize}.txt', 'a') as f:
                f.write(f"####### Zero Shot CLIP performance #########\n")
                f.write(f"Epoch {ep} - Top1: {top1:.2f} Top5: {top5:.2f}\n")
                f.write(f"Class label {idx}: {cifar_classes[idx]} corruption predictions - {predictions}\n")
                f.write(100*"-" + "\n")

        if ((ep + 1) % 5 == 0 or ep == 0):
            top1, top5, predictions = validate(model)
            print(f"Epoch {ep} - Top1: {top1:.2f} Top5: {top5:.2f}\n")
            print(f"Class label {idx}: {cifar_classes[idx]} corruption predictions - {predictions}\n")

            with open(f'checkpoints/{clipname}/chk_fs{fontsize}.txt', 'a') as f:
                f.write(f"Epoch {ep} - Top1: {top1:.2f} Top5: {top5:.2f}\n")
                f.write(f"Class label {idx}: {cifar_classes[idx]} corruption predictions - {predictions}\n")

            model_weights = model.state_dict()
            torch.save(model_weights, f'checkpoints/{clipname}/chk_ep{ep}.pth')

        train_loss = train(model)
        print(f"Epoch {ep} - Train loss: {train_loss:.2f}")
        with open(f'checkpoints/{clipname}/chk_ep_fs{fontsize}.txt', 'a') as f:
            f.write(f"Epoch {ep} - Train loss: {train_loss:.2f}\n")
        
