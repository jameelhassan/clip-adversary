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


def train(model):
    model.train()
    train_loss = 0
    total = 0
    correct=0
    with tqdm(train_loader, unit="batch") as tepoch:
        for batch_idx, (data, target) in enumerate(tepoch):
            # send to device
            data, target = data.to(device), target.to(device)
            # print(target.shape)
            text_corrupt = clip.tokenize(["a photo of a {corrupt_text}"]).to(device)

            
            # clear the gradients of all optimized variables
            optimizer.zero_grad()
            # forward pass: compute predicted outputs by passing inputs to the model
            output = model(data)
            # print(output.shape)

            z = featurizer.encode_image(data)
            z_hat = featurizer.encode_image(output)
            t_neg = featurizer.encode_text(text_corrupt)
            # t_neg = t_neg.expand #if needed
            # loss = F.mse_loss(z_hat, t_neg) #MSE loss on the corrupted image and text embeddings

            loss = criterion(z_hat, z, t_neg)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

            # update running training loss
        #     train_loss += loss.item()*data.size(0)
        #     _, predicted = output.max(1)
        #     total += target.size(0)
        #     correct += predicted.eq(target).sum().item()
        #     train_accuracy = 100.*correct/total
        # print(' train loss: {:.4f} accuracy: {:.4f}'.format(train_loss/(batch_idx+1), 100.*correct/total))

    return train_loss


def validate(model):   
    model.eval()

    test_loss = 0
    correct = 0
    total = 0
    with tqdm(test_loader, unit="batch") as tepoch:
        for batch_idx, (data, target) in enumerate(tepoch):
            # send to device
            data, target = data.to(device), target.to(device)
    
            output = model(data)
            loss = nn.CrossEntropyLoss()(output, target)
            test_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
            test_accuracy = 100.*correct/total
        print(' test loss: {:.4f} accuracy: {:.4f}'.format(test_loss/(batch_idx+1), 100.*correct/total))

    return test_loss, test_accuracy


if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"
    clip_models = clip.available_models()[0:1] + clip.available_models()[6:7]
    featurizer, preprocess = clip.load('ViT-B/16')
    featurizer = featurizer.float().to(device)
    print("Loaded clip model")

    model = GeneratorResnet().to(device)
    model.to(device)
    print("Loaded generator model")

    criterion = ContrastiveLoss()
    optimizer = torch.optim.AdamW(list(model.parameters()), lr=5e-6)
    batch_size = 24
    epochs = 10

    trainset = torchvision.datasets.CIFAR10(root='./data/cifar10', train=True, download=False, transform=preprocess)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./data/cifar10', train=False, download=False, transform=preprocess)
    test_loader = torch.utils.data.DataLoader(dataset=testset, batch_size=batch_size, shuffle=False, num_workers=2)

    for ep in range(epochs):
        loss, acc = train(model)

        if ep % 5 == 0:
            val_loss, val_acc = validate(model)