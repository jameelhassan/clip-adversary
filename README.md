# clip-adversary
AI701 project - Language as the adversary for CLIP

Use meta files in *data/* folder for ```get_cifar_classes()``` function (both cifar10 and cifar100). 

Set the data in the directory and set it accordingly when using the code here. You can set download to TRUE for the first time to download the data using pytorch. The data will be in root folder inside *data/* folder. 
``` 
    testset = torchvision.datasets.CIFAR10(root='/home/jameel.hassan/Documents/AI701/data/cifar10', train=False, download=False, transform=preprocess
    testloader = torch.utils.data.DataLoader(dataset=testset, batch_size=batch_size, shuffle=False, num_workers=2) 
```

To get CLIP evaluation on CIFAR10 and CIFAR100 run ```eval.py``` \
To get evaluation with text corruption set ```TEXT_CORRUPT=True``` in file. For Zero shot set it to **False**. 