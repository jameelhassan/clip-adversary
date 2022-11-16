# clip-adversary
AI701 project - Language as the adversary for CLIP

Set the data in the directory and set it accordingly when using the code here
``` 
    testset = torchvision.datasets.CIFAR10(root='/home/jameel.hassan/Documents/AI701/data/cifar10', train=False, download=False, transform=preprocess <br>
    testloader = torch.utils.data.DataLoader(dataset=testset, batch_size=batch_size, shuffle=False, num_workers=2) 
```
