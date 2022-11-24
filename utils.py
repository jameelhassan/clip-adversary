import numpy as np

def accuracy(output, target, topk=(1,)):
    pred = output.topk(max(topk), 1, True, True)[1].t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    return [float(correct[:k].reshape(-1).float().sum(0, keepdim=True).cpu().numpy()) for k in topk]


def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='latin1')
    return dict


def get_cifar10_classes(file):
    """
    Get the Cifar10 classes as a list for AddText transform
    """
    data = unpickle(file)
    classes = data['label_names']
    return classes


def get_cifar100_classes(file):
    """
    Get the Cifar100 classes as a list for AddText transform
    """
    data = unpickle(file)
    classes = data['fine_label_names']
    return classes


def prepare_caltech101(path, transforms=None):
    image_paths = list(paths.list_images(path + '/101_ObjectCategories'))

    data = []
    labels = []
    for img_path in tqdm(image_paths):
        label = img_path.split(os.path.sep)[-2]
        if label == "BACKGROUND_Google":
            continue
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        data.append(img)
        labels.append(label)
        
    data = np.array(data)
    labels = np.array(labels)
    return data, labels

