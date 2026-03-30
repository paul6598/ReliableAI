import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torchvision import datasets, transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from algorithm.fgsm import fgsm
from algorithm.pgd import pgd

class cnn_for_mnist(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
class pretrained_for_cifar10(nn.Module):
    def __init__(self, pretrained_model):
        super().__init__()
        self.pretrained_model = pretrained_model 

        self.register_buffer('mean', torch.tensor([0.4914, 0.4822, 0.4465]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.2023, 0.1994, 0.2010]).view(1, 3, 1, 1))    

    def forward(self, x):
            x_normalized = (x - self.mean) / self.std
            return self.pretrained_model(x_normalized) 


def preprocess(dataset_name, batch_size = 128):
    """
    dataset_name: "mnist" 또는 "cifar10"
    return: train_loader, test_loader
    """
    
    if dataset_name == "mnist":
        transform = transforms.Compose([
            transforms.ToTensor(),
        ])

        train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
        test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    elif dataset_name == "cifar10":
        transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        
        train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
        test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader


def train(dataset_name, train_loader, test_loader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if dataset_name == "mnist":
        model = cnn_for_mnist().to(device)
        loss_function = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9) 
        for epoch in range(5):  # loop over the dataset multiple times

            running_loss = 0.0
            for i, data in enumerate(train_loader, 0):
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data[0].to(device), data[1].to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = model(inputs)
                loss = loss_function(outputs, labels)
                loss.backward()
                optimizer.step()

                # print statistics
                running_loss += loss.item()
                # if i % 100 == 99:    
                #     print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
                #     running_loss = 0.0
        model.eval()
    
    if dataset_name == "cifar10":
        pretrained_model = torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar10_resnet20", pretrained=True)
        pretrained_model.eval()
        model = pretrained_for_cifar10(pretrained_model).to(device)
        model.eval()

    return model

def adversarial_attack(model, loader, dataset, targeted, attack_function, **kwargs):
    """
    model : the neural network
    loader : the data loader
    x : input image tensor
    targeted : whether the attack is targeted
    attack_function : the attack function to use
    attack_method : the attack method to use (e.g., "fgsm", "pgd")
    kwargs : additional arguments for the attack method
    return : adversarial image x_adv
    """
    save_path = f"./results/{dataset}/{'targeted' if targeted else 'untargeted'}/{attack_function.__name__}"
    os.makedirs(save_path, exist_ok=True)
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    total = 0
    correct = 0
    attack_success = 0
    
    images, labels = next(iter(loader))

    images, labels = images.to(device), labels.to(device)
    total = labels.size(0)
    with torch.no_grad():
        outputs = model(images)
        preds = outputs.argmax(dim=1)
        correct_mask = (preds == labels)
        correct += correct_mask.sum().item()

    if targeted:
        target_labels = targeted_label(labels)
    else:
        target_labels = labels
    attacked_images = attack_function(model, images, target_labels, **kwargs)
    save_images(attacked_images.cpu(), save_path)
    with torch.no_grad():
        outputs = model(attacked_images)
        attacked_preds = outputs.argmax(dim=1)
        if targeted:
            attack_success += (attacked_preds == target_labels).sum().item()
        else:
            attack_success += (attacked_preds != labels).sum().item()

    print(f"Attack success rate: {attack_success / total * 100:.2f}%")

def save_images(images, path):
    """
    images : a batch of image tensors
    path : the path to save the images
    """
    for i in range(10):
        save_image(images[i], os.path.join(path, f"result_{i}.png"))

def targeted_label(true_label, num_classes=10):
    """
    true_label: the correct class label
    num_classes: total number of classes in the dataset
    return: a random target label different from the true label
    """
    targets = torch.randint(0, num_classes, true_label.shape).to(true_label.device)
    mask = (targets == true_label)

    targets[mask] = (targets[mask] + 1) % num_classes
    
    return targets
