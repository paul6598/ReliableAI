import os

import torch
from torchvision import utils

from utils import preprocess, train, targeted_label, adversarial_attack
from algorithm.fgsm import fgsm
from algorithm.pgd import pgd

def main():
    eps = 0.05
    eps_step = 0.05/30
    k = 200
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for dataset in ["cifar10", "mnist"]:
        train_loader, test_loader = preprocess(dataset)
        model = train(dataset, train_loader, test_loader)
        for targeted in [True, False]:
            for attack_function in [fgsm, pgd]:
                print("=" * 50)
                print(f"Processing dataset: {dataset}")
                print(f"Attack method: {'targeted' if targeted else 'untargeted'} {attack_function.__name__}")
                adversarial_attack(model, test_loader, dataset, targeted, attack_function, eps = eps, eps_step = eps_step, k = k, device = device)
if __name__ == "__main__":
    main()