import os
import torch
from torchvision import utils

from utils import preprocess, train
from algorithm.fgsm import fgsm
from algorithm.pgd import pgd

def main():
    eps = 0.1
    dataset = "mnist" #mnist, cifar10
    save_path = f"./results/{dataset}_images"
    os.makedirs(save_path, exist_ok=True)

    train_loader, test_loader = preprocess(dataset)
    model = train(dataset, train_loader, test_loader)


    # for targeted in [True, False]:
    #     FGSM_image = fgsm(model, x, target, eps, targeted)
    #     PGD_image = pgd(model, x, target, eps, targeted)
    #     utils.save_image(FGSM_image, os.path.join(save_path, 'FGSM_result.png'))
    #     utils.save_image(PGD_image, os.path.join(save_path, 'PGD_result.png'))


if __name__ == "main":
    main()