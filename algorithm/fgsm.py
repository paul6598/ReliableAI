import torch
import torch.nn as F


def fgsm_targeted(model, x, target, eps):
    """
    model : the neural network
    x : input image tensor (requires_grad should be set)
    target : the desired (wrong) class label
    eps : perturbation magnitude (e.g., 0.1, 0.3)
    return : adversarial image x_adv
    """
    x.requires_grad_(True)
    prediction = model(x)
    ce = F.CrossEntropyLoss()
    loss = ce(prediction, target)

    model.zero_grad()
    loss.backward()

    data_grad = x.grad.data
    perturbed_image = x - eps * data_grad.sign()
    perturbed_image = torch.clamp(perturbed_image, 0, 1)

    return perturbed_image
