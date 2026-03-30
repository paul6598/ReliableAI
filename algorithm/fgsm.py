import torch
import torch.nn as nn


def fgsm(model, x, target, eps, targeted):
    """
    model : the neural network
    x : input image tensor (requires_grad should be set)
    target : the desired (wrong) class label
    eps : perturbation magnitude (e.g., 0.1, 0.3)
    return : adversarial image x_adv
    """
    x.requires_grad_(True)
    prediction = model(x)
    ce = nn.CrossEntropyLoss()
    loss = ce(prediction, target)

    model.zero_grad()
    loss.backward()

    data_grad = x.grad.data
    if targeted:
        perturbed_image = x - eps * data_grad.sign()
    else:
        perturbed_image = x + eps * data_grad.sign()
    perturbed_image = torch.clamp(perturbed_image, 0, 1)

    return perturbed_image

# def fgsm_untargeted(model, x, label, eps):
#     """
#     model : the neural network
#     x : input image tensor
#     label : the correct class label
#     eps : perturbation magnitude
#     return : adversarial image x_adv
#     """
#     x.requires_grad_(True)
#     prediction = model(x)
#     ce = F.CrossEntropyLoss()
#     loss = ce(prediction, label)

#     model.zero_grad()
#     loss.backward()

#     data_grad = x.grad.data
#     perturbed_image = x + eps * data_grad.sign()
#     perturbed_image = torch.clamp(perturbed_image, 0, 1)

#     return perturbed_image
