import torch
import torch.nn as nn



def pgd(model, x, target, k, eps = 0.1, eps_step = 0.01, targeted = False, device=None, **kwargs):
    """
    model : the neural network
    x : input image tensor
    target : desired (wrong) class label
    k : number of iterations (e.g., 10, 40)
    eps : total perturbation budget
    eps_step : step size per iteration
    return : adversarial image x_adv
    """
    device = device
    x = x.clone().detach().to(device)
    perturbed_image = x.clone().detach().to(device)
    target = target.clone().detach().to(device)

    x.requires_grad_(True)
    for _ in range(k):
        perturbed_image.requires_grad_(True)

        prediction = model(perturbed_image).to(device)
        ce = nn.CrossEntropyLoss()
        loss = ce(prediction, target)

        model.zero_grad()
        loss.backward()

        data_grad = perturbed_image.grad.data
        if targeted:
            perturbed_image = perturbed_image - eps_step * data_grad.sign()
        else:
            perturbed_image = perturbed_image + eps_step * data_grad.sign()
        perturbation = torch.clamp(perturbed_image - x, min=-eps, max=eps)
        perturbed_image = torch.clamp(x + perturbation, 0, 1).detach()
        
        
    return perturbed_image