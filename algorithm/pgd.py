import torch
import torch.nn as nn



def pgd(model, x, target, eps = 0.3, eps_step = 0.01, k = 40, targeted = False, device=None, **kwargs):
    """
    model : the neural network
    x : input image tensor (requires_grad should be set)
    target : the desired (wrong) class label
    eps : perturbation magnitude (e.g., 0.1, 0.3)
    eps_step : step size for each iteration
    k : number of iterations
    return : adversarial image 
    """
    device = device
    x = x.clone().detach().to(device)
    perturbed_image = x.clone().detach().to(device)
    target = target.clone().detach().to(device)
    ce = nn.CrossEntropyLoss()

    x.requires_grad_(True)
    for _ in range(k):
        perturbed_image.requires_grad_(True)

        prediction = model(perturbed_image).to(device)
        
        loss = ce(prediction, target)

        model.zero_grad()
        loss.backward()
        with torch.no_grad():
            data_grad = perturbed_image.grad.data
            if targeted:
                perturbed_image = perturbed_image - eps_step * data_grad.sign()
            else:
                perturbed_image = perturbed_image + eps_step * data_grad.sign()
            perturbation = torch.clamp(perturbed_image - x, min=-eps, max=eps)
            perturbed_image = torch.clamp(x + perturbation, 0, 1).detach()
        perturbed_image = perturbed_image.detach()
        
    return perturbed_image