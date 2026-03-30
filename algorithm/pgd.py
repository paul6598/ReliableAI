


def pgd(model, x, target, k, eps, eps_step):
    """
    model : the neural network
    x : input image tensor
    target : desired (wrong) class label
    k : number of iterations (e.g., 10, 40)
    eps : total perturbation budget
    eps_step : step size per iteration
    return : adversarial image x_adv
    """
    return 0