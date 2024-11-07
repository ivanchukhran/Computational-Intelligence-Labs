import torch

def l1_regularization(model, factor=0.01):
    l1_loss = 0
    for param in model.parameters():
        l1_loss += torch.norm(param, 1)
    return factor * l1_loss

def l2_regularization(model, factor=0.01):
    l2_loss = 0
    for param in model.parameters():
        l2_loss += torch.norm(param, 2)
    return factor * l2_loss