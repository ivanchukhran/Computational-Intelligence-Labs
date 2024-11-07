import torch
from torch import nn

from torch.optim import Optimizer
from torch.utils.data import DataLoader


from tqdm.notebook import tqdm

from dataset import FlowersDataset, Split

from dataclasses import dataclass

from typing import Any, Callable

@dataclass
class TrainParams:
    model: nn.Module
    optimizer: Optimizer
    loaders: dict[str, DataLoader]
    loss_fn: nn.Module
    num_epochs: int
    device: torch.device
    regularization: dict[Callable, float] = None

@dataclass 
class StepStatistics:
    loss: float
    accuracy: float


def train_step(params: TrainParams) -> StepStatistics:
    final_loss = 0.0
    final_accuracy = 0.0

    model = params.model
    optimizer = params.optimizer
    loader = params.loaders["train"]
    loss_fn = params.loss_fn
    device = params.device
    
    model = model.train()
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        
        optimizer.zero_grad()
        
        y_hat = model(x)
        loss = loss_fn(y_hat, y)

        if params.regularization:
            for reg, factor in params.regularization.items():
                loss += reg(model, factor)

        y = y.int()
        accuracy = (torch.argmax(y_hat, dim=1) == y).float().mean()
        
        loss.backward()
        optimizer.step()
        
        final_loss += loss.item()
        final_accuracy += accuracy.item()
    
    return StepStatistics(
        loss=final_loss / len(loader), 
        accuracy=final_accuracy / len(loader)
    )

def validation_step(params: TrainParams) -> StepStatistics:
    final_loss = 0.0
    final_accuracy = 0.0

    model = params.model
    loader = params.loaders["validation"]
    loss_fn = params.loss_fn
    device = params.device

    # disables weight updates
    model.eval()

    # disables gradient computation for memory efficiency
    with torch.no_grad():
        for x, y in loader: 
            x, y = x.to(device), y.to(device)
            
            y_hat = model(x)
            loss = loss_fn(y_hat, y)

            y = y.int()
            accuracy = (torch.argmax(y_hat, dim=1) == y).float().mean()

            final_loss += loss.item()
            final_accuracy += accuracy.item()

    return StepStatistics( 
        loss=final_loss / len(loader), 
        accuracy=final_accuracy / len(loader)
    )
        

def train(params: TrainParams):
    logs: dict[str, list[float]] = {
        "train_loss": [],
        "train_accuracy": [],
        "validation_loss": [],
        "validation_accuracy": []
    }
    params.model.to(params.device)
    for epoch in tqdm(
        range(params.num_epochs), 
        desc="Epochs", 
        total=params.num_epochs,
        leave=False
        ): 
        train_stats = train_step(params)
        validation_stats = validation_step(params)
        logs["train_loss"].append(train_stats.loss)
        logs["train_accuracy"].append(train_stats.accuracy)
        logs["validation_loss"].append(validation_stats.loss)
        logs["validation_accuracy"].append(validation_stats.accuracy)
    return logs


