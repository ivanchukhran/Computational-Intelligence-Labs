import os

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import v2

from deap import base, creator, tools, algorithms

from PIL import Image

import numpy as np
from sklearn.model_selection import train_test_split
import random

from tqdm import tqdm

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# search space parameters
PARAM_RANGES = {
    "conv1_filters": (16, 64),
    "conv2_filters": (32, 128),
    "dropout_rate": (0.2, 0.7),
    "linear_units": (64, 256),
    "learning_rate": (0.0001, 0.01),
}

PARAM_BOUNDS = [
    PARAM_RANGES["conv1_filters"],  # (16, 64)
    PARAM_RANGES["conv2_filters"],  # (32, 128)
    PARAM_RANGES["dropout_rate"],  # (0.2, 0.7)
    PARAM_RANGES["linear_units"],  # (64, 256)
    PARAM_RANGES["learning_rate"],  # (0.0001, 0.01)
]


def bounded_gaussian_mutation(individual, mu, sigma, param_bounds, indpb):
    for i in range(len(individual)):
        if random.random() < indpb:
            # Get bounds for this parameter
            lower, upper = param_bounds[i]
            # Apply mutation and clip to bounds
            individual[i] += random.gauss(mu, sigma)
            individual[i] = max(lower, min(upper, individual[i]))
    return (individual,)


class FlexibleConvNet(nn.Module):
    def __init__(self, params):
        super().__init__()
        self.conv1_filters = params["conv1_filters"]
        self.conv2_filters = params["conv2_filters"]
        self.dropout_rate = params["dropout_rate"]
        self.linear_units = params["linear_units"]

        self.network = nn.Sequential(
            nn.Conv2d(3, self.conv1_filters, 3),
            nn.BatchNorm2d(self.conv1_filters),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),
            nn.MaxPool2d(2),
            nn.Conv2d(self.conv1_filters, self.conv2_filters, 3),
            nn.BatchNorm2d(self.conv2_filters),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(self.conv2_filters * 30 * 30, self.linear_units),
            nn.ReLU(),
            nn.Linear(self.linear_units, 5),
        )

    def forward(self, x):
        return self.network(x)


def evaluate_network(individual, train_loader, val_loader, num_epochs=10):
    params = {
        "conv1_filters": int(individual[0]),
        "conv2_filters": int(individual[1]),
        "dropout_rate": float(individual[2]),
        "linear_units": int(individual[3]),
        "learning_rate": float(individual[4]),
    }

    model = FlexibleConvNet(params).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=params["learning_rate"])
    criterion = nn.CrossEntropyLoss()

    for epoch in range(num_epochs):
        model.train()
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    accuracy = correct / total
    return (accuracy,)  # DEAP requires a tuple


def create_genetic_optimizer(train_loader, val_loader):
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()

    toolbox.register(
        "conv1_filters",
        random.randint,
        PARAM_RANGES["conv1_filters"][0],
        PARAM_RANGES["conv1_filters"][1],
    )
    toolbox.register(
        "conv2_filters",
        random.randint,
        PARAM_RANGES["conv2_filters"][0],
        PARAM_RANGES["conv2_filters"][1],
    )
    toolbox.register(
        "dropout_rate",
        random.uniform,
        PARAM_RANGES["dropout_rate"][0],
        PARAM_RANGES["dropout_rate"][1],
    )
    toolbox.register(
        "linear_units",
        random.randint,
        PARAM_RANGES["linear_units"][0],
        PARAM_RANGES["linear_units"][1],
    )
    toolbox.register(
        "learning_rate",
        random.uniform,
        PARAM_RANGES["learning_rate"][0],
        PARAM_RANGES["learning_rate"][1],
    )

    toolbox.register(
        "individual",
        tools.initCycle,
        creator.Individual,
        (
            toolbox.conv1_filters,
            toolbox.conv2_filters,
            toolbox.dropout_rate,
            toolbox.linear_units,
            toolbox.learning_rate,
        ),
        n=1,
    )
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    toolbox.register(
        "evaluate",
        evaluate_network,
        train_loader=train_loader,
        val_loader=val_loader,
    )
    toolbox.register("mate", tools.cxTwoPoint)
    # toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.2)
    toolbox.register(
        "mutate",
        bounded_gaussian_mutation,
        mu=0,
        sigma=0.2,  # Reduced sigma for more controlled mutations
        param_bounds=PARAM_BOUNDS,
        indpb=0.2,
    )
    toolbox.register("select", tools.selTournament, tournsize=3)

    return toolbox


def run_optimization(toolbox, n_generations=20, population_size=10):
    population = toolbox.population(n=population_size)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("max", np.max)

    final_pop, logbook = algorithms.eaSimple(
        population,
        toolbox,
        cxpb=0.7,  # crossover probability
        mutpb=0.3,  # mutation probability
        ngen=n_generations,
        stats=stats,
        verbose=True,
    )

    return tools.selBest(final_pop, k=1)[0], logbook


def optimize_network(train_loader, val_loader):
    toolbox = create_genetic_optimizer(train_loader, val_loader)
    best_individual, log = run_optimization(toolbox)

    best_params = {
        "conv1_filters": int(best_individual[0]),
        "conv2_filters": int(best_individual[1]),
        "dropout_rate": float(best_individual[2]),
        "linear_units": int(best_individual[3]),
        "learning_rate": float(best_individual[4]),
    }

    return best_params, log


class FlowersDataset(Dataset):
    def __init__(self, path: str, transform: v2.Compose | nn.Module | None = None):
        self.path = path
        self.classes = os.listdir(self.path)
        self.filenames = []
        self.targets = []
        for i, cls in enumerate(self.classes):
            cls_path = os.path.join(self.path, cls)
            for filename in os.listdir(cls_path):
                self.filenames.append(os.path.join(cls_path, filename))
                self.targets.append(i)
        self.transform = transform

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        img = Image.open(self.filenames[idx])
        target = self.targets[idx]
        if self.transform:
            img = self.transform(img)
        return img, target


def main():
    dataset_dir = "/home/chukhran/datasets/flowers"
    datapath = os.path.join(dataset_dir, "data")

    transform = v2.Compose(
        [
            v2.Resize((128, 128)),
            v2.RandomVerticalFlip(),
            v2.RandomHorizontalFlip(),
            v2.ToTensor(),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    dataset = FlowersDataset(datapath, transform=transform)
    train_ds, test_ds = train_test_split(dataset, test_size=0.2)

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=32)

    best_params, logs = optimize_network(train_loader, test_loader)
    print("Best parameters found:", best_params)
    print("Logbook:", logs)


if __name__ == "__main__":
    main()
