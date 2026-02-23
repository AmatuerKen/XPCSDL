import numpy as np
import zarr
import matplotlib.pyplot as plt

import optuna
from functools import partial

import torch
from torch.utils.data import Dataset, Subset, random_split, DataLoader
from tqdm import tqdm
import torch.nn as nn
from torchmetrics import MeanSquaredError
from torch.optim.lr_scheduler import ReduceLROnPlateau

from functions_dataset import *
from functions_training import train_with_scheduler_and_early_stopping
from functions_bulk_runner import CorrZarrDataset
from functions_finetuning import *


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

#the goal is to test max learnable gamma (dt_max), keep N_target the the same as dt_max, but keep the total number of samples the same 

dt_min = 1

dt_max = 8
#N_target = dt_max

#total number of samples
N_sample = 20000

binsize = 8

sp_zarr_path = "A209.zarr"
directory = "./step3_finetuning/"
#create a boolean mask for the center of the image, keep the center of the image
#xc = 57
#yc = 777
bounds = (57, 117, 777, 837)


shear_rate = 0.107
tau0 = 0.0005

N = N_sample

train_frac = 0.7
val_frac   = 0.15
test_frac  = 0.15

n_train = int(train_frac * N)
n_val   = int(val_frac * N)
n_test  = N - n_train - n_val

N_target = dt_max
N_entry = int(N_sample/N_target)


#create_single_correlation_dataset(directory, sp_zarr_path, dt_min, dt_max, N_target, N_entry, binsize, bounds, show_examples=False)


class ConvRegressor(nn.Module):
    def __init__(self, n_conv, base_channels, kernel_size):
        super().__init__()

        #padding = kernel_size // 2
        padding = 1
        stride = 1
        layers = []

        in_ch = 1
        out_ch = base_channels

        for i in range(n_conv):
            layers.append(nn.Conv2d(in_ch, out_ch, kernel_size, stride = stride, padding=padding))
            layers.append(nn.ReLU())
            #layers.append(nn.MaxPool2d(2))
            in_ch = out_ch
            out_ch *= 2
        
        layers.append(nn.AdaptiveAvgPool2d((1, 1)))

        self.conv = nn.Sequential(*layers)


        N_neuron1 = int(out_ch/2)
        N_neuron2 = int(np.power(N_neuron1, 2/3))
        N_neuron3 = int(np.power(N_neuron1, 1/3))

        self.fc = nn.Sequential(
            nn.Linear(N_neuron1, N_neuron2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(N_neuron2, N_neuron3),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(N_neuron3, 1)
        )

    def forward(self, x):
        x = x.unsqueeze(1)      # (B, 1, H, W)

        h = self.conv(x).flatten(1)
        out = self.fc(h)
        return out.squeeze(-1)


def objective(trial, train_set, val_set):
    print('new trial')
    n_conv = trial.suggest_int("n_conv", 3, 9)
    base_channels = trial.suggest_categorical("base_channels", [4, 16, 32, 64])
    kernel_size = trial.suggest_categorical("kernel_size", [3, 5, 7])

    #n_conv = 3
    #base_channels = 4
    #kernel_size = 3
    batch_size = trial.suggest_categorical("batch_size", [8, 16, 32])

    model = ConvRegressor(
            n_conv=n_conv,
            base_channels=base_channels,
            kernel_size=kernel_size,
        ).to(device)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    #test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)


    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=2)
    criterion = torch.nn.MSELoss()
    metric = MeanSquaredError().to(device)

    #best_validation_accuracy = np.inf #loss

    n_epochs = 20

    #for epoch in range(n_epochs):
    history, _ = train_with_scheduler_and_early_stopping(model, optimizer, criterion, metric, train_loader, valid_loader,
                    n_epochs, device, scheduler, show_progress=False)

    validation_accuracy = min(history["valid_metrics"])

        #if validation_accuracy < best_validation_accuracy:
        #        best_validation_accuracy = validation_accuracy
        #trial.report(validation_accuracy, epoch)
        #if trial.should_prune():
        #    raise optuna.TrialPruned()

    return validation_accuracy


bin = binsize

dataset = CorrZarrDataset(
    corr_zarr_path_features = directory + f"corr_dataset_maxdt{dt_max:02d}_bin{bin:02d}_feature.zarr",
    corr_zarr_path_target = directory + f"corr_dataset_maxdt{dt_max:02d}_bin{bin:02d}_target.npz",
    shear_rate=shear_rate * tau0 * 100 * 100,
)

targets = [dataset[i][1] for i in range(len(dataset))]

idx_train, idx_val, idx_test = stratified_regression_split(targets)

train_set = Subset(dataset, idx_train)
val_set   = Subset(dataset, idx_val)
test_set  = Subset(dataset, idx_test)
#plot_target_distribution(train_set, val_set, test_set)


objective_with_data = lambda trial: objective(
    trial, train_set=train_set, val_set=val_set)

objective_with_data = partial(objective, train_set=train_set, val_set=val_set)

torch.manual_seed(42)
sampler = optuna.samplers.TPESampler(seed=42)
pruner = optuna.pruners.MedianPruner()

study = optuna.create_study(direction="minimize", sampler=sampler, pruner=pruner)

study.optimize(objective_with_data, n_trials=30)

print("Best trial:")
print(study.best_params)

print("Best validation loss:", study.best_value)
np.savez('./optuna_study.npz', study = study)