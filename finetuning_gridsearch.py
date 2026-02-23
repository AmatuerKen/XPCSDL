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
#from functions_training import train_with_scheduler_and_early_stopping_memory_monitor
from functions_bulk_runner import CorrZarrDataset
from functions_finetuning import *

from optuna.samplers import GridSampler
from torch.amp import autocast, GradScaler

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Check if BF16 is supported
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"BF16 supported: {torch.cuda.is_bf16_supported()}")
else:
    print("CUDA not available")

dt_max = 8

binsize = 8
bin = binsize

directory = "./step3_finetuning/"

shear_rate = 0.107
tau0 = 0.0005

dataset = CorrZarrDataset(
    corr_zarr_path_features = directory + f"corr_dataset_maxdt{dt_max:02d}_bin{bin:02d}_feature.zarr",
    corr_zarr_path_target = directory + f"corr_dataset_maxdt{dt_max:02d}_bin{bin:02d}_target.npz",
    shear_rate=shear_rate * tau0 * 100 * 100,
)

idx_data = np.load("./step3_finetuning/dataset_split_index.npz")
idx_train = idx_data['idx_train']
idx_val = idx_data['idx_val']
idx_test = idx_data['idx_test']

train_set = Subset(dataset, idx_train)
val_set   = Subset(dataset, idx_val)
test_set  = Subset(dataset, idx_test)
#plot_target_distribution(train_set, val_set, test_set)

def objective(trial, train_set, val_set, saving_memory = False):

    n_conv = trial.suggest_int("n_conv", 3, 9)
    base_channels = trial.suggest_categorical("base_channels", [4, 8, 16, 32, 64])
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

    total_params = sum(p.numel() for p in model.parameters())
    
    print(f"Total parameters: {total_params/1000000:.2f} M")   
    print(f"Parameter memory (GB): {total_params * 4 / 1024**3:.2f} GB")
    #print("Parameter type:", next(model.parameters()).dtype)
    

    print("before training")
    #print(torch.cuda.memory_summary())
    print_memory(torch.cuda.memory_allocated())

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
    #history, _ = train_with_scheduler_and_early_stopping(model, optimizer, criterion, metric, train_loader, valid_loader,
    #                n_epochs, device, scheduler, show_progress=False)

    if saving_memory:
        history, _ = train_with_scheduler_bestaccurary_memory_monitor_mixedprecision(model, optimizer, criterion, metric, train_loader, valid_loader,
                        n_epochs, device, scheduler, show_progress=False, show_memory= True)
    else:
        history, _ = train_with_scheduler_bestaccurary_memory_monitor(model, optimizer, criterion, metric, train_loader, valid_loader,
                        n_epochs, device, scheduler, show_progress=False, show_memory= True)

    validation_accuracy = min(history["valid_metrics"])

        #if validation_accuracy < best_validation_accuracy:
        #        best_validation_accuracy = validation_accuracy
        #trial.report(validation_accuracy, epoch)
        #if trial.should_prune():
        #    raise optuna.TrialPruned()

    return validation_accuracy

objective_with_data = lambda trial: objective(
    trial, train_set=train_set, val_set=val_set, saving_memory=True)

objective_with_data = partial(objective, train_set=train_set, val_set=val_set, saving_memory=False)

torch.manual_seed(42)

search_space = {
    "n_conv": [3],
    "base_channels": [16],
    "kernel_size": [3, 5, 7],
    "batch_size": [8, 16, 32],
}

sampler = GridSampler(search_space)

study = optuna.create_study(direction="minimize", sampler=sampler)

study.optimize(objective_with_data)

print("Best trial:")
print(study.best_params)

print("Best validation loss:", study.best_value)
np.savez('./step3_finetuning/optuna_best_study_10.npz', study = study)

#3, 4 ...1
#4, 4 ...2
#5, 4 ...3
#6, 4 ...4
#7, 4 ...5
#8, 4 ...6
#3, 8 ...7
#4, 8 ...8
#5-8, 8 ...9
#can't run 9 layers locally
#3, 16 ...10