import numpy as np
import zarr
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from copy import deepcopy

from torch.amp import autocast

from functions_dataset import create_frame_pairs, create_correlation_zarr  
from functions_training import evaluate_tm

def create_single_correlation_dataset(directory, sp_zarr_path, dt_min, dt_max, N_target, N_entry, bin, bounds, show_examples=True):

    N_frames = 10000
    pairs = create_frame_pairs(N_frames, dt_min, dt_max, N_target, N_entry) 
    #print(pairs.shape)
    #print(pairs[2])
    pair_filename = f"A209.imm.dataset_pair_maxdt{dt_max:02d}.npy"
    np.save(directory + pair_filename, pairs)



    corr_zarr_path_feature = directory +  f"corr_dataset_maxdt{dt_max:02d}_bin{bin:02d}_feature.zarr"
    corr_zarr_path_target = directory +  f"corr_dataset_maxdt{dt_max:02d}_bin{bin:02d}_target.npz"

    create_correlation_zarr(sp_zarr_path, corr_zarr_path_feature, corr_zarr_path_target, pairs, bin, bounds)

    if show_examples:
        X = zarr.open(corr_zarr_path_feature, mode="r")

        # load 5 examples (first 5 along axis 0)
        n_list = np.array(range(N_target) )* N_entry

        fig, axes = plt.subplots(1, N_target, figsize=(4*N_target, 4))
        for i, ax in enumerate(axes):
            ax.imshow(X[n_list[i]], cmap="viridis")
            ax.axis("off")
            ax.set_title(f"dt = {pairs[n_list[i], 2]:02d}")

        plt.savefig(directory + f"corr_maxdt{dt_max:02d}_bin{bin:02d}_feature.png")
        plt.close()

    print(f"mmaxdt{dt_max:02d} is done.")

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


def train_with_scheduler_bestaccurary_memory_monitor(model, optimizer, loss_fn, metric, train_loader, valid_loader,
          n_epochs, device, scheduler, show_progress=True, show_memory = True):
    history = {"train_losses": [], "train_metrics": [], "valid_metrics": []}

    best_val_metric = np.inf
    
    for epoch in range(n_epochs):
        total_loss = 0.0
        metric.reset()
        model.train()

        for X_batch, y_batch in train_loader:

            torch.cuda.reset_peak_memory_stats()
            
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            y_pred = model(X_batch)

            if show_memory:
                print("after forward, memory allocated")
                #print(torch.cuda.memory_summary())
                print_memory(torch.cuda.memory_allocated())
                print("max memory allocated")
                print_memory(torch.cuda.max_memory_allocated())

            loss = loss_fn(y_pred, y_batch)
            total_loss += loss.item()
            loss.backward()
            
            if show_memory:
                print("after backward")
                print_memory(torch.cuda.memory_allocated())
                print("max memory allocated")
                print_memory(torch.cuda.max_memory_allocated())

            optimizer.step()
            optimizer.zero_grad()

            if show_memory:
                print("after optimizer step")
                print_memory(torch.cuda.memory_allocated())
                print("max memory allocated")
                print_memory(torch.cuda.max_memory_allocated())
            #only show one round
            show_memory = False

            metric.update(y_pred, y_batch)

        history["train_losses"].append(total_loss / len(train_loader))
        history["train_metrics"].append(metric.compute().item())

        val_metric = evaluate_tm(model, valid_loader, metric, device).item()

        history["valid_metrics"].append(val_metric)
        scheduler.step(val_metric)

        if val_metric < best_val_metric:
            best_val_metric = val_metric
            best_model = deepcopy(model)

        if show_progress:
            print(f"Epoch {epoch + 1}/{n_epochs}, "
                  f"train loss: {history['train_losses'][-1]:.4f}, "
                  f"train metric: {history['train_metrics'][-1]:.4f}, "
                  f"valid metric: {history['valid_metrics'][-1]:.4f}")
                  
    return history, best_model

def print_memory(memory):
    print(f"{memory / 1024**2:.2f} MB")

def train_with_scheduler_bestaccurary_memory_monitor_mixedprecision(model, optimizer, loss_fn, metric, train_loader, valid_loader,
          n_epochs, device, scheduler, show_progress=True, show_memory = True):

    history = {"train_losses": [], "train_metrics": [], "valid_metrics": []}

    best_val_metric = np.inf
    
    for epoch in range(n_epochs):
        total_loss = 0.0
        metric.reset()
        model.train()


        for X_batch, y_batch in train_loader:

            torch.cuda.reset_peak_memory_stats()

            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            with autocast('cuda', dtype=torch.bfloat16):
                y_pred = model(X_batch)

                if show_memory:
                    print("after forward")
                    print_memory(torch.cuda.memory_allocated())
                    print("max memory allocated")
                    print_memory(torch.cuda.max_memory_allocated())

                loss = loss_fn(y_pred, y_batch)

            total_loss += loss.item()
            loss.backward()
            
            if show_memory:
                print("after backward")
                print_memory(torch.cuda.memory_allocated())
                print("max memory allocated")
                print_memory(torch.cuda.max_memory_allocated())

            optimizer.step()
            optimizer.zero_grad()

            if show_memory:
                print("after optimizer step")
                print_memory(torch.cuda.memory_allocated())
                print("max memory allocated")
                print_memory(torch.cuda.max_memory_allocated())
            #only show one round
            show_memory = False

            metric.update(y_pred, y_batch)

        history["train_losses"].append(total_loss / len(train_loader))
        history["train_metrics"].append(metric.compute().item())

        val_metric = evaluate_tm(model, valid_loader, metric, device).item()

        history["valid_metrics"].append(val_metric)
        scheduler.step(val_metric)

        if val_metric < best_val_metric:
            best_val_metric = val_metric
            best_model = deepcopy(model)

        if show_progress:
            print(f"Epoch {epoch + 1}/{n_epochs}, "
                  f"train loss: {history['train_losses'][-1]:.4f}, "
                  f"train metric: {history['train_metrics'][-1]:.4f}, "
                  f"valid metric: {history['valid_metrics'][-1]:.4f}")
                  
    return history, best_model