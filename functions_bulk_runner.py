import numpy as np
import zarr
import matplotlib.pyplot as plt
import torch
from torch.nn import Module
from torch.utils.data import Dataset
from torch import nn
from functions_dataset import create_frame_pairs, create_correlation_zarr  
import matplotlib.gridspec as gridspec

class CorrZarrDataset(Dataset):
    def __init__(self, corr_zarr_path_features, corr_zarr_path_target, shear_rate):
        root_features = zarr.open(corr_zarr_path_features, mode="r")
        self.X = root_features
        #root_target = zarr.open(corr_zarr_path_target, mode="r")
        #self.Y = root_target
        target_npz = np.load(corr_zarr_path_target)
        self.Y = target_npz["target"]
        self.shear_rate = shear_rate

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        x = torch.from_numpy(self.X[idx])
        #y = torch.tensor(self.Y[idx] * self.shear_rate, dtype=torch.float32)
        y = torch.tensor(self.Y[idx] * self.shear_rate, dtype=torch.float32)

        return x, y

def create_correlation_dataset(directory, sp_zarr_path, dt_min, dt_max, N_target, N_entry, binsize_list, bounds, show_examples=True):

    N_frames = 10000
    pairs = create_frame_pairs(N_frames, dt_min, dt_max, N_target, N_entry) 
    #print(pairs.shape)
    #print(pairs[2])
    pair_filename = f"A209.imm.dataset_pair_maxdt{dt_max:02d}.npy"
    np.save(directory + pair_filename, pairs)

    for bin in binsize_list:

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


class CovNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=16, stride=2, padding=2),
            nn.ReLU(),
            #resultant size is (60 + 4 - 10)/2 + 1 = 28
            nn.MaxPool2d(kernel_size=2, stride=1),
            #(B, 8, 27)
        
            nn.Conv2d(16, 32, kernel_size=16, stride=2, padding=2),
            nn.ReLU(),
            #resultant size is (27 + 4 - 10)/2 + 1 = 11
            nn.MaxPool2d(kernel_size=2, stride=1),
            #(B, 16, 10)

            nn.Conv2d(32, 64, kernel_size=8, stride=1, padding=2),
            nn.ReLU(),
            #resultant size is (10 + 4 - 8)/2 + 1 = 4
            nn.MaxPool2d(kernel_size=2, stride=1),
            #(B, 16, 3)

            nn.Conv2d(64, 128, kernel_size=4, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=1),

            nn.Conv2d(128, 256, kernel_size=4, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=1),

            nn.Conv2d(256, 512, kernel_size=2, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=1),
            

            nn.AdaptiveAvgPool2d((1, 1))
        )

        self.fc = nn.Sequential(
            nn.Linear(512, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 16),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(16, 1)
        )

    def forward(self, x):
        # x: (B, H, W) H = W = 60
        x = x.unsqueeze(1)      # (B, 1, H, W)

        h = self.conv(x).flatten(1)
        out = self.fc(h)
        return out.squeeze(-1)


def predict_test_set(model_filename, prediction_filename, test_set, device):
    loaded_model = torch.load(model_filename, weights_only=False)
    loaded_model.eval()

    y_true = []
    y_pred = []

    device = next(loaded_model.parameters()).device

    with torch.no_grad():
        for x, y in test_set:
            x = x.unsqueeze(0).to(device)
            y = y.to(device)

            pred = loaded_model(x)

            y_true.append(y.item())
            y_pred.append(pred.item())

    np.savez(prediction_filename, y_pred=y_pred, y_true=y_true)

def show_results(history, n_epochs, prediction_filename):
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.5))
    axes[0].plot(np.array(range(n_epochs))+1, history['train_losses'], label = 'training loss')
    axes[0].plot(np.array(range(n_epochs))+1, history['valid_metrics'], label = 'validation loss')
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Loss during training")
    axes[0].legend(loc = "upper right")
    axes[0].set_xlim(0, n_epochs + 1)
    axes[0].set_ylim(0, 7)


    data = np.load(prediction_filename)
    y_true = data["y_true"]
    y_pred = data["y_pred"]
    
    axes[1].scatter(y_true, y_pred, alpha=0.2, s=15)

    # compute mean/std per distinct target
    unique_targets = np.unique(y_true)

    means = []
    stds = []

    for t in unique_targets:
        preds = y_pred[y_true == t]
        means.append(preds.mean())
        stds.append(preds.std())

    means = np.array(means)
    stds = np.array(stds)

    # overlay mean (square) and std (error bars)
    axes[1].errorbar(
        unique_targets,
        means,
        yerr=stds,
        fmt='s',                 # square marker
        markersize=5,
        markerfacecolor='none',  # hollow square
        markeredgewidth=1.5,
        markeredgecolor='black',   # square color
        ecolor='black',           # error bar color
        capsize=3,
        linewidth=1,
    )
    
    vmin = min(y_true.min(), y_pred.min()) - 1
    vmax = max(y_true.max(), y_pred.max()) + 1
    axes[1].plot([vmin, vmax], [vmin, vmax], 'k--')
    axes[1].set_xlim(vmin, vmax)
    axes[1].set_ylim(vmin, vmax)

    axes[1].set_xlabel("Target")
    axes[1].set_ylabel("Prediction")
    axes[1].set_title("Prediction vs Target (Test Set)")
    
    plt.tight_layout()
    #plt.savefig(prediction_filename + ".png")
    plt.show()




def plot_all(N_target, N_entry, history, n_epochs, prediction_filename, corr_zarr_path_feature, show_figures = True):

    n_cols = 4
    n_target_rows = N_target // n_cols
    n_rows = n_target_rows + 1  # +1 for the special row

    fig = plt.figure(figsize=(4 * n_cols, 3 * n_rows))
    gs = gridspec.GridSpec(n_rows, n_cols, figure=fig)

    # Target subplots (4 per row)
    axes_targets = []
    idx = 0
    for r in range(n_target_rows):
        for c in range(n_cols):
            ax = fig.add_subplot(gs[r, c])
            axes_targets.append(ax)
            idx += 1

    X = zarr.open(corr_zarr_path_feature, mode="r")

    # load 5 examples (first 5 along axis 0)
    n_list = np.array(range(N_target) )* N_entry

    for i, ax in enumerate(axes_targets):
        ax.imshow(X[n_list[i]], cmap="viridis")
        ax.axis("off")
        ax.set_title(f"dt = {(i+1):02d}")


    # Second row: 2 subplots spanning columns
    ax_bottom_left = fig.add_subplot(gs[-1, :1])
    ax_bottom_right = fig.add_subplot(gs[-1, 1:2])

    ax_bottom_left.plot(np.array(range(n_epochs))+1, history['train_losses'], label = 'training loss')
    ax_bottom_left.plot(np.array(range(n_epochs))+1, history['valid_metrics'], label = 'validation loss')
    ax_bottom_left.set_xlabel("Epoch")
    ax_bottom_left.set_ylabel("Loss")
    ax_bottom_left.set_title("Loss during training")
    ax_bottom_left.legend(loc = "upper right")
    ax_bottom_left.set_xlim(0, n_epochs + 1)
    ax_bottom_left.set_ylim(0, 7)


    data = np.load(prediction_filename)
    y_true = data["y_true"]
    y_pred = data["y_pred"]
    
    ax_bottom_right.scatter(y_true, y_pred, alpha=0.2, s=15)

    # compute mean/std per distinct target
    unique_targets = np.unique(y_true)

    means = []
    stds = []

    for t in unique_targets:
        preds = y_pred[y_true == t]
        means.append(preds.mean())
        stds.append(preds.std())

    means = np.array(means)
    stds = np.array(stds)

    # overlay mean (square) and std (error bars)
    ax_bottom_right.errorbar(
        unique_targets,
        means,
        yerr=stds,
        fmt='s',                 # square marker
        markersize=5,
        markerfacecolor='none',  # hollow square
        markeredgewidth=1.5,
        markeredgecolor='black',   # square color
        ecolor='black',           # error bar color
        capsize=3,
        linewidth=1,
    )
    
    vmin = min(y_true.min(), y_pred.min()) - 1
    vmax = max(y_true.max(), y_pred.max()) + 1
    ax_bottom_right.plot([vmin, vmax], [vmin, vmax], 'k--')
    ax_bottom_right.set_xlim(vmin, vmax)
    ax_bottom_right.set_ylim(vmin, vmax)

    ax_bottom_right.set_xlabel("Target")
    ax_bottom_right.set_ylabel("Prediction")
    ax_bottom_right.set_title("Prediction vs Target (Test Set)")

    plt.savefig(prediction_filename + ".png")
    plt.tight_layout()
    if show_figures:
        plt.show()
    plt.close()