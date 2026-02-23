import numpy as np
import torch
from torch.utils.data import Dataset
import zarr
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from numcodecs import Blosc
import time

'''
class ZarrShearDataset(Dataset):
    def __init__(self, zarr_path, pairs, shear_rate, normalize=True):
        self.z = zarr.open(zarr_path, mode="r")   # read-only
        self.pairs = pairs
        self.shear_rate = shear_rate
        self.normalize = normalize

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        i, j, dt = self.pairs[idx]

        img1 = self.z[i]
        img2 = self.z[j]

        if self.normalize:
            #img1 = img1.astype(np.float32) / 65535.0
            #img2 = img2.astype(np.float32) / 65535.0
            img1 = img1.astype(np.float32)
            img2 = img2.astype(np.float32)

        x = np.stack([img1, img2], axis=0)   # (2, H, W)
        y = self.shear_rate * dt             # target

        return (
            torch.from_numpy(x),
            torch.tensor(y, dtype=torch.float32),
        )
'''

def stratified_regression_split(
    targets,
    train_ratio=0.7,
    val_ratio=0.15,
    test_ratio=0.15,
    n_bins=10,
    random_state=42
):
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6

    targets = np.asarray(targets)

    # Bin targets using quantiles
    bins = np.quantile(targets, np.linspace(0, 1, n_bins + 1))
    y_binned = np.digitize(targets, bins[1:-1])

    # Train / temp split
    idx = np.arange(len(targets))
    idx_train, idx_temp, y_train, y_temp = train_test_split(
        idx,
        y_binned,
        test_size=(1.0 - train_ratio),
        stratify=y_binned,
        random_state=random_state,
    )

    # Validation / test split
    val_ratio_adj = val_ratio / (val_ratio + test_ratio)
    idx_val, idx_test = train_test_split(
        idx_temp,
        test_size=(1.0 - val_ratio_adj),
        stratify=y_temp,
        random_state=random_state,
    )

    return idx_train, idx_val, idx_test


def plot_target_distribution(train_set, val_set, test_set):

    def get_targets(subset):
        return [subset[i][1] for i in range(len(subset))]

    train_targets = get_targets(train_set)
    val_targets   = get_targets(val_set)
    test_targets  = get_targets(test_set)

    fig, axes = plt.subplots(1, 3, figsize=(12, 3))  # 1 row, 3 columns

    # Train set
    axes[0].hist(train_targets, bins=30, color='skyblue')
    axes[0].set_title("Train Set")
    axes[0].set_xlabel("Target value")
    axes[0].set_ylabel("Count")

    # Validation set
    axes[1].hist(val_targets, bins=30, color='lightgreen')
    axes[1].set_title("Validation Set")
    axes[1].set_xlabel("Target value")
    axes[1].set_ylabel("Count")

    # Test set
    axes[2].hist(test_targets, bins=30, color='salmon')
    axes[2].set_title("Test Set")
    axes[2].set_xlabel("Target value")
    axes[2].set_ylabel("Count")

    plt.tight_layout()
    plt.show()


def create_frame_pairs(N_frames, min_dt, max_dt, N_dt, pairs_per_dt):# total number of frames in the IMM file
    #N_frames = 10000  
    
    # desired dt range
    #min_dt = 1
    #max_dt = 1000
    
    # how many different dt values
    #N_dt = 500  # can be adjusted (50–200 reasonable)
    # how many examples to sample per dt
    #pairs_per_dt = 200
    
    # generate Δt values uniformly
    delta_ts = np.linspace(min_dt, max_dt, N_dt).astype(int)
    delta_ts = np.unique(delta_ts)
    print(delta_ts)
    
    pairs = []   # list of (i, j, dt)
    
    for dt in delta_ts:
        max_i = N_frames - dt - 1 - 500 # remove the last 100 frames in case of further averaging
        if max_i <= 0:
            continue
            
        # choose indices uniformly over valid range
        iset = np.random.randint(0, max_i, size=pairs_per_dt)
        
        for i in iset:
            j = i + dt
            pairs.append((i, j, dt))
    
    # convert to array for convenience
    pairs = np.array(pairs)

    return pairs


    
def create_correlation_zarr(sp_zarr_path, corr_zarr_path_features, corr_zarr_path_target, pairs, bin, bounds):

    z = zarr.open(sp_zarr_path, mode="r")

    compressor = Blosc(
        cname="zstd",
        clevel=3,
        shuffle=Blosc.BITSHUFFLE
        )

    N = len(pairs)
    #print(N, pairs.shape[0])
    xl, xh, yl, yh = bounds

    H, W = xh - xl, yh - yl
    
    '''
    Y = zarr.open(
        corr_zarr_path_target,
        mode="w",
        shape=(N,),
        chunks=(1024,),
        dtype="float32",
        zarr_format=2,        # ⭐ THIS FIXES IT
    )
    attempt = 0

    for idx in range(pairs.shape[0]):

        _, _, dt = pairs[idx]

        if attempt == 10:
            
            time.sleep(0.1)
            attempt += 1
            attempt = attempt % 10

        Y[idx] = dt.astype(np.float32)
    '''
    Y = np.zeros(N, dtype = np.float32)

    X = zarr.open(
        corr_zarr_path_features,
        mode="w",
        shape=(N, H, W),
        chunks=(1, H, W),
        dtype="float32",
        compressor=compressor,
        zarr_format=2,        # ⭐ THIS FIXES IT
    )
    
    for idx in range(pairs.shape[0]):

        i, j, dt = pairs[idx]

        Y[idx] = dt

        #Y[idx] = dt.astype(np.float32)

        img1_full = z[i : i + bin].astype(np.float32)
        img2_full = z[j : j + bin].astype(np.float32)
        #print(img1_full.shape)
        img1 = img1_full[:, xl:xh, yl:yh]
        img2 = img2_full[:, xl:xh, yl:yh]
        #print(img1.shape)

        # Means
        I1 = img1.mean(axis=0)
        I2 = img2.mean(axis=0)

        # Correlation term
        corr = (img1 * img2).mean(axis=0)

        # Compute C
        denom = (I1 + I2) ** 2
        x = np.zeros_like(denom)
        valid = denom != 0
        x[valid] = 4* corr[valid] / denom[valid]

        '''
        # Min–max normalize to [0, 1]
        x_min = x.min()
        x_max = x.max()
        x = (x - x_min) / (x_max - x_min + 1e-6)
        '''
        # Target
        #y = np.float32(dt)

        X[idx] = x.astype(np.float32)
        
    np.savez(corr_zarr_path_target, target = Y)
