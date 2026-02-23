import torch
import matplotlib.pyplot as plt
import numpy as np
from copy import deepcopy

def evaluate_tm(model, data_loader, metric, device):
    model.eval()
    metric.reset()
    with torch.no_grad():
        for X_batch, y_batch in data_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            y_pred = model(X_batch)
            metric.update(y_pred, y_batch)
    return metric.compute()

def train(model, optimizer, loss_fn, metric, train_loader, valid_loader,
          n_epochs, device):
    history = {"train_losses": [], "train_metrics": [], "valid_metrics": []}
    for epoch in range(n_epochs):
        total_loss = 0.0
        metric.reset()
        model.train()
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            y_pred = model(X_batch)
            loss = loss_fn(y_pred, y_batch)
            total_loss += loss.item()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            metric.update(y_pred, y_batch)
        history["train_losses"].append(total_loss / len(train_loader))
        history["train_metrics"].append(metric.compute().item())
        history["valid_metrics"].append(
            evaluate_tm(model, valid_loader, metric, device).item())
        print(f"Epoch {epoch + 1}/{n_epochs}, "
              f"train loss: {history['train_losses'][-1]:.4f}, "
              f"train metric: {history['train_metrics'][-1]:.4f}, "
              f"valid metric: {history['valid_metrics'][-1]:.4f}")
    return history


def train_with_scheduler(model, optimizer, loss_fn, metric, train_loader, valid_loader,
          n_epochs, device, scheduler, show_progress=True):
    history = {"train_losses": [], "train_metrics": [], "valid_metrics": []}

    for epoch in range(n_epochs):
        total_loss = 0.0
        metric.reset()
        model.train()

        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            y_pred = model(X_batch)
            loss = loss_fn(y_pred, y_batch)
            total_loss += loss.item()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            metric.update(y_pred, y_batch)

        history["train_losses"].append(total_loss / len(train_loader))
        history["train_metrics"].append(metric.compute().item())

        val_metric = evaluate_tm(model, valid_loader, metric, device).item()

        history["valid_metrics"].append(val_metric)
        scheduler.step(val_metric)

        if show_progress:
            print(f"Epoch {epoch + 1}/{n_epochs}, "
                  f"train loss: {history['train_losses'][-1]:.4f}, "
                  f"train metric: {history['train_metrics'][-1]:.4f}, "
                  f"valid metric: {history['valid_metrics'][-1]:.4f}")

    return history



def train_with_scheduler_and_early_stopping(model, optimizer, loss_fn, metric, train_loader, valid_loader,
          n_epochs, device, scheduler, show_progress=True):
    history = {"train_losses": [], "train_metrics": [], "valid_metrics": []}

    best_val_metric = np.inf
    
    for epoch in range(n_epochs):
        total_loss = 0.0
        metric.reset()
        model.train()

        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            y_pred = model(X_batch)

            loss = loss_fn(y_pred, y_batch)
            total_loss += loss.item()
            loss.backward()

            optimizer.step()
            optimizer.zero_grad()

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

    data = np.load(prediction_filename)
    y_true = data["y_true"]
    y_pred = data["y_pred"]

    fig = plt.figure(figsize=(4, 3))

    # scatter cloud
    plt.scatter(y_true, y_pred, alpha=0.2, s=15)

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
    plt.errorbar(
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
    plt.plot([vmin, vmax], [vmin, vmax], 'k--')
    plt.xlim(vmin, vmax)
    plt.ylim(vmin, vmax)

    plt.xlabel("Target")
    plt.ylabel("Prediction")
    plt.title("Prediction vs Target (Test Set)")
    plt.show()

