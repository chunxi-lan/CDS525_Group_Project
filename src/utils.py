import os
import json
import random
import numpy as np
import torch
from config import Config


def set_seed(seed=Config.RANDOM_SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_metrics(metrics, filename, save_dir=Config.RESULTS_DIR):
    os.makedirs(save_dir, exist_ok=True)
    filepath = os.path.join(save_dir, filename)
    with open(filepath, 'w') as f:
        json.dump(metrics, f, indent=4)
    return filepath


def load_metrics(filepath):
    with open(filepath, 'r') as f:
        return json.load(f)


def save_checkpoint(model, optimizer, scheduler, epoch, val_acc,
                    filename='best_model.pth', save_dir=Config.CHECKPOINT_DIR):
    os.makedirs(save_dir, exist_ok=True)
    filepath = os.path.join(save_dir, filename)
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict() if optimizer else None,
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'epoch': epoch,
        'val_acc': val_acc
    }
    torch.save(checkpoint, filepath)
    return filepath


def load_checkpoint(model, optimizer=None, scheduler=None,
                    filename='best_model.pth', save_dir=Config.CHECKPOINT_DIR,
                    device=Config.DEVICE):
    filepath = os.path.join(save_dir, filename)
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Checkpoint not found: {filepath}")
    
    checkpoint = torch.load(filepath, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    if scheduler and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    return model, optimizer, scheduler, checkpoint['epoch'], checkpoint['val_acc']


def _safe_makedirs(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _is_number(s: str) -> bool:
    try:
        float(s)
        return True
    except ValueError:
        return False


def _parse_experiment_from_filename(filename: str) -> dict:
    base = os.path.splitext(os.path.basename(filename))[0]
    if base.startswith("metrics_"):
        base = base[len("metrics_"):]

    if base == "quick_sgd" or ("quick" in base and "sgd" in base):
        return {"label": "opt=sgd(quick)", "group": "opt", "value": "sgd_quick"}

    if base.startswith("lr") and _is_number(base[2:]):
        value_str = base[2:]
        return {"label": f"lr={value_str}", "group": "lr", "value": float(value_str)}

    if base.startswith("bs") and base[2:].isdigit():
        value_str = base[2:]
        return {"label": f"bs={value_str}", "group": "bs", "value": float(value_str)}

    parts = base.split("_")
    if not parts:
        return {"label": base, "group": "misc", "value": None}

    group = parts[0]
    rest = "_".join(parts[1:]) if len(parts) > 1 else ""
    value = None

    if group == "lr":
        value_str = rest
        value = float(value_str) if _is_number(value_str) else None
        label = f"lr={value_str}" if value_str else base
        return {"label": label, "group": "lr", "value": value}

    if group == "bs":
        value_str = rest
        value = float(value_str) if value_str.isdigit() else None
        label = f"bs={value_str}" if value_str else base
        return {"label": label, "group": "bs", "value": value}

    if group in {"adam", "sgd"}:
        return {"label": f"opt={group}", "group": "opt", "value": group}

    return {"label": base, "group": "misc", "value": None}


def plot_metrics_from_json_dir(
    json_dir,
    output_dir: str = None,
    max_epochs: int | None = None,
):
    import matplotlib.pyplot as plt

    if output_dir is None:
        output_dir = os.path.join(Config.RESULTS_DIR, "plots")
    _safe_makedirs(output_dir)

    json_sources = json_dir if isinstance(json_dir, (list, tuple, set)) else [json_dir]
    json_files_by_name = {}
    for source in json_sources:
        if source is None:
            continue
        if isinstance(source, str) and source.lower().endswith(".json") and os.path.isfile(source):
            name = os.path.basename(source)
            if name not in json_files_by_name:
                json_files_by_name[name] = source
            continue
        if isinstance(source, str) and os.path.isdir(source):
            for f in os.listdir(source):
                if not f.lower().endswith(".json"):
                    continue
                if f not in json_files_by_name:
                    json_files_by_name[f] = os.path.join(source, f)
            continue

    json_files = [json_files_by_name[k] for k in sorted(json_files_by_name.keys())]

    runs = []
    for path in json_files:
        metrics = load_metrics(path)
        exp = _parse_experiment_from_filename(path)
        runs.append(
            {
                "path": path,
                "filename": os.path.basename(path),
                "label": exp["label"],
                "group": exp["group"],
                "value": exp["value"],
                "metrics": metrics,
            }
        )

    def _clip(values):
        if max_epochs is None:
            return values
        return values[:max_epochs]

    def _plot_single(run):
        metrics = run["metrics"]
        epochs = len(metrics.get("train_loss", []))
        x = list(range(1, len(_clip(list(range(epochs)))) + 1))

        fig = plt.figure(figsize=(12, 5))
        ax1 = fig.add_subplot(1, 2, 1)
        ax2 = fig.add_subplot(1, 2, 2)

        for key, name in [("train_loss", "train"), ("val_loss", "val"), ("test_loss", "test")]:
            if key in metrics:
                y = _clip(metrics[key])
                ax1.plot(x[: len(y)], y, label=name)
        ax1.set_title(f"Loss - {run['label']}")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss")
        ax1.grid(True, alpha=0.3)
        ax1.legend()

        for key, name in [("train_acc", "train"), ("val_acc", "val"), ("test_acc", "test")]:
            if key in metrics:
                y = _clip(metrics[key])
                ax2.plot(x[: len(y)], y, label=name)
        ax2.set_title(f"Accuracy - {run['label']}")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Accuracy (%)")
        ax2.grid(True, alpha=0.3)
        ax2.legend()

        fig.tight_layout()
        out_path = os.path.join(output_dir, f"curves_{os.path.splitext(run['filename'])[0]}.png")
        fig.savefig(out_path, dpi=200)
        plt.close(fig)
        return out_path

    def _plot_group_overlay(group_name, metric_key, title_suffix, out_name, sort_key):
        group_runs = [r for r in runs if r["group"] == group_name and metric_key in r["metrics"]]
        if len(group_runs) <= 1:
            return None
        group_runs.sort(key=sort_key)

        fig = plt.figure(figsize=(8, 5))
        ax = fig.add_subplot(1, 1, 1)
        for r in group_runs:
            y = _clip(r["metrics"][metric_key])
            x = list(range(1, len(y) + 1))
            ax.plot(x, y, label=r["label"])
        ax.set_title(f"{metric_key} overlay - {title_suffix}")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Accuracy (%)" if "acc" in metric_key else "Loss")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
        fig.tight_layout()
        out_path = os.path.join(output_dir, out_name)
        fig.savefig(out_path, dpi=200)
        plt.close(fig)
        return out_path

    def _plot_final_bar(group_name, metric_key, title, out_name, sort_key, reducer):
        group_runs = [r for r in runs if r["group"] == group_name and metric_key in r["metrics"]]
        if len(group_runs) <= 1:
            return None
        group_runs.sort(key=sort_key)

        labels = [r["label"] for r in group_runs]
        values = [reducer(_clip(r["metrics"][metric_key])) for r in group_runs]

        fig = plt.figure(figsize=(8, 4))
        ax = fig.add_subplot(1, 1, 1)
        ax.bar(labels, values)
        ax.set_title(title)
        ax.set_ylabel("Accuracy (%)" if "acc" in metric_key else "Loss")
        ax.grid(True, axis="y", alpha=0.3)
        ax.set_ylim(0, max(values) * 1.1 if values else 1)
        for i, v in enumerate(values):
            ax.text(i, v, f"{v:.2f}" if isinstance(v, float) else str(v), ha="center", va="bottom", fontsize=8)
        fig.tight_layout()
        out_path = os.path.join(output_dir, out_name)
        fig.savefig(out_path, dpi=200)
        plt.close(fig)
        return out_path

    outputs = {"single_run_curves": [], "comparisons": []}

    for r in runs:
        outputs["single_run_curves"].append(_plot_single(r))

    outputs["comparisons"].append(
        _plot_group_overlay(
            "lr",
            "val_acc",
            "learning rate",
            "overlay_lr_val_acc.png",
            sort_key=lambda r: (r["value"] is None, r["value"]),
        )
    )
    outputs["comparisons"].append(
        _plot_group_overlay(
            "lr",
            "test_acc",
            "learning rate",
            "overlay_lr_test_acc.png",
            sort_key=lambda r: (r["value"] is None, r["value"]),
        )
    )
    outputs["comparisons"].append(
        _plot_final_bar(
            "lr",
            "test_acc",
            "Final Test Acc by Learning Rate",
            "bar_lr_final_test_acc.png",
            sort_key=lambda r: (r["value"] is None, r["value"]),
            reducer=lambda ys: ys[-1] if ys else float("nan"),
        )
    )
    outputs["comparisons"].append(
        _plot_final_bar(
            "lr",
            "val_acc",
            "Best Val Acc by Learning Rate",
            "bar_lr_best_val_acc.png",
            sort_key=lambda r: (r["value"] is None, r["value"]),
            reducer=lambda ys: max(ys) if ys else float("nan"),
        )
    )

    outputs["comparisons"].append(
        _plot_group_overlay(
            "bs",
            "val_acc",
            "batch size",
            "overlay_bs_val_acc.png",
            sort_key=lambda r: (r["value"] is None, r["value"]),
        )
    )
    outputs["comparisons"].append(
        _plot_group_overlay(
            "bs",
            "test_acc",
            "batch size",
            "overlay_bs_test_acc.png",
            sort_key=lambda r: (r["value"] is None, r["value"]),
        )
    )
    outputs["comparisons"].append(
        _plot_final_bar(
            "bs",
            "test_acc",
            "Final Test Acc by Batch Size",
            "bar_bs_final_test_acc.png",
            sort_key=lambda r: (r["value"] is None, r["value"]),
            reducer=lambda ys: ys[-1] if ys else float("nan"),
        )
    )
    outputs["comparisons"].append(
        _plot_final_bar(
            "bs",
            "val_acc",
            "Best Val Acc by Batch Size",
            "bar_bs_best_val_acc.png",
            sort_key=lambda r: (r["value"] is None, r["value"]),
            reducer=lambda ys: max(ys) if ys else float("nan"),
        )
    )

    outputs["comparisons"].append(
        _plot_group_overlay(
            "opt",
            "val_acc",
            "optimizer",
            "overlay_opt_val_acc.png",
            sort_key=lambda r: str(r["value"]),
        )
    )
    outputs["comparisons"].append(
        _plot_group_overlay(
            "opt",
            "test_acc",
            "optimizer",
            "overlay_opt_test_acc.png",
            sort_key=lambda r: str(r["value"]),
        )
    )
    outputs["comparisons"].append(
        _plot_final_bar(
            "opt",
            "test_acc",
            "Final Test Acc by Optimizer",
            "bar_opt_final_test_acc.png",
            sort_key=lambda r: str(r["value"]),
            reducer=lambda ys: ys[-1] if ys else float("nan"),
        )
    )
    outputs["comparisons"].append(
        _plot_final_bar(
            "opt",
            "val_acc",
            "Best Val Acc by Optimizer",
            "bar_opt_best_val_acc.png",
            sort_key=lambda r: str(r["value"]),
            reducer=lambda ys: max(ys) if ys else float("nan"),
        )
    )

    outputs["comparisons"] = [p for p in outputs["comparisons"] if p is not None]
    return outputs


def visualize_first_n_test_samples(
    test_loader,
    classes,
    output_path: str,
    n: int = 100,
    device=Config.DEVICE,
    model=None,
):
    import matplotlib.pyplot as plt

    mean = torch.tensor([0.5071, 0.4867, 0.4408], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.2675, 0.2565, 0.2761], device=device).view(1, 3, 1, 1)

    _safe_makedirs(os.path.dirname(output_path) or ".")

    imgs = []
    ys_true = []
    ys_pred = []

    if model is not None:
        model.eval()

    collected = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            if model is not None:
                outputs = model(inputs)
                preds = outputs.argmax(dim=1)
            else:
                preds = None

            batch_size = inputs.size(0)
            take = min(n - collected, batch_size)
            if take <= 0:
                break

            imgs.append(inputs[:take])
            ys_true.append(labels[:take])
            if preds is not None:
                ys_pred.append(preds[:take])

            collected += take
            if collected >= n:
                break

    if not imgs:
        raise ValueError("test_loader yielded no data")

    images = torch.cat(imgs, dim=0)
    y_true = torch.cat(ys_true, dim=0)
    y_pred = torch.cat(ys_pred, dim=0) if ys_pred else None

    images = (images * std + mean).clamp(0, 1).cpu()
    y_true = y_true.cpu().tolist()
    y_pred = y_pred.cpu().tolist() if y_pred is not None else None

    cols = 10
    rows = int(np.ceil(n / cols))
    fig = plt.figure(figsize=(cols * 1.6, rows * 1.6))

    for i in range(n):
        ax = fig.add_subplot(rows, cols, i + 1)
        img = images[i].permute(1, 2, 0).numpy()
        ax.imshow(img)
        ax.axis("off")
        t = classes[y_true[i]] if 0 <= y_true[i] < len(classes) else str(y_true[i])
        if y_pred is None:
            ax.set_title(f"T:{t}", fontsize=7)
        else:
            p = classes[y_pred[i]] if 0 <= y_pred[i] < len(classes) else str(y_pred[i])
            ax.set_title(f"T:{t}\nP:{p}", fontsize=7)

    fig.tight_layout(pad=0.2)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return output_path
