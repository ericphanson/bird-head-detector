#!/usr/bin/env python3
# ruff: noqa: E402
"""
Train YOLOv8n model for bird head detection on M1 MacBook Pro with Comet.ml tracking.
"""

import os

import comet_ml
import torch
import torch.nn as nn
import tqdm.auto as _tqdm_auto
from ultralytics import YOLO


# hacks for MPS especially for yolov12
def safe_view(t, *shape):
    return t.view(*shape) if t.is_contiguous() else t.reshape(*shape)


torch.Tensor.safe_view = safe_view  # type: ignore[attr-defined]
# save original implementation
_orig_bn_forward = nn.BatchNorm2d.forward


def _mps_safe_forward(self, input):
    # MPS requires contiguous input, CUDA/CPU ignore the copy flag
    if not input.is_contiguous():
        input = input.contiguous()
    return _orig_bn_forward(self, input)


# patch every existing and future BatchNorm2d
nn.BatchNorm2d.forward = _mps_safe_forward

# NMS prefilter patch will be applied later based on TRAINING_CONFIG

os.environ["TORCH_SHOW_CPP_STACKTRACES"] = "1"  # C++ sites as well
torch.autograd.set_detect_anomaly(True)  # Python call chain

if torch.backends.cuda.is_built():
    torch.backends.cudnn.benchmark = True

# Training Configuration
TRAINING_CONFIG = {
    "model": "yolov8n",
    "model_file": "yolov8n.pt",
    "model_yaml": "yolov8n.yaml",
    "data": "../data/yolo-4-class/dataset.yaml",
    "epochs": 100,
    "imgsz": 960,
    "batch": 8,
    "project": "runs/multi-detect",
    "name": "bird_multi_yolov8n",
    "workers": 0,  # Prevent multiprocessing issues on M1
    "verbose": True,
    # Inference/eval knobs
    "conf": 0.10,
    "iou": 0.50,
    "max_nms": 12000,
    "max_det": 200,
    "plots": True,
    # Enable/disable Fast NMS prefilter monkey patch
    "fast_nms_prefilter": True,
    # Per-epoch prediction logging
    "log_epoch_predictions": True,  # set True to enable
    "pred_samples": 8,  # number of val images to predict/log per epoch
    "pred_log_interval": 1,  # log every N epochs
    # Progress bar verbosity / ETA
    "progress_bar_eta": True,
    "tqdm_bar_format": "{l_bar}{bar:10}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
    # Epoch sample logging to Comet
    "log_epoch_samples": True,
    "task": "bird_detection",
    "dataset": "CUB-200-2011",
    "architecture": "YOLOv8n",
    # Debug Configuration
    "debug_run": False,  # Set to True for quick testing
    "debug_epochs": 10,  # Reduced epochs for debug
    "debug_fraction": 0.1,  # Use 10% of data for debug (0.1 = 10%)
}

# if TRAINING_CONFIG["debug_run"]:
# print("DEBUG: Setting image size to 640 for debug run")
# TRAINING_CONFIG["imgsz"] = 640  # Smaller size for debug

# --- Conditionally apply Fast NMS prefilter (speeds up validation on MPS/CPU) ---
if TRAINING_CONFIG.get("fast_nms_prefilter", True):
    try:
        from ultralytics.utils import ops as _uops

        _orig_nms = _uops.non_max_suppression

        def _prefilter_topk(prediction, nc, max_k):
            # prediction: (B, N, 5+nc)
            if prediction is None or prediction.ndim != 3:
                return prediction
            outs = []
            for x in prediction:  # per image (N, 5+nc)
                if x.numel() == 0:
                    outs.append(x)
                    continue
                # combined score = obj * max_cls
                if nc > 0 and x.shape[1] >= 5 + nc:
                    scores = x[:, 4] * x[:, 5 : 5 + nc].amax(1)
                else:
                    scores = x[:, 4]
                K = min(int(max_k), x.shape[0])
                if K < x.shape[0]:
                    topk_idx = scores.topk(K).indices
                    x = x[topk_idx]
                outs.append(x)
            # pad to equal length if shapes differ before stacking
            maxN = max(t.shape[0] for t in outs) if outs else 0
            if maxN == 0:
                return prediction
            padded = []
            for t in outs:
                if t.shape[0] < maxN:
                    pad = torch.zeros(
                        (maxN - t.shape[0], t.shape[1]), device=t.device, dtype=t.dtype
                    )
                    t = torch.cat([t, pad], 0)
                padded.append(t)
            return torch.stack(padded, 0)

        def non_max_suppression_fast(
            prediction,
            conf_thres=0.001,
            iou_thres=0.6,
            classes=None,
            agnostic=False,
            multi_label=False,
            labels=(),
            max_det=300,
            nc=0,
            max_time_img=2.5,
            max_nms=30000,
            max_wh=7680,
        ):
            # Prefilter to top‑K candidates per image to cut NMS workload dramatically
            try:
                K = min(12000, max_nms)
                prediction = _prefilter_topk(prediction, nc, K)
            except Exception:
                pass  # fall back silently
            return _orig_nms(
                prediction,
                conf_thres,
                iou_thres,
                classes,
                agnostic,
                multi_label,
                labels,
                max_det,
                nc,
                max_time_img,
                max_nms,
                max_wh,
            )

        _uops.non_max_suppression = non_max_suppression_fast
        print("⚡ Patched Ultralytics NMS with top‑K prefilter (K≤12k).")
    except Exception as _e:
        print(f"⚠️ NMS prefilter patch skipped: {_e}")
else:
    print("ℹ️ NMS prefilter patch disabled via TRAINING_CONFIG.")


def print_checkpoint_metrics(trainer):
    """Print trainer metrics and loss details after each checkpoint is saved."""
    print(
        f"Model details\n"
        f"Best fitness: {trainer.best_fitness}, "
        f"Loss names: {trainer.loss_names}, "  # List of loss names
        f"Metrics: {trainer.metrics}, "
        f"Total loss: {trainer.tloss}"  # Total loss value
    )


CALLBACKS = {
    "on_model_save": print_checkpoint_metrics,
}

# ----------------- Run directory helpers -----------------


def setup_comet(device, experiment_key: str | None = None):
    """Setup Comet.ml experiment tracking."""
    api_key = os.getenv("COMET_API_KEY")
    if not api_key:
        print("⚠️  COMET_API_KEY not found. Comet.ml tracking will be disabled.")
        print("   Set your API key: export COMET_API_KEY='your-api-key'")
        print("   Get your API key from: https://www.comet.ml/api/my/settings/")
        return None

    project_name = os.getenv("COMET_PROJECT_NAME", "bird-head-detector")
    workspace = os.getenv("COMET_WORKSPACE")

    try:
        experiment = comet_ml.Experiment(
            api_key=api_key,
            project_name=project_name,
            workspace=workspace,
        )
        print(f"✅ Comet.ml experiment started: {experiment.url}")

        # Log hyperparameters from global config
        log_params = TRAINING_CONFIG.copy()
        log_params["device"] = device
        experiment.log_parameters(log_params)
        return experiment

    except Exception as e:
        print(f"❌ Failed to initialize Comet.ml: {e}")
        print("   Training will continue without experiment tracking.")
        return None


def create_debug_dataset():
    """Create a subset dataset configuration for debug runs."""
    import random
    import shutil
    from pathlib import Path

    import yaml

    debug_dir = Path("../data/yolo-4-class-debug")

    # Remove existing debug directory to prevent accumulation of old files
    if debug_dir.exists():
        shutil.rmtree(debug_dir)
        print(f"🗑️  Removed existing debug directory: {debug_dir}")

    debug_dir.mkdir(exist_ok=True)

    # Create debug directories
    for split in ["train", "val"]:
        (debug_dir / split / "images").mkdir(parents=True, exist_ok=True)
        (debug_dir / split / "labels").mkdir(parents=True, exist_ok=True)

    # Copy subset of files
    original_dir = Path("../data/yolo-4-class")
    fraction = TRAINING_CONFIG["debug_fraction"]

    for split in ["train", "val"]:
        original_images = list((original_dir / split / "images").glob("*.jpg"))
        sample_size = max(1, int(len(original_images) * fraction))
        sampled_images = random.sample(original_images, sample_size)

        print(
            f"📊 Debug: Using {len(sampled_images)}/{len(original_images)} {split} images ({fraction * 100:.1f}%)"
        )

        for img_path in sampled_images:
            # Copy image
            dst_img = debug_dir / split / "images" / img_path.name
            if not dst_img.exists():
                shutil.copy2(img_path, dst_img)

            # Copy corresponding label
            label_name = img_path.stem + ".txt"
            src_label = original_dir / split / "labels" / label_name
            dst_label = debug_dir / split / "labels" / label_name
            if src_label.exists() and not dst_label.exists():
                shutil.copy2(src_label, dst_label)

    # Create debug dataset.yaml
    debug_yaml = {
        "path": str(debug_dir.absolute()),  # Use absolute path
        "train": "train/images",
        "val": "val/images",
        "nc": 4,
        "names": ["bird", "head", "eye", "beak"],
    }

    yaml_path = debug_dir / "dataset.yaml"
    with open(yaml_path, "w") as f:
        yaml.dump(debug_yaml, f, default_flow_style=False)

    return str(yaml_path)


def main():
    # Check if MPS is available
    if torch.backends.mps.is_available():
        print("✅ MPS (Metal Performance Shaders) is available")
        device = "mps"  # Re-enable MPS for speed
        print("� Using MPS for accelerated training")
    else:
        print("❌ MPS not available, falling back to CPU")
        device = "cpu"

    # Check if running in debug mode
    is_debug = TRAINING_CONFIG["debug_run"]
    if is_debug:
        print("🐛 DEBUG MODE: Quick training run enabled")
        print(
            f"   - Epochs: {TRAINING_CONFIG['debug_epochs']} (vs {TRAINING_CONFIG['epochs']} normal)"
        )
        print(
            f"   - Data subset: {TRAINING_CONFIG['debug_fraction'] * 100:.1f}% of full dataset"
        )

        # Create debug dataset
        data_config = create_debug_dataset()
        epochs = TRAINING_CONFIG["debug_epochs"]
        run_name = f"{TRAINING_CONFIG['name']}_debug"
    else:
        print("🚀 FULL TRAINING MODE")
        data_config = TRAINING_CONFIG["data"]
        epochs = TRAINING_CONFIG["epochs"]
        run_name = TRAINING_CONFIG["name"]

    # Setup Comet.ml tracking
    experiment = setup_comet(device)

    # Log debug info to Comet.ml
    if experiment:
        experiment.log_parameter("run_name", run_name)
        experiment.log_parameter("debug_mode", is_debug)
        if is_debug:
            experiment.log_parameter("debug_epochs", epochs)
            experiment.log_parameter(
                "debug_fraction", TRAINING_CONFIG["debug_fraction"]
            )

    # Load a pretrained YOLO model
    print(f"📦 Loading {TRAINING_CONFIG['model']} pretrained model...")
    model = YOLO(TRAINING_CONFIG["model_yaml"]).load(TRAINING_CONFIG["model_file"])

    # Ensure model is on the intended device (important for MPS)
    try:
        if hasattr(model, "to"):
            model.to(device)
    except Exception:
        try:
            inner = getattr(model, "model", None)
            if isinstance(inner, nn.Module):
                inner.to(device)
        except Exception as _e:
            print(f"⚠️ Could not move model to device '{device}': {_e}")

    # Store model reference for callbacks
    global GLOBAL_YOLO
    GLOBAL_YOLO = model

    # Register callbacks
    for name, value in CALLBACKS.items():
        model.add_callback(name, value)

    # Configure Comet.ml integration for YOLO
    if experiment:
        os.environ["COMET_MODE"] = "online"

    # Train the model on bird head dataset
    mode_text = "DEBUG" if is_debug else "FULL"
    print(f"🚀 Starting {mode_text} training...")
    results = model.train(
        data=data_config,
        epochs=epochs,
        imgsz=TRAINING_CONFIG["imgsz"],
        batch=TRAINING_CONFIG["batch"],
        device=device,
        project=TRAINING_CONFIG["project"],
        name=run_name,
        workers=TRAINING_CONFIG["workers"],
        verbose=TRAINING_CONFIG["verbose"],
        conf=TRAINING_CONFIG["conf"],
        iou=TRAINING_CONFIG["iou"],
        max_det=TRAINING_CONFIG["max_det"],
        plots=TRAINING_CONFIG["plots"],
        amp=True,
        save_json=True,
        save_period=1,
    )

    # Log final results to Comet.ml
    if experiment and results:
        try:
            # Log final metrics
            final_metrics = (
                results.results_dict if hasattr(results, "results_dict") else {}
            )
            for key, value in final_metrics.items():
                if isinstance(value, int | float):
                    experiment.log_metric(f"final_{key}", value)

            # Log model artifacts
            best_model_path = f"{TRAINING_CONFIG['project']}/{run_name}/weights/best.pt"
            if os.path.exists(best_model_path):
                experiment.log_model("best_model", best_model_path)
                print("📤 Model uploaded to Comet.ml")

            experiment.end()
            print("✅ Comet.ml experiment completed")

        except Exception as e:
            print(f"⚠️ Error logging to Comet.ml: {e}")

    print(f"✅ {mode_text} training completed!")
    print(
        f"📊 Best model saved to: {TRAINING_CONFIG['project']}/{run_name}/weights/best.pt"
    )

    return results


# --- Conditionally patch tqdm to show ETA ---
if TRAINING_CONFIG.get("progress_bar_eta", True):
    try:
        from tqdm.auto import tqdm as _tqdm

        _orig_tqdm = _tqdm

        def _tqdm_with_eta(*args, **kwargs):
            kwargs.setdefault("dynamic_ncols", True)
            kwargs.setdefault("smoothing", 0.05)
            kwargs.setdefault(
                "bar_format",
                TRAINING_CONFIG.get(
                    "tqdm_bar_format",
                    "{l_bar}{bar:10}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
                ),
            )
            return _orig_tqdm(*args, **kwargs)

        _tqdm_auto.tqdm = _tqdm_with_eta  # type: ignore[attr-defined]
        print("📈 Patched tqdm progress bar to show ETA.")
    except Exception as _e:
        print(f"⚠️ Progress bar patch skipped: {_e}")


# --- Progress bar ETA tweaks via callbacks ---


def _format_seconds(secs: float) -> str:
    try:
        s = max(0, int(secs))
        h, s = divmod(s, 3600)
        m, s = divmod(s, 60)
        if h:
            return f"{h:d}:{m:02d}:{s:02d}"
        return f"{m:02d}:{s:02d}"
    except Exception:
        return "--:--"


def _set_eta_postfix_from_pbar(pbar):
    try:
        if pbar is None:
            return
        # tqdm maintains format_dict with elapsed, remaining, rate
        fd = getattr(pbar, "format_dict", None) or {}
        remaining = fd.get("remaining")
        rate = fd.get("rate")
        eta = _format_seconds(remaining) if remaining is not None else None
        postfix = None
        if eta is not None:
            postfix = f"ETA {eta}"
            if rate:
                postfix += f" | {rate:.2f} it/s"
        elif rate:
            postfix = f"{rate:.2f} it/s"
        if postfix:
            # reduce spam: only update every few steps
            n = getattr(pbar, "n", 0)
            if n % 5 == 0:
                pbar.set_postfix_str(postfix, refresh=False)
    except Exception:
        pass


def on_train_batch_end_set_eta(trainer):
    if not TRAINING_CONFIG.get("progress_bar_eta", True):
        return
    try:
        pbar = getattr(trainer, "pbar", None)
        _set_eta_postfix_from_pbar(pbar)
    except Exception:
        pass


def on_val_batch_end_set_eta(trainer):
    if not TRAINING_CONFIG.get("progress_bar_eta", True):
        return
    try:
        # trainer may have its own pbar
        pbar = getattr(trainer, "pbar", None)
        _set_eta_postfix_from_pbar(pbar)
        # and also a validator with its own pbar
        validator = getattr(trainer, "validator", None)
        if validator is not None:
            _set_eta_postfix_from_pbar(getattr(validator, "pbar", None))
    except Exception:
        pass


CALLBACKS["on_train_batch_end"] = on_train_batch_end_set_eta
CALLBACKS["on_val_batch_end"] = on_val_batch_end_set_eta


if __name__ == "__main__":
    main()
