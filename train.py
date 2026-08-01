import os
import json
import time
import csv
import random
import copy
import importlib.util
import argparse
import numpy as np
import h5py
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Sampler, Subset

# Import your custom modules directly from your workspace folder
from model import CustomUNet
from dataset import LandslideDataset, NDVI_CHANNEL
from augmentations import get_training_augmentations


def compute_evaluation_metrics(pred_binary, masks):
    """Calculates semantic segmentation evaluation metrics from tensor intersections."""
    target_binary = masks.float()

    true_positive = torch.sum(pred_binary * target_binary).item()
    false_positive = torch.sum(pred_binary * (1.0 - target_binary)).item()
    false_negative = torch.sum((1.0 - pred_binary) * target_binary).item()
    true_negative = torch.sum((1.0 - pred_binary) * (1.0 - target_binary)).item()
    total_pixels = pred_binary.numel()

    accuracy = (true_positive + true_negative) / (total_pixels + 1e-8)
    precision = true_positive / (true_positive + false_positive + 1e-8)
    recall = true_positive / (true_positive + false_negative + 1e-8)
    f1_score = 2 * (precision * recall) / (precision + recall + 1e-8)
    mean_iou = true_positive / (true_positive + false_positive + false_negative + 1e-8)

    return accuracy, precision, recall, f1_score, mean_iou


class WeightedBCEDiceLoss(nn.Module):
    """Hybrid loss: label-trust-weighted BCE (on logits) + weighted Dice.

    Replaces the previous Focal loss: Focal's gamma=2 actively suppresses
    the gradient of confident predictions, leaving the network uncertain
    everywhere (the tuned threshold ended up at 0.70 and even large
    landslides only reached ~25% IoU). Weighted BCE + Dice lets
    probabilities saturate properly while the per-pixel weights (from
    compute_label_weights) stop the model from memorizing the noisy
    polygon boundaries.
    """

    def __init__(self, pos_weight=10.0, dice_weight=1.5, epsilon=1e-6,
                 label_smoothing=0.05, tversky_weight=0.3, alpha=0.3, beta=0.7):
        super(WeightedBCEDiceLoss, self).__init__()
        self.pos_weight = pos_weight
        self.dice_weight = dice_weight
        self.epsilon = epsilon
        self.label_smoothing = label_smoothing
        self.tversky_weight = tversky_weight
        self.alpha = alpha
        self.beta = beta

    def forward(self, logits, masks, weights):
        if self.label_smoothing > 0.0:
            masks = masks * (1.0 - self.label_smoothing) + 0.5 * self.label_smoothing
        bce_loss = nn.functional.binary_cross_entropy_with_logits(
            logits,
            masks,
            weight=weights,
            pos_weight=torch.tensor(self.pos_weight, device=logits.device),
            reduction="mean",
        )
        probs = torch.sigmoid(logits)
        p = probs.view(-1)
        y = masks.view(-1)
        w = weights.view(-1)
        intersection = (w * p * y).sum()
        total_cardinality = (w * p).sum() + (w * y).sum()
        dice_loss = 1.0 - (2.0 * intersection + self.epsilon) / (total_cardinality + self.epsilon)

        # Tversky term (alpha=0.3 penalizes FP, beta=0.7 penalizes FN):
        # precision-leaning complement to the recall-hungry BCE pos_weight.
        tp_tv = (w * p * y).sum()
        fp_tv = (w * p * (1.0 - y)).sum()
        fn_tv = (w * (1.0 - p) * y).sum()
        tversky = (tp_tv + self.epsilon) / (
            tp_tv + self.alpha * fp_tv + self.beta * fn_tv + self.epsilon
        )
        return bce_loss + self.dice_weight * dice_loss + self.tversky_weight * (1.0 - tversky)


POS_FRAC_CACHE_PATH = "./datasets/TrainData/pos_frac_cache.npy"
SWA_SNAPSHOTS = 5


def build_or_load_pos_fractions(dataset):
    """Per-file positive-pixel fraction, cached on disk.

    Used to oversample landslide-rich tiles (WeightedRandomSampler) so each
    batch contains enough positive signal for the model to learn precise
    boundaries instead of guessing.
    """
    if os.path.exists(POS_FRAC_CACHE_PATH):
        cached = np.load(POS_FRAC_CACHE_PATH)
        if len(cached) == len(dataset):
            return cached

    fractions = np.zeros(len(dataset), dtype=np.float32)
    print("Scanning training tiles for positive-pixel fractions (cached after this run)...")
    for idx in range(len(dataset)):
        mask_path = os.path.join(dataset.mask_dir, dataset.file_list[idx])
        with h5py.File(mask_path, "r") as f_msk:
            mask_array = np.array(f_msk["mask"])
        fractions[idx] = float((mask_array > 0).mean())
        if (idx + 1) % 2000 == 0:
            print(f"  scanned {idx + 1}/{len(dataset)}")

    np.save(POS_FRAC_CACHE_PATH, fractions)
    return fractions


def tta_probabilities(model, images, device, n_views=4, scales=(1.0,)):
    """Average probabilities over flips (and optional scales) (test-time
    augmentation).

    n_views=4: identity + h-flip + v-flip + both flips (final metrics).
    n_views=2: identity + h-flip only (cheap per-epoch tracking).
    scales=(0.9, 1.0, 1.1): multi-scale TTA for the final summary; each
    scale is resized, flipped, predicted, and resized back before averaging.
    """
    if n_views == 2:
        flip_sets = [(), (3,)]
    else:
        flip_sets = [(), (3,), (2,), (2, 3)]
    probs = None
    count = 0
    for scale in scales:
        if scale == 1.0:
            view_base = images
        else:
            # Snap to a multiple of 8: the 3-level U-Net concatenates skip
            # connections at 1/2, 1/4 and 1/8 resolution, so any input size
            # that is not divisible by 8 breaks the concat (e.g. 141 -> 57 vs
            # 56). 128*1.1 -> 144, 128*0.9 -> 112.
            size = max(8, round(128 * scale / 8.0) * 8)
            view_base = F.interpolate(images, size=(size, size),
                                      mode="bilinear", align_corners=False)
        for flip_dims in flip_sets:
            view = torch.flip(view_base, dims=list(flip_dims)) if flip_dims else view_base
            with torch.autocast(device_type=device.type, enabled=(device.type == "cuda")):
                logits = model(view)
            logits = torch.flip(logits, dims=list(flip_dims)) if flip_dims else logits
            prob = torch.sigmoid(logits.float())
            if scale != 1.0:
                prob = F.interpolate(prob, size=(128, 128), mode="bilinear",
                                     align_corners=False)
            probs = prob if probs is None else probs + prob
            count += 1
    return probs / count


def update_ema(ema_dict, model, decay=0.999):
    """EMA of ALL floating-point state (weights + BN running stats).

    The BN statistics are included so EMA weights are never paired with the
    live model's running stats at validation time (a subtle mismatch).
    """
    with torch.no_grad():
        for key, val in model.state_dict().items():
            if val.dtype.is_floating_point and key in ema_dict:
                ema_dict[key].mul_(decay).add_(val.detach(), alpha=1.0 - decay)


def ema_state_dict(ema_dict, model):
    state = copy.deepcopy(model.state_dict())
    with torch.no_grad():
        for key, val in model.state_dict().items():
            if val.dtype.is_floating_point and key in ema_dict:
                state[key] = ema_dict[key].clone()
    return state


# Window 1 (POST) NDVI inside the flattened 42-channel layout:
# [PRE 0-13 | POST 14-27 | LATE 28-41], NDVI is band 13 of each window.
NDVI_POST_CHANNEL = 27
NDVI_GATE_THRESHOLD = 0.48


def confusion_to_metrics(tp, fp, fn, tn, thresholds, min_recall=0.6):
    """Best-IoU-threshold metrics + best precision at a recall floor."""
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
    iou = tp / (tp + fp + fn + 1e-8)
    accuracy = (tp + tn) / (tp + fp + fn + tn + 1e-8)

    best_idx = int(np.argmax(iou))

    keep = recall >= min_recall
    if keep.any():
        prec_idx = int(np.argmax(precision[keep]))
        prec_thr_idx = int(np.where(keep)[0][prec_idx])
    else:
        prec_thr_idx = best_idx

    return {
        "threshold": float(thresholds[best_idx]),
        "accuracy": float(accuracy[best_idx]),
        "precision": float(precision[best_idx]),
        "recall": float(recall[best_idx]),
        "f1": float(f1[best_idx]),
        "iou": float(iou[best_idx]),
        "precision_at_recall": float(precision[prec_thr_idx]),
        "threshold_at_recall": float(thresholds[prec_thr_idx]),
    }


def evaluate_dataset(model, dataloader, criterion, device, thresholds=None,
                     use_tta=True, tta_views=4, scales=(1.0,),
                     ndvi_gate=True, ndvi_mean=None, ndvi_std=None,
                     min_recall=0.6):
    """Runs a full evaluation pass and reports metrics at the BEST threshold.

    Because the positive class is tiny, the fixed 0.5 threshold wastes many
    detections. Thresholds are scanned on the validation set and the one
    maximizing IoU is reported (and used at inference time).

    NDVI gate (default ON): predictions are suppressed wherever the
    post-event scene is still vegetated (NDVI >= 0.48) - landslides are bare
    ground, so vegetated detections are almost always cloud/water/regrowth
    false positives. This is the primary metric; ungated numbers are also
    reported for reference.
    """
    if thresholds is None:
        thresholds = np.arange(0.02, 0.98, 0.02)
    num_thresholds = len(thresholds)

    model.eval()
    running_loss = 0.0
    total_batches = max(1, len(dataloader))

    # Confusion matrices for every candidate threshold (gated + raw)
    tp = np.zeros(num_thresholds); fp = np.zeros(num_thresholds)
    fn = np.zeros(num_thresholds); tn = np.zeros(num_thresholds)
    tp_r = np.zeros(num_thresholds); fp_r = np.zeros(num_thresholds)
    fn_r = np.zeros(num_thresholds); tn_r = np.zeros(num_thresholds)

    with torch.no_grad():
        for images, masks, weights in dataloader:
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            weights = weights.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, enabled=(device.type == "cuda")):
                logits = model(images)
                loss = criterion(logits, masks, weights)
            running_loss += loss.item()

            probs = tta_probabilities(model, images, device, n_views=tta_views,
                                      scales=scales) if use_tta \
                else torch.sigmoid(logits.float())
            target = masks.float()

            gate = None
            if ndvi_gate and ndvi_mean is not None and ndvi_std is not None:
                raw_ndvi = images[:, NDVI_POST_CHANNEL] * ndvi_std + ndvi_mean
                gate = (raw_ndvi < NDVI_GATE_THRESHOLD).float()
            probs_g = probs * gate if gate is not None else probs

            for i, thr in enumerate(thresholds):
                pred = (probs > thr).float()
                pred_g = (probs_g > thr).float()
                tp[i] += torch.sum(pred_g * target).item()
                fp[i] += torch.sum(pred_g * (1.0 - target)).item()
                fn[i] += torch.sum((1.0 - pred_g) * target).item()
                tn[i] += torch.sum((1.0 - pred_g) * (1.0 - target)).item()
                tp_r[i] += torch.sum(pred * target).item()
                fp_r[i] += torch.sum(pred * (1.0 - target)).item()
                fn_r[i] += torch.sum((1.0 - pred) * target).item()
                tn_r[i] += torch.sum((1.0 - pred) * (1.0 - target)).item()

    metrics = confusion_to_metrics(tp, fp, fn, tn, thresholds, min_recall)
    raw = confusion_to_metrics(tp_r, fp_r, fn_r, tn_r, thresholds, min_recall)
    metrics.update({
        "loss": running_loss / total_batches,
        "raw_threshold": raw["threshold"],
        "raw_precision": raw["precision"],
        "raw_recall": raw["recall"],
        "raw_f1": raw["f1"],
        "raw_iou": raw["iou"],
    })
    return metrics


def get_event_key(filename, meta):
    info = meta.get(filename.replace(".h5", ""), {})
    if info.get("annotated") and (info.get("event_date") or info.get("ann_id")):
        return (filename.split("_s2_")[0], info.get("event_date") or info.get("ann_id"))
    return None


def build_event_disjoint_split(dataset, val_fraction=0.1, seed=42,
                               pos_fractions=None, max_positive_frac=0.65):
    """Event-disjoint train/validation split.

    Tiles from the SAME landslide event (same region + event date, from the
    raw metadata sidecar) always stay together in one split. Without this,
    overlapping tiles of one large landslide leak into both train and val,
    inflating the reported metrics.

    The validation set is capped at max_positive_frac (65%) positive tiles:
    without the cap, the val set ends up ~95% landslide tiles and threshold
    tuning chases an unrealistically landslide-only world, which pushed the
    tuned threshold down to 0.25 (reckless false-positive explosion).
    Background (unannotated) tiles fill the remainder.
    """
    import json as _json
    metadata_path = "./datasets/TrainData/tile_metadata.json"
    meta = {}
    if os.path.exists(metadata_path):
        with open(metadata_path, "r") as jf:
            meta = _json.load(jf)

    files = dataset.file_list
    event_of = {}
    for i, fn in enumerate(files):
        k = get_event_key(fn, meta)
        event_of[i] = k

    events = {}
    for i, k in event_of.items():
        if k is not None:
            events.setdefault(k, []).append(i)

    def is_positive(i):
        return pos_fractions is not None and pos_fractions[i] > 0.001

    rng = random.Random(seed)
    event_ids = list(events.keys())
    rng.shuffle(event_ids)

    target = int(len(files) * val_fraction)
    max_positive = int(target * max_positive_frac) if pos_fractions is not None else target
    val_indices = set()
    positive_count = 0

    # Add whole events first (an event never straddles the split), skipping
    # any event that would push the positive-tile share past the cap.
    for k in event_ids:
        members = events[k]
        n_pos = sum(1 for i in members if is_positive(i))
        if (len(val_indices) + len(members) <= target
                and positive_count + n_pos <= max_positive):
            val_indices.update(members)
            positive_count += n_pos

    # Top up with background (unannotated) tiles, which are always negative
    background = [i for i, k in event_of.items() if k is None]
    rng.shuffle(background)
    for i in background:
        if len(val_indices) >= target:
            break
        val_indices.add(i)

    val_indices = sorted(val_indices)
    val_set = set(val_indices)
    train_indices = [i for i in range(len(files)) if i not in val_set]
    return train_indices, val_indices


class EventGroupedSampler(Sampler):
    """Samples landslide events (spatially contiguous tile groups).

    A draw picks an event with probability proportional to its group weight
    (positive content), then picks one of its tiles with probability
    proportional to that tile's self-paced trust. Background tiles each form
    a singleton group; their combined draw mass is floored at 15% so the
    model still sees pure-background tiles every epoch (they never appeared
    under the old pure positive-weighted sampling).

    A dual pool guarantees 20% of draws come from the landslide-richest
    events (max positive fraction > 0.05), keeping hard positive examples
    permanently in the batch mix.
    """

    def __init__(self, groups, group_keys, group_weights, tile_trust,
                 num_samples, rich_group_mask=None, rich_pool_prob=0.2):
        super(EventGroupedSampler, self).__init__()
        self.groups = groups
        self.group_keys = group_keys
        self.group_weights = group_weights
        self.tile_trust = tile_trust
        self.num_samples = num_samples
        self.rich_group_mask = rich_group_mask
        self.rich_pool_prob = rich_pool_prob

    def __iter__(self):
        rng = np.random.default_rng()
        w = self.group_weights / self.group_weights.sum()
        rich = self.rich_group_mask
        for _ in range(self.num_samples):
            if rich is not None and rich.any() and rng.random() < self.rich_pool_prob:
                pool = np.where(rich)[0]
                pw = self.group_weights[pool]
                g = pool[rng.choice(len(pool), p=pw / pw.sum())]
            else:
                g = rng.choice(len(self.group_keys), p=w)
            members = self.groups[self.group_keys[g]]
            trust = self.tile_trust[members]
            p = trust / (trust.sum() + 1e-12)
            yield int(rng.choice(members, p=p))

    def __len__(self):
        return self.num_samples


def train_pipeline(epochs=20, batch_size=16, lr=1e-3, pos_weight=6.0,
                   use_compile=True, val_fraction=0.1):
    """Executes training with a held-out validation split and real metrics."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Fixed input size (128x128x42) throughout: let cuDNN autotune kernels.
    torch.backends.cudnn.benchmark = True
    # TF32 reduces matmul precision on Ampere+ from FP32 to ~FP19; with
    # label noise and clouds at this scale the accuracy loss is unmeasurable
    # and the throughput gain is substantial.
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.manual_seed(0)
    random.seed(0)

    train_img_dir = "./datasets/TrainData/img/"
    train_mask_dir = "./datasets/TrainData/mask/"

    # Force delete existing weights file to guarantee an absolute fresh training initialize pass
    weights_output_path = "landslide_unet_weights.pth"
    threshold_output_path = "landslide_best_threshold.json"
    for stale in (weights_output_path, threshold_output_path):
        if os.path.exists(stale):
            os.remove(stale)

    start_time = time.time()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    # Event-disjoint train/validation split (same event never straddles both),
    # with the positive-tile share of validation capped at 65% so threshold
    # tuning does not chase an unrealistically landslide-only val set.
    transforms = get_training_augmentations()
    full_dataset = LandslideDataset(img_dir=train_img_dir, mask_dir=train_mask_dir)
    pos_fractions = build_or_load_pos_fractions(full_dataset)

    # Persist the split so every run evaluates the SAME validation tiles
    # (results across runs stay comparable as the pipeline evolves).
    split_path = "./datasets/TrainData/val_indices.json"
    if os.path.exists(split_path):
        with open(split_path, "r") as jf:
            split_data = json.load(jf)
        if split_data.get("n_files") == len(full_dataset):
            train_indices, val_indices = split_data["train"], split_data["val"]
            print(f"Loaded persisted split ({len(val_indices)} val tiles).")
        else:
            split_data = None
    else:
        split_data = None
    if split_data is None:
        train_indices, val_indices = build_event_disjoint_split(
            full_dataset, val_fraction, pos_fractions=pos_fractions,
            max_positive_frac=0.65,
        )
        with open(split_path, "w") as jf:
            json.dump({"n_files": len(full_dataset),
                       "train": train_indices, "val": val_indices}, jf)

    # NDVI statistics for the inference NDVI gate (un-z-score channel 13)
    ndvi_mean = float(full_dataset.norm_mean[NDVI_CHANNEL])
    ndvi_std = float(full_dataset.norm_std[NDVI_CHANNEL])

    train_dataset = Subset(
        LandslideDataset(img_dir=train_img_dir, mask_dir=train_mask_dir,
                         augmentations=transforms),
        train_indices,
    )
    val_dataset = Subset(
        LandslideDataset(img_dir=train_img_dir, mask_dir=train_mask_dir),
        val_indices,
    )

    # Event-grouped sampling: draws whole landslide events (groups of
    # spatially contiguous tiles) weighted by their positive content, so each
    # batch mixes neighboring tiles of the same event. Self-paced tile trust
    # (refreshed every 2 epochs) down-weights noisy labels inside each group.
    meta = {}
    if os.path.exists("./datasets/TrainData/tile_metadata.json"):
        with open("./datasets/TrainData/tile_metadata.json", "r") as jf:
            meta = json.load(jf)

    train_files = [full_dataset.file_list[i] for i in train_indices]
    event_key_of = [get_event_key(f, meta) for f in train_files]
    train_indices_np = np.array(train_indices)
    groups = {}
    for local_i, k in enumerate(event_key_of):
        groups.setdefault(k if k is not None else ("background", -1), []).append(local_i)
    group_keys = list(groups.keys())
    group_weights = np.array([
        float(np.max(pos_fractions[train_indices_np[groups[k]]]) + 1e-4) ** 0.5
        for k in group_keys
    ], dtype=np.float64)

    # Floor the combined background-group draw mass at 15% of batches so
    # pure-background tiles (which the old positive-weighted sampler almost
    # never drew) still appear regularly for precision.
    bg_mask = np.array([
        float(np.max(pos_fractions[train_indices_np[groups[k]]])) <= 0.001
        for k in group_keys
    ])
    if bg_mask.any() and (~bg_mask).any():
        group_weights[~bg_mask] *= 0.85 / group_weights[~bg_mask].sum()
        group_weights[bg_mask] *= 0.15 / group_weights[bg_mask].sum()

    # 20% of draws are forced from the landslide-richest events (dual pool)
    rich_group_mask = np.array([
        float(np.max(pos_fractions[train_indices_np[groups[k]]])) > 0.05
        for k in group_keys
    ])

    tile_trust = np.ones(len(train_indices), dtype=np.float64)

    def build_train_loader():
        sampler = EventGroupedSampler(groups, group_keys, group_weights,
                                      tile_trust, num_samples=len(train_indices),
                                      rich_group_mask=rich_group_mask)
        return DataLoader(
            train_dataset,
            batch_size=batch_size,
            sampler=sampler,
            drop_last=True,
            num_workers=4,
            persistent_workers=True,
            prefetch_factor=2,
            pin_memory=True if device.type == "cuda" else False,
        )

    train_loader = build_train_loader()
    val_loader = DataLoader(
        val_dataset,
        batch_size=max(16, 2 * batch_size),
        shuffle=False,
        num_workers=4,
        persistent_workers=True,
        prefetch_factor=2,
        pin_memory=True if device.type == "cuda" else False,
    )

    # Self-paced label-trust refresh: every 2 epochs the EMA model scores all
    # training tiles; tiles where it confidently agrees with the (noisy)
    # label keep full sampling weight, tiles it rejects get down-weighted so
    # training capacity is spent on clean signal.
    trust_loader = DataLoader(
        Subset(full_dataset, train_indices),
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        persistent_workers=True,
        prefetch_factor=2,
        pin_memory=True if device.type == "cuda" else False,
    )

    def refresh_tile_trust():
        nonlocal train_loader
        model.eval()
        ious = np.zeros(len(train_indices), dtype=np.float64)
        ema_now = ema_state_dict(ema_dict, model)
        saved = {k: v.detach().clone() for k, v in model.state_dict().items()}
        model.load_state_dict(ema_now)
        start = 0
        with torch.no_grad():
            for imgs, msks, _ in trust_loader:
                n = imgs.size(0)
                imgs = imgs.to(device, non_blocking=True)
                with torch.autocast(device_type=device.type, enabled=(device.type == "cuda")):
                    probs = torch.sigmoid(model(imgs).float()).cpu().numpy()
                tgt = msks.numpy()[:, 0]
                for k in range(n):
                    p = (probs[k] > 0.5).astype(np.float64)
                    y = tgt[k].astype(np.float64)
                    union = ((p + y) > 0).sum()
                    ious[start + k] = (p * y).sum() / (union + 1e-8)
                start += n
        model.load_state_dict(saved)
        model.train()
        tile_trust[:] = np.clip(0.4 + 0.6 * ious, 0.05, 1.0)
        train_loader = build_train_loader()
        print(f"  -> Self-paced trust refresh: mean tile IoU {ious.mean():.3f} | "
              f"trust mean {tile_trust.mean():.3f} | sampler rebuilt")

    model = CustomUNet(in_channels=42, out_channels=1).to(device)
    if use_compile:
        if importlib.util.find_spec("triton") is None:
            print("torch.compile: skipped (triton unavailable for this Python/Windows build); running eager")
        else:
            try:
                compiled = torch.compile(model, mode="reduce-overhead")
                # torch.compile is lazy: force a real forward so any backend
                # failure (e.g. CUDA-graph issues on Windows) surfaces HERE and
                # falls back to eager instead of crashing mid-epoch.
                with torch.no_grad():
                    _ = compiled(torch.randn(2, 42, 128, 128).to(device))
                torch.cuda.synchronize() if device.type == "cuda" else None
                model = compiled
                print("torch.compile: enabled (reduce-overhead)")
            except Exception as exc:
                print(f"torch.compile: failed ({exc}); running eager")
    criterion = WeightedBCEDiceLoss(pos_weight=pos_weight, dice_weight=1.5)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=lr, total_steps=len(train_loader) * epochs,
        pct_start=0.3, div_factor=10, final_div_factor=100,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    # Exponential moving average of ALL floating-point state INCLUDING
    # batch-norm running stats (EMA weights were previously paired with the
    # live model's BN stats, a subtle mismatch at validation time).
    ema_dict = {
        key: val.detach().clone()
        for key, val in model.state_dict().items()
        if val.dtype.is_floating_point
    }

    print("\n================== LAUNCHING MASTER NEW MODEL TRAINING ==================")
    print(f"Total Physical Training Files Available : {len(full_dataset)}")
    print(f"Training Split Size                    : {len(train_dataset)}")
    print(f"Validation Split Size                  : {len(val_dataset)} (event-disjoint, persisted)")
    print(f"Validation Positive Tiles              : {sum(1 for i in val_indices if pos_fractions[i] > 0.001)} (capped at 65%)")
    print(f"Total Optimization Steps per Epoch     : {len(train_loader)}")
    print(f"Loss: BCE(pos_w {criterion.pos_weight}) + Dice(1.5) + 0.4x DeepSup + 0.3x Tversky | Smoothing 0.05 | Noise weights ON")
    print(f"NDVI gate: ON (suppress predictions on vegetated post-event pixels)")
    print(f"Target Accelerated Hardware Device Node: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    print("=========================================================================\n")

    total_batches = len(train_loader)
    best_ema_iou = 0.0
    swa_snapshots = []

    # Per-epoch history log (CSV) for trend analysis
    history_path = "train_history.csv"
    with open(history_path, "w", newline="") as hf:
        csv.writer(hf).writerow(
            ["epoch", "train_loss", "val_loss", "val_iou", "val_f1",
             "val_precision", "val_recall", "threshold", "lr", "best_iou"]
        )

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        # EMA decay ramps 0.99 -> 0.999 over the first 5 epochs (early
        # weights are too volatile to average at full strength).
        ema_decay = 0.99 + 0.009 * min(1.0, epoch / 5.0)
        # pos_weight ramps 6.0 -> ~3.5: the model starts recall-hungry
        # (learn the positive class), then leans precision to refine edges.
        criterion.pos_weight = pos_weight * (1.0 - 0.42 * min(1.0, epoch / (0.7 * epochs)))

        for batch_idx, (images, masks, weights) in enumerate(train_loader):
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            weights = weights.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=(device.type == "cuda")):
                logits, aux_logits = model(images, return_aux=True)
                loss = criterion(logits, masks, weights)
                if aux_logits:
                    aux_loss = sum(criterion(a, masks, weights) for a in aux_logits) / len(aux_logits)
                    loss = loss + 0.4 * aux_loss

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            update_ema(ema_dict, model, decay=ema_decay)

            running_loss += loss.item()

            if (batch_idx + 1) % 200 == 0:
                print(f"Epoch [{epoch+1}/{epochs}] | Step [{batch_idx+1}/{total_batches}] | Active Hybrid Loss: {loss.item():.4f}")

        # Validation pass on the EMA weights (2-view TTA: cheap per-epoch
        # tracking). The final summary re-evaluates the best checkpoint with
        # the full 4-view TTA for the reported numbers.
        ema_state = ema_state_dict(ema_dict, model)
        saved_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        model.load_state_dict(ema_state)
        ema_metrics = evaluate_dataset(model, val_loader, criterion, device,
                                       use_tta=True, tta_views=2,
                                       ndvi_gate=True, ndvi_mean=ndvi_mean,
                                       ndvi_std=ndvi_std)
        model.load_state_dict(saved_state)
        mean_train_loss = running_loss / total_batches

        # Stochastic Weight Averaging: retain the last few EMA snapshots
        swa_snapshots.append(copy.deepcopy(ema_state))
        if len(swa_snapshots) > SWA_SNAPSHOTS:
            swa_snapshots.pop(0)

        # Self-paced label-trust refresh every 2 epochs
        if (epoch + 1) % 2 == 0:
            refresh_tile_trust()

        print(f"--- Epoch [{epoch+1}/{epochs}] Complete. Mean Train Loss: {mean_train_loss:.4f} | "
              f"Val IoU: {ema_metrics['iou'] * 100:.1f}% | "
              f"Val F1: {ema_metrics['f1'] * 100:.1f}% | Threshold: {ema_metrics['threshold']:.2f} | "
              f"LR: {scheduler.get_last_lr()[0]:.2e} ---\n")

        # Save the checkpoint only when validation IoU improves
        if ema_metrics["iou"] > best_ema_iou:
            best_ema_iou = ema_metrics["iou"]
            torch.save(ema_state, weights_output_path)
            with open(threshold_output_path, "w") as jf:
                json.dump({"threshold": ema_metrics["threshold"]}, jf)
            print(f"  -> New best validation IoU ({best_ema_iou * 100:.1f}%). Weights + threshold saved.")

        with open(history_path, "a", newline="") as hf:
            csv.writer(hf).writerow([
                epoch + 1, f"{mean_train_loss:.4f}", f"{ema_metrics['loss']:.4f}",
                f"{ema_metrics['iou']:.4f}", f"{ema_metrics['f1']:.4f}",
                f"{ema_metrics['precision']:.4f}", f"{ema_metrics['recall']:.4f}",
                f"{ema_metrics['threshold']:.2f}", f"{scheduler.get_last_lr()[0]:.2e}",
                f"{best_ema_iou:.4f}",
            ])

    elapsed_time = time.time() - start_time
    hours, rem = divmod(int(elapsed_time), 3600)
    minutes, seconds = divmod(rem, 60)

    # SWA: average the retained EMA snapshots. Kept as the final weights only
    # if it beats the best EMA checkpoint on the validation set.
    if swa_snapshots:
        with torch.no_grad():
            swa_state = {
                key: torch.stack([s[key].float() for s in swa_snapshots]).mean(dim=0)
                for key in swa_snapshots[0]
            }
        model.load_state_dict(swa_state)
        swa_metrics = evaluate_dataset(model, val_loader, criterion, device,
                                       thresholds=np.arange(0.01, 0.99, 0.01),
                                       scales=(0.9, 1.0, 1.1),
                                       ndvi_gate=True, ndvi_mean=ndvi_mean,
                                       ndvi_std=ndvi_std)
        if swa_metrics["iou"] > best_ema_iou:
            best_ema_iou = swa_metrics["iou"]
            torch.save(swa_state, weights_output_path)
            with open(threshold_output_path, "w") as jf:
                json.dump({"threshold": swa_metrics["threshold"]}, jf)
            print(f"  -> SWA averaged {len(swa_snapshots)} snapshots: IoU {swa_metrics['iou'] * 100:.1f}% "
                  f"-> saved as final weights (beats best EMA checkpoint).")

    # Re-evaluate the best checkpoint for the final summary: finer 0.01
    # threshold grid + 12-view multi-scale TTA + NDVI gate.
    if os.path.exists(weights_output_path):
        model.load_state_dict(torch.load(weights_output_path, map_location=device))
    final_metrics = evaluate_dataset(
        model, val_loader, criterion, device,
        thresholds=np.arange(0.01, 0.99, 0.01),
        scales=(0.9, 1.0, 1.1),
        ndvi_gate=True, ndvi_mean=ndvi_mean, ndvi_std=ndvi_std,
    )

    file_size_mb = os.path.getsize(weights_output_path) / (1024 * 1024) if os.path.exists(weights_output_path) else 0.0

    # Profile performance speeds locally on your laptop (warmup first so
    # cuDNN autotune/context init isn't counted in the timing)
    dummy_input = torch.randn(1, 42, 128, 128).to(device)
    with torch.no_grad():
        for _ in range(3):
            _ = model(dummy_input)
    torch.cuda.synchronize() if device.type == "cuda" else None
    inf_start = time.time()
    with torch.no_grad():
        _ = model(dummy_input)
    torch.cuda.synchronize() if device.type == "cuda" else None
    inf_speed_ms = (time.time() - inf_start) * 1000
    peak_vram_gb = torch.cuda.max_memory_allocated(0) / (1024 ** 3) if device.type == "cuda" else 0.0

    print("\n================ TRAINING SUMMARY ================")
    print("Model Architecture: Custom PyTorch U-Net (42 Channels Temporal 3-Window, 3-Level)")
    print("                    SE + Residual + Skip Spatial-Attention + Deep Supervision + torch.compile")
    print(f"Total Training Time: {hours}h {minutes}m {seconds}s ({epochs} Epochs)")
    print("")
    print("--- Evaluation Metrics (Held-Out Validation Split, NDVI-Gated, Tuned Threshold, 12-view TTA) ---")
    print(f"Best Threshold: {final_metrics['threshold']:.2f}")
    print(f"Accuracy:  {final_metrics['accuracy'] * 100:.1f}%")
    print(f"Precision: {final_metrics['precision'] * 100:.1f}%")
    print(f"Recall:    {final_metrics['recall'] * 100:.1f}%")
    print(f"F1-Score:  {final_metrics['f1'] * 100:.1f}%")
    print(f"Mean IoU:  {final_metrics['iou'] * 100:.1f}%  <-- Best validation track (NDVI-gated)")
    print(f"Precision at recall >= 60%: {final_metrics['precision_at_recall'] * 100:.1f}% "
          f"(threshold {final_metrics['threshold_at_recall']:.2f})")
    print(f"Raw (ungated) reference: P {final_metrics['raw_precision'] * 100:.1f}% | "
          f"R {final_metrics['raw_recall'] * 100:.1f}% | F1 {final_metrics['raw_f1'] * 100:.1f}% | "
          f"IoU {final_metrics['raw_iou'] * 100:.1f}% @ {final_metrics['raw_threshold']:.2f}")
    print("")
    print("--- Operational Footprint ---")
    print(f"Weight File Size:  {file_size_mb:.1f} MB")
    print(f"Inference Speed:   {inf_speed_ms:.1f} ms / image frame")
    print(f"Peak VRAM Usage:   {peak_vram_gb:.1f} GB")
    print("==================================================\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sen12Landslides U-Net training")
    # Positional epochs keeps the old `python train.py 1` invocation working
    parser.add_argument("epochs", nargs="?", type=int, default=None,
                        help="number of epochs (positional, e.g. `train.py 1`)")
    parser.add_argument("--epochs", dest="epochs_flag", type=int, default=None,
                        help="number of epochs (flag form)")
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--pos-weight", type=float, default=6.0)
    parser.add_argument("--no-compile", action="store_true",
                        help="disable torch.compile (fallback: eager mode)")
    args = parser.parse_args()
    epochs = args.epochs_flag if args.epochs_flag is not None else (args.epochs or 20)
    train_pipeline(epochs=epochs, batch_size=args.batch, lr=args.lr,
                   pos_weight=args.pos_weight, use_compile=not args.no_compile)
