import os
import sys

# Add src/ to the import path so all library modules are importable from root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import json
import time
import csv
import datetime
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

from model import CustomUNet
from dataset import LandslideDataset, NDVI_CHANNEL
from augmentations import get_training_augmentations
from paths import PATHS


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
        # Label smoothing belongs to the BCE term ONLY. The Dice/Tversky terms
        # are region-overlap ratios: smoothing gives all ~16k background pixels
        # a target of 0.025, and at a ~1.5% positive rate that phantom mass
        # (16138 * 0.025 = 403) outweighs the real positives (246 * 0.975 = 240)
        # in the cardinality denominator, so the overlap terms were dominated
        # by background and barely responded to the actual prediction.
        soft_masks = masks
        if self.label_smoothing > 0.0:
            soft_masks = masks * (1.0 - self.label_smoothing) + 0.5 * self.label_smoothing
        bce_loss = nn.functional.binary_cross_entropy_with_logits(
            logits,
            soft_masks,
            weight=weights,
            pos_weight=torch.tensor(self.pos_weight, device=logits.device),
            reduction="mean",
        )
        probs = torch.sigmoid(logits)
        p = probs.view(-1)
        y = masks.view(-1)  # hard targets: region terms must stay exact
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


POS_FRAC_CACHE_PATH = PATHS.POS_FRAC_CACHE
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


def export_state_dict(state):
    """Strip torch.compile's '_orig_mod.' prefix before writing a checkpoint.

    torch.compile returns a wrapper module, so state_dict() keys come out as
    '_orig_mod.<name>'. Saving those verbatim yields a checkpoint that
    predict.py - which builds a plain CustomUNet - can never load. Stripping
    at save time keeps the on-disk format identical whether or not compile
    was active.
    """
    prefix = "_orig_mod."
    return {
        (key[len(prefix):] if key.startswith(prefix) else key): val
        for key, val in state.items()
    }


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
                # Slice as a RANGE to keep the channel dimension. Indexing with
                # a bare int gives (B, H, W), and multiplying that against the
                # (B, 1, H, W) probability map broadcasts to (B, B, H, W) -
                # every image gated by every other image's NDVI. That silently
                # corrupted every gated metric before it started crashing here.
                raw_ndvi = (images[:, NDVI_POST_CHANNEL:NDVI_POST_CHANNEL + 1]
                            * ndvi_std + ndvi_mean)
                gate = (raw_ndvi < NDVI_GATE_THRESHOLD).float()
            probs_g = probs * gate if gate is not None else probs
            if probs_g.shape != probs.shape:
                raise RuntimeError(
                    f"NDVI gate broadcast the probability map: {probs.shape} -> "
                    f"{probs_g.shape}. Gate and probs must share a shape."
                )

            # Vectorized threshold sweep. The previous per-threshold Python
            # loop issued 8 blocking .item() syncs per threshold - 784 GPU
            # stalls per batch on the final 0.01 grid - which dominated the
            # whole evaluation. One (T, N) comparison per map replaces that
            # with 2 transfers, and the arithmetic is identical.
            thr_t = torch.as_tensor(thresholds, device=probs.device,
                                    dtype=probs.dtype).view(-1, 1)
            target_flat = target.reshape(1, -1) > 0.5
            n_pix = target_flat.numel()
            n_pos = int(target_flat.sum().item())
            # Accumulators are unpacked into names: `acc[0] += x` on a tuple is
            # item assignment and raises TypeError, even though the element is
            # a mutable ndarray. `a_tp += x` mutates the array in place.
            for prob_map, (a_tp, a_fp, a_fn, a_tn) in (
                    (probs_g, (tp, fp, fn, tn)),
                    (probs, (tp_r, fp_r, fn_r, tn_r))):
                pred = prob_map.reshape(1, -1) > thr_t       # (T, N) bool
                tp_np = (pred & target_flat).sum(dim=1).cpu().numpy()
                pred_pos_np = pred.sum(dim=1).cpu().numpy()
                fn_np = n_pos - tp_np
                a_tp += tp_np
                a_fp += pred_pos_np - tp_np
                a_fn += fn_np
                a_tn += n_pix - pred_pos_np - fn_np

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
    metadata_path = PATHS.TILE_METADATA
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
                 num_samples, rich_group_mask=None, rich_pool_prob=0.2,
                 seed=0):
        super(EventGroupedSampler, self).__init__()
        self.groups = groups
        self.group_keys = group_keys
        self.group_weights = group_weights
        # Held by reference: refresh_tile_trust() updates this array in place,
        # so every epoch re-reads the current self-paced trust with no rebuild.
        self.tile_trust = tile_trust
        self.num_samples = num_samples
        self.rich_group_mask = rich_group_mask
        self.rich_pool_prob = rich_pool_prob
        # Seeded per epoch: an unseeded default_rng() made every run draw a
        # different tile order, so two runs of the same code were not
        # comparable and no change could be attributed to the change itself.
        self.seed = seed
        self.epoch = 0

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        self.epoch += 1
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


def train_pipeline(epochs=20, batch_size=16, lr=2e-4, pos_weight=6.0,
                   use_compile=True, val_fraction=0.1, loader_workers=2,
                   refine_threshold=None, base_filters=64, dropout=0.2,
                   weight_decay=3e-4, patience=6, pos_weight_decay=0.0):
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

    train_img_dir = PATHS.TRAIN_IMG_DIR
    train_mask_dir = PATHS.TRAIN_MASK_DIR

    # Start from a genuinely fresh checkpoint, but ARCHIVE rather than delete.
    weights_output_path = PATHS.WEIGHTS
    threshold_output_path = PATHS.THRESHOLD_JSON
    for stale in (weights_output_path, threshold_output_path):
        if os.path.exists(stale):
            base, ext = os.path.splitext(stale)
            previous = f"{base}.prev{ext}"
            if os.path.exists(previous):
                os.remove(previous)
            os.replace(stale, previous)
            print(f"Archived previous {stale} -> {previous}")

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
    split_path = PATHS.VAL_INDICES
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
                         augmentations=transforms,
                         refine_threshold=refine_threshold),
        train_indices,
    )
    val_dataset = Subset(
        LandslideDataset(img_dir=train_img_dir, mask_dir=train_mask_dir,
                         refine_threshold=None),
        val_indices,
    )

    # Event-grouped sampling: draws whole landslide events (groups of
    # spatially contiguous tiles) weighted by their positive content, so each
    # batch mixes neighboring tiles of the same event. Self-paced tile trust
    # (refreshed every 2 epochs) down-weights noisy labels inside each group.
    meta = {}
    if os.path.exists(PATHS.TILE_METADATA):
        with open(PATHS.TILE_METADATA, "r") as jf:
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

    # Worker budget. On Windows, DataLoader workers are SPAWNED, so each one is
    # a fresh process that re-imports torch and reserves GB-scale commit charge
    # for the CUDA DLLs before it loads a single tile. The old layout ran three
    # loaders at 4 persistent workers each = 12 such processes, which exhausted
    # the system commit limit mid-run ("[WinError 1455] The paging file is too
    # small"). Windows does auto-grow the pagefile, but far slower than 12 torch
    # processes allocate, so the burst wins the race.
    #   train : `loader_workers`, persistent (hot path, needs the throughput)
    #   val   : 0 - runs in the main process; it is only ~1.4k tiles and the
    #           wall time is dominated by multi-view TTA on the GPU anyway
    #   trust : `loader_workers`, NOT persistent, so the processes are released
    #           between refreshes instead of idling for the whole run
    # Peak concurrent workers: 2 * loader_workers, versus 12 before.
    def loader_kwargs(n_workers, persistent):
        if n_workers <= 0:
            # persistent_workers/prefetch_factor are invalid when workers == 0
            return {"num_workers": 0}
        return {"num_workers": n_workers,
                "persistent_workers": persistent,
                "prefetch_factor": 2}

    pin = device.type == "cuda"

    def build_train_loader():
        sampler = EventGroupedSampler(groups, group_keys, group_weights,
                                      tile_trust, num_samples=len(train_indices),
                                      rich_group_mask=rich_group_mask, seed=0)
        return DataLoader(
            train_dataset,
            batch_size=batch_size,
            sampler=sampler,
            drop_last=True,
            pin_memory=pin,
            **loader_kwargs(loader_workers, persistent=True),
        )

    train_loader = build_train_loader()
    val_loader = DataLoader(
        val_dataset,
        batch_size=max(16, 2 * batch_size),
        shuffle=False,
        pin_memory=pin,
        **loader_kwargs(0, persistent=False),
    )

    # Self-paced label-trust refresh: every 2 epochs the EMA model scores all
    # training tiles; tiles where it confidently agrees with the (noisy)
    # label keep full sampling weight, tiles it rejects get down-weighted so
    # training capacity is spent on clean signal.
    # 0 workers, deliberately. The trust refresh is the ONLY moment when two
    # loaders are live at once: the train loader's persistent workers idle
    # while this one spawns its own. Measured on this machine, each spawned
    # torch worker costs ~1.9 GB of commit, and that overlap drove headroom to
    # 0.4 GB even at --workers 2. Scoring in the main process removes the
    # spike entirely; it costs wall time, not stability.
    trust_loader = DataLoader(
        Subset(full_dataset, train_indices),
        batch_size=max(32, 2 * batch_size),
        shuffle=False,
        pin_memory=pin,
        **loader_kwargs(0, persistent=False),
    )

    def refresh_tile_trust():
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
        # In-place update: the live sampler holds a reference to this array,
        # so the next epoch picks the new trust up automatically. Rebuilding
        # the DataLoader here (as before) stranded a full set of
        # persistent_workers processes every second epoch.
        tile_trust[:] = np.clip(0.4 + 0.6 * ious, 0.05, 1.0)
        print(f"  -> Self-paced trust refresh: mean tile IoU {ious.mean():.3f} | "
              f"trust mean {tile_trust.mean():.3f}")

    model = CustomUNet(in_channels=42, out_channels=1,
                       base_filters=base_filters, dropout=dropout).to(device)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
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
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    # pct_start 0.15 (was 0.30): validation IoU peaked during WARMUP at the
    # lowest LR the model ever saw and fell as the LR climbed, so the schedule
    # should spend less time near the peak and more time annealing.
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=lr, total_steps=len(train_loader) * epochs,
        pct_start=0.15, div_factor=10, final_div_factor=100,
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
    print(f"Model: {n_params:.1f}M params (base_filters {base_filters}, dropout {dropout})")
    print(f"Optim: AdamW lr {lr:.1e} (OneCycle, 15% warmup) | weight_decay {weight_decay:.1e} "
          f"| early stop patience {patience if patience > 0 else 'off'}")
    print(f"Loss: BCE(pos_w {criterion.pos_weight:.2f}"
          f"{', constant' if pos_weight_decay == 0 else f', decaying {pos_weight_decay:.0%}'}) "
          f"+ Dice(1.5) + 0.4x DeepSup + 0.3x Tversky | Smoothing 0.05 | Noise weights ON")
    print(f"NDVI gate: ON (suppress predictions on vegetated post-event pixels)")
    print(f"DataLoader workers: {loader_workers} train (persistent) + 0 val + 0 trust "
          f"| peak {loader_workers} spawned processes (~1.9 GB commit each)")
    print(f"Target Accelerated Hardware Device Node: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    print("=========================================================================\n")

    total_batches = len(train_loader)
    best_ema_iou = 0.0
    epochs_since_best = 0
    swa_snapshots = []

    # Per-epoch history log (CSV) for trend analysis. Every run gets its own
    # timestamped directory so records from previous runs are never
    # overwritten; all runs stay available under training_runs/.
    run_tag = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(PATHS.TRAINING_RUNS_DIR, run_tag)
    os.makedirs(run_dir, exist_ok=True)
    history_path = os.path.join(run_dir, "train_history.csv")
    epoch_log_path = os.path.join(run_dir, "epoch_log.txt")
    summary_path = os.path.join(run_dir, "train_summary.txt")
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
        # pos_weight is CONSTANT by default (pos_weight_decay=0.0).
        # The old ramp (6.0 -> 3.5) was meant to trade recall for precision late
        # in training, but it was engineering the exact degradation we measured:
        # over 20 epochs recall fell 0.45 -> 0.24 while precision only rose
        # 0.25 -> 0.35, so IoU dropped monotonically, and the tuned threshold
        # collapsed 0.32 -> 0.02 as the model drifted toward predicting
        # background everywhere. A moving objective also makes per-epoch IoU
        # non-comparable: each epoch was scored against a different loss.
        criterion.pos_weight = pos_weight * (
            1.0 - pos_weight_decay * min(1.0, epoch / max(1e-9, 0.7 * epochs))
        )

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
                                       ndvi_gate=False, ndvi_mean=ndvi_mean,
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

        epoch_line = (f"--- Epoch [{epoch+1}/{epochs}] Complete. Mean Train Loss: {mean_train_loss:.4f} | "
                      f"Val IoU: {ema_metrics['iou'] * 100:.1f}% | "
                      f"Val F1: {ema_metrics['f1'] * 100:.1f}% | Threshold: {ema_metrics['threshold']:.2f} | "
                      f"LR: {scheduler.get_last_lr()[0]:.2e} ---")
        print(epoch_line + "\n")
        with open(epoch_log_path, "a") as ef:
            ef.write(epoch_line + "\n")

        # Save the checkpoint only when validation IoU improves
        if ema_metrics["iou"] > best_ema_iou:
            best_ema_iou = ema_metrics["iou"]
            epochs_since_best = 0
            torch.save(export_state_dict(ema_state), weights_output_path)
            with open(threshold_output_path, "w") as jf:
                json.dump({"threshold": ema_metrics["threshold"]}, jf)
            print(f"  -> New best validation IoU ({best_ema_iou * 100:.1f}%). Weights + threshold saved.")
        else:
            epochs_since_best += 1
            print(f"  -> No improvement for {epochs_since_best} epoch(s) "
                  f"(best {best_ema_iou * 100:.1f}%)")

        with open(history_path, "a", newline="") as hf:
            csv.writer(hf).writerow([
                epoch + 1, f"{mean_train_loss:.4f}", f"{ema_metrics['loss']:.4f}",
                f"{ema_metrics['iou']:.4f}", f"{ema_metrics['f1']:.4f}",
                f"{ema_metrics['precision']:.4f}", f"{ema_metrics['recall']:.4f}",
                f"{ema_metrics['threshold']:.2f}", f"{scheduler.get_last_lr()[0]:.2e}",
                f"{best_ema_iou:.4f}",
            ])

        # Early stopping. The 20-epoch run peaked at epoch 1 and then spent 19
        # epochs getting worse; there is no reason to keep burning hours once
        # validation has clearly stopped improving.
        if patience > 0 and epochs_since_best >= patience:
            print(f"\nEarly stop: no validation improvement for {patience} epochs "
                  f"(best IoU {best_ema_iou * 100:.1f}% at epoch {epoch + 1 - epochs_since_best}). "
                  f"Stopping at epoch {epoch + 1}/{epochs}.")
            break

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
                                       ndvi_gate=False, ndvi_mean=ndvi_mean,
                                       ndvi_std=ndvi_std)
        if swa_metrics["iou"] > best_ema_iou:
            best_ema_iou = swa_metrics["iou"]
            torch.save(export_state_dict(swa_state), weights_output_path)
            with open(threshold_output_path, "w") as jf:
                json.dump({"threshold": swa_metrics["threshold"]}, jf)
            print(f"  -> SWA averaged {len(swa_snapshots)} snapshots: IoU {swa_metrics['iou'] * 100:.1f}% "
                  f"-> saved as final weights (beats best EMA checkpoint).")

    # Single-epoch / never-improved runs: the canonical weights file is only
    # written when validation IoU improves. Guarantee ONE .pth ALWAYS exists
    # at the model path so even a 1-epoch training run ends with a usable
    # checkpoint (alongside the history CSV) for predict.py / app.py.
    if not os.path.exists(weights_output_path) and swa_snapshots:
        torch.save(export_state_dict(swa_state), weights_output_path)
        print("  -> No validation improvement during the run; saved the last "
              "epoch's weights as the canonical checkpoint.")

    # Re-evaluate the best checkpoint for the final summary: finer 0.01
    # threshold grid + 12-view multi-scale TTA + NDVI gate.
    if os.path.exists(weights_output_path):
        # Checkpoints are saved unprefixed; load into the real module so this
        # works whether or not torch.compile wrapped the model.
        getattr(model, "_orig_mod", model).load_state_dict(
            torch.load(weights_output_path, map_location=device)
        )
    final_metrics = evaluate_dataset(
        model, val_loader, criterion, device,
        thresholds=np.arange(0.01, 0.99, 0.01),
        scales=(0.9, 1.0, 1.1),
        ndvi_gate=False, ndvi_mean=ndvi_mean, ndvi_std=ndvi_std,
    )

    # Persist the threshold that belongs to the numbers printed below. The
    # per-epoch value written earlier was tuned on 2-view TTA and a coarse
    # 0.02 grid; predict.py then applied it to a different TTA setting. The
    # operating point must come from the same evaluation that is reported.
    with open(threshold_output_path, "w") as jf:
        json.dump({"threshold": final_metrics["threshold"],
                   "ndvi_gate": False,
                   "ndvi_gate_threshold": NDVI_GATE_THRESHOLD}, jf)

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

    summary_text = f"""
================ TRAINING SUMMARY ================
Run Directory: {run_dir}
Model Architecture: Custom PyTorch U-Net (42 Channels Temporal 3-Window, 3-Level)
                    SE + Residual + Skip Spatial-Attention + Deep Supervision + torch.compile
Total Training Time: {hours}h {minutes}m {seconds}s ({epochs} Epochs)

--- Evaluation Metrics (Held-Out Validation Split, Raw/Ungated, Tuned Threshold, 12-view TTA) ---
Best Threshold: {final_metrics['threshold']:.2f}
Accuracy:  {final_metrics['accuracy'] * 100:.1f}%
Precision: {final_metrics['precision'] * 100:.1f}%
Recall:    {final_metrics['recall'] * 100:.1f}%
F1-Score:  {final_metrics['f1'] * 100:.1f}%
Mean IoU:  {final_metrics['iou'] * 100:.1f}%  <-- Best validation track (Raw/Ungated)
Precision at recall >= 60%: {final_metrics['precision_at_recall'] * 100:.1f}% (threshold {final_metrics['threshold_at_recall']:.2f})
Reference raw checks: P {final_metrics['raw_precision'] * 100:.1f}% | R {final_metrics['raw_recall'] * 100:.1f}% | F1 {final_metrics['raw_f1'] * 100:.1f}% | IoU {final_metrics['raw_iou'] * 100:.1f}% @ {final_metrics['raw_threshold']:.2f}

--- Operational Footprint ---
Weight File Size:  {file_size_mb:.1f} MB
Inference Speed:   {inf_speed_ms:.1f} ms / image frame
Peak VRAM Usage:   {peak_vram_gb:.1f} GB
==================================================
"""
    print(summary_text)
    with open(summary_path, "w") as sf:
        sf.write(summary_text)

    # Cumulative per-run summary so every training session stays on record
    runs_summary_path = PATHS.RUNS_SUMMARY
    runs_header = ["run", "epochs", "train_time_s", "threshold", "accuracy",
                   "precision", "recall", "f1", "iou", "weights_mb",
                   "inf_speed_ms", "peak_vram_gb"]
    is_new = not os.path.exists(runs_summary_path)
    with open(runs_summary_path, "a", newline="") as rf:
        w = csv.writer(rf)
        if is_new:
            w.writerow(runs_header)
        w.writerow([run_tag, epochs, int(elapsed_time),
                    f"{final_metrics['threshold']:.2f}",
                    f"{final_metrics['accuracy'] * 100:.1f}",
                    f"{final_metrics['precision'] * 100:.1f}",
                    f"{final_metrics['recall'] * 100:.1f}",
                    f"{final_metrics['f1'] * 100:.1f}",
                    f"{final_metrics['iou'] * 100:.1f}",
                    f"{file_size_mb:.1f}", f"{inf_speed_ms:.1f}",
                    f"{peak_vram_gb:.1f}"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sen12Landslides U-Net training")
    # Positional epochs keeps the old `python train.py 1` invocation working
    parser.add_argument("epochs", nargs="?", type=int, default=None,
                        help="number of epochs (positional, e.g. `train.py 1`)")
    parser.add_argument("--epochs", dest="epochs_flag", type=int, default=None,
                        help="number of epochs (flag form)")
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-4,
                        help="max LR for OneCycle (default 2e-4). The old 1e-3 "
                             "default overfit: validation IoU peaked during "
                             "warmup at the LOWEST LR and fell as LR climbed.")
    parser.add_argument("--pos-weight", type=float, default=6.0)
    parser.add_argument("--pos-weight-decay", type=float, default=0.0,
                        help="fraction by which pos_weight decays across the run "
                             "(default 0.0 = constant). The old hard-coded 0.42 "
                             "drove recall from 0.45 down to 0.24.")
    parser.add_argument("--base-filters", type=int, default=64,
                        help="model width (default 64, ~8.2M params). The 96 "
                             "setting is ~18.3M and overfits this dataset.")
    parser.add_argument("--dropout", type=float, default=0.2,
                        help="bottleneck dropout; decoder levels use half this")
    parser.add_argument("--weight-decay", type=float, default=3e-4)
    parser.add_argument("--patience", type=int, default=6,
                        help="early stop after N epochs without val improvement "
                             "(0 disables)")
    parser.add_argument("--no-compile", action="store_true",
                        help="disable torch.compile (fallback: eager mode)")
    parser.add_argument("--refine-threshold", type=float, default=None,
                        help="label refinement: strip mask positives whose "
                             "POST-window NDVI >= this value (e.g. 0.48). "
                             "Applied to TRAIN masks only; validation stays "
                             "on the raw labels for honest evaluation.")
    parser.add_argument("--workers", type=int, default=2,
                        help="DataLoader worker processes (default 2). Each one "
                             "re-imports torch and reserves GB-scale commit on "
                             "Windows; raise only if you have commit headroom, "
                             "use 0 to load entirely in the main process.")
    args = parser.parse_args()
    epochs = args.epochs_flag if args.epochs_flag is not None else (args.epochs or 20)
    train_pipeline(epochs=epochs, batch_size=args.batch, lr=args.lr,
                   pos_weight=args.pos_weight, use_compile=not args.no_compile,
                   refine_threshold=args.refine_threshold,
                   loader_workers=args.workers,
                   base_filters=args.base_filters, dropout=args.dropout,
                   weight_decay=args.weight_decay, patience=args.patience,
                   pos_weight_decay=args.pos_weight_decay)
