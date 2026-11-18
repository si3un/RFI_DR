import os
import re
import csv
import yaml
import json
import random
import argparse

import numpy as np
from sklearn.metrics import roc_curve, auc, confusion_matrix
from scipy.stats import norm

class AverageMeter:
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


# ------------------------------------------------------------
# [Added] Environment-variable interpolation
#  If a config value contains ${VAR} or $VAR, it is replaced with the environment variable
#  at run time. This lets server-specific mount paths be passed via environment variables
#  without editing the config. e.g. data_root: "${DATA_ROOT}"
# ------------------------------------------------------------
_ENV_PATTERN = re.compile(r"\$\{[^}]+\}|\$[A-Za-z_][A-Za-z0-9_]*")

def _expand_env(value):
    if isinstance(value, str) and _ENV_PATTERN.search(value):
        return os.path.expandvars(value)
    if isinstance(value, dict):
        return {k: _expand_env(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand_env(v) for v in value]
    return value

def load_yaml(filename):
    with open(filename, 'r') as f:
        data = yaml.load(f, Loader=yaml.FullLoader)
    # [Added] Substitute environment variables right after loading.
    return _expand_env(data)

def save_yaml(filename, data):
    with open(filename, 'w') as f:
        yaml.dump(data, f, Dumper=yaml.Dumper)

def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace


# ------------------------------------------------------------
# [Added] Helper to safely read "optional" keys that may be missing from the config.
#  Wrapped with getattr + a default for compatibility with older configs.
# ------------------------------------------------------------
def cfg_get(namespace, name, default=None):
    return getattr(namespace, name, default)


# ------------------------------------------------------------
# [Added] Fix seeds for reproducibility.
#  Running with the same seed makes data shuffling/augmentation/initialisation identical,
#  so experiment results (AUC etc.) can be reproduced.
#  * torch is imported only inside this function (for pure-logic testing).
# ------------------------------------------------------------
def set_seed(seed, deterministic=False):
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        # Fully deterministic (may be slightly slower). Set True only for debugging/exact reproduction.
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        # With fixed input sizes, benchmark=True is faster.
        torch.backends.cudnn.benchmark = True

def seed_worker(worker_id):
    # [Added] Also fix the seed of each DataLoader worker process.
    #  (so numpy/random-based augmentation is reproducible per worker)
    import torch
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def _peek_first_image_path(meta_path, input_label):
    # [Added] Peek at one image-path column value from the first CSV row (reads only header + first row).
    #  Used in path validation (fail-fast) to check that an image actually opens.
    if not input_label:
        return None
    try:
        with open(meta_path, newline='') as f:
            reader = csv.DictReader(f)
            if input_label not in (reader.fieldnames or []):
                return None
            for row in reader:
                val = row.get(input_label)
                if val:
                    return val
    except Exception:
        return None
    return None


# ------------------------------------------------------------
# [Added] Pre-flight path validation (fail-fast).
#  Check that the required paths actually exist before training starts, so that
#  a wrong mount/path exits immediately with a clear error before grabbing GPUs.
#  (Previously a FileNotFoundError only appeared late, while reading the first batch.)
# ------------------------------------------------------------
def validate_paths(cfg):
    problems = []

    # [Changed] data_root is "optional" (kept backward compatible).
    #  - If set, check that the folder exists and join it with the CSV's relative paths to find images (recommended).
    #  - If not set (None), use the CSV paths as absolute paths as-is.
    #  Either way, "whether the first CSV image actually opens" is checked below, so
    #  path problems are caught before training regardless of whether data_root is set.
    data_root = cfg_get(cfg.data, 'data_root', None)
    if data_root is not None and not os.path.isdir(data_root):
        problems.append(f"data.data_root path does not exist: {data_root}")

    meta_path = cfg.data.meta_path
    if not os.path.isfile(meta_path):
        problems.append(f"data.meta_path (CSV) file does not exist: {meta_path}")
    else:
        # [Added] Check that the first CSV image path opens, regardless of whether data_root is set
        input_label = cfg_get(cfg.data, 'input_label', None)
        if not input_label:
            problems.append("data.input_label is not set. (the CSV's image-path column name)")
        else:
            first = _peek_first_image_path(meta_path, input_label)
            if first is None:
                problems.append(f"CSV '{meta_path}': could not read column '{input_label}'.")
            else:
                resolved = os.path.join(data_root, first) if data_root else first
                if not os.path.exists(resolved):
                    hint = ("(data_root not set -> using CSV paths as absolute paths)"
                            if not data_root else "(data_root + CSV relative path)")
                    problems.append(f"First CSV image path not found: {resolved} {hint}")

    pretrained_root = cfg_get(cfg, 'pretrained_root', None)
    if pretrained_root is not None and not os.path.isdir(pretrained_root):
        problems.append(f"pretrained_root path does not exist: {pretrained_root}")

    # save_root will be created if missing, so only check that the parent folder is writable
    save_root = cfg_get(cfg, 'save_root', None)
    if save_root is not None:
        probe = save_root
        while probe and not os.path.exists(probe):
            probe = os.path.dirname(probe)
        if probe and not os.access(probe, os.W_OK):
            problems.append(f"No write permission for save_root: {save_root}")

    if problems:
        msg = "\n".join(f"  - {p}" for p in problems)
        raise FileNotFoundError(
            "Check the configured paths (pre-training validation failed):\n" + msg
        )


def validate_select_metric(cfg):
    # [Added] select_metric validation (fail-fast).
    #  If select_metric has the form "<class index>_<metric>" (e.g. "2_auc"), check that the class index
    #  is within the range of actual metric keys. With num_classes=1 (binary, BCE) only 0_* metric keys
    #  are produced, so a value like "2_auc" raises a KeyError at model selection. This blocks it up front.
    sm = cfg_get(cfg, 'select_metric', None)
    num_classes = cfg_get(cfg, 'num_classes', None)
    if sm is None or num_classes is None:
        return
    m = re.match(r'^(\d+)_', str(sm))
    if not m:
        return
    idx = int(m.group(1))
    n_metric_classes = 1 if num_classes == 1 else num_classes
    if idx >= n_metric_classes:
        raise ValueError(
            f"select_metric='{sm}' has class index {idx}, which is out of range for num_classes={num_classes}. "
            f"Available indices: 0..{n_metric_classes - 1}. "
            f"(e.g. for binary classification with num_classes=1, change select_metric to '0_auc' or similar.)"
        )


def update_prefix(dict, split):
    return {f'{split}/{k}': v for k, v in dict.items()}

def remove_prefix(dict):
    ret = {}
    for k, v in dict.items():
        new_k = k.split('/', 1)[-1] if '/' in k else k
        ret[new_k] = v
    return ret

def save_dictionary_as_json(dic, path):
    _dic = {k: float(v) if isinstance(v, (np.floating, np.integer)) else v for k, v in dic.items()}
    with open(path, 'w') as f:
        json.dump(_dic, f)

def confidence_interval(metric, n, alpha=0.05):
    z = norm.ppf(1 - alpha / 2)
    se = np.sqrt((metric * (1 - metric)) / n)
    ci_lower = metric - z * se
    ci_upper = metric + z * se
    return max(0, ci_lower), min(1, ci_upper)

def compute_basic_metrics(y_scores, y_true, thresholds=None):
    """
    Compute per-class one-vs-rest AUC / sensitivity / specificity.

    [Changed] Added the thresholds argument (prevents data leakage)
      - thresholds=None  : find each class's optimal threshold directly on "the data passed in"
                           using Youden's J. (used for valid evaluation)
      - thresholds=dict  : use predetermined thresholds of the form {class index: threshold}
                           as-is. (For test evaluation, pass the values found on valid to avoid
                           the optimistic bias of choosing thresholds on test data.)

    Returns: (metrics, used_thresholds, roc_curves)
      - metrics        : metrics dict (includes per-class confusion-matrix cell counts tn/fp/fn/tp)
      - used_thresholds: {class index: threshold used} - to pass on to the next stage (test)
      - roc_curves     : {class index: (fpr, tpr, auc)} - for ROC curve plotting only
                         (raw arrays, so not put in JSON; used only for the report's ROC figure)
    """
    N = y_true.shape[0]
    res = {}
    used_thresholds = {}
    roc_curves = {}

    for i in range(y_scores.shape[1]):
        fpr, tpr, roc_thresholds = roc_curve(y_true[:, i], y_scores[:, i])
        _auc = auc(fpr, tpr)
        _auc_lower, _auc_upper = confidence_interval(_auc, N)
        res[f'{i}_auc'] = _auc
        res[f'{i}_auc_lower'] = _auc_lower
        res[f'{i}_auc_upper'] = _auc_upper
        roc_curves[i] = (fpr, tpr, _auc)

        if thresholds is not None and i in thresholds:
            # [Changed] Use the predetermined threshold (the value found on valid)
            best_thr = thresholds[i]
        else:
            # If no threshold is given, search for the optimum on this data with Youden's J
            J = tpr - fpr
            best_thr = roc_thresholds[np.argmax(J)]
        used_thresholds[i] = float(best_thr)

        tn, fp, fn, tp = confusion_matrix(
            y_true[:, i], (y_scores[:, i] >= best_thr).astype(np.int64),
            labels=[0, 1],
        ).ravel()
        _sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        _sensitivity_lower, _sensitivity_upper = confidence_interval(_sensitivity, N)
        _specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        _specificity_lower, _specificity_upper = confidence_interval(_specificity, N)

        res[f'{i}_best_thr'] = best_thr
        res[f'{i}_sensitivity'] = _sensitivity
        res[f'{i}_sensitivity_lower'] = _sensitivity_lower
        res[f'{i}_sensitivity_upper'] = _sensitivity_upper
        res[f'{i}_specificity'] = _specificity
        res[f'{i}_specificity_lower'] = _specificity_lower
        res[f'{i}_specificity_upper'] = _specificity_upper
        # [Added] Raw confusion-matrix cell counts (one-vs-rest). Sensitivity/specificity are
        #  already computed from these, but they are also kept as-is for when the report should
        #  show "how many were actually correct" as raw numbers.
        res[f'{i}_tn'] = int(tn)
        res[f'{i}_fp'] = int(fp)
        res[f'{i}_fn'] = int(fn)
        res[f'{i}_tp'] = int(tp)

    # ------------------------------------------------------------
    # [Added] Two summary metrics for true multi-class (one-hot y_true) with more than 2 classes.
    #  - weighted_auc: per-class one-vs-rest AUC averaged with weights equal to the number of actual
    #    positives (data) in each class. Used as the default select_metric when you want to summarise
    #    "overall performance" in a single number rather than looking at one class.
    #  - acc: fraction of argmax predictions that match the ground truth. For typical multi-class
    #    problems trained with CrossEntropy (e.g. CIFAR-10) this is more familiar than per-class AUC.
    # ------------------------------------------------------------
    if y_scores.shape[1] > 1:
        supports = y_true.sum(axis=0).astype(np.float64)
        aucs = np.array([res[f'{i}_auc'] for i in range(y_scores.shape[1])])
        res['weighted_auc'] = (
            float(np.average(aucs, weights=supports)) if supports.sum() > 0 else float(aucs.mean())
        )
        res['acc'] = float((y_scores.argmax(axis=1) == y_true.argmax(axis=1)).mean())

    res['n_samples'] = int(N)

    return res, used_thresholds, roc_curves
