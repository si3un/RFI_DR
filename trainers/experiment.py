"""
[Added] Provenance-recording tools for experiment reproducibility.

Records the following so that each training run (save_root/desc/ folder) is a "complete experiment record".
 - Code state: git commit hash + uncommitted diff (exactly which code was used)
 - Full terminal output: train.log (everything printed to the console via print/tqdm etc.)
 - report.html generated automatically at the end of training: data/model/hyperparameters/start-end time (KST)/
   learning curves/final test metrics, collected in one file that is easy to view in a browser.
"""
import base64
import html
import re
import subprocess
import sys
from datetime import datetime, timezone, timedelta
from io import BytesIO
from pathlib import Path

import yaml
from PIL import Image

import matplotlib
matplotlib.use('Agg')  # save to file only, safe on servers without a display
import matplotlib.pyplot as plt

KST = timezone(timedelta(hours=9))


def now_kst():
    return datetime.now(KST)


def format_kst(dt):
    return dt.astimezone(KST).strftime('%Y-%m-%d %H:%M:%S KST')


class Tee:
    """Write everything sent to stdout/stderr to both the console and a log file."""
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            s.write(data)

    def flush(self):
        for s in self.streams:
            s.flush()

    def isatty(self):
        return False


def start_terminal_log(log_path, resume=False):
    """
    Redirect stdout/stderr to also write to log_path, and return a restore function.
    Note: always call the restore function later. When the trainer is built several times in the
    same process (e.g. a hyperparameter sweep in a notebook), not restoring keeps nesting the
    Tee and the next experiment's output leaks into the log file of an experiment that already finished.

    [Changed] Append to the existing log only when resume=True (actually resuming via --resume).
    Otherwise (a new run) the log is written fresh even if the same desc folder is reused, so it
    does not mix with a previous run's log (same kind of bug as metrics.csv rows piling up).
    """
    log_file = open(log_path, 'a' if resume else 'w')
    orig_stdout, orig_stderr = sys.stdout, sys.stderr
    sys.stdout = Tee(orig_stdout, log_file)
    sys.stderr = Tee(orig_stderr, log_file)

    def restore():
        sys.stdout = orig_stdout
        sys.stderr = orig_stderr
        log_file.close()

    return restore


def capture_git_info(save_path, repo_root=None):
    """
    Save the current code state (commit hash + uncommitted diff) to save_path.
    This makes it possible to reproduce "exactly which code this experiment ran" later with
    git checkout <commit> + git apply git_diff.patch.
    """
    repo_root = repo_root or Path(__file__).resolve().parent.parent
    save_path = Path(save_path)
    info = {'commit': None, 'dirty': False, 'error': None}
    try:
        commit = subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'], cwd=repo_root, text=True, stderr=subprocess.DEVNULL
        ).strip()
        status = subprocess.check_output(
            ['git', 'status', '--porcelain'], cwd=repo_root, text=True, stderr=subprocess.DEVNULL
        )
        diff = subprocess.check_output(
            ['git', 'diff', 'HEAD'], cwd=repo_root, text=True, stderr=subprocess.DEVNULL
        )
        info['commit'] = commit
        info['dirty'] = bool(status.strip())

        (save_path / 'git_commit.txt').write_text(
            f"commit: {commit}\ndirty: {info['dirty']}\n\n"
            f"--- git status --porcelain ---\n{status}"
        )
        if diff.strip():
            (save_path / 'git_diff.patch').write_text(diff)
    except Exception as e:
        info['error'] = str(e)
    return info


def capture_code_snapshot(save_path, repo_root=None):
    """
    [Added] Reproducing code from git commit/diff alone requires checking out that commit and
    applying the patch, which is cumbersome. Instead, the source code (.py) actually used by this
    experiment is copied as-is under save_path/code_snapshot/ - a complete copy that can be opened
    directly, without git and without switching commits.
    """
    repo_root = Path(repo_root or Path(__file__).resolve().parent.parent)
    save_path = Path(save_path)
    dst_root = save_path / 'code_snapshot'

    for sub in ('trainers', 'models', 'data', 'interpreters'):
        src_dir = repo_root / sub
        if not src_dir.is_dir():
            continue
        for py_file in src_dir.rglob('*.py'):
            if '__pycache__' in py_file.parts:
                continue
            rel = py_file.relative_to(repo_root)
            dst_file = dst_root / rel
            dst_file.parent.mkdir(parents=True, exist_ok=True)
            dst_file.write_bytes(py_file.read_bytes())


def capture_pip_freeze(save_path):
    """
    [Added] Record the library versions actually installed at training time
    (`pip freeze`). Questions like "which torch/transformers version did this experiment use?"
    cannot be answered from the code diff alone, so this is saved separately for every experiment.
    """
    save_path = Path(save_path)
    try:
        out = subprocess.check_output(
            [sys.executable, '-m', 'pip', 'freeze'], text=True,
            stderr=subprocess.DEVNULL, timeout=60,
        )
        (save_path / 'pip_freeze.txt').write_text(out)
    except Exception as e:
        (save_path / 'pip_freeze.txt').write_text(f'(pip freeze failed: {e})')


def save_predictions_csv(save_path, test_dataset, y_scores, save_name='test_predictions.csv'):
    """
    [Added] Save the final test inference results (from the model rebuilt from the best checkpoint) as CSV:
    test_dataset.df (the dataframe BaseDataset read from the CSV) as-is, with per-class raw output
    columns (output_0, output_1, ...) appended.

    Note: the row order of y_scores must match the row order of test_dataset.df exactly - the test
    DataLoader is built with shuffle=False/drop_last=False (base_trainer._init_dataloaders), so
    this always holds when one full epoch is run in order.

    Silently skipped for trainers that do not use BaseDataset (i.e. have no .df).
    """
    if not hasattr(test_dataset, 'df'):
        return
    df = test_dataset.df.copy()
    if len(df) != len(y_scores):
        return
    for c in range(y_scores.shape[1]):
        df[f'output_{c}'] = y_scores[:, c]
    df.to_csv(Path(save_path) / save_name, index=False)


def _dataset_summary(datasets):
    """
    Extract per-split counts and per-class counts from ClassificationTrainer's self.datasets
    (dict[str, BaseDataset]). Silently skipped for trainers that use a Dataset other than
    BaseDataset (the rest of the report is still generated normally).
    """
    summary = {}
    for split, ds in datasets.items():
        entry = {'n': len(ds), 'counts': None}
        try:
            counts = ds.df[ds.target_label].value_counts().sort_index()
            entry['counts'] = {k: int(v) for k, v in counts.items()}
        except Exception:
            pass
        summary[split] = entry
    return summary


def _sample_images_by_label(datasets, split='train', max_px=320, n_per_label=3):
    """
    [Changed] Pick n_per_label (default 3) representative images per label (target_label) and return them
    as base64 data URIs. Text alone (class-count numbers) does not convey "which images/labels this
    experiment actually trained on", so actual samples are shown directly in the report.
    (Previously there was only 1 image per label, which was not enough to judge what images of that
    label generally look like, so it was increased to 3.)

    Note: images are fully embedded (base64) inside report.html - the report still opens even if the
    original data folder is later moved or deleted (e.g. examples/data/ is a regenerated folder).
    Silently skipped for trainers that use a Dataset other than BaseDataset.

    Returns: {label: [b64_1, b64_2, ...]} (up to n_per_label per label; fewer if less data is available)
    """
    ds = datasets.get(split)
    if ds is None or not hasattr(ds, 'df'):
        return {}
    samples = {}
    try:
        for label, group in ds.df.groupby(ds.target_label):
            b64_list = []
            for _, row in group.head(n_per_label).iterrows():
                img_path = ds._resolve_path(row[ds.input_label])
                with Image.open(img_path) as im:
                    im = im.convert('RGB')
                    im.thumbnail((max_px, max_px))
                    buf = BytesIO()
                    im.save(buf, format='JPEG', quality=92)
                b64_list.append(base64.b64encode(buf.getvalue()).decode('ascii'))
            samples[label] = b64_list
    except Exception:
        return {}
    return samples


def _e(x):
    """Short helper for html.escape."""
    return html.escape(str(x))


_CLASS_METRIC_RE = re.compile(
    r'^(\d+)_(auc|auc_lower|auc_upper|best_thr|sensitivity|sensitivity_lower|'
    r'sensitivity_upper|specificity|specificity_lower|specificity_upper|tn|fp|fn|tp)$'
)


def _pivot_test_metrics(test_metrics):
    """
    [Added] Split the flat dict produced by compute_basic_metrics()
    ({'0_auc':.., '0_auc_lower':.., ..., '1_sensitivity':.., 'acc':.., 'loss':..}) into two parts
    so the report can show "classes as rows, metrics as columns".

    Returns: (class_rows, overall)
      - class_rows: {class_idx: {'AUC': '0.9123 (0.85–0.97)', 'Best Threshold': '0.51',
                     'Sensitivity': .., 'Specificity': .., 'N (actual positives)': .., 'TN': .., 'FP': ..,
                     'FN': .., 'TP': ..}}  (confidence intervals shown in parentheses next to the value)
      - overall   : {'loss': 0.1234, 'weighted_auc': 0.91, 'acc': 0.88, 'n_samples': 250, ...}
                    i.e. the remaining metrics with no class index (summaries over all classes)
    """
    by_class = {}
    overall = {}
    for k, v in test_metrics.items():
        m = _CLASS_METRIC_RE.match(k)
        if not m:
            overall[k] = v
            continue
        idx, name = int(m.group(1)), m.group(2)
        by_class.setdefault(idx, {})[name] = v

    def _fmt_with_ci(d, base):
        if base not in d:
            return '-'
        val = d[base]
        lo, hi = d.get(f'{base}_lower'), d.get(f'{base}_upper')
        if lo is not None and hi is not None:
            return f"{val:.4f} ({lo:.4f}–{hi:.4f})"
        return f"{val:.4f}"

    class_rows = {}
    for idx, d in by_class.items():
        # [Added] Raw confusion-matrix (one-vs-rest) cell counts + number of actual positives (N) for this class.
        #  Sensitivity/specificity are already ratios computed from these, but the table also shows
        #  the raw counts so "how many exactly" can be checked.
        tn, fp, fn, tp = d.get('tn'), d.get('fp'), d.get('fn'), d.get('tp')
        n_pos = tp + fn if tp is not None and fn is not None else None
        class_rows[idx] = {
            'AUC': _fmt_with_ci(d, 'auc'),
            'Best Threshold': f"{d['best_thr']:.4f}" if 'best_thr' in d else '-',
            'Sensitivity': _fmt_with_ci(d, 'sensitivity'),
            'Specificity': _fmt_with_ci(d, 'specificity'),
            'N (actual positives)': str(n_pos) if n_pos is not None else '-',
            'TN': str(tn) if tn is not None else '-',
            'FP': str(fp) if fp is not None else '-',
            'FN': str(fn) if fn is not None else '-',
            'TP': str(tp) if tp is not None else '-',
        }
    return class_rows, overall


def _plot_roc_curves(save_path, roc_curves, save_name='roc_curve.png'):
    """
    [Added] Plot per-class (one-vs-rest) ROC curves for the final test metrics (from ckpt_best) and
    save them to save_path/roc_curve.png. As with training_curves.png, the image is fully embedded in
    report.html as base64, so report.html still opens even if this PNG file is later moved or
    deleted.
    """
    if not roc_curves:
        return None
    fig, ax = plt.subplots(figsize=(5, 5))
    for idx in sorted(roc_curves):
        fpr, tpr, auc_val = roc_curves[idx]
        ax.plot(fpr, tpr, label=f'class {idx} (AUC={auc_val:.3f})')
    ax.plot([0, 1], [0, 1], linestyle='--', color='gray', linewidth=1)
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC curve (test set, ckpt_best)')
    ax.legend(fontsize=8, loc='lower right')
    fig.tight_layout()
    path = Path(save_path) / save_name
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


_REPORT_CSS = """
body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif;
       max-width: 960px; margin: 2rem auto; padding: 0 1.5rem; line-height: 1.55; color: #1a1a1a; }
h1 { border-bottom: 3px solid #2a5db0; padding-bottom: 0.4rem; }
h2 { border-bottom: 1px solid #ddd; padding-bottom: 0.3rem; margin-top: 2.2rem; color: #2a5db0; }
code { background: #f2f2f2; padding: 0.1rem 0.35rem; border-radius: 3px; font-size: 0.92em; }
pre { background: #f7f7f7; border: 1px solid #e0e0e0; border-radius: 6px; padding: 1rem; overflow-x: auto; }
pre code { background: none; padding: 0; }
table { border-collapse: collapse; margin: 0.8rem 0; }
th, td { border: 1px solid #ddd; padding: 0.35rem 0.9rem; text-align: left; }
th { background: #f2f2f2; }
ul { padding-left: 1.3rem; }
img.curves { max-width: 100%; border: 1px solid #ddd; border-radius: 6px; margin-top: 0.5rem; }
.warn { color: #b04a2a; font-weight: 600; }
.ok { color: #2a7a3a; }
.filelist code { white-space: nowrap; }
.sample-table img { width: 200px; height: 200px; object-fit: contain; background: #f2f2f2;
               border: 1px solid #ddd; border-radius: 6px; display: block; }
details { margin: 0.6rem 0; border: 1px solid #e0e0e0; border-radius: 6px; padding: 0.5rem 0.8rem; }
details summary { cursor: pointer; font-weight: 600; color: #2a5db0; }
details table { margin-top: 0.6rem; max-height: 400px; display: block; overflow-y: auto; }
"""


def generate_report(
    save_path, *, desc, cfg_dict, model, datasets,
    train_start, train_end, git_info,
    epochs_configured, epochs_run,
    select_metric_key, best_select_metric,
    test_metrics=None,
    test_roc_curves=None,
    test_model_ckpt_path=None,
    initial_requires_grad=None,
    check_history=None,
    env_info=None,
):
    save_path = Path(save_path)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    body = [f"<h1>Experiment report: {_e(desc)}</h1>"]

    body += [
        "<h2>Run time</h2>",
        "<ul>",
        f"<li>Start: {_e(format_kst(train_start))}</li>",
        f"<li>End: {_e(format_kst(train_end))}</li>",
        f"<li>Duration: {_e(train_end - train_start)}</li>",
        "</ul>",
    ]

    body += ["<h2>Code state</h2>", "<ul>"]
    if git_info.get('commit'):
        if git_info['dirty']:
            dirty_note = ' <span class="warn">uncommitted changes present (see git_diff.patch)</span>'
        else:
            dirty_note = ' <span class="ok">(clean)</span>'
        body.append(f"<li>git commit: <code>{_e(git_info['commit'])}</code>{dirty_note}</li>")
    else:
        err = git_info.get('error')
        body.append("<li>Could not retrieve git info" + (f" ({_e(err)})" if err else "") + "</li>")
    body.append("</ul>")

    body.append("<h2>Data</h2>")
    try:
        ds_summary = _dataset_summary(datasets)
        # [Changed] Instead of a string list like "0=8, 1=10, 2=9", show an actual split x label table
        #  (to compare counts and label distribution at a glance).
        all_labels = sorted(
            {k for s in ds_summary.values() for k in (s.get('counts') or {})}, key=str
        )
        if all_labels:
            body += [
                "<table>",
                "<tr><th>split</th>" + "".join(f"<th>label {_e(l)}</th>" for l in all_labels)
                + "<th>Total</th></tr>",
            ]
            for split, s in ds_summary.items():
                counts = s.get('counts') or {}
                cells = "".join(f"<td>{counts.get(l, 0)}</td>" for l in all_labels)
                body.append(f"<tr><td><b>{_e(split)}</b></td>{cells}<td><b>{s['n']}</b></td></tr>")
            body.append("</table>")
        else:
            for split, s in ds_summary.items():
                body.append(f"<p><b>{_e(split)}</b>: {s['n']} images</p>")
    except Exception:
        body.append("<p>(Could not retrieve dataset summary)</p>")

    sample_images = _sample_images_by_label(datasets)
    if sample_images:
        n_cols = max(len(v) for v in sample_images.values())
        body.append(f"<h3>Sample images (labels as rows, up to {n_cols} per label, from train)</h3>")
        body.append('<table class="sample-table"><tr><th>label</th>'
                     + "".join(f"<th>Sample {i+1}</th>" for i in range(n_cols)) + "</tr>")
        for label in sorted(sample_images, key=str):
            b64_list = sample_images[label]
            cells = "".join(
                f'<td><img src="data:image/jpeg;base64,{b64}" alt="label {_e(label)} sample {i+1}"></td>'
                for i, b64 in enumerate(b64_list)
            )
            # Labels may have different sample counts (classes with few images); pad missing cells with empty cells
            cells += "<td></td>" * (n_cols - len(b64_list))
            body.append(f"<tr><td><b>{_e(label)}</b></td>{cells}</tr>")
        body.append("</table>")

    frozen_params = total_params - trainable_params
    body += [
        "<h2>Model</h2>",
        "<ul>",
        f"<li>backbone (<code>enc_name</code>): <code>{_e(cfg_dict.get('enc_name'))}</code></li>",
        f"<li>Fine-tuning method (<code>ft.method</code>): <code>{_e(cfg_dict.get('ft', {}).get('method'))}</code></li>",
        "</ul>",
        "<table>",
        "<tr><th>Group</th><th>Parameters</th><th>Ratio</th></tr>",
        f"<tr><td>Total</td><td>{total_params:,}</td><td>100.00%</td></tr>",
        f"<tr><td>Trainable</td><td>{trainable_params:,}</td>"
        f"<td>{trainable_params / total_params * 100:.2f}%</td></tr>",
        f"<tr><td>Not trained (frozen)</td><td>{frozen_params:,}</td>"
        f"<td>{frozen_params / total_params * 100:.2f}%</td></tr>",
        "</table>",
    ]

    # [Added] model_checker results: (1) the full requires_grad state at the start of training (whether the
    #  freeze settings were applied as intended), (2) per-check_freq results of "which parameters actually changed".
    #  There may be many parameters, so they are collapsed in <details>; expand to see the full table.
    if initial_requires_grad:
        n_trainable = sum(1 for _, g in initial_requires_grad if g)
        body.append("<h3>Initial freeze state (model_checker)</h3>")
        body.append(
            f"<p>Of {len(initial_requires_grad)} parameters, {n_trainable} are set as trainable "
            f"(<code>requires_grad=True</code>).</p>"
        )
        body.append("<details><summary>Expand full per-parameter list</summary>")
        body.append("<table><tr><th>Parameter</th><th>requires_grad</th></tr>")
        for name, grad in initial_requires_grad:
            grad_cell = '<span class="ok">True</span>' if grad else 'False'
            body.append(f"<tr><td><code>{_e(name)}</code></td><td>{grad_cell}</td></tr>")
        body.append("</table></details>")

    if check_history:
        body.append("<h3>Freeze verification during training (every check_freq, vs. initial weights)</h3>")
        body.append(
            "<p>Every <code>check_freq</code> epochs, compares against the initial weights to verify that frozen "
            "parameters really did not change and that trainable parameters are actually "
            "being updated (readme.md section 9).</p>"
        )
        for entry in check_history:
            n_updated = sum(1 for _, status, _ in entry['results'] if status == 'updated')
            body.append(
                f"<details><summary>epoch {entry['epoch']} - "
                f"{n_updated}/{len(entry['results'])} parameters changed from initial values (expand)</summary>"
            )
            body.append("<table><tr><th>Parameter</th><th>Status</th><th>Sum of diff from initial</th></tr>")
            for name, status, diff in entry['results']:
                if status == 'updated':
                    status_cell = '<span class="warn">🔥 updated</span>'
                else:
                    status_cell = '<span class="ok">🥶 freezed</span>'
                body.append(f"<tr><td><code>{_e(name)}</code></td><td>{status_cell}</td><td>{diff:.3g}</td></tr>")
            body.append("</table></details>")

    body.append("<h2>Training progress</h2>")
    early_note = " (stopped early by early stopping)" if epochs_run < epochs_configured else ""
    progress_rows = [
        ("Configured epochs", str(epochs_configured)),
        ("Last epoch reached", f"{epochs_run}{early_note}"),
    ]
    if best_select_metric is not None:
        progress_rows.append((f"best {select_metric_key} (on valid)", f"{best_select_metric:.4f}"))

    img_size = (cfg_dict.get('data') or {}).get('img_size')
    if img_size is not None:
        progress_rows.append(("Image size (img_size)", f"{img_size} x {img_size}"))

    n_gpus = (env_info or {}).get('n_gpus') or 0
    batch_size = (cfg_dict.get('training') or {}).get('batch_size')
    if batch_size is not None:
        if n_gpus > 1:
            progress_rows.append((
                "Batch size (per GPU x number of GPUs = total)",
                f"{batch_size} x {n_gpus} = {batch_size * n_gpus}",
            ))
        else:
            progress_rows.append(("Batch size", str(batch_size)))

    if env_info:
        if n_gpus > 0:
            gpu_note = f"{n_gpus} GPU(s)" + (f" ({env_info['gpu_name']})" if env_info.get('gpu_name') else "")
        else:
            gpu_note = "No GPU (CPU)"
        progress_rows.append(("GPU setup", gpu_note))

        avg_sec = env_info.get('avg_epoch_seconds')
        if avg_sec is not None:
            progress_rows.append(("Mean time per epoch", f"{avg_sec:.1f} s"))
        progress_rows.append(("Total training time (start to end)", str(train_end - train_start)))

        # [Changed] Shown as time "per epoch" rather than per batch. env_info's avg_data/fwd/bwd_seconds
        #  are already (the per-epoch dataloader/forward/backward totals) averaged over epochs
        #  (changed to AverageMeter.sum in classification_trainer.py._train_step), so here they are
        #  only shown in seconds as-is.
        avg_data, avg_fwd, avg_bwd = (
            env_info.get('avg_data_seconds'), env_info.get('avg_fwd_seconds'), env_info.get('avg_bwd_seconds')
        )
        if avg_data is not None and avg_fwd is not None and avg_bwd is not None:
            total = avg_data + avg_fwd + avg_bwd
            progress_rows.append((
                "Dataloader per epoch (IO+augmentation)",
                f"{avg_data:.2f} s ({avg_data / total * 100:.0f}%)",
            ))
            progress_rows.append((
                "Forward per epoch", f"{avg_fwd:.2f} s ({avg_fwd / total * 100:.0f}%)",
            ))
            progress_rows.append((
                "Backward per epoch", f"{avg_bwd:.2f} s ({avg_bwd / total * 100:.0f}%)",
            ))

        peak_mem = env_info.get('peak_gpu_memory_bytes')
        if peak_mem is not None:
            mem_label = "Peak memory per GPU" if n_gpus > 1 else "Peak GPU memory"
            progress_rows.append((mem_label, f"{peak_mem / 1024**3:.2f} GB"))

    body.append("<table><tr><th>Item</th><th>Value</th></tr>")
    for k, v in progress_rows:
        body.append(f"<tr><td>{_e(k)}</td><td>{_e(v)}</td></tr>")
    body.append("</table>")

    curves_path = save_path / 'training_curves.png'
    if curves_path.exists():
        # [Changed] Embed fully as base64 instead of a relative path (src="training_curves.png").
        #  With a relative path, the image breaks when only report.html is copied/downloaded on its own.
        curves_b64 = base64.b64encode(curves_path.read_bytes()).decode('ascii')
        body += [
            "<h2>Learning curves</h2>",
            f'<img class="curves" src="data:image/png;base64,{curves_b64}" alt="training curves">',
        ]

    if test_metrics:
        body.append("<h2>Final test metrics (best checkpoint)</h2>")

        # [Added] State, as an absolute path, exactly which checkpoint file these metrics came from.
        #  (test_metrics.json is always pure inference from a model rebuilt from ckpt_best.pth alone.)
        if test_model_ckpt_path:
            body.append(
                f"<p>Model used for evaluation (absolute path): <code>{_e(test_model_ckpt_path)}</code></p>"
            )

        class_rows, overall = _pivot_test_metrics(test_metrics)

        # [Changed] Metrics without a class index (summaries over all classes: loss/weighted_auc/acc/n_samples etc.)
        #  are shown first in a separate small table. Per-class AUC/threshold/sensitivity/specificity/
        #  confusion matrix are laid out below in a table with "classes as rows, metrics as columns".
        if overall:
            overall_keys = sorted(overall)
            body.append("<table><tr>" + "".join(f"<th>{_e(k)}</th>" for k in overall_keys) + "</tr>")
            body.append(
                "<tr>" + "".join(
                    f"<td>{overall[k]:.4f}</td>" if isinstance(overall[k], float) else f"<td>{_e(overall[k])}</td>"
                    for k in overall_keys
                ) + "</tr></table>"
            )

        if class_rows:
            # [Added] N (actual positives)/TN/FP/FN/TP: shown next to AUC/threshold/sensitivity/specificity
            #  so the confusion matrix (one-vs-rest) can also be seen as raw counts.
            cols = ['AUC', 'Best Threshold', 'Sensitivity', 'Specificity',
                    'N (actual positives)', 'TN', 'FP', 'FN', 'TP']
            body.append("<table><tr><th>class</th>" + "".join(f"<th>{c}</th>" for c in cols) + "</tr>")
            for idx in sorted(class_rows):
                cells = "".join(f"<td>{_e(class_rows[idx][c])}</td>" for c in cols)
                body.append(f"<tr><td><b>{idx}</b></td>{cells}</tr>")
            body.append("</table>")

        # [Added] Per-class (one-vs-rest) ROC curves. Like training_curves.png, fully embedded as base64
        #  so they display correctly even if only report.html is copied.
        if test_roc_curves:
            roc_path = _plot_roc_curves(save_path, test_roc_curves)
            if roc_path and roc_path.exists():
                roc_b64 = base64.b64encode(roc_path.read_bytes()).decode('ascii')
                body += [
                    "<h3>ROC curve (test)</h3>",
                    f'<img class="curves" src="data:image/png;base64,{roc_b64}" alt="ROC curve">',
                ]

    yaml_str = yaml.dump(cfg_dict, allow_unicode=True, sort_keys=False).rstrip()
    body += ["<h2>Hyperparameters (full config)</h2>", f"<pre><code>{_e(yaml_str)}</code></pre>"]

    body += [
        "<h2>Files in this folder</h2>",
        f"<p>Folder path (absolute): <code>{_e(str(save_path.resolve()))}</code></p>",
        '<ul class="filelist">',
        "<li><code>config.yml</code>: full settings of this experiment (values after environment-variable substitution)</li>",
        "<li><code>ckpt_best.pth</code>: best checkpoint on valid (the final model to actually use)</li>",
        "<li><code>ckpt_last.pth</code>: checkpoint at the end of training (state after re-evaluating best)</li>",
        "<li><code>metrics.csv</code> / <code>training_curves.png</code>: per-epoch learning curves (kept even without wandb)</li>",
        "<li><code>test_metrics.json</code>: final test metrics (raw json) - pure inference from a model "
        "rebuilt from ckpt_best.pth alone</li>",
        "<li><code>test_predictions.csv</code>: per-sample raw outputs of the final inference above "
        "(test dataframe + per-class raw output columns <code>output_0, output_1, ...</code>)</li>",
        "<li><code>roc_curve.png</code>: per-class (one-vs-rest) ROC curves from the same final inference</li>",
        "<li><code>train.log</code>: everything printed to the terminal during training</li>",
        "<li><code>git_commit.txt</code> (+ <code>git_diff.patch</code>): exact code state that produced this experiment</li>",
        "</ul>",
    ]

    html_doc = (
        "<!doctype html>\n<html lang=\"ko\"><head><meta charset=\"utf-8\">"
        f"<title>Experiment report: {_e(desc)}</title><style>{_REPORT_CSS}</style></head><body>\n"
        + "\n".join(body) + "\n</body></html>\n"
    )
    (save_path / 'report.html').write_text(html_doc)
