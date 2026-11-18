"""
[Added] Save each epoch's metrics to CSV regardless of whether wandb is used, and redraw the
plots on the fly (every epoch) and save them as PNG.

wandb needs internet access/an account to view, but this logger only writes local files (under
save_path: metrics.csv / training_curves.png). Re-reading the CSV or reopening the PNG during
training immediately shows the curves up to that point (the files are updated every
epoch), effectively giving "live monitoring" - even when connected to a server over SSH you can
simply download the files, with no separate service/network needed as with wandb.
"""
import csv
from pathlib import Path

import matplotlib
matplotlib.use('Agg')  # save to file only, safe on servers without a display
import matplotlib.pyplot as plt


def _to_float(v):
    if v in (None, ''):
        return None
    try:
        return float(v)
    except ValueError:
        return None


class MetricsLogger:
    def __init__(self, save_path, select_metric_key=None, resume=False):
        self.save_path = Path(save_path)
        self.csv_path = self.save_path / 'metrics.csv'
        self.plot_path = self.save_path / 'training_curves.png'
        # [Added] Knowing select_metric (e.g. '0_auc') lets the "metric" panel of the plot
        #  point exactly at valid/{select_metric_key}.
        self.select_metric_key = select_metric_key

        self._fieldnames = []
        self._rows = []

        # [Changed] Append to the existing CSV only when resume=True (actually resuming via --resume).
        #  Unconditionally reading and appending to an existing CSV without resuming means that,
        #  when training again from scratch with a previous run's results left in the same desc folder
        #  (e.g. re-running the same config), new rows from epoch 1 are appended after the old ones and epochs pile up/duplicate.
        #  Without resume it is a new run, so always start empty (the file is overwritten on the first log() call).
        if resume and self.csv_path.exists():
            with open(self.csv_path, newline='') as f:
                reader = csv.DictReader(f)
                self._fieldnames = list(reader.fieldnames or [])
                for row in reader:
                    self._rows.append({k: _to_float(v) for k, v in row.items()})

    def log(self, epoch, metrics: dict):
        row = {'epoch': float(epoch), **{k: float(v) for k, v in metrics.items()}}
        self._rows.append(row)

        # [Added] valid exists only every eval_freq, so the set of columns may differ per epoch;
        #  the header is built from all keys seen so far and missing values are left blank.
        fieldnames = ['epoch'] + sorted(k for k in row if k != 'epoch')
        for k in self._fieldnames:
            if k not in fieldnames:
                fieldnames.append(k)
        self._fieldnames = fieldnames

        with open(self.csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self._fieldnames, restval='')
            writer.writeheader()
            writer.writerows(self._rows)
            f.flush()

    def plot(self):
        if not self._rows:
            return

        loss_keys = [k for k in self._fieldnames if k.endswith('/loss')]
        lr_key = 'train/learning_rate'
        metric_key = f'valid/{self.select_metric_key}' if self.select_metric_key else None
        if metric_key not in self._fieldnames:
            metric_key = None

        n_panels = 1 + int(metric_key is not None) + int(lr_key in self._fieldnames)
        fig, axes = plt.subplots(1, n_panels, figsize=(4.5 * n_panels, 3.5))
        if n_panels == 1:
            axes = [axes]
        panel = iter(axes)

        ax = next(panel)
        for k in loss_keys:
            xs, ys = self._series(k)
            if xs:
                ax.plot(xs, ys, marker='o', markersize=3, label=k)
        ax.set_xlabel('epoch'); ax.set_ylabel('loss'); ax.set_title('Loss'); ax.legend(fontsize=8)

        if metric_key is not None:
            ax = next(panel)
            xs, ys = self._series(metric_key)
            ax.plot(xs, ys, marker='o', markersize=3, color='tab:orange', label='valid')

            test_key = f'test/{self.select_metric_key}'
            if ys:
                best_i = max(range(len(ys)), key=lambda i: ys[i])
                best_epoch, best_valid = xs[best_i], ys[best_i]
                ax.scatter([best_epoch], [best_valid], color='red', zorder=3, label=f'best={best_valid:.4f}')

                # [Changed] Draw a reference line parallel to the y-axis (= vertical) at x=best_epoch
                #  so "which epoch" valid was best is visible immediately, and annotate the test
                #  value at that epoch - so valid/test can be compared directly at "the epoch the model
                #  was selected" (to check the generalisation gap).
                ax.axvline(x=best_epoch, color='red', linestyle=':', linewidth=1, alpha=0.5)
                test_at_best = next(
                    (r.get(test_key) for r in self._rows if r.get('epoch') == best_epoch), None
                )
                label = f'epoch={best_epoch:g}'
                if test_at_best is not None:
                    label += f', test={test_at_best:.4f}'
                ax.annotate(
                    label, xy=(best_epoch, best_valid), xytext=(5, 5),
                    textcoords='offset points', fontsize=8, color='tab:blue',
                )

            # [Added] train/test are logged at the same interval as valid, so if present they are
            #  overlaid on the same panel to compare at a glance how closely (or differently) the three move.
            for split, color in [('train', 'tab:green'), ('test', 'tab:blue')]:
                split_key = f'{split}/{self.select_metric_key}'
                if split_key in self._fieldnames:
                    sxs, sys_ = self._series(split_key)
                    if sxs:
                        ax.plot(sxs, sys_, marker='s', markersize=3, color=color,
                                linestyle='--', alpha=0.8, label=split)
            ax.set_xlabel('epoch'); ax.set_ylabel(self.select_metric_key)
            ax.set_title(f'select_metric ({self.select_metric_key})'); ax.legend(fontsize=8)

        if lr_key in self._fieldnames:
            ax = next(panel)
            xs, ys = self._series(lr_key)
            ax.plot(xs, ys, marker='o', markersize=3, color='tab:green')
            ax.set_xlabel('epoch'); ax.set_ylabel('learning rate'); ax.set_title('LR schedule')

        fig.tight_layout()
        fig.savefig(self.plot_path, dpi=120)
        plt.close(fig)

    def _series(self, key):
        xs, ys = [], []
        for r in self._rows:
            v = r.get(key)
            if v is not None:
                xs.append(r['epoch'])
                ys.append(v)
        return xs, ys
