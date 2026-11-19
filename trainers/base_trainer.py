import time
from abc import ABC, abstractmethod
from pathlib import Path
from tqdm import trange
import wandb

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from trainers.model_checker import ModelChecker
from models import build_encoder
from trainers.metrics_logger import MetricsLogger
from trainers.experiment import (
    start_terminal_log, capture_git_info, capture_code_snapshot, capture_pip_freeze,
    now_kst, generate_report, save_predictions_csv,
)
from trainers.utils import save_yaml, load_yaml
from trainers.utils import save_dictionary_as_json
from trainers.utils import update_prefix
from trainers.utils import dict2namespace
from trainers.utils import cfg_get, set_seed, seed_worker, validate_paths, validate_select_metric
from trainers.dist_utils import (
    setup_distributed, is_main_process, barrier, cleanup_distributed, broadcast_bool, get_world_size,
)


# ------------------------------------------------------------
# Choose the AMP (automatic mixed precision) dtype
#  - bf16: supported on recent GPUs (A100/RTX30 and newer). Stable, no GradScaler needed.
#  - fp16: compatible with older GPUs (V100/T4, etc.). Requires a GradScaler.
#  - "auto": inspect the hardware and pick bf16 if supported, otherwise fp16.
#  This way AMP behaves correctly on any server.
# ------------------------------------------------------------
def resolve_amp_dtype(name):
    if name == 'bf16':
        return torch.bfloat16
    if name == 'fp16':
        return torch.float16
    # auto
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


class BaseTrainer(ABC):
    def __init__(self, args):
        super().__init__()
        cfg = load_yaml(args.config_path)
        self.cfg = dict2namespace(cfg)

        # Reproducibility: fix random seeds
        self.seed = cfg_get(self.cfg, 'seed', 42)
        set_seed(self.seed, deterministic=cfg_get(self.cfg, 'deterministic', False))

        # Distributed/device setup (DDP under torchrun, otherwise a single device)
        self.distributed, self.device, self.local_rank = setup_distributed()

        # Pre-flight validation (fail-fast). Exit immediately on bad paths/config, before grabbing GPUs.
        validate_paths(self.cfg)
        validate_select_metric(self.cfg)

        # How often (in epochs) to print model_checker (freeze-state check). Default 10.
        self.check_freq = int(cfg_get(self.cfg.training, 'check_freq', 10) or 10)

        # Early stopping. If select_metric does not improve for this many consecutive
        #  evaluations (counted in eval_freq intervals), training stops early. 0/unset disables it.
        #  Note: the lr scheduler (cosine etc.) keeps stepping every epoch regardless of this.
        #    (The scheduler is not made to react to this metric like ReduceLROnPlateau.)
        self.early_stop_patience = int(cfg_get(self.cfg.training, 'early_stop_patience', 0) or 0)
        self.epochs_without_improve = 0

        # Training-environment info for report.html: number of GPUs / mean time per epoch /
        #  peak GPU memory. epoch_durations holds the actual wall time (s) of each epoch;
        #  GPU memory is reset with reset_peak_memory_stats() at the start of training and
        #  the true peak for this run is read with max_memory_allocated() at the end.
        self.epoch_durations = []
        # Per-epoch accumulation of the average time spent in dataloader (IO+augmentation)/
        #  forward/backward within a train step (collects train/data_time, train/fwd_time,
        #  train/bwd_time as returned by _train_step).
        self.data_times = []
        self.fwd_times = []
        self.bwd_times = []
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        self._init_loss()
        self._init_model()
        self._init_optimizer()
        self._init_dataset()
        self._init_dataloader()

        self.start_epoch = 1
        self.resume = bool(args.resume)
        self.resume_path = args.resume
        if args.resume:
            self.load_model(args.resume)

        self._init_log(cfg)
        # higher is better
        self.val_best_select_metric = -float('inf')
        # Keep the classification threshold found on valid so it can be reused on test
        #  (prevents optimistic bias/leakage from choosing the threshold on test data)
        self.best_val_thresholds = None

    @abstractmethod
    def _init_loss(self):
        raise NotImplementedError

    def _init_model(self):
        self.model = build_encoder(self.cfg)

        # Move to self.device (works for CPU or any specific GPU)
        self.model = self.model.to(self.device)
        self.criterion = self.criterion.to(self.device)

        # Snapshot initial weights before DDP wrapping (i.e. with the original parameter names).
        #  Used to periodically verify that frozen parameters are not accidentally updated during training.
        self.model_checker = ModelChecker(self.model)
        # Keep the initial requires_grad state and the check_with_init result at every check_freq
        #  in structured form so they can be copied into report.html as-is.
        self.initial_requires_grad = self.model_checker.requires_grad_table()
        self.check_history = []

        if self.distributed:
            # DistributedDataParallel (DDP): one process per GPU (run: torchrun --nproc_per_node=N ...)
            #  find_unused_parameters: default False (fast). The current models (linear/partial_ft/lora)
            #  always produce grads for every trainable parameter on each forward, so False is enough.
            #  (Smoke test confirmed "no unused parameters".) If a model has conditional branches in
            #  forward so that some parameters may be skipped on some steps, enable
            #  ddp_find_unused_parameters: true in the config (slower, but prevents hangs).
            self.model = nn.parallel.DistributedDataParallel(
                self.model,
                device_ids=[self.local_rank],
                find_unused_parameters=cfg_get(self.cfg, 'ddp_find_unused_parameters', False),
            )

    def unwrap_model(self):
        # Unwrap the DDP/DP wrapper and return the underlying model (used for save/eval/load)
        if isinstance(self.model, (nn.DataParallel, nn.parallel.DistributedDataParallel)):
            return self.model.module
        return self.model

    def _init_optimizer(self):
        cfg = self.cfg.training
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = getattr(optim, cfg.optim)(
            params=trainable_params,
            lr=cfg.lr,
            weight_decay=cfg.weight_decay
        )

        # Warmup: if warmup=True, lr starts near 0, increases linearly over warmup_epochs,
        #  then decays with cosine.
        eta_min = cfg.lr * (cfg.lr_decay_rate ** 2)
        warmup = cfg_get(cfg, 'warmup', False)
        warmup_epochs = int(cfg_get(cfg, 'warmup_epochs', 0) or 0)
        if warmup and warmup_epochs > 0:
            warmup_sched = optim.lr_scheduler.LinearLR(
                self.optimizer, start_factor=1e-3, total_iters=warmup_epochs
            )
            cosine_sched = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=max(1, cfg.epochs - warmup_epochs), eta_min=eta_min
            )
            self.scheduler = optim.lr_scheduler.SequentialLR(
                self.optimizer, [warmup_sched, cosine_sched], milestones=[warmup_epochs]
            )
        else:
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=cfg.epochs, eta_min=eta_min
            )

        # Automatic AMP dtype selection + GradScaler only for fp16
        self.enable_amp = cfg.enable_amp
        self.amp_dtype = resolve_amp_dtype(cfg_get(cfg, 'amp_dtype', 'auto'))
        use_scaler = self.enable_amp and (self.amp_dtype == torch.float16)
        self.scaler = torch.amp.GradScaler('cuda', enabled=use_scaler)

    @abstractmethod
    def _init_dataset(self):
        raise NotImplementedError

    def _init_dataloader(self):
        cfg = self.cfg.training
        num_workers = cfg.num_workers
        # Apply persistent_workers/prefetch_factor only when there are workers
        persistent = num_workers > 0
        prefetch = cfg_get(cfg, 'prefetch_factor', 4) if num_workers > 0 else None

        # Generator for reproducible shuffling (fixed seed)
        generator = torch.Generator()
        generator.manual_seed(self.seed)

        self.loaders = {}
        self.train_sampler = None
        for sp in ['train', 'valid', 'test']:
            if sp == 'train' and self.distributed:
                # Under DDP, DistributedSampler splits the data per GPU (process).
                sampler = DistributedSampler(
                    self.datasets[sp], shuffle=True, seed=self.seed, drop_last=True
                )
                shuffle = False
                self.train_sampler = sampler
            else:
                sampler = None
                shuffle = (sp == 'train')

            self.loaders[sp] = DataLoader(
                self.datasets[sp],
                batch_size=cfg.batch_size,
                num_workers=num_workers,
                pin_memory=(self.device.type == 'cuda'),  # pin_memory only on CUDA (avoids a warning on CPU)
                shuffle=shuffle,
                sampler=sampler,
                persistent_workers=persistent, # avoid re-creating workers every epoch (less overhead)
                prefetch_factor=prefetch,      # batches prefetched per worker (hides I/O)
                drop_last=(sp == 'train'),     # drop the last partial batch (stable stats/DDP balance)
                worker_init_fn=seed_worker,    # fixed worker seeds (reproducible augmentation)
                generator=generator,
            )

    def _init_log(self, _cfg):
        # Since the same desc may be run repeatedly (re-running the same config, reproducibility checks, etc.),
        #  append the actual run time (KST) to the folder name so each run gets a new folder.
        #  Note: with --resume we must keep using "that" folder (the one created by the original run,
        #  whose name already carries a timestamp), so no new timestamp is appended.
        self.train_start_time = now_kst()
        if self.resume:
            # Continue in the folder that actually contains the resume checkpoint.
            #  (Its name already has the timestamp from the first run, so it must not be rebuilt
            #  from desc - the naming rule differs from the else branch below.)
            self.save_path = Path(self.resume_path).resolve().parent
        else:
            ts = self.train_start_time.strftime('%Y%m%d_%H%M%S')
            self.save_path = Path(self.cfg.save_root)/f"{self.cfg.desc}_{ts}"

        # Saving/logging/wandb only on rank 0 (the main process).
        #  Under DDP, all processes writing at once would cause file conflicts/duplicates.
        if is_main_process():
            self.save_path.mkdir(parents=True, exist_ok=True)

            if self.cfg.use_wandb:
                wandb.init(
                    entity=self.cfg.wandb_entity,
                    project=self.cfg.wandb_project,
                    name=self.cfg.desc,
                    config=_cfg,  # also record this experiment's hyperparameters in the wandb run
                )

            # Save per-epoch metrics locally as CSV/plot regardless of whether wandb is used.
            #  save_path/metrics.csv and save_path/training_curves.png are updated every epoch, so
            #  the learning curves can be followed from files without wandb (or alongside it).
            #  An existing metrics.csv is appended to only when resuming (--resume).
            self.metrics_logger = MetricsLogger(
                self.save_path, select_metric_key=self.cfg.select_metric, resume=self.resume
            )

            # Record experiment provenance.
            #  - train.log: everything printed to the terminal (print/tqdm) during training.
            #  - git_commit.txt/git_diff.patch: exact code state that produced this experiment (commit + uncommitted changes).
            #  - code_snapshot/: copy of the actual source code (.py) that can be opened without git.
            #  - pip_freeze.txt: full list of library versions installed at that time.
            #  (self.train_start_time was already recorded above, same time as the folder-name timestamp)
            self._log_restore = start_terminal_log(self.save_path / 'train.log', resume=self.resume)
            self._git_info = capture_git_info(self.save_path)
            capture_code_snapshot(self.save_path)
            capture_pip_freeze(self.save_path)
            self._raw_cfg = _cfg  # kept so it can be embedded as-is in report.html

            save_yaml(self.save_path/'config.yml', _cfg)
        # Synchronise all processes after folder creation
        barrier()

    @abstractmethod
    def _train_step(self):
        raise NotImplementedError

    @abstractmethod
    def _eval_step(self):
        raise NotImplementedError

    def fit(self):
        # Wrapped so the stdout/stderr Tee (for train.log) is always restored even if
        #  training raises. Otherwise (especially when one process builds the trainer several times,
        #  e.g. a hyperparameter sweep in a notebook) all later print() output keeps flowing into
        #  the failed experiment's log file. restore() is safe to call twice.
        try:
            self._fit_loop()
        finally:
            log_restore = getattr(self, '_log_restore', None)
            if log_restore is not None:
                log_restore()

    def _fit_loop(self):
        stop_training = False
        for epoch in trange(self.start_epoch, self.cfg.training.epochs + 1,
                            desc='training epochs', disable=not is_main_process()):
            # Expose the last epoch actually reached (e.g. to check "after how many epochs
            #  it stopped" when sweeping several settings). Updated on every loop iteration, so it is
            #  always accurate even if early stopping ends training midway.
            self.last_epoch = epoch
            epoch_start_time = time.time()

            # The DDP sampler needs set_epoch every epoch to change the shuffle seed,
            #  so the data is shuffled in a different order each epoch.
            if self.train_sampler is not None:
                self.train_sampler.set_epoch(epoch)

            metrics = self._train_step()

            if epoch % self.cfg.training.eval_freq == 0:
                # Evaluation/model selection/saving only on rank 0.
                if is_main_process():
                    valid_metrics, valid_thresholds = self._eval_step()
                    metrics.update(valid_metrics)

                    # Test is also computed at the same interval as valid (eval_freq) and stored in
                    #  metrics.csv, so its trajectory during training can be plotted.
                    #  Note: this is a "reference record" that never takes part in model selection (choosing
                    #  the best checkpoint); the final saved/reported test_metrics.json is still computed after
                    #  training below by reloading the best checkpoint. The threshold found on this epoch's
                    #  valid set is reused as-is (choosing a new threshold on this epoch's test data
                    #  would be leakage, so it is not allowed).
                    # [DR] Test is evaluated only once after training by _final_inference with ckpt_best -> per-epoch test removed.
                    # Train eval for curves only every curve_freq. Valid every epoch -> selection/early stop unchanged.
                    curve_freq = cfg_get(self.cfg.training, 'curve_freq', self.cfg.training.eval_freq)
                    if epoch % curve_freq == 0:

                        # Also compute train metrics (in eval mode, reusing the valid threshold) so
                        #  train/valid/test can all be compared in metrics.csv/training_curves.png.
                        #  Note: train/loss (the actual training loss from _train_step above) is kept; only
                        #  classification metrics such as select_metric are taken from this eval-mode pass, without overwriting it.
                        train_metrics_epoch, _ = self._eval_step(split='train', thresholds=valid_thresholds)
                        train_metrics_epoch.pop('train/loss', None)
                        metrics.update(train_metrics_epoch)

                    # Clear error if the select_metric key is missing (config mistake)
                    key = f'valid/{self.cfg.select_metric}'
                    if key not in valid_metrics:
                        raise KeyError(
                            f"select_metric '{self.cfg.select_metric}' -> '{key}' is not among the evaluation metrics. "
                            f"Available keys: {sorted(valid_metrics)}"
                        )
                    select_metric = valid_metrics[key]
                    if select_metric > self.val_best_select_metric:
                        self.val_best_select_metric = select_metric
                        self.epochs_without_improve = 0
                        # Save the valid threshold together with the best model -> reused on test
                        self.best_val_thresholds = valid_thresholds
                        self.save_model(epoch, fn=self.save_path/'ckpt_best.pth',
                                        extra={'val_thresholds': valid_thresholds})
                    else:
                        self.epochs_without_improve += 1

                    # Early stopping. The lr scheduler (cosine etc.) is step()ed every epoch
                    #  below regardless of this decision, so early stopping does not touch
                    #  the lr schedule (not made to behave like ReduceLROnPlateau).
                    if (self.early_stop_patience > 0
                            and self.epochs_without_improve >= self.early_stop_patience):
                        print(f"[epoch {epoch}] Early stopping: select_metric did not improve over "
                              f"{self.epochs_without_improve} evaluations "
                              f"(patience={self.early_stop_patience}).", flush=True)
                        stop_training = True
                barrier()  # other processes wait for rank 0's evaluation/saving

            if is_main_process() and self.cfg.use_wandb:
                wandb.log(metrics, step=epoch)

            # Always append to the CSV and redraw/save the plot, regardless of wandb.
            #  Both files are updated every epoch, so reopening them during training shows
            #  the curves up to that point (effectively live monitoring).
            if is_main_process():
                self.metrics_logger.log(epoch, metrics)
                self.metrics_logger.plot()

            # Print an epoch summary to stdout so progress is visible without wandb
            if is_main_process():
                msg = f"[epoch {epoch}] train/loss={metrics.get('train/loss', float('nan')):.4f}"
                key = f'valid/{self.cfg.select_metric}'
                if key in metrics:
                    msg += (f" | valid/loss={metrics.get('valid/loss', float('nan')):.4f}"
                            f" | {key}={metrics[key]:.4f} | best={self.val_best_select_metric:.4f}")
                print(msg, flush=True)

            # model_checker: check that frozen parameters really do not change, printed only
            #  every check_freq (default 10) epochs rather than every epoch (too much log otherwise).
            if epoch % self.check_freq == 0 and is_main_process():
                print(f"[epoch {epoch}] model parameter check (vs. initial weights):", flush=True)
                check_result = self.model_checker.check_with_init(self.model)
                self.check_history.append({'epoch': epoch, 'results': check_result})

            if epoch % self.cfg.training.save_freq == 0 and is_main_process():
                self.save_model(epoch,
                                fn=None if self.cfg.training.save_last_only else f'ckpt_{epoch}.pth')

            self.scheduler.step()  # all processes step identically (always called, regardless of early stop)

            # Record for "mean time per epoch" in report.html (includes train+eval, measured)
            if is_main_process():
                self.epoch_durations.append(time.time() - epoch_start_time)
                if 'train/data_time' in metrics:
                    self.data_times.append(metrics['train/data_time'])
                    self.fwd_times.append(metrics['train/fwd_time'])
                    self.bwd_times.append(metrics['train/bwd_time'])

            # Broadcast the early-stop decision made on rank 0 to all processes, then exit.
            #  (Under DDP all ranks must follow the same epoch flow, so rank 0 alone must not break)
            stop_training = broadcast_bool(stop_training, self.device)
            if stop_training:
                break

        '''
        Final evaluation on test set
        '''
        barrier()
        if is_main_process():
            # Instead of reusing self.model/self.loaders from the training loop, rebuild
            #  the model "only from what is stored in" ckpt_best.pth and run inference on the test set
            #  once more, cleanly (trainers/classification_trainer.py._final_inference).
            print("Final results: rebuilding the model from ckpt_best.pth only and re-running inference on the test set...")
            test_metrics_clean, test_y_scores, test_roc_curves = self._final_inference()
            save_dictionary_as_json(
                test_metrics_clean,
                self.save_path/'test_metrics.json'
            )
            save_predictions_csv(self.save_path, self.report_datasets['test'], test_y_scores)

            # [DR] Evaluate the external validation set (re-inference with ckpt_best, valid threshold applied).
            #  Only runs when cfg.data.external_meta_path is set. Saves external_predictions.csv
            #  (image_path/label/pred) used by saliency/ensemble.
            _ext_fn = getattr(self, '_external_inference', None)
            ext_metrics, ext_y_scores = _ext_fn() if _ext_fn else (None, None)
            if ext_metrics is not None:
                save_dictionary_as_json(ext_metrics, self.save_path/'external_metrics.json')
                import pandas as pd
                edf = self.external_dataset.df.copy()
                edf['pred'] = ext_y_scores[:, 0]   # binary (num_classes=1) positive-class probability
                keep = [c for c in ['image_path', 'label', 'grade', 'dataset', 'pred'] if c in edf.columns]
                edf[keep].to_csv(self.save_path / 'external_predictions.csv', index=False)
                print("Saved external evaluation: external_metrics.json / external_predictions.csv")
            # Record in the report, as an absolute path, exactly which checkpoint file produced the
            #  final test metrics (test_metrics.json is always pure inference from a model rebuilt from
            #  this file alone - see "Why are the final results re-inferred?" in
            #  trainers/experiment.py).
            test_model_ckpt_path = str((self.save_path / 'ckpt_best.pth').resolve())
            if self.cfg.use_wandb:
                wandb.log(update_prefix(test_metrics_clean, 'test'), step=epoch)
            # Training may have ended earlier than cfg.training.epochs because of early stopping,
            #  so record the last epoch actually reached (the loop's epoch variable, which keeps its
            #  last value even after break) rather than "the configured total number of epochs".
            self.save_model(epoch)

            # Training-environment info: number of GPUs / mean time per epoch / measured peak GPU memory.
            #  On CPU (as in these examples) the GPU-related values naturally become 0/None.
            def _avg(xs):
                return sum(xs) / len(xs) if xs else None

            env_info = {
                'n_gpus': get_world_size() if torch.cuda.is_available() else 0,
                'gpu_name': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                'avg_epoch_seconds': _avg(self.epoch_durations),
                # Time per epoch spent in dataloader (IO+augmentation)/forward/backward within
                #  train steps (mean over epochs of the per-epoch totals for the whole run).
                'avg_data_seconds': _avg(self.data_times),
                'avg_fwd_seconds': _avg(self.fwd_times),
                'avg_bwd_seconds': _avg(self.bwd_times),
                'peak_gpu_memory_bytes': (
                    torch.cuda.max_memory_allocated() if torch.cuda.is_available() else None
                ),
            }

            # Automatic experiment report: data/model/hyperparameters/start-end time (KST)/
            #  learning curves/final test metrics, collected in a single report.html.
            generate_report(
                self.save_path,
                desc=self.cfg.desc,
                cfg_dict=self._raw_cfg,
                model=self.unwrap_model(),
                # fast_linear replaces self.datasets with cached feature Datasets
                #  (which lack ds.df etc., leaving the report's data table/sample images empty), so if
                #  report_datasets holding the original image Datasets exists, use it instead.
                datasets=getattr(self, 'report_datasets', None) or getattr(self, 'datasets', {}),
                train_start=self.train_start_time,
                train_end=now_kst(),
                git_info=self._git_info,
                epochs_configured=self.cfg.training.epochs,
                epochs_run=epoch,
                select_metric_key=self.cfg.select_metric,
                best_select_metric=(
                    self.val_best_select_metric
                    if self.val_best_select_metric != -float('inf') else None
                ),
                test_metrics=test_metrics_clean,
                test_roc_curves=test_roc_curves,
                test_model_ckpt_path=test_model_ckpt_path,
                initial_requires_grad=self.initial_requires_grad,
                check_history=self.check_history,
                env_info=env_info,
            )
            # Restore stdout/stderr. When one process builds the trainer several times
            #  (e.g. a hyperparameter sweep in a notebook), logs keep nesting unless restored.
            self._log_restore()

        # Tear down the distributed process group
        cleanup_distributed()

    def _model_state_to_save(self):
        # Select which model parameters to save.
        #  With save_trainable_only=True, the frozen backbone is excluded and only "trainable"
        #  parameters are saved. linear/lora shrink to a few KB to a few MB, saving a lot of disk.
        #  (The original backbone weights live in pretrained_root, so on restore they are reloaded
        #   in the build step and only this trained part is applied on top -> load_model uses strict=False)
        model = self.unwrap_model()
        full = model.state_dict()
        if cfg_get(self.cfg.training, 'save_trainable_only', True):
            trainable = {n for n, p in model.named_parameters() if p.requires_grad}
            return {k: v for k, v in full.items() if k in trainable}
        return full

    def save_model(self, epoch, fn=None, extra=None):
        ckpt = {
            'model': self._model_state_to_save(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'scaler': self.scaler.state_dict(),
            'epoch': epoch,
            # Save-mode metadata (for restoring). Trainable-only checkpoints must be loaded with strict=False.
            'save_trainable_only': bool(cfg_get(self.cfg.training, 'save_trainable_only', True)),
        }
        if extra:
            ckpt.update(extra)
        torch.save(ckpt,
                   self.save_path/fn if fn is not None else self.save_path/f'ckpt_last.pth'
                   )

    def load_model(self, fn):
        # map_location=self.device, weights_only=False (a trusted checkpoint we created).
        ckpt = torch.load(fn, map_location=self.device, weights_only=False)
        model = self.unwrap_model()
        # strict=False: trainable-only checkpoints have no backbone keys.
        #  (the backbone is already loaded from pretrained weights in build_encoder)
        model.load_state_dict(ckpt['model'], strict=False)
        self.optimizer.load_state_dict(ckpt['optimizer'])
        self.scheduler.load_state_dict(ckpt['scheduler'])
        self.scaler.load_state_dict(ckpt['scaler'])
        self.start_epoch = ckpt['epoch'] + 1
        # Restore the saved valid threshold (reused for test evaluation)
        if ckpt.get('val_thresholds') is not None:
            self.best_val_thresholds = ckpt['val_thresholds']
