import argparse
import time
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from data.base_dataset import BaseDataset
from trainers.base_trainer import BaseTrainer
from trainers.utils import AverageMeter
from trainers.utils import get_lr, update_prefix
from trainers.utils import compute_basic_metrics
from trainers.utils import cfg_get
from trainers.dist_utils import is_main_process


class _CachedFeatureDataset(Dataset):
    """[Added] For ft.method='fast_linear': a lightweight Dataset that simply holds (backbone feature, target)
    tensors. It shares only the interface of BaseDataset (which opens image files); in practice it just
    indexes tensors that are already in memory."""
    def __init__(self, features, targets):
        self.features = features
        self.targets = targets

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, i):
        return self.features[i], self.targets[i]


class ClassificationTrainer(BaseTrainer):
    def __init__(self, args):
        super().__init__(args)

    def _init_loss(self):
        # [Changed] Support binary classification (num_classes == 1).
        #  Previously the loss was always CrossEntropyLoss, and with num_classes=1
        #  CrossEntropy + one_hot(num_classes=1) broke.
        #  Now num_classes==1 uses a single logit + BCEWithLogitsLoss.
        #  (Regular multi-class still uses CrossEntropyLoss)
        self.num_classes = self.cfg.num_classes
        if self.num_classes == 1:
            self.criterion = nn.BCEWithLogitsLoss()
        else:
            self.criterion = nn.CrossEntropyLoss(
                label_smoothing=self.cfg.training.label_smoothing
            )
        # [DR] Aux grade head (multi-task). use_aux=True -> ckpts_aux family.
        #  main (referable BCE) + aux_weight * grade (CrossEntropy). No aux for external.
        self.use_aux = cfg_get(self.cfg, 'use_aux', False)
        self.aux_weight = cfg_get(self.cfg, 'aux_weight', 0.1)
        if self.use_aux:
            self.grade_criterion = nn.CrossEntropyLoss()

    def _init_dataset(self):
        self.datasets = {}
        for sp in ['train', 'valid', 'test']:
            self.datasets[sp] = BaseDataset(
                cfg=self.cfg.data,
                split=sp,
                use_aux=self.use_aux,
            )
        # [DR] External validation set (evaluation only, no split/aux). Requires cfg.data.external_meta_path.
        self.external_dataset = None
        if cfg_get(self.cfg.data, 'external_meta_path', None):
            self.external_dataset = BaseDataset(cfg=self.cfg.data, split='external', use_aux=False)

        # [Added] report.html uses ds.df/ds.target_label/ds._resolve_path (BaseDataset attributes)
        #  to build the data summary (counts/label distribution) and sample images.
        #  fast_linear (below) replaces self.datasets with cached feature Datasets, which lack
        #  those attributes, so the report's data table/sample images were dropped entirely
        #  (bug). Keep the "real" image Datasets from before the replacement separately.
        self.report_datasets = self.datasets

        # [Added] ft.method='fast_linear': the backbone is fully frozen (it gets no gradient
        #  anyway), so there is no need to re-run it every epoch. Pass the data through the backbone
        #  only once here, cache the features (right before the head), and replace self.datasets with that cache.
        #  Afterwards _train_step/_eval_step check self.fast_linear and pass only through the head, without the backbone.
        #  (By contrast, ft.method='linear' re-runs the backbone every epoch as before and keeps
        #  train-time augmentation - use it when accuracy/augmentation matter more than speed.)
        self.fast_linear = (cfg_get(self.cfg.ft, 'method', None) == 'fast_linear')
        if self.fast_linear:
            if self.distributed:
                raise NotImplementedError(
                    "ft.method='fast_linear' does not support DDP (multi-GPU) yet. "
                    "If you need multiple GPUs, use ft.method='linear' "
                    "(it re-runs the backbone every epoch but is DDP-compatible)."
                )
            self.datasets = self._extract_and_cache_features(self.datasets)

    def _extract_and_cache_features(self, image_datasets):
        """
        [Added] Pass each split's images through the backbone(+head) forward only once and
        capture the input right before the head (= fixed backbone features) with a forward hook
        for caching. Every backbone implementation (dino/retfound/rad_dino/openclip/chexagent)
        passes through `model.head` (nn.Linear) last, so this single hook works the same way
        without knowing model-specific code.

        Note: caching only makes sense if features are fixed, so the train split is also re-read
        with the augmentation-free eval transform (force_eval_tfms=True) - i.e.
        fast_linear trades away train-time augmentation to eliminate the per-epoch backbone
        forward cost entirely.
        Note: this implementation keeps all features in memory. With very large data (hundreds of
        thousands of images) it uses a lot of memory - use ft.method='linear' in that case.
        """
        model = self.unwrap_model()
        model.eval()

        captured = {}

        def _hook(module, inp, out):
            captured['feat'] = inp[0].detach()

        handle = model.head.register_forward_hook(_hook)

        cached = {}
        try:
            for sp, ds in image_datasets.items():
                if sp == 'train':
                    # Must be re-read in eval style without augmentation so the features are fixed.
                    ds = BaseDataset(cfg=self.cfg.data, split=sp, force_eval_tfms=True)
                loader = DataLoader(
                    ds, batch_size=self.cfg.training.batch_size,
                    num_workers=self.cfg.training.num_workers, shuffle=False,
                )
                feats, targets = [], []
                with torch.no_grad():
                    for _b in tqdm(loader, desc=f'caching {sp} features',
                                             disable=not is_main_process()):
                        img, target = _b[0], _b[1]
                        img = img.to(self.device, non_blocking=True)
                        model(img)  # forward only - output discarded, only the hooked pre-head features are used
                        feats.append(captured['feat'].cpu())
                        targets.append(target)
                cached[sp] = _CachedFeatureDataset(torch.cat(feats, dim=0), torch.cat(targets, dim=0))
        finally:
            handle.remove()

        return cached

    def _prepare_target(self, target):
        # [Added] Shape targets to match the loss function.
        #  - BCE(num_classes==1): float, shape [N, 1]
        #  - CrossEntropy       : long,  shape [N]
        if self.num_classes == 1:
            return target.float().unsqueeze(1)
        return target

    def _sync_device(self):
        # [Added] Synchronisation to split timing intervals accurately. CUDA ops are asynchronous, so
        #  timing with time.perf_counter() without a sync makes "forward" only launch kernels while
        #  the actual computation finishes in the next interval (backward), creating the illusion
        #  that time is concentrated in backward.
        if self.device.type == 'cuda':
            torch.cuda.synchronize(self.device)

    def _train_step(self):
        self.model.train()

        losses = AverageMeter()
        # [Added] Break "time spent" into dataloader (IO+augmentation) / forward / backward.
        #  Dataloader time is measured as "time the batch was actually received - time the previous
        #  backward finished", which includes the time the DataLoader spends reading images and
        #  applying augmentation (with worker processes this overlaps, so the actual wait
        #  can be shorter).
        data_time = AverageMeter()
        fwd_time = AverageMeter()
        bwd_time = AverageMeter()
        loader = self.loaders['train']

        # [Added] Show the progress bar only on rank 0 (avoids duplicate progress bars under DDP)
        iterator = tqdm(loader, desc='train one epoch', total=len(loader),
                        disable=not is_main_process())

        end = time.perf_counter()
        for batch in iterator:
            data_time.update(time.perf_counter() - end)

            # [DR] (img, label, grade) if aux, otherwise (img, label)
            if self.use_aux and not self.fast_linear:
                batch_x, target, grade = batch
                grade = grade.to(self.device, non_blocking=True)
            else:
                batch_x, target = batch[0], batch[1]
            batch_x = batch_x.to(self.device, non_blocking=True)
            target = self._prepare_target(target).to(self.device, non_blocking=True)

            # ────────────────────────────────────────────────
            # [Changed - key fix] Apply AMP autocast.
            #  Previously only a GradScaler was created and forward was not wrapped in autocast,
            #  so computation actually ran in fp32 with no AMP speed-up at all.
            #  Now forward/loss are wrapped in autocast and computed in bf16/fp16.
            # ────────────────────────────────────────────────
            t0 = time.perf_counter()
            with torch.autocast(device_type=self.device.type,
                                dtype=self.amp_dtype,
                                enabled=self.enable_amp):
                # [Added] With fast_linear, batch_x is a cached backbone feature rather than an image, so
                #  pass it only through the head without the backbone (= skip the per-epoch backbone forward).
                if self.fast_linear:
                    pred = self.unwrap_model().head(batch_x)
                    loss = self.criterion(pred, target)
                elif self.use_aux:
                    # [DR] main(referable) + aux_weight * grade
                    main_out, grade_out = self.model(batch_x, return_aux=True)
                    loss = self.criterion(main_out, target) \
                        + self.aux_weight * self.grade_criterion(grade_out, grade)
                else:
                    pred = self.model(batch_x)
                    loss = self.criterion(pred, target)
            self._sync_device()
            fwd_time.update(time.perf_counter() - t0)

            t0 = time.perf_counter()
            self.optimizer.zero_grad(set_to_none=True)
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self._sync_device()
            bwd_time.update(time.perf_counter() - t0)

            losses.update(loss.item(), batch_x.size(0))
            end = time.perf_counter()

        metrics = {
            'loss': losses.avg,
            'learning_rate': get_lr(self.optimizer),
            # [Changed] Time "summed over this epoch" rather than per-batch mean (for per-epoch timing analysis).
            #  AverageMeter.sum is exactly the total time of all batches processed in this epoch.
            'data_time': data_time.sum,
            'fwd_time': fwd_time.sum,
            'bwd_time': bwd_time.sum,
        }
        return update_prefix(metrics, 'train')

    @torch.no_grad()
    def _eval_step(self, split='valid', thresholds=None):
        # [Changed] Evaluate with the underlying model unwrapped from DDP (rank-0-only evaluation, no sync needed)
        # [Changed] Generalised to take a split string instead of a bool test flag
        #  (supports 'train'/'valid'/'test' - to draw train/valid/test curves).
        #  Note: even with split='train', self.loaders['train'] is the Dataset with training augmentation
        #  (RandomResizedCrop/RandAugment) applied (only the model is in eval() mode), so these train metrics
        #  may be slightly lower and noisier than the "true train performance" without augmentation.
        #  They are still enough to see the trend (whether training is progressing).
        model = self.unwrap_model()
        model.eval()

        losses = AverageMeter()
        loader = self.loaders[split]
        running_pred, running_target = [], []

        for batch in tqdm(loader, desc=f'{split} one epoch', total=len(loader),
                                disable=not is_main_process()):
            batch_x, target = batch[0], batch[1]   # [DR] with aux, grade (=batch[2]) is ignored in evaluation
            batch_x = batch_x.to(self.device, non_blocking=True)
            t = self._prepare_target(target).to(self.device, non_blocking=True)

            with torch.autocast(device_type=self.device.type,
                                dtype=self.amp_dtype,
                                enabled=self.enable_amp):
                pred = model.head(batch_x) if self.fast_linear else model(batch_x)
                loss = self.criterion(pred, t)
            losses.update(loss.item(), batch_x.size(0))

            # pred may be bf16/fp16 due to autocast, so cast to float and move to CPU
            running_pred.append(pred.float().cpu())
            running_target.append(target.cpu())

        preds = torch.cat(running_pred, dim=0)
        targets = torch.cat(running_target, dim=0)
        y_scores, y_true = self._scores_and_true(preds, targets)

        # [Changed] If thresholds from valid are given, use them as-is (prevents test leakage);
        #  otherwise (= valid evaluation) compute thresholds on this data and return them as well.
        metrics, used_thresholds, _ = compute_basic_metrics(y_scores, y_true, thresholds=thresholds)
        metrics['loss'] = losses.avg

        return update_prefix(metrics, split), used_thresholds

    def _scores_and_true(self, preds, targets):
        """[Added] Convert the model's raw output (preds) and ground truth (targets) into the
        (y_scores, y_true) form expected by compute_basic_metrics. Shared by _eval_step/_final_inference."""
        if self.num_classes == 1:
            y_scores = torch.sigmoid(preds).numpy()
            y_true = targets.long().view(-1, 1).numpy()
        else:
            y_scores = preds.numpy()
            y_true = F.one_hot(targets, num_classes=self.num_classes).numpy()
        return y_scores, y_true

    @torch.no_grad()
    def _final_inference(self):
        """
        [Added] Instead of reusing self.model from the training loop (which may be a DDP wrapper, and
        with fast_linear is adapted to cached features rather than images) or self.loaders, rebuild
        the model from scratch "using only what is stored in" ckpt_best.pth and run a clean
        inference on the test set once more:
          1) build a fresh backbone with build_encoder(cfg) (reloaded from pretrained_root),
          2) apply only the trained parameters stored in ckpt_best.pth on top (strict=False -
             the backbone is already loaded above),
          3) use the valid thresholds (val_thresholds) stored alongside in ckpt_best.pth as-is, and
          4) re-read report_datasets['test'] (original images, not the cache, even with fast_linear)
             end to end with a new DataLoader, performing pure inference only.
        Returns: (test_metrics, y_scores, roc_curves) - y_scores is used to save test_predictions.csv,
        roc_curves to save the ROC figure in the report.
        """
        from models import build_encoder

        ckpt = torch.load(self.save_path / 'ckpt_best.pth', map_location=self.device, weights_only=False)

        model = build_encoder(self.cfg)
        model.load_state_dict(ckpt['model'], strict=False)
        model = model.to(self.device)
        model.eval()

        test_ds = self.report_datasets['test']
        loader = DataLoader(
            test_ds, batch_size=self.cfg.training.batch_size,
            num_workers=self.cfg.training.num_workers, shuffle=False,
        )

        losses = AverageMeter()
        running_pred, running_target = [], []
        for _b in tqdm(loader, desc='final inference (test, ckpt_best only)',
                                     total=len(loader), disable=not is_main_process()):
            batch_x, target = _b[0], _b[1]
            batch_x = batch_x.to(self.device, non_blocking=True)
            t = self._prepare_target(target).to(self.device, non_blocking=True)

            with torch.autocast(device_type=self.device.type,
                                dtype=self.amp_dtype,
                                enabled=self.enable_amp):
                pred = model(batch_x)
                loss = self.criterion(pred, t)
            losses.update(loss.item(), batch_x.size(0))

            running_pred.append(pred.float().cpu())
            running_target.append(target.cpu())

        preds = torch.cat(running_pred, dim=0)
        targets = torch.cat(running_target, dim=0)
        y_scores, y_true = self._scores_and_true(preds, targets)

        thresholds = ckpt.get('val_thresholds')
        metrics, _, roc_curves = compute_basic_metrics(y_scores, y_true, thresholds=thresholds)
        metrics['loss'] = losses.avg

        return metrics, y_scores, roc_curves

    @torch.no_grad()
    def _external_inference(self):
        """[DR] After training, rebuild the model from ckpt_best.pth only and run inference on the external set.
        Applies the threshold chosen on valid (val_thresholds) as-is (same principle as test, prevents leakage).
        Returns: (metrics, y_scores) - (None, None) if there is no external_dataset.
        """
        from models import build_encoder
        if getattr(self, 'external_dataset', None) is None:
            return None, None

        ckpt = torch.load(self.save_path / 'ckpt_best.pth', map_location=self.device, weights_only=False)
        model = build_encoder(self.cfg)
        model.load_state_dict(ckpt['model'], strict=False)
        model = model.to(self.device)
        model.eval()

        loader = DataLoader(
            self.external_dataset, batch_size=self.cfg.training.batch_size,
            num_workers=self.cfg.training.num_workers, shuffle=False,
        )
        running_pred, running_target = [], []
        for batch in tqdm(loader, desc='external inference (ckpt_best only)',
                          total=len(loader), disable=not is_main_process()):
            batch_x, target = batch[0], batch[1]
            batch_x = batch_x.to(self.device, non_blocking=True)
            with torch.autocast(device_type=self.device.type,
                                dtype=self.amp_dtype, enabled=self.enable_amp):
                pred = model(batch_x)
            running_pred.append(pred.float().cpu())
            running_target.append(target.cpu())

        preds = torch.cat(running_pred, dim=0)
        targets = torch.cat(running_target, dim=0)
        y_scores, y_true = self._scores_and_true(preds, targets)
        thresholds = ckpt.get('val_thresholds')
        metrics, _, _ = compute_basic_metrics(y_scores, y_true, thresholds=thresholds)
        return metrics, y_scores


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config-path', type=str, required=True,
        help='Path to the config file.'
    )
    parser.add_argument(
        '--resume', type=str, default=None,
        help='Path to the checkpoint to resume from.'
    )
    args = parser.parse_args()

    trainer = ClassificationTrainer(args)
    trainer.fit()
