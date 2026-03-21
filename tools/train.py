"""
Main training entry point for Vision3D.

Uses Hydra for configuration management and PyTorch Lightning for training.
All sub-components (backbone, neck, encoder, head, loss, matcher, evaluator)
are recursively instantiated by Hydra from the config tree.

Usage:
    # Train with default config (dummy dataset, BEVFormer):
    python tools/train.py

    # Train on NuScenes with a custom experiment override:
    python tools/train.py dataset=nuscenes experiment=exp_01 max_epochs=24

    # Multi-run hyperparameter sweep:
    python tools/train.py --multirun model.learning_rate=1e-4,2e-4 max_epochs=12,24
"""

from __future__ import annotations

import logging

import hydra
import pytorch_lightning as pl
from hydra.utils import instantiate
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from vision3d.data.dataset import Vision3DDataset
from vision3d.utils.foxglove import FoxgloveMCAPLogger

log = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../configs", config_name="train")
def main(cfg: DictConfig) -> None:
    """Hydra entry point: instantiate all components and launch training."""
    pl.seed_everything(cfg.seed)
    log.info("Starting training with config: %s", cfg)
    model = instantiate(cfg.model)
    train_dataset = instantiate(cfg.dataset, split="train")
    val_dataset = instantiate(cfg.dataset, split="val")
    persistent = cfg.num_workers > 0 and bool(cfg.get("persistent_workers", False))
    dataloader_kwargs = {
        "batch_size": cfg.batch_size,
        "num_workers": cfg.num_workers,
        "persistent_workers": persistent,
        "collate_fn": Vision3DDataset.collate_fn,
    }
    if cfg.num_workers > 0:
        dataloader_kwargs["prefetch_factor"] = max(1, int(cfg.get("prefetch_factor", 1)))

    train_loader = DataLoader(train_dataset, shuffle=True, **dataloader_kwargs)
    val_loader = DataLoader(val_dataset, shuffle=False, **dataloader_kwargs)
    foxglove_logger = FoxgloveMCAPLogger(output_dir=cfg.output_dir + "/mcap")
    checkpoint_cb = pl.callbacks.ModelCheckpoint(
        monitor="val/mAP",
        mode="max",
        save_top_k=3,
        filename="{epoch:02d}-{val/mAP:.3f}",
    )
    lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval="step")
    trainer = pl.Trainer(
        max_epochs=cfg.max_epochs,
        callbacks=[foxglove_logger, checkpoint_cb, lr_monitor],
        logger=pl.loggers.TensorBoardLogger(cfg.output_dir),
    )
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    log.info("Training complete. Best checkpoint saved at: %s", checkpoint_cb.best_model_path)


if __name__ == "__main__":
    main()
