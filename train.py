"""
Main training entry point for Vision3D.

Uses Hydra for configuration management and PyTorch Lightning for training.
All sub-components (backbone, neck, encoder, head, loss, matcher, evaluator)
are recursively instantiated by Hydra from the config tree.

Usage:
    # Train with default config (dummy dataset, BEVFormer):
    python train.py

    # Train on NuScenes with a custom experiment override:
    python train.py dataset=nuscenes experiment=exp_01 max_epochs=24

    # Multi-run hyperparameter sweep:
    python train.py --multirun model.learning_rate=1e-4,2e-4 max_epochs=12,24
"""

from __future__ import annotations

import logging

import hydra
import pytorch_lightning as pl
from hydra.utils import instantiate
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from vision3d.config.schema import TrainConfig
from vision3d.data.dataset import Vision3DDataset
from vision3d.utils.foxglove import FoxgloveMCAPLogger

log = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="configs", config_name="train")
def main(cfg: DictConfig) -> None:
    """Hydra entry point: instantiate all components and launch training.

    Steps:
      1. Set the global random seed for reproducibility.
      2. Instantiate the full LightningModule (model + loss + evaluator) via Hydra.
      3. Instantiate training and validation datasets and DataLoaders.
      4. Build Lightning callbacks (FoxgloveMCAPLogger, ModelCheckpoint, …).
      5. Instantiate and run the Lightning Trainer.

    Args:
        cfg: Hydra DictConfig constructed from `configs/train.yaml` and any
             command-line overrides.
    """
    # TODO: pl.seed_everything(cfg.seed)

    # TODO: log = logging.getLogger(__name__); log.info("Starting training with config: %s", cfg)

    # TODO: instantiate the full LightningModule:
    #         model = instantiate(cfg.model)

    # TODO: instantiate train and val datasets:
    #         train_dataset = instantiate(cfg.dataset, split="train")
    #         val_dataset   = instantiate(cfg.dataset, split="val")

    # TODO: build DataLoaders:
    #         train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size,
    #                                   shuffle=True,  num_workers=cfg.num_workers,
    #                                   collate_fn=Vision3DDataset.collate_fn)
    #         val_loader   = DataLoader(val_dataset,   batch_size=cfg.batch_size,
    #                                   shuffle=False, num_workers=cfg.num_workers,
    #                                   collate_fn=Vision3DDataset.collate_fn)

    # TODO: build callbacks:
    #         foxglove_logger = FoxgloveMCAPLogger(output_dir=cfg.output_dir + "/mcap")
    #         checkpoint_cb   = pl.callbacks.ModelCheckpoint(...)
    #         lr_monitor      = pl.callbacks.LearningRateMonitor(logging_interval="step")

    # TODO: instantiate the Lightning Trainer:
    #         trainer = pl.Trainer(
    #             max_epochs=cfg.max_epochs,
    #             callbacks=[foxglove_logger, checkpoint_cb, lr_monitor],
    #             logger=pl.loggers.TensorBoardLogger(cfg.output_dir),
    #             ...
    #         )

    # TODO: trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    raise NotImplementedError


if __name__ == "__main__":
    main()
