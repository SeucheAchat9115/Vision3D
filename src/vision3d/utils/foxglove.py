"""
Foxglove MCAP visualisation logger for Vision3D.

Provides `FoxgloveMCAPLogger`, a PyTorch Lightning Callback that hooks into
`on_validation_epoch_end` to serialise ground-truth and predicted 3-D bounding
boxes into an `.mcap` file. The file can be opened directly in Foxglove Studio
for drag-and-drop inspection of model outputs.

MCAP format reference: https://mcap.dev
Foxglove schemas: https://github.com/foxglove/schemas
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, List, Optional

import pytorch_lightning as pl
import torch

from vision3d.config.schema import BatchData, BoundingBox3DPrediction, BoundingBox3DTarget


class FoxgloveMCAPLogger(pl.Callback):
    """Lightning Callback that writes GT and predicted boxes to an MCAP file.

    Hooks into the validation loop to collect fully populated `FrameData`
    objects (containing targets, predictions, and optionally matches) and
    serialises them to a single `.mcap` file at the end of each validation
    epoch. The file can then be opened in Foxglove Studio for visual debugging.

    Foxglove topic layout:
      - `/gt/boxes3d`: Ground-truth 3-D bounding boxes.
      - `/pred/boxes3d`: Predicted 3-D bounding boxes with confidence scores.
      - `/cameras/<name>`: Camera images (optional, configurable).

    Args:
        output_dir: Directory where `.mcap` files are written.
            Files are named `epoch_{epoch:04d}.mcap`.
        max_frames: Maximum number of frames to write per epoch. Set to None
            to write all validation frames (may produce large files).
        write_images: If True, also serialise camera images into the MCAP.
            This significantly increases file size.
        score_threshold: Minimum confidence score for a predicted box to be
            included in the MCAP visualisation.
    """

    def __init__(
        self,
        output_dir: str = "outputs/mcap",
        max_frames: Optional[int] = 100,
        write_images: bool = False,
        score_threshold: float = 0.3,
    ) -> None:
        super().__init__()
        # TODO: store output_dir (as Path), max_frames, write_images, score_threshold
        # TODO: initialise a list self._frame_buffer to accumulate frames during validation
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Lightning Callback hooks
    # ------------------------------------------------------------------

    def on_validation_epoch_start(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
    ) -> None:
        """Clear the frame buffer at the start of each validation epoch."""
        # TODO: self._frame_buffer.clear()
        raise NotImplementedError

    def on_validation_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: Any,
        batch: BatchData,
        batch_idx: int,
    ) -> None:
        """Collect frames from the current validation batch into the buffer.

        Stops collecting once `max_frames` have been accumulated to avoid
        unbounded memory growth.

        Args:
            batch: The `BatchData` for the current validation step, which
                should already have `predictions` populated by the forward pass.
        """
        # TODO: if max_frames is set and len(self._frame_buffer) >= max_frames, return early
        # TODO: for each frame in batch.frames:
        #         append frame to self._frame_buffer (detach all tensors first)
        raise NotImplementedError

    def on_validation_epoch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
    ) -> None:
        """Write all buffered frames to an MCAP file.

        Constructs the output path, opens an MCAP writer, and serialises each
        frame's GT boxes, predicted boxes (above threshold), and optionally
        camera images using Foxglove's standard message schemas.

        Args:
            trainer: The Lightning Trainer (used to get current epoch number).
            pl_module: The LightningModule (unused but required by the hook).
        """
        # TODO: build output path: self.output_dir / f"epoch_{trainer.current_epoch:04d}.mcap"
        # TODO: create output_dir if it does not exist
        # TODO: open an mcap.Writer context manager
        # TODO: register Foxglove Boxes3d and (optionally) CompressedImage schemas
        # TODO: iterate self._frame_buffer; for each frame:
        #         a. Encode ground-truth boxes in Foxglove Boxes3d message format
        #         b. Filter predictions by score_threshold
        #         c. Encode predicted boxes in Foxglove Boxes3d message format
        #         d. If write_images, encode each camera image as CompressedImage
        #         e. Write all messages to the MCAP with the frame timestamp
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Serialisation helpers
    # ------------------------------------------------------------------

    def _encode_boxes3d(
        self,
        boxes: torch.Tensor,
        labels: torch.Tensor,
        scores: Optional[torch.Tensor] = None,
    ) -> bytes:
        """Serialise a set of 3-D boxes to the Foxglove Boxes3d binary schema.

        Args:
            boxes: Box parameters tensor of shape (N, 10) in the Vision3D 10-DOF
                format [x, y, z, w, l, h, sin(θ), cos(θ), vx, vy].
            labels: Integer class indices. Shape: (N,).
            scores: Optional confidence scores. Shape: (N,). Included in the
                metadata field when provided.

        Returns:
            Serialised bytes ready to be written to the MCAP channel.
        """
        # TODO: convert sin/cos heading encoding back to yaw angle (atan2)
        # TODO: build list of Foxglove Pose + PackedElementField messages per box
        # TODO: serialise using the Foxglove Boxes3d schema (protobuf or JSON)
        # TODO: return the serialised bytes
        raise NotImplementedError

    def _encode_image(
        self,
        image: torch.Tensor,
        camera_name: str,
    ) -> bytes:
        """Serialise a camera image tensor to a Foxglove CompressedImage message.

        Args:
            image: Float32 image tensor of shape (3, H, W) in [0, 1] range.
            camera_name: Camera identifier embedded in the message metadata.

        Returns:
            Serialised bytes for the CompressedImage channel.
        """
        # TODO: convert float32 tensor to uint8 numpy array (scale by 255)
        # TODO: encode as JPEG using PIL or cv2 for file-size efficiency
        # TODO: wrap in a Foxglove CompressedImage message struct
        # TODO: return the serialised bytes
        raise NotImplementedError
