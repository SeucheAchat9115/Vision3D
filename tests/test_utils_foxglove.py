"""Unit tests for FoxgloveMCAPLogger."""

from __future__ import annotations

import importlib.util
from unittest.mock import MagicMock

import pytest
import pytorch_lightning as pl
import torch

_MCAP_AVAILABLE = importlib.util.find_spec("mcap") is not None

from vision3d.config.schema import (
    BatchData,
    BoundingBox3DPrediction,
    BoundingBox3DTarget,
    CameraExtrinsics,
    CameraIntrinsics,
    CameraView,
    FrameData,
)
from vision3d.utils.foxglove import FoxgloveMCAPLogger

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_frame(
    frame_id: str = "frame_0",
    timestamp: float = 0.0,
    include_targets: bool = True,
    include_predictions: bool = False,
    num_boxes: int = 3,
    score_threshold: float = 0.0,
) -> FrameData:
    torch.manual_seed(0)
    cameras = {
        "front": CameraView(
            image=torch.rand(3, 32, 32),
            intrinsics=CameraIntrinsics(matrix=torch.eye(3)),
            extrinsics=CameraExtrinsics(
                translation=torch.zeros(3), rotation=torch.tensor([1.0, 0.0, 0.0, 0.0])
            ),
            name="front",
        )
    }
    targets = None
    predictions = None
    if include_targets:
        boxes = torch.randn(num_boxes, 10)
        boxes[:, 3:6] = 1.0  # Positive dims
        targets = BoundingBox3DTarget(
            boxes=boxes,
            labels=torch.zeros(num_boxes, dtype=torch.long),
            instance_ids=[f"id_{i}" for i in range(num_boxes)],
        )
    if include_predictions:
        boxes = torch.randn(num_boxes, 10)
        predictions = BoundingBox3DPrediction(
            boxes=boxes,
            scores=torch.ones(num_boxes) * (score_threshold + 0.1),
            labels=torch.zeros(num_boxes, dtype=torch.long),
        )
    return FrameData(
        frame_id=frame_id,
        timestamp=timestamp,
        cameras=cameras,
        targets=targets,
        predictions=predictions,
    )


def _make_batch(frames: list[FrameData]) -> BatchData:
    return BatchData(batch_size=len(frames), frames=frames)


def _make_trainer(epoch: int = 0) -> pl.Trainer:
    trainer = MagicMock(spec=pl.Trainer)
    trainer.current_epoch = epoch
    return trainer


def _make_pl_module() -> pl.LightningModule:
    return MagicMock(spec=pl.LightningModule)


# ---------------------------------------------------------------------------
# Interface tests
# ---------------------------------------------------------------------------


class TestFoxgloveMCAPLoggerInterface:
    """Verify FoxgloveMCAPLogger interface."""

    def test_is_pl_callback(self):
        assert issubclass(FoxgloveMCAPLogger, pl.Callback)

    def test_has_on_validation_epoch_end(self):
        assert hasattr(FoxgloveMCAPLogger, "on_validation_epoch_end")

    def test_has_on_validation_epoch_start(self):
        assert hasattr(FoxgloveMCAPLogger, "on_validation_epoch_start")

    def test_has_on_validation_batch_end(self):
        assert hasattr(FoxgloveMCAPLogger, "on_validation_batch_end")

    def test_init_stores_params(self):
        logger = FoxgloveMCAPLogger(
            output_dir="/tmp/mcap_test",
            max_frames=50,
            write_images=True,
            score_threshold=0.5,
        )
        assert logger.max_frames == 50
        assert logger.write_images is True
        assert logger.score_threshold == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# Buffer management
# ---------------------------------------------------------------------------


class TestFoxgloveMCAPLoggerBuffer:
    """Test the frame buffer lifecycle."""

    def test_buffer_empty_at_init(self):
        logger = FoxgloveMCAPLogger()
        assert len(logger._frame_buffer) == 0

    def test_on_validation_epoch_start_clears_buffer(self):
        logger = FoxgloveMCAPLogger()
        logger._frame_buffer.append(_make_frame())
        logger.on_validation_epoch_start(_make_trainer(), _make_pl_module())
        assert len(logger._frame_buffer) == 0

    def test_on_validation_batch_end_adds_frames(self):
        logger = FoxgloveMCAPLogger(max_frames=100)
        frames = [_make_frame(f"f_{i}") for i in range(3)]
        batch = _make_batch(frames)
        logger.on_validation_batch_end(_make_trainer(), _make_pl_module(), None, batch, 0)
        assert len(logger._frame_buffer) == 3

    def test_max_frames_respected(self):
        max_frames = 2
        logger = FoxgloveMCAPLogger(max_frames=max_frames)
        frames = [_make_frame(f"f_{i}") for i in range(5)]
        batch = _make_batch(frames)
        logger.on_validation_batch_end(_make_trainer(), _make_pl_module(), None, batch, 0)
        assert len(logger._frame_buffer) <= max_frames

    def test_none_max_frames_keeps_all(self):
        logger = FoxgloveMCAPLogger(max_frames=None)
        frames = [_make_frame(f"f_{i}") for i in range(10)]
        for i in range(0, 10, 2):
            batch = _make_batch(frames[i : i + 2])
            logger.on_validation_batch_end(_make_trainer(), _make_pl_module(), None, batch, 0)
        assert len(logger._frame_buffer) == 10


# ---------------------------------------------------------------------------
# Encoding helpers
# ---------------------------------------------------------------------------


class TestFoxgloveMCAPLoggerEncoding:
    """Test _encode_boxes3d and _encode_image helpers."""

    def test_encode_boxes3d_returns_bytes(self):
        logger = FoxgloveMCAPLogger()
        boxes = torch.randn(3, 10)
        labels = torch.zeros(3, dtype=torch.long)
        result = logger._encode_boxes3d(boxes, labels)
        assert isinstance(result, bytes)

    def test_encode_boxes3d_valid_json(self):
        import json

        logger = FoxgloveMCAPLogger()
        boxes = torch.randn(2, 10)
        labels = torch.zeros(2, dtype=torch.long)
        data = json.loads(logger._encode_boxes3d(boxes, labels))
        assert "boxes" in data
        assert len(data["boxes"]) == 2

    def test_encode_boxes3d_with_scores(self):
        import json

        logger = FoxgloveMCAPLogger()
        boxes = torch.randn(2, 10)
        labels = torch.zeros(2, dtype=torch.long)
        scores = torch.tensor([0.8, 0.6])
        data = json.loads(logger._encode_boxes3d(boxes, labels, scores))
        for box_entry in data["boxes"]:
            assert "score" in box_entry

    def test_encode_boxes3d_contains_position(self):
        import json

        logger = FoxgloveMCAPLogger()
        boxes = torch.randn(1, 10)
        labels = torch.zeros(1, dtype=torch.long)
        data = json.loads(logger._encode_boxes3d(boxes, labels))
        assert "position" in data["boxes"][0]
        assert "x" in data["boxes"][0]["position"]

    def test_encode_boxes3d_contains_size(self):
        import json

        logger = FoxgloveMCAPLogger()
        boxes = torch.randn(1, 10)
        labels = torch.zeros(1, dtype=torch.long)
        data = json.loads(logger._encode_boxes3d(boxes, labels))
        assert "size" in data["boxes"][0]

    def test_encode_boxes3d_empty_boxes(self):
        import json

        logger = FoxgloveMCAPLogger()
        boxes = torch.zeros(0, 10)
        labels = torch.zeros(0, dtype=torch.long)
        data = json.loads(logger._encode_boxes3d(boxes, labels))
        assert data["boxes"] == []

    def test_encode_image_returns_bytes(self):
        logger = FoxgloveMCAPLogger()
        image = torch.rand(3, 32, 32)
        result = logger._encode_image(image, "front")
        assert isinstance(result, bytes)
        assert len(result) > 0


# ---------------------------------------------------------------------------
# MCAP writing
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _MCAP_AVAILABLE, reason="mcap not installed (install vision3d[viz])")
class TestFoxgloveMCAPLoggerWriting:
    """Test that MCAP files are written correctly."""

    def test_on_validation_epoch_end_writes_file(self, tmp_path):
        logger = FoxgloveMCAPLogger(output_dir=str(tmp_path / "mcap"))
        frame = _make_frame(include_targets=True)
        logger._frame_buffer.append(frame)
        trainer = _make_trainer(epoch=1)
        logger.on_validation_epoch_end(trainer, _make_pl_module())
        mcap_files = list((tmp_path / "mcap").glob("*.mcap"))
        assert len(mcap_files) == 1

    def test_empty_buffer_does_not_write_file(self, tmp_path):
        logger = FoxgloveMCAPLogger(output_dir=str(tmp_path / "mcap"))
        logger.on_validation_epoch_end(_make_trainer(), _make_pl_module())
        mcap_dir = tmp_path / "mcap"
        if mcap_dir.exists():
            mcap_files = list(mcap_dir.glob("*.mcap"))
            assert len(mcap_files) == 0

    def test_output_dir_created(self, tmp_path):
        logger = FoxgloveMCAPLogger(output_dir=str(tmp_path / "new_dir" / "mcap"))
        logger._frame_buffer.append(_make_frame())
        logger.on_validation_epoch_end(_make_trainer(epoch=0), _make_pl_module())
        assert (tmp_path / "new_dir" / "mcap").exists()

    def test_file_named_by_epoch(self, tmp_path):
        logger = FoxgloveMCAPLogger(output_dir=str(tmp_path))
        logger._frame_buffer.append(_make_frame())
        logger.on_validation_epoch_end(_make_trainer(epoch=5), _make_pl_module())
        expected = tmp_path / "epoch_0005.mcap"
        assert expected.exists()

    def test_frame_with_predictions_written(self, tmp_path):
        logger = FoxgloveMCAPLogger(output_dir=str(tmp_path), score_threshold=0.3)
        frame = _make_frame(include_predictions=True, score_threshold=0.3)
        logger._frame_buffer.append(frame)
        logger.on_validation_epoch_end(_make_trainer(epoch=0), _make_pl_module())
        assert any(tmp_path.glob("*.mcap"))
