"""Unit tests for config schema dataclasses."""

from __future__ import annotations

import dataclasses

import torch

from vision3d.config.schema import (
    BackboneConfig,
    BatchData,
    BEVFormerModelConfig,
    BoundingBox3DPrediction,
    BoundingBox3DTarget,
    CameraExtrinsics,
    CameraIntrinsics,
    CameraView,
    DatasetConfig,
    EncoderConfig,
    EvaluatorConfig,
    FrameData,
    HeadConfig,
    LitModuleConfig,
    LossConfig,
    MatcherConfig,
    MatchingResult,
    NeckConfig,
    TrainConfig,
)

# ---------------------------------------------------------------------------
# Hydra config tests
# ---------------------------------------------------------------------------


class TestHydraConfigTargets:
    """Verify _target_ fields point to the correct class paths."""

    def test_backbone_target(self):
        assert BackboneConfig._target_ == "vision3d.models.backbones.ResNetBackbone"

    def test_neck_target(self):
        assert NeckConfig._target_ == "vision3d.models.necks.FPNNeck"

    def test_encoder_target(self):
        assert EncoderConfig._target_ == "vision3d.models.encoders.BEVEncoder"

    def test_head_target(self):
        assert HeadConfig._target_ == "vision3d.models.heads.DetectionHead"

    def test_loss_target(self):
        assert LossConfig._target_ == "vision3d.core.losses.DetectionLoss"

    def test_matcher_target(self):
        assert MatcherConfig._target_ == "vision3d.core.matchers.HungarianMatcher"

    def test_evaluator_target(self):
        assert EvaluatorConfig._target_ == "vision3d.core.evaluators.Vision3DEvaluator"

    def test_bevformer_model_target(self):
        assert BEVFormerModelConfig._target_ == "vision3d.models.bevformer.BEVFormerModel"

    def test_lit_module_target(self):
        assert LitModuleConfig._target_ == "vision3d.engine.lit_module.Vision3DLightningModule"

    def test_dataset_target(self):
        assert DatasetConfig._target_ == "vision3d.data.dataset.Vision3DDataset"


class TestHydraConfigDefaults:
    """Verify default field values of Hydra configs."""

    def test_backbone_defaults(self):
        cfg = BackboneConfig()
        assert cfg.depth == 50
        assert cfg.out_indices == [1, 2, 3]

    def test_neck_defaults(self):
        cfg = NeckConfig()
        assert cfg.in_channels == [512, 1024, 2048]
        assert cfg.out_channels == 256

    def test_encoder_defaults(self):
        cfg = EncoderConfig()
        assert cfg.bev_h == 200
        assert cfg.bev_w == 200
        assert cfg.embed_dims == 256
        assert cfg.num_layers == 6

    def test_head_defaults(self):
        cfg = HeadConfig()
        assert cfg.num_classes == 10
        assert cfg.in_channels == 256

    def test_loss_defaults(self):
        cfg = LossConfig()
        assert cfg.cls_weight == 2.0
        assert cfg.bbox_weight == 0.25

    def test_matcher_defaults(self):
        cfg = MatcherConfig()
        assert cfg.cost_class == 2.0
        assert cfg.cost_bbox == 0.25

    def test_evaluator_defaults(self):
        cfg = EvaluatorConfig()
        assert cfg.eval_range == 50.0

    def test_train_config_defaults(self):
        cfg = TrainConfig()
        assert cfg.max_epochs == 24
        assert cfg.batch_size == 1

    def test_dataset_defaults(self):
        cfg = DatasetConfig()
        assert cfg.split == "train"
        assert cfg.num_cameras == 6
        assert cfg.num_past_frames == 2


class TestLitModuleConfigFields:
    """Verify LitModuleConfig uses model field, not backbone/neck/etc."""

    def test_has_model_field(self):
        field_names = {f.name for f in dataclasses.fields(LitModuleConfig)}
        assert "model" in field_names

    def test_has_loss_field(self):
        field_names = {f.name for f in dataclasses.fields(LitModuleConfig)}
        assert "loss" in field_names

    def test_has_matcher_field(self):
        field_names = {f.name for f in dataclasses.fields(LitModuleConfig)}
        assert "matcher" in field_names

    def test_has_evaluator_field(self):
        field_names = {f.name for f in dataclasses.fields(LitModuleConfig)}
        assert "evaluator" in field_names

    def test_no_backbone_field(self):
        field_names = {f.name for f in dataclasses.fields(LitModuleConfig)}
        assert "backbone" not in field_names

    def test_no_neck_field(self):
        field_names = {f.name for f in dataclasses.fields(LitModuleConfig)}
        assert "neck" not in field_names

    def test_no_encoder_field(self):
        field_names = {f.name for f in dataclasses.fields(LitModuleConfig)}
        assert "encoder" not in field_names

    def test_no_head_field(self):
        field_names = {f.name for f in dataclasses.fields(LitModuleConfig)}
        assert "head" not in field_names

    def test_model_field_is_bev_former_model_config(self):
        cfg = LitModuleConfig()
        assert isinstance(cfg.model, BEVFormerModelConfig)


# ---------------------------------------------------------------------------
# Runtime dataclass tests
# ---------------------------------------------------------------------------


class TestCameraExtrinsics:
    """Test CameraExtrinsics dataclass."""

    def test_instantiation(self):
        ext = CameraExtrinsics(
            translation=torch.zeros(3),
            rotation=torch.tensor([1.0, 0.0, 0.0, 0.0]),
        )
        assert ext.translation.shape == (3,)
        assert ext.rotation.shape == (4,)

    def test_fields(self):
        field_names = {f.name for f in dataclasses.fields(CameraExtrinsics)}
        assert "translation" in field_names
        assert "rotation" in field_names


class TestCameraIntrinsics:
    """Test CameraIntrinsics dataclass."""

    def test_instantiation(self):
        intr = CameraIntrinsics(matrix=torch.eye(3))
        assert intr.matrix.shape == (3, 3)

    def test_fields(self):
        field_names = {f.name for f in dataclasses.fields(CameraIntrinsics)}
        assert "matrix" in field_names


class TestCameraView:
    """Test CameraView dataclass."""

    def test_instantiation(self):
        view = CameraView(
            image=torch.rand(3, 32, 32),
            intrinsics=CameraIntrinsics(matrix=torch.eye(3)),
            extrinsics=CameraExtrinsics(
                translation=torch.zeros(3),
                rotation=torch.tensor([1.0, 0.0, 0.0, 0.0]),
            ),
            name="front",
        )
        assert view.name == "front"
        assert view.image.shape == (3, 32, 32)

    def test_fields(self):
        field_names = {f.name for f in dataclasses.fields(CameraView)}
        for expected in ("image", "intrinsics", "extrinsics", "name"):
            assert expected in field_names


class TestBoundingBox3DTarget:
    """Test BoundingBox3DTarget dataclass."""

    def test_instantiation(self):
        tgt = BoundingBox3DTarget(
            boxes=torch.zeros(3, 10),
            labels=torch.zeros(3, dtype=torch.long),
            instance_ids=["a", "b", "c"],
        )
        assert tgt.boxes.shape == (3, 10)
        assert len(tgt.instance_ids) == 3

    def test_fields(self):
        field_names = {f.name for f in dataclasses.fields(BoundingBox3DTarget)}
        assert "boxes" in field_names
        assert "labels" in field_names
        assert "instance_ids" in field_names


class TestBoundingBox3DPrediction:
    """Test BoundingBox3DPrediction dataclass."""

    def test_instantiation(self):
        pred = BoundingBox3DPrediction(
            boxes=torch.zeros(5, 10),
            scores=torch.ones(5),
            labels=torch.zeros(5, dtype=torch.long),
        )
        assert pred.boxes.shape == (5, 10)
        assert pred.scores.shape == (5,)
        assert pred.labels.shape == (5,)

    def test_fields(self):
        field_names = {f.name for f in dataclasses.fields(BoundingBox3DPrediction)}
        assert "boxes" in field_names
        assert "scores" in field_names
        assert "labels" in field_names


class TestMatchingResult:
    """Test MatchingResult dataclass."""

    def test_instantiation(self):
        mr = MatchingResult(
            pred_indices=torch.arange(3),
            gt_indices=torch.tensor([0, 2, 1]),
        )
        assert mr.pred_indices.shape == (3,)
        assert mr.gt_indices.shape == (3,)

    def test_empty_matching(self):
        mr = MatchingResult(
            pred_indices=torch.zeros(0, dtype=torch.long),
            gt_indices=torch.zeros(0, dtype=torch.long),
        )
        assert mr.pred_indices.numel() == 0


class TestFrameData:
    """Test FrameData dataclass."""

    def test_instantiation_minimal(self):
        frame = FrameData(frame_id="f0", timestamp=0.0, cameras={})
        assert frame.frame_id == "f0"
        assert frame.timestamp == 0.0
        assert frame.targets is None
        assert frame.predictions is None
        assert frame.matches is None
        assert frame.past_frames == []

    def test_fields(self):
        field_names = {f.name for f in dataclasses.fields(FrameData)}
        for expected in (
            "frame_id",
            "timestamp",
            "cameras",
            "targets",
            "predictions",
            "matches",
            "past_frames",
        ):
            assert expected in field_names

    def test_optional_fields_default_none(self):
        frame = FrameData(frame_id="f0", timestamp=0.0, cameras={})
        assert frame.targets is None
        assert frame.predictions is None
        assert frame.matches is None


class TestBatchData:
    """Test BatchData dataclass and to() method."""

    def _make_frame(self) -> FrameData:
        return FrameData(
            frame_id="f0",
            timestamp=0.0,
            cameras={
                "front": CameraView(
                    image=torch.rand(3, 8, 8),
                    intrinsics=CameraIntrinsics(matrix=torch.eye(3)),
                    extrinsics=CameraExtrinsics(
                        translation=torch.zeros(3),
                        rotation=torch.tensor([1.0, 0.0, 0.0, 0.0]),
                    ),
                    name="front",
                )
            },
            targets=BoundingBox3DTarget(
                boxes=torch.zeros(2, 10),
                labels=torch.zeros(2, dtype=torch.long),
                instance_ids=["a", "b"],
            ),
        )

    def test_instantiation(self):
        batch = BatchData(batch_size=2, frames=[self._make_frame(), self._make_frame()])
        assert batch.batch_size == 2
        assert len(batch.frames) == 2

    def test_fields(self):
        field_names = {f.name for f in dataclasses.fields(BatchData)}
        assert "batch_size" in field_names
        assert "frames" in field_names

    def test_to_cpu(self):
        batch = BatchData(batch_size=1, frames=[self._make_frame()])
        result = batch.to(torch.device("cpu"))
        assert result is batch
        cam = result.frames[0].cameras["front"]
        assert cam.image.device.type == "cpu"

    def test_to_moves_all_camera_tensors(self):
        batch = BatchData(batch_size=1, frames=[self._make_frame()])
        batch.to(torch.device("cpu"))
        cam = batch.frames[0].cameras["front"]
        assert cam.image.device.type == "cpu"
        assert cam.intrinsics.matrix.device.type == "cpu"
        assert cam.extrinsics.translation.device.type == "cpu"

    def test_to_moves_target_tensors(self):
        batch = BatchData(batch_size=1, frames=[self._make_frame()])
        batch.to(torch.device("cpu"))
        tgt = batch.frames[0].targets
        assert tgt is not None
        assert tgt.boxes.device.type == "cpu"
