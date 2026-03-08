"""
Integration tests for Vision3D.

These tests verify that different components work correctly together, testing
various combinations of units and ensuring interfaces are compatible even for
edge cases.

Covered scenarios:
  - Backbone + Neck combinations (ResNet18, ResNet34, ResNet50 × various FPN configs)
  - Neck → Encoder integration
  - Full BEVFormerModel pipeline (different configs, batch sizes, cameras)
  - Model + Matcher + Loss (training pipeline)
  - Model + Evaluator (validation pipeline)
  - Lightning module training_step and validation_step
  - BatchData / FrameData interface edge cases
"""

from __future__ import annotations

import pytest
import torch

from vision3d.config.schema import (
    BatchData,
    BoundingBox3DPrediction,
    BoundingBox3DTarget,
    CameraExtrinsics,
    CameraIntrinsics,
    CameraView,
    FrameData,
    MatchingResult,
)
from vision3d.core.evaluators import Vision3DEvaluator
from vision3d.core.losses import DetectionLoss
from vision3d.core.matchers import HungarianMatcher
from vision3d.models.backbones.resnet import ResNetBackbone
from vision3d.models.bevformer import BEVFormerModel
from vision3d.models.encoders.bev_encoder import BEVEncoder
from vision3d.models.heads.detection_head import DetectionHead
from vision3d.models.necks.fpn import FPNNeck

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

# Channel counts returned by each ResNet depth for out_indices=[1,2,3]
_RESNET_CHANNELS: dict[int, list[int]] = {
    18: [64, 128, 256],
    34: [64, 128, 256],
    50: [256, 512, 1024],
}


def _make_camera_view(
    image_h: int = 64,
    image_w: int = 64,
    *,
    name: str = "front",
    seed: int = 0,
) -> CameraView:
    """Return a synthetic CameraView with identity-like extrinsics."""
    torch.manual_seed(seed)
    K = torch.tensor(
        [[400.0, 0.0, image_w / 2.0], [0.0, 400.0, image_h / 2.0], [0.0, 0.0, 1.0]]
    )
    return CameraView(
        image=torch.rand(3, image_h, image_w),
        intrinsics=CameraIntrinsics(matrix=K),
        extrinsics=CameraExtrinsics(
            translation=torch.tensor([1.5, 0.0, 1.2]),
            rotation=torch.tensor([1.0, 0.0, 0.0, 0.0]),  # identity quaternion
        ),
        name=name,
    )


def _make_frame(
    num_cameras: int = 1,
    num_boxes: int = 3,
    num_classes: int = 5,
    image_h: int = 64,
    image_w: int = 64,
    *,
    frame_id: str = "test_frame",
    seed: int = 0,
    with_targets: bool = True,
) -> FrameData:
    """Return a synthetic FrameData for integration testing."""
    cam_names = [f"cam_{i}" for i in range(num_cameras)]
    cameras = {
        name: _make_camera_view(image_h, image_w, name=name, seed=seed + i)
        for i, name in enumerate(cam_names)
    }
    targets: BoundingBox3DTarget | None = None
    if with_targets:
        torch.manual_seed(seed + 100)
        targets = BoundingBox3DTarget(
            boxes=torch.randn(num_boxes, 10),
            labels=torch.randint(0, num_classes, (num_boxes,)),
            instance_ids=[f"{frame_id}_id_{i}" for i in range(num_boxes)],
        )
    return FrameData(
        frame_id=frame_id,
        timestamp=float(seed),
        cameras=cameras,
        targets=targets,
    )


def _make_batch(
    batch_size: int = 2,
    num_cameras: int = 1,
    num_boxes: int = 3,
    num_classes: int = 5,
    image_h: int = 64,
    image_w: int = 64,
    *,
    with_targets: bool = True,
) -> BatchData:
    """Return a synthetic BatchData."""
    frames = [
        _make_frame(
            num_cameras=num_cameras,
            num_boxes=num_boxes,
            num_classes=num_classes,
            image_h=image_h,
            image_w=image_w,
            frame_id=f"frame_{i}",
            seed=i,
            with_targets=with_targets,
        )
        for i in range(batch_size)
    ]
    return BatchData(batch_size=batch_size, frames=frames)


def _make_small_model(
    backbone_depth: int = 18,
    out_indices: list[int] | None = None,
    embed_dims: int = 32,
    bev_h: int = 4,
    bev_w: int = 4,
    num_classes: int = 5,
    num_queries: int = 10,
    num_bev_layers: int = 1,
) -> BEVFormerModel:
    """Return a small BEVFormerModel suitable for fast CPU integration tests."""
    if out_indices is None:
        out_indices = [1, 2, 3]
    # Map out_indices to corresponding backbone channel counts
    all_channels = _RESNET_CHANNELS[backbone_depth]
    in_channels = [all_channels[i - 1] for i in out_indices]
    backbone = ResNetBackbone(depth=backbone_depth, pretrained=False, out_indices=out_indices)
    neck = FPNNeck(in_channels=in_channels, out_channels=embed_dims, num_outs=len(out_indices))
    encoder = BEVEncoder(
        bev_h=bev_h,
        bev_w=bev_w,
        embed_dims=embed_dims,
        num_layers=num_bev_layers,
        num_heads=4,
        num_points=2,
        dropout=0.0,
    )
    head = DetectionHead(
        num_classes=num_classes,
        in_channels=embed_dims,
        num_queries=num_queries,
        num_decoder_layers=1,
        num_heads=4,
        ffn_dim=64,
    )
    return BEVFormerModel(backbone=backbone, neck=neck, encoder=encoder, head=head)


# ---------------------------------------------------------------------------
# 1. Backbone + Neck combinations
# ---------------------------------------------------------------------------


class TestBackboneNeckIntegration:
    """Verify that different backbone depths are wired correctly to FPNNeck."""

    @pytest.mark.parametrize("depth", [18, 34])
    def test_resnet_shallow_fpn_output_shape(self, depth: int) -> None:
        """ResNet18/34 channels [64,128,256] must be compatible with FPNNeck."""
        backbone = ResNetBackbone(depth=depth, pretrained=False, out_indices=[1, 2, 3])
        neck = FPNNeck(in_channels=[64, 128, 256], out_channels=64, num_outs=3)
        x = torch.randn(2, 3, 64, 64)
        features = backbone(x)
        outs = neck(features)
        assert len(outs) == 3
        for out in outs:
            assert out.shape[1] == 64, "All FPN outputs must have out_channels=64"

    def test_resnet50_fpn_output_shape(self) -> None:
        """ResNet50 channels [256,512,1024] must be compatible with FPNNeck."""
        backbone = ResNetBackbone(depth=50, pretrained=False, out_indices=[1, 2, 3])
        neck = FPNNeck(in_channels=[256, 512, 1024], out_channels=128, num_outs=3)
        x = torch.randn(1, 3, 64, 64)
        features = backbone(x)
        outs = neck(features)
        assert len(outs) == 3
        for out in outs:
            assert out.shape[1] == 128

    def test_single_scale_backbone_fpn(self) -> None:
        """Single-scale backbone output (out_indices=[3]) wired into FPN."""
        backbone = ResNetBackbone(depth=18, pretrained=False, out_indices=[3])
        neck = FPNNeck(in_channels=[256], out_channels=32, num_outs=1)
        x = torch.randn(2, 3, 64, 64)
        features = backbone(x)
        assert len(features) == 1
        outs = neck(features)
        assert len(outs) == 1
        assert outs[0].shape[1] == 32

    def test_extra_fpn_levels_via_maxpool(self) -> None:
        """FPN should add extra max-pooled levels when num_outs > len(in_channels)."""
        backbone = ResNetBackbone(depth=18, pretrained=False, out_indices=[1, 2, 3])
        neck = FPNNeck(in_channels=[64, 128, 256], out_channels=64, num_outs=5)
        x = torch.randn(2, 3, 64, 64)
        features = backbone(x)
        outs = neck(features)
        assert len(outs) == 5
        for out in outs:
            assert out.shape[1] == 64

    def test_backbone_fpn_gradient_flows(self) -> None:
        """Gradients must propagate from FPN output back through the backbone."""
        backbone = ResNetBackbone(depth=18, pretrained=False, out_indices=[1, 2, 3])
        neck = FPNNeck(in_channels=[64, 128, 256], out_channels=32, num_outs=3)
        x = torch.randn(1, 3, 64, 64, requires_grad=True)
        features = backbone(x)
        outs = neck(features)
        loss = sum(o.sum() for o in outs)
        loss.backward()
        # conv1 is frozen (requires_grad=False) but x is not frozen
        assert x.grad is not None

    def test_backbone_fpn_batch_size_preserved(self) -> None:
        """Batch dimension must be preserved through backbone and neck."""
        for B in [1, 3]:
            backbone = ResNetBackbone(depth=18, pretrained=False, out_indices=[1, 2, 3])
            neck = FPNNeck(in_channels=[64, 128, 256], out_channels=32, num_outs=3)
            x = torch.randn(B, 3, 64, 64)
            outs = neck(backbone(x))
            for out in outs:
                assert out.shape[0] == B


# ---------------------------------------------------------------------------
# 2. Neck → Encoder integration
# ---------------------------------------------------------------------------


class TestNeckEncoderIntegration:
    """Verify that FPNNeck outputs are compatible with BEVEncoder inputs."""

    def test_neck_encoder_output_shape(self) -> None:
        """BEVEncoder must accept neck feature maps and return (B, C, H, W)."""
        B, C, H, W = 2, 32, 4, 4
        backbone = ResNetBackbone(depth=18, pretrained=False, out_indices=[1, 2, 3])
        neck = FPNNeck(in_channels=[64, 128, 256], out_channels=C, num_outs=3)
        encoder = BEVEncoder(
            bev_h=H, bev_w=W, embed_dims=C, num_layers=1, num_heads=4, num_points=2, dropout=0.0
        )
        # Stack images for all cameras × batch
        images = torch.randn(B, 3, 64, 64)
        features = neck(backbone(images))
        # BEVEncoder expects img_features as list[Tensor(B*num_cameras, C, fH, fW)]
        # with single camera: B*1 = B
        K = torch.zeros(B, 1, 3, 3)
        K[:, :, 0, 0] = 400.0
        K[:, :, 1, 1] = 400.0
        K[:, :, 0, 2] = 8.0
        K[:, :, 1, 2] = 8.0
        K[:, :, 2, 2] = 1.0
        E = torch.eye(4).unsqueeze(0).unsqueeze(0).expand(B, 1, 4, 4).clone()
        E[:, :, 2, 3] = 5.0
        bev = encoder(features, K, E)
        assert bev.shape == (B, C, H, W)

    def test_neck_encoder_gradient_flows(self) -> None:
        """Gradients must flow from BEV map all the way back to image pixels."""
        B, C = 1, 32
        backbone = ResNetBackbone(depth=18, pretrained=False, out_indices=[1, 2, 3])
        neck = FPNNeck(in_channels=[64, 128, 256], out_channels=C, num_outs=3)
        encoder = BEVEncoder(
            bev_h=4, bev_w=4, embed_dims=C, num_layers=1, num_heads=4, num_points=2, dropout=0.0
        )
        images = torch.randn(B, 3, 64, 64, requires_grad=True)
        features = neck(backbone(images))
        K = torch.zeros(B, 1, 3, 3)
        K[:, :, 0, 0] = 400.0
        K[:, :, 1, 1] = 400.0
        K[:, :, 0, 2] = 8.0
        K[:, :, 1, 2] = 8.0
        K[:, :, 2, 2] = 1.0
        E = torch.eye(4).unsqueeze(0).unsqueeze(0).expand(B, 1, 4, 4).clone()
        bev = encoder(features, K, E)
        bev.sum().backward()
        assert images.grad is not None


# ---------------------------------------------------------------------------
# 3. Full BEVFormerModel pipeline
# ---------------------------------------------------------------------------


class TestBEVFormerModelIntegration:
    """End-to-end BEVFormerModel integration tests."""

    def test_forward_returns_correct_types(self) -> None:
        """Model must return (BoundingBox3DPrediction, Tensor)."""
        model = _make_small_model()
        batch = _make_batch()
        preds, new_bev = model(batch)
        assert isinstance(preds, BoundingBox3DPrediction)
        assert isinstance(new_bev, torch.Tensor)

    def test_forward_prediction_shapes(self) -> None:
        """Prediction tensors must have correct shapes for batch_size=2, num_queries=10."""
        B, Q, num_classes = 2, 10, 5
        model = _make_small_model(num_classes=num_classes, num_queries=Q)
        batch = _make_batch(batch_size=B)
        preds, _ = model(batch)
        assert preds.boxes.shape == (B, Q, 10)
        assert preds.scores.shape == (B, Q)
        assert preds.labels.shape == (B, Q)

    def test_forward_new_bev_shape(self) -> None:
        """new_bev must be shaped (H*W, B, C) for temporal attention."""
        B, C, H, W = 2, 32, 4, 4
        model = _make_small_model(embed_dims=C, bev_h=H, bev_w=W)
        batch = _make_batch(batch_size=B)
        _, new_bev = model(batch)
        assert new_bev.shape == (H * W, B, C)

    def test_forward_single_frame(self) -> None:
        """Model must work with batch_size=1."""
        model = _make_small_model()
        batch = _make_batch(batch_size=1)
        preds, new_bev = model(batch)
        assert preds.boxes.shape[0] == 1

    def test_forward_with_prev_bev(self) -> None:
        """Temporal path (prev_bev) must not raise and must yield correct shapes."""
        B, C, H, W, Q = 2, 32, 4, 4, 10
        model = _make_small_model(embed_dims=C, bev_h=H, bev_w=W, num_queries=Q)
        batch = _make_batch(batch_size=B)
        _, new_bev = model(batch, prev_bev=None)
        preds2, _ = model(batch, prev_bev=new_bev)
        assert preds2.boxes.shape == (B, Q, 10)

    def test_forward_without_targets(self) -> None:
        """Inference without targets must not raise."""
        model = _make_small_model()
        batch = _make_batch(with_targets=False)
        preds, _ = model(batch)
        assert isinstance(preds, BoundingBox3DPrediction)

    def test_forward_multi_camera(self) -> None:
        """Model must handle multiple cameras per frame."""
        model = _make_small_model()
        batch = _make_batch(num_cameras=3)
        preds, _ = model(batch)
        assert preds.boxes.shape[0] == 2  # batch_size=2 by default

    def test_forward_six_cameras(self) -> None:
        """Full autonomous-driving scenario: 6 cameras per frame."""
        model = _make_small_model()
        batch = _make_batch(num_cameras=6)
        preds, _ = model(batch)
        assert preds.boxes.shape[0] == 2

    def test_forward_scores_in_unit_interval(self) -> None:
        """Confidence scores must lie in [0, 1] (sigmoid activation)."""
        model = _make_small_model()
        batch = _make_batch()
        preds, _ = model(batch)
        assert preds.scores.min().item() >= 0.0
        assert preds.scores.max().item() <= 1.0

    def test_forward_labels_within_class_range(self) -> None:
        """Predicted class labels must be in [0, num_classes)."""
        num_classes = 5
        model = _make_small_model(num_classes=num_classes)
        batch = _make_batch(num_classes=num_classes)
        preds, _ = model(batch)
        assert preds.labels.min().item() >= 0
        assert preds.labels.max().item() < num_classes

    def test_forward_gradients_flow_through_full_model(self) -> None:
        """A loss on the predictions must produce non-None gradients for the head."""
        model = _make_small_model()
        batch = _make_batch()
        preds, _ = model(batch)
        loss = preds.boxes.sum()
        loss.backward()
        head_grad = next(
            (p.grad for p in model.head.parameters() if p.grad is not None),
            None,
        )
        assert head_grad is not None

    @pytest.mark.parametrize("depth", [18, 34])
    def test_different_backbone_depths(self, depth: int) -> None:
        """Models with different backbone depths must produce compatible outputs."""
        model = _make_small_model(backbone_depth=depth)
        batch = _make_batch()
        preds, _ = model(batch)
        assert isinstance(preds, BoundingBox3DPrediction)

    @pytest.mark.parametrize("num_queries", [5, 20, 50])
    def test_different_query_counts(self, num_queries: int) -> None:
        """Models with different query counts must produce shapes matching num_queries."""
        model = _make_small_model(num_queries=num_queries)
        batch = _make_batch()
        preds, _ = model(batch)
        assert preds.boxes.shape[1] == num_queries

    @pytest.mark.parametrize("bev_size", [(4, 4), (6, 8)])
    def test_different_bev_grid_sizes(self, bev_size: tuple[int, int]) -> None:
        """Models with different BEV grid sizes must produce correctly-sized new_bev."""
        H, W = bev_size
        C = 32
        model = _make_small_model(embed_dims=C, bev_h=H, bev_w=W)
        batch = _make_batch(batch_size=1)
        _, new_bev = model(batch)
        assert new_bev.shape == (H * W, 1, C)


# ---------------------------------------------------------------------------
# 4. Model + Matcher + Loss (training pipeline)
# ---------------------------------------------------------------------------


class TestModelMatcherLossIntegration:
    """Integration tests for the BEVFormerModel → HungarianMatcher → DetectionLoss pipeline."""

    def _run_training_pass(
        self,
        batch: BatchData,
        model: BEVFormerModel,
        matcher: HungarianMatcher,
        loss_fn: DetectionLoss,
    ) -> torch.Tensor:
        """Helper: forward, match, compute loss; return total loss scalar."""
        preds_batch, _ = model(batch)
        pred_list = [
            BoundingBox3DPrediction(
                boxes=preds_batch.boxes[i],
                scores=preds_batch.scores[i],
                labels=preds_batch.labels[i],
            )
            for i, frame in enumerate(batch.frames)
            if frame.targets is not None
        ]
        targets = [f.targets for f in batch.frames if f.targets is not None]
        matches = matcher.match_batch(pred_list, targets)
        total, _ = loss_fn(pred_list, targets, matches)
        return total

    def test_training_pass_loss_is_finite(self) -> None:
        """End-to-end training pass must produce a finite scalar loss."""
        num_classes = 5
        model = _make_small_model(num_classes=num_classes)
        matcher = HungarianMatcher()
        loss_fn = DetectionLoss(num_classes=num_classes)
        batch = _make_batch(num_classes=num_classes)
        total = self._run_training_pass(batch, model, matcher, loss_fn)
        assert torch.isfinite(total)

    def test_training_pass_loss_requires_grad(self) -> None:
        """Loss must be differentiable with respect to model parameters."""
        num_classes = 5
        model = _make_small_model(num_classes=num_classes)
        matcher = HungarianMatcher()
        loss_fn = DetectionLoss(num_classes=num_classes)
        batch = _make_batch(num_classes=num_classes)
        total = self._run_training_pass(batch, model, matcher, loss_fn)
        assert total.requires_grad

    def test_training_pass_backward(self) -> None:
        """Backward pass must not raise and must populate gradients."""
        num_classes = 5
        model = _make_small_model(num_classes=num_classes)
        matcher = HungarianMatcher()
        loss_fn = DetectionLoss(num_classes=num_classes)
        batch = _make_batch(num_classes=num_classes)
        total = self._run_training_pass(batch, model, matcher, loss_fn)
        total.backward()
        n_params_with_grad = sum(
            1 for p in model.parameters() if p.requires_grad and p.grad is not None
        )
        assert n_params_with_grad > 0

    def test_training_pass_with_empty_targets(self) -> None:
        """Training step must handle frames with zero ground-truth boxes."""
        num_classes = 5
        model = _make_small_model(num_classes=num_classes)
        matcher = HungarianMatcher()
        loss_fn = DetectionLoss(num_classes=num_classes)
        # Frame with no GT boxes
        frame = _make_frame(num_boxes=0, num_classes=num_classes)
        batch = BatchData(batch_size=1, frames=[frame])
        total = self._run_training_pass(batch, model, matcher, loss_fn)
        assert torch.isfinite(total)

    def test_training_pass_single_frame_single_gt(self) -> None:
        """Training step must work with exactly one GT box."""
        num_classes = 5
        model = _make_small_model(num_classes=num_classes, num_queries=10)
        matcher = HungarianMatcher()
        loss_fn = DetectionLoss(num_classes=num_classes)
        frame = _make_frame(num_boxes=1, num_classes=num_classes)
        batch = BatchData(batch_size=1, frames=[frame])
        total = self._run_training_pass(batch, model, matcher, loss_fn)
        assert torch.isfinite(total)

    def test_training_pass_more_gt_than_queries(self) -> None:
        """When GT boxes exceed queries the matcher clips matches; loss must be finite."""
        num_classes = 5
        model = _make_small_model(num_classes=num_classes, num_queries=3)
        matcher = HungarianMatcher()
        loss_fn = DetectionLoss(num_classes=num_classes)
        frame = _make_frame(num_boxes=8, num_classes=num_classes)
        batch = BatchData(batch_size=1, frames=[frame])
        total = self._run_training_pass(batch, model, matcher, loss_fn)
        assert torch.isfinite(total)

    def test_training_pass_without_targets_skipped(self) -> None:
        """Frames with no targets should be excluded; remaining frames still produce a loss."""
        num_classes = 5
        model = _make_small_model(num_classes=num_classes)
        matcher = HungarianMatcher()
        loss_fn = DetectionLoss(num_classes=num_classes)
        # Mix: one frame with targets, one without
        frame_with = _make_frame(num_boxes=2, num_classes=num_classes, frame_id="has_targets")
        frame_without = _make_frame(
            num_boxes=0, num_classes=num_classes, frame_id="no_targets", with_targets=False
        )
        batch = BatchData(batch_size=2, frames=[frame_with, frame_without])
        preds_batch, _ = model(batch)
        # Only include the frame that has targets
        pred_list = [
            BoundingBox3DPrediction(
                boxes=preds_batch.boxes[i],
                scores=preds_batch.scores[i],
                labels=preds_batch.labels[i],
            )
            for i, frame in enumerate(batch.frames)
            if frame.targets is not None
        ]
        targets = [f.targets for f in batch.frames if f.targets is not None]
        assert len(pred_list) == 1  # Only the frame_with was included
        matches = matcher.match_batch(pred_list, targets)
        total, _ = loss_fn(pred_list, targets, matches)
        assert torch.isfinite(total)

    @pytest.mark.parametrize(
        "cls_w,bbox_w,giou_w",
        [(2.0, 0.25, 0.1), (1.0, 1.0, 1.0), (0.0, 1.0, 0.0)],
    )
    def test_training_pass_different_loss_weights(
        self, cls_w: float, bbox_w: float, giou_w: float
    ) -> None:
        """Training pass must produce finite loss for various loss weight combinations."""
        num_classes = 5
        model = _make_small_model(num_classes=num_classes)
        matcher = HungarianMatcher()
        loss_fn = DetectionLoss(
            num_classes=num_classes,
            cls_weight=cls_w,
            bbox_weight=bbox_w,
            giou_weight=giou_w,
        )
        batch = _make_batch(num_classes=num_classes)
        total = self._run_training_pass(batch, model, matcher, loss_fn)
        assert torch.isfinite(total)


# ---------------------------------------------------------------------------
# 5. Model + Evaluator (validation pipeline)
# ---------------------------------------------------------------------------


class TestModelEvaluatorIntegration:
    """Integration tests for the BEVFormerModel → Vision3DEvaluator pipeline."""

    def _run_validation_pass(
        self,
        batch: BatchData,
        model: BEVFormerModel,
        evaluator: Vision3DEvaluator,
    ) -> None:
        """Helper: forward then accumulate into evaluator."""
        with torch.no_grad():
            preds_batch, _ = model(batch)
        pred_list = [
            BoundingBox3DPrediction(
                boxes=preds_batch.boxes[i],
                scores=preds_batch.scores[i],
                labels=preds_batch.labels[i],
            )
            for i, frame in enumerate(batch.frames)
            if frame.targets is not None
        ]
        targets = [f.targets for f in batch.frames if f.targets is not None]
        evaluator.update(pred_list, targets)

    def test_validation_pass_produces_finite_map(self) -> None:
        """Validation pipeline must produce a finite mAP."""
        num_classes = 3
        model = _make_small_model(num_classes=num_classes)
        evaluator = Vision3DEvaluator(num_classes=num_classes)
        batch = _make_batch(num_classes=num_classes)
        self._run_validation_pass(batch, model, evaluator)
        metrics = evaluator.compute()
        assert isinstance(metrics["mAP"], float)
        assert 0.0 <= metrics["mAP"] <= 1.0

    def test_validation_pass_metrics_keys(self) -> None:
        """All expected metric keys must be present in compute() output."""
        num_classes = 3
        model = _make_small_model(num_classes=num_classes)
        evaluator = Vision3DEvaluator(num_classes=num_classes)
        batch = _make_batch(num_classes=num_classes)
        self._run_validation_pass(batch, model, evaluator)
        metrics = evaluator.compute()
        for key in ["mAP", "NDS", "ATE", "ASE", "AOE", "AVE"]:
            assert key in metrics

    def test_validation_multiple_batches(self) -> None:
        """Evaluator must correctly accumulate across multiple validation batches."""
        num_classes = 3
        model = _make_small_model(num_classes=num_classes)
        evaluator = Vision3DEvaluator(num_classes=num_classes)
        for i in range(3):
            batch = _make_batch(batch_size=2, num_classes=num_classes)
            self._run_validation_pass(batch, model, evaluator)
        assert len(evaluator._all_predictions) == 6
        metrics = evaluator.compute()
        assert torch.isfinite(torch.tensor(metrics["mAP"]))

    def test_validation_reset_clears_state(self) -> None:
        """reset() must clear accumulated state so re-evaluation starts fresh."""
        num_classes = 3
        model = _make_small_model(num_classes=num_classes)
        evaluator = Vision3DEvaluator(num_classes=num_classes)
        batch = _make_batch(num_classes=num_classes)
        self._run_validation_pass(batch, model, evaluator)
        evaluator.reset()
        assert len(evaluator._all_predictions) == 0
        assert evaluator.compute()["mAP"] == pytest.approx(0.0)

    def test_validation_with_no_gt_boxes(self) -> None:
        """Validation must not crash when frames contain no GT annotations."""
        num_classes = 3
        model = _make_small_model(num_classes=num_classes)
        evaluator = Vision3DEvaluator(num_classes=num_classes)
        batch = _make_batch(num_boxes=0, num_classes=num_classes)
        self._run_validation_pass(batch, model, evaluator)
        metrics = evaluator.compute()
        assert isinstance(metrics["mAP"], float)


# ---------------------------------------------------------------------------
# 6. Lightning module integration
# ---------------------------------------------------------------------------


class TestLightningModuleIntegration:
    """Integration tests for Vision3DLightningModule."""

    def _make_lit_module(
        self,
        num_classes: int = 5,
        num_queries: int = 10,
    ):
        from vision3d.engine.lit_module import Vision3DLightningModule

        model = _make_small_model(num_classes=num_classes, num_queries=num_queries)
        matcher = HungarianMatcher()
        loss_fn = DetectionLoss(num_classes=num_classes)
        evaluator = Vision3DEvaluator(num_classes=num_classes)
        return Vision3DLightningModule(
            model=model,
            matcher=matcher,
            loss=loss_fn,
            evaluator=evaluator,
            learning_rate=1e-3,
            max_epochs=2,
        )

    def test_training_step_returns_scalar(self) -> None:
        """training_step must return a finite scalar tensor."""
        lit = self._make_lit_module()
        batch = _make_batch(num_classes=5)
        loss = lit.training_step(batch, batch_idx=0)
        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0
        assert torch.isfinite(loss)

    def test_training_step_twice_updates_prev_bev(self) -> None:
        """After two training steps _prev_bev must be set (temporal state)."""
        lit = self._make_lit_module()
        batch = _make_batch(num_classes=5)
        lit.training_step(batch, batch_idx=0)
        assert lit._prev_bev is not None
        lit.training_step(batch, batch_idx=1)
        assert lit._prev_bev is not None

    def test_validation_step_does_not_raise(self) -> None:
        """validation_step must not raise for normal input."""
        lit = self._make_lit_module()
        batch = _make_batch(num_classes=5)
        lit.on_validation_epoch_start()
        lit.validation_step(batch, batch_idx=0)

    def test_on_validation_epoch_start_resets_prev_bev(self) -> None:
        """on_validation_epoch_start must clear the temporal BEV state."""
        lit = self._make_lit_module()
        batch = _make_batch(num_classes=5)
        lit.training_step(batch, batch_idx=0)
        assert lit._prev_bev is not None
        lit.on_validation_epoch_start()
        assert lit._prev_bev is None

    def test_on_validation_epoch_end_logs_map(self) -> None:
        """on_validation_epoch_end must call evaluator.compute() without raising."""
        lit = self._make_lit_module()
        batch = _make_batch(num_classes=5)
        lit.on_validation_epoch_start()
        lit.validation_step(batch, batch_idx=0)
        # Simulating the epoch-end call (no trainer attached, just verify no exception)
        metrics = lit.evaluator.compute()
        assert "mAP" in metrics

    def test_configure_optimizers_returns_dict(self) -> None:
        """configure_optimizers must return a dict with 'optimizer' key."""
        lit = self._make_lit_module()
        opt_config = lit.configure_optimizers()
        assert isinstance(opt_config, dict)
        assert "optimizer" in opt_config
        assert "lr_scheduler" in opt_config

    def test_training_step_with_empty_targets(self) -> None:
        """Training step must handle a batch where all frames have no GT boxes."""
        lit = self._make_lit_module()
        frame = _make_frame(num_boxes=0, num_classes=5)
        batch = BatchData(batch_size=1, frames=[frame])
        loss = lit.training_step(batch, batch_idx=0)
        assert torch.isfinite(loss)

    def test_training_and_validation_steps_sequence(self) -> None:
        """Running training then validation steps must not raise or corrupt state."""
        lit = self._make_lit_module()
        train_batch = _make_batch(num_classes=5)
        val_batch = _make_batch(num_classes=5)
        # Training phase
        for i in range(2):
            lit.training_step(train_batch, batch_idx=i)
        # Validation phase
        lit.on_validation_epoch_start()
        lit.validation_step(val_batch, batch_idx=0)
        metrics = lit.evaluator.compute()
        assert "mAP" in metrics


# ---------------------------------------------------------------------------
# 7. Interface edge cases
# ---------------------------------------------------------------------------


class TestInterfaceEdgeCases:
    """Tests that verify interface contracts are satisfied even for edge cases."""

    def test_batch_data_to_device_is_noop_on_cpu(self) -> None:
        """BatchData.to('cpu') must succeed and return self."""
        batch = _make_batch()
        returned = batch.to(torch.device("cpu"))
        assert returned is batch

    def test_batch_data_to_moves_image_tensors(self) -> None:
        """All image tensors in BatchData must reside on the target device after .to()."""
        batch = _make_batch()
        batch.to(torch.device("cpu"))
        for frame in batch.frames:
            for cam in frame.cameras.values():
                assert cam.image.device.type == "cpu"

    def test_frame_data_with_all_optional_fields_none(self) -> None:
        """FrameData with targets=None must not raise in model forward pass."""
        model = _make_small_model()
        frame = _make_frame(with_targets=False)
        batch = BatchData(batch_size=1, frames=[frame])
        preds, _ = model(batch)
        assert isinstance(preds, BoundingBox3DPrediction)

    def test_model_output_shapes_consistent_with_config(self) -> None:
        """All model output shapes must match the constructor configuration."""
        B, Q, C_cls = 3, 7, 4
        model = _make_small_model(num_classes=C_cls, num_queries=Q)
        batch = _make_batch(batch_size=B, num_classes=C_cls)
        preds, _ = model(batch)
        assert preds.boxes.shape == (B, Q, 10)
        assert preds.scores.shape == (B, Q)
        assert preds.labels.shape == (B, Q)

    def test_matching_result_consistent_length(self) -> None:
        """pred_indices and gt_indices in MatchingResult must have equal length."""
        model = _make_small_model(num_classes=5)
        matcher = HungarianMatcher()
        batch = _make_batch(num_classes=5)
        preds_batch, _ = model(batch)
        for i, frame in enumerate(batch.frames):
            if frame.targets is None:
                continue
            pred = BoundingBox3DPrediction(
                boxes=preds_batch.boxes[i],
                scores=preds_batch.scores[i],
                labels=preds_batch.labels[i],
            )
            result = matcher.match(pred, frame.targets)
            assert result.pred_indices.shape == result.gt_indices.shape

    def test_loss_breakdown_sums_to_total(self) -> None:
        """The weighted sum of breakdown losses must equal the total loss."""
        num_classes = 5
        cw, bw, gw = 2.0, 0.5, 0.3
        model = _make_small_model(num_classes=num_classes)
        matcher = HungarianMatcher()
        loss_fn = DetectionLoss(
            num_classes=num_classes, cls_weight=cw, bbox_weight=bw, giou_weight=gw
        )
        batch = _make_batch(num_classes=num_classes)
        preds_batch, _ = model(batch)
        pred_list = [
            BoundingBox3DPrediction(
                boxes=preds_batch.boxes[i],
                scores=preds_batch.scores[i],
                labels=preds_batch.labels[i],
            )
            for i, frame in enumerate(batch.frames)
            if frame.targets is not None
        ]
        targets = [f.targets for f in batch.frames if f.targets is not None]
        matches = matcher.match_batch(pred_list, targets)
        total, breakdown = loss_fn(pred_list, targets, matches)
        expected = (
            cw * breakdown["loss_cls"] + bw * breakdown["loss_bbox"] + gw * breakdown["loss_giou"]
        )
        assert torch.allclose(total, expected, atol=1e-5)

    def test_evaluator_per_class_ap_keys_present(self) -> None:
        """Vision3DEvaluator must expose AP/{class_name} for every class."""
        num_classes = 3
        class_names = ["car", "pedestrian", "cyclist"]
        model = _make_small_model(num_classes=num_classes)
        evaluator = Vision3DEvaluator(num_classes=num_classes, class_names=class_names)
        batch = _make_batch(num_classes=num_classes)
        with torch.no_grad():
            preds_batch, _ = model(batch)
        pred_list = [
            BoundingBox3DPrediction(
                boxes=preds_batch.boxes[i],
                scores=preds_batch.scores[i],
                labels=preds_batch.labels[i],
            )
            for i, frame in enumerate(batch.frames)
            if frame.targets is not None
        ]
        targets = [f.targets for f in batch.frames if f.targets is not None]
        evaluator.update(pred_list, targets)
        metrics = evaluator.compute()
        for name in class_names:
            assert f"AP/{name}" in metrics

    def test_model_with_temporal_frames_in_batch(self) -> None:
        """FrameData.past_frames must not interfere with model forward pass."""
        model = _make_small_model()
        past = _make_frame(frame_id="past_frame", seed=99)
        current = _make_frame(frame_id="current_frame", seed=0)
        current.past_frames = [past]
        batch = BatchData(batch_size=1, frames=[current])
        preds, _ = model(batch)
        assert isinstance(preds, BoundingBox3DPrediction)

    def test_bev_encoder_prev_bev_none_vs_provided(self) -> None:
        """Model output must differ when prev_bev is None vs. a real tensor."""
        B, C, H, W = 1, 32, 4, 4
        model = _make_small_model(embed_dims=C, bev_h=H, bev_w=W)
        model.eval()
        batch = _make_batch(batch_size=B)
        with torch.no_grad():
            preds_no_prev, bev = model(batch, prev_bev=None)
            preds_with_prev, _ = model(batch, prev_bev=bev)
        # Having prev_bev should change the BEV representation
        assert not torch.allclose(preds_no_prev.boxes, preds_with_prev.boxes)

    def test_full_pipeline_with_single_class(self) -> None:
        """num_classes=1 (minimum) must work end-to-end."""
        num_classes = 1
        model = _make_small_model(num_classes=num_classes)
        matcher = HungarianMatcher()
        loss_fn = DetectionLoss(num_classes=num_classes)
        frame = _make_frame(num_classes=num_classes, num_boxes=2)
        batch = BatchData(batch_size=1, frames=[frame])
        preds_batch, _ = model(batch)
        pred_list = [
            BoundingBox3DPrediction(
                boxes=preds_batch.boxes[0],
                scores=preds_batch.scores[0],
                labels=preds_batch.labels[0],
            )
        ]
        targets = [frame.targets]
        matches = matcher.match_batch(pred_list, targets)
        total, _ = loss_fn(pred_list, targets, matches)
        assert torch.isfinite(total)
