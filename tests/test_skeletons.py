"""
Skeleton tests for Vision3D.

These tests verify the class interface (i.e., that the expected classes exist
and expose the correct methods/signatures). They do NOT test functionality
since the classes contain only skeleton stubs.
"""


class TestDataLayer:
    """Interface tests for the data layer classes."""

    def test_vision3d_dataset_exists(self):
        """Vision3DDataset should be importable and expose the correct interface."""
        from torch.utils.data import Dataset

        from vision3d.data.dataset import Vision3DDataset

        assert issubclass(Vision3DDataset, Dataset)
        assert hasattr(Vision3DDataset, "__len__")
        assert hasattr(Vision3DDataset, "__getitem__")
        assert hasattr(Vision3DDataset, "collate_fn")

    def test_json_loader_exists(self):
        """JsonLoader should be importable and expose a load method."""
        from vision3d.data.loaders import JsonLoader

        assert hasattr(JsonLoader, "load")

    def test_image_loader_exists(self):
        """ImageLoader should be importable and expose a load method."""
        from vision3d.data.loaders import ImageLoader

        assert hasattr(ImageLoader, "load")

    def test_box_filter_exists(self):
        """BoxFilter should be importable and expose a filter method."""
        from vision3d.data.filters import BoxFilter

        assert hasattr(BoxFilter, "filter")

    def test_image_filter_exists(self):
        """ImageFilter should be importable and expose a should_keep method."""
        from vision3d.data.filters import ImageFilter

        assert hasattr(ImageFilter, "should_keep")

    def test_data_augmenter_exists(self):
        """DataAugmenter should be importable and be callable."""
        from vision3d.data.augmentations import DataAugmenter

        assert callable(DataAugmenter)


class TestModelArchitecture:
    """Interface tests for the model architecture classes."""

    def test_resnet_backbone_exists(self):
        """ResNetBackbone should be importable and be an nn.Module subclass."""
        import torch.nn as nn

        from vision3d.models.backbones.resnet import ResNetBackbone

        assert issubclass(ResNetBackbone, nn.Module)
        assert hasattr(ResNetBackbone, "forward")

    def test_fpn_neck_exists(self):
        """FPNNeck should be importable and be an nn.Module subclass."""
        import torch.nn as nn

        from vision3d.models.necks.fpn import FPNNeck

        assert issubclass(FPNNeck, nn.Module)
        assert hasattr(FPNNeck, "forward")

    def test_bev_encoder_exists(self):
        """BEVEncoder should be importable and be an nn.Module subclass."""
        import torch.nn as nn

        from vision3d.models.encoders.bev_encoder import BEVEncoder

        assert issubclass(BEVEncoder, nn.Module)
        assert hasattr(BEVEncoder, "forward")

    def test_detection_head_exists(self):
        """DetectionHead should be importable and be an nn.Module subclass."""
        import torch.nn as nn

        from vision3d.models.heads.detection_head import DetectionHead

        assert issubclass(DetectionHead, nn.Module)
        assert hasattr(DetectionHead, "forward")

    def test_bevformer_model_exists(self):
        """BEVFormerModel should be importable and be an nn.Module subclass."""
        import torch.nn as nn

        from vision3d.models.bevformer import BEVFormerModel

        assert issubclass(BEVFormerModel, nn.Module)
        assert hasattr(BEVFormerModel, "forward")


class TestCoreLogic:
    """Interface tests for the core logic classes."""

    def test_hungarian_matcher_exists(self):
        """HungarianMatcher should be importable and expose match / match_batch."""
        from vision3d.core.matchers import HungarianMatcher

        assert hasattr(HungarianMatcher, "match")
        assert hasattr(HungarianMatcher, "match_batch")

    def test_detection_loss_exists(self):
        """DetectionLoss should be importable and be an nn.Module subclass."""
        import torch.nn as nn

        from vision3d.core.losses import DetectionLoss

        assert issubclass(DetectionLoss, nn.Module)
        assert hasattr(DetectionLoss, "forward")

    def test_vision3d_evaluator_exists(self):
        """Vision3DEvaluator should be importable and expose reset/update/compute."""
        from vision3d.core.evaluators import Vision3DEvaluator

        assert hasattr(Vision3DEvaluator, "reset")
        assert hasattr(Vision3DEvaluator, "update")
        assert hasattr(Vision3DEvaluator, "compute")


class TestEngineAndUtils:
    """Interface tests for the engine and utility classes."""

    def test_lightning_module_exists(self):
        """Vision3DLightningModule should subclass pl.LightningModule and accept a BEVFormerModel."""
        import inspect

        import pytorch_lightning as pl

        from vision3d.engine.lit_module import Vision3DLightningModule

        assert issubclass(Vision3DLightningModule, pl.LightningModule)
        assert hasattr(Vision3DLightningModule, "training_step")
        assert hasattr(Vision3DLightningModule, "validation_step")
        assert hasattr(Vision3DLightningModule, "configure_optimizers")

        # The __init__ should accept 'model' (BEVFormerModel), not backbone/neck/encoder/head
        sig = inspect.signature(Vision3DLightningModule.__init__)
        params = set(sig.parameters.keys())
        assert "model" in params
        assert "matcher" in params
        assert "loss" in params
        assert "evaluator" in params
        assert "backbone" not in params
        assert "neck" not in params
        assert "encoder" not in params
        assert "head" not in params

    def test_camera_projector_exists(self):
        """CameraProjector should be importable and expose a project method."""
        from vision3d.utils.geometry import CameraProjector

        assert hasattr(CameraProjector, "project")
        assert hasattr(CameraProjector, "quaternion_to_rotation_matrix")

    def test_foxglove_logger_exists(self):
        """FoxgloveMCAPLogger should be importable and subclass pl.Callback."""
        import pytorch_lightning as pl

        from vision3d.utils.foxglove import FoxgloveMCAPLogger

        assert issubclass(FoxgloveMCAPLogger, pl.Callback)
        assert hasattr(FoxgloveMCAPLogger, "on_validation_epoch_end")


class TestConfigSchema:
    """Interface tests for the config and data interface dataclasses."""

    def test_hydra_configs_importable(self):
        """All Hydra config dataclasses should be importable from config.schema."""
        from vision3d.config.schema import (
            BackboneConfig,
            BEVFormerModelConfig,
            EncoderConfig,
            EvaluatorConfig,
            HeadConfig,
            LitModuleConfig,
            LossConfig,
            MatcherConfig,
            NeckConfig,
        )

        # Verify _target_ fields are present on model-related configs
        assert BackboneConfig._target_ == "vision3d.models.backbones.ResNetBackbone"
        assert NeckConfig._target_ == "vision3d.models.necks.FPNNeck"
        assert EncoderConfig._target_ == "vision3d.models.encoders.BEVEncoder"
        assert HeadConfig._target_ == "vision3d.models.heads.DetectionHead"
        assert LossConfig._target_ == "vision3d.core.losses.DetectionLoss"
        assert MatcherConfig._target_ == "vision3d.core.matchers.HungarianMatcher"
        assert EvaluatorConfig._target_ == "vision3d.core.evaluators.Vision3DEvaluator"
        assert BEVFormerModelConfig._target_ == "vision3d.models.bevformer.BEVFormerModel"

        # LitModuleConfig should have a 'model' field (BEVFormerModelConfig),
        # not individual backbone/neck/encoder/head fields
        import dataclasses

        lit_fields = {f.name for f in dataclasses.fields(LitModuleConfig)}
        assert "model" in lit_fields
        assert "loss" in lit_fields
        assert "matcher" in lit_fields
        assert "evaluator" in lit_fields
        assert "backbone" not in lit_fields
        assert "neck" not in lit_fields
        assert "encoder" not in lit_fields
        assert "head" not in lit_fields

    def test_data_interface_dataclasses_importable(self):
        """All runtime data interface dataclasses should be importable."""
        # Verify key field names exist via dataclass fields introspection
        import dataclasses

        from vision3d.config.schema import (
            BatchData,
            FrameData,
        )

        frame_fields = {f.name for f in dataclasses.fields(FrameData)}
        assert "frame_id" in frame_fields
        assert "timestamp" in frame_fields
        assert "cameras" in frame_fields
        assert "targets" in frame_fields
        assert "predictions" in frame_fields
        assert "matches" in frame_fields
        assert "past_frames" in frame_fields

        batch_fields = {f.name for f in dataclasses.fields(BatchData)}
        assert "batch_size" in batch_fields
        assert "frames" in batch_fields


class TestScripts:
    """Interface tests for the standalone scripts."""

    def test_nuscenes_converter_exists(self):
        """NuScenesConverter should be importable and expose a convert method."""
        from scripts.convert_nuscenes import NuScenesConverter

        assert hasattr(NuScenesConverter, "convert")

    def test_dummy_generator_exists(self):
        """DummyDatasetGenerator should be importable and expose a generate method."""
        from scripts.generate_dummy_dataset import DummyDatasetGenerator

        assert hasattr(DummyDatasetGenerator, "generate")
