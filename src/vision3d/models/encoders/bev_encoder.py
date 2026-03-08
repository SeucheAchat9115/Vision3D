"""
BEVFormer encoder for Vision3D.

Provides `BEVEncoder`, which is the core of the BEVFormer architecture. It
maintains a learnable Bird's-Eye-View (BEV) query grid and iteratively refines
it through stacked encoder layers.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class BEVEncoderLayer(nn.Module):
    """A single BEVFormer encoder layer combining TSA and SCA."""

    def __init__(
        self,
        embed_dims: int = 256,
        num_heads: int = 8,
        num_points: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.num_points = num_points
        self.tsa = nn.MultiheadAttention(embed_dims, num_heads, dropout=dropout, batch_first=False)
        self.tsa_norm = nn.LayerNorm(embed_dims)
        self.sca_proj = nn.Linear(embed_dims, embed_dims)
        self.sca_norm = nn.LayerNorm(embed_dims)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dims, embed_dims * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dims * 4, embed_dims),
            nn.Dropout(dropout),
        )
        self.ffn_norm = nn.LayerNorm(embed_dims)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        bev_queries: torch.Tensor,
        prev_bev: torch.Tensor | None,
        image_features: list[torch.Tensor],
        reference_points: torch.Tensor,
        camera_intrinsics: torch.Tensor,
        camera_extrinsics: torch.Tensor,
        spatial_shapes: torch.Tensor,
    ) -> torch.Tensor:
        """Process BEV queries through TSA → SCA → FFN."""
        if prev_bev is not None:
            tsa_out, _ = self.tsa(bev_queries, prev_bev, prev_bev)
            bev_queries = self.tsa_norm(bev_queries + self.dropout(tsa_out))

        HW, B, C = bev_queries.shape
        num_cameras = camera_intrinsics.shape[1]

        R = camera_extrinsics[:, :, :3, :3]  # (B, C, 3, 3)
        t = camera_extrinsics[:, :, :3, 3]  # (B, C, 3)
        R_e2c = R.transpose(-1, -2)
        t_e2c = -(R_e2c @ t.unsqueeze(-1)).squeeze(-1)

        ref = reference_points.unsqueeze(0).unsqueeze(0)  # (1, 1, HW, 3)
        pts_cam = (R_e2c.unsqueeze(2) @ ref.unsqueeze(-1)).squeeze(-1) + t_e2c.unsqueeze(2)
        pts_img = (camera_intrinsics.unsqueeze(2) @ pts_cam.unsqueeze(-1)).squeeze(-1)
        uv = pts_img[:, :, :, :2] / (pts_img[:, :, :, 2:3].clamp(min=1e-5))

        feat = image_features[0]
        _, _, H_feat, W_feat = feat.shape
        u_norm = 2.0 * uv[:, :, :, 0] / W_feat - 1.0
        v_norm = 2.0 * uv[:, :, :, 1] / H_feat - 1.0
        grid = torch.stack([u_norm, v_norm], dim=-1)  # (B, num_cameras, HW, 2)

        feat_reshaped = feat.reshape(B, num_cameras, C, H_feat, W_feat)
        sampled = []
        for cam_i in range(num_cameras):
            g = grid[:, cam_i, :, :].unsqueeze(2)  # (B, HW, 1, 2)
            f = feat_reshaped[:, cam_i]  # (B, C, H, W)
            s = F.grid_sample(f, g, align_corners=False, padding_mode="zeros")  # (B, C, HW, 1)
            sampled.append(s.squeeze(-1))
        sampled_feat = torch.stack(sampled, dim=1).mean(dim=1)  # (B, C, HW)
        sampled_feat = sampled_feat.permute(2, 0, 1)  # (HW, B, C)

        sca_out = self.sca_proj(sampled_feat)
        bev_queries = self.sca_norm(bev_queries + self.dropout(sca_out))

        ffn_out = self.ffn(bev_queries)
        bev_queries = self.ffn_norm(bev_queries + ffn_out)
        return bev_queries


class BEVEncoder(nn.Module):
    """BEVFormer encoder: lifts multi-view image features to a BEV grid."""

    def __init__(
        self,
        bev_h: int = 200,
        bev_w: int = 200,
        embed_dims: int = 256,
        num_layers: int = 6,
        num_heads: int = 8,
        num_points: int = 4,
        dropout: float = 0.1,
        pc_range: list[float] | None = None,
    ) -> None:
        super().__init__()
        self.bev_h = bev_h
        self.bev_w = bev_w
        self.embed_dims = embed_dims
        self.num_layers = num_layers
        if pc_range is None:
            pc_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
        self.pc_range = pc_range
        self.bev_embedding = nn.Embedding(bev_h * bev_w, embed_dims)
        self.bev_pos = nn.Embedding(bev_h * bev_w, embed_dims)
        xs = torch.linspace(pc_range[0], pc_range[3], bev_w)
        ys = torch.linspace(pc_range[1], pc_range[4], bev_h)
        z_center = (pc_range[2] + pc_range[5]) / 2.0
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
        ref_pts = torch.stack(
            [grid_x.flatten(), grid_y.flatten(), torch.full((bev_h * bev_w,), z_center)], dim=-1
        )
        self.register_buffer("reference_points", ref_pts)
        self.layers = nn.ModuleList(
            [BEVEncoderLayer(embed_dims, num_heads, num_points, dropout) for _ in range(num_layers)]
        )

    def forward(
        self,
        image_features: list[torch.Tensor],
        camera_intrinsics: torch.Tensor,
        camera_extrinsics: torch.Tensor,
        prev_bev: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute the BEV feature map from multi-view image features."""
        B = camera_intrinsics.shape[0]
        HW = self.bev_h * self.bev_w
        indices = torch.arange(HW, device=camera_intrinsics.device)
        bev_queries = self.bev_embedding(indices) + self.bev_pos(indices)
        bev_queries = bev_queries.unsqueeze(1).expand(-1, B, -1)  # (HW, B, C)
        spatial_shapes = torch.tensor(
            [[f.shape[2], f.shape[3]] for f in image_features],
            dtype=torch.long,
            device=camera_intrinsics.device,
        )
        ref: torch.Tensor = self.reference_points  # type: ignore[assignment]
        for layer in self.layers:
            bev_queries = layer(
                bev_queries,
                prev_bev,
                image_features,
                ref,
                camera_intrinsics,
                camera_extrinsics,
                spatial_shapes,
            )
        bev_map: torch.Tensor = bev_queries.permute(1, 2, 0).reshape(
            B, self.embed_dims, self.bev_h, self.bev_w
        )
        return bev_map
