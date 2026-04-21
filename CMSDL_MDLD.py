import math
from typing import Optional, List, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBNReLU(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, k: int = 3, p: int = 1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=k, padding=p, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class Encoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        patch_size: int,
        use_diff_attn: bool = False,
        attn_num_heads: int = 8,
        attn_scale: float = 0.5,
        attn_chunk_size: int = 1,
    ):
        super().__init__()

        c1 = out_channels // 2
        c2 = out_channels // 2
        c3 = out_channels
        c4 = out_channels

        self.conv1 = ConvBNReLU(in_channels, c1, 3, 1)
        self.conv2 = ConvBNReLU(c1, c2, 3, 1)
        self.conv3 = ConvBNReLU(c2, c3, 3, 1)
        self.conv4 = ConvBNReLU(c3, c4, 3, 1)

        self.out_channels_list = [c1, c2, c3, c4]

    def forward(self, x: torch.Tensor, return_features: bool = False):
        f1 = self.conv1(x)
        f2 = self.conv2(f1)
        f3 = self.conv3(f2)
        f4 = self.conv4(f3)

        if return_features:
            return f4, [f1, f2, f3, f4]
        return f4


class DINOHead(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 512,
        bottleneck_dim: int = 256,
        out_dim: int = 256,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.bottleneck_dim = bottleneck_dim
        self.out_dim = out_dim

        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.act1 = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, bottleneck_dim)
        self.act2 = nn.GELU()

        self.last_layer = nn.utils.weight_norm(
            nn.Linear(bottleneck_dim, out_dim, bias=False)
        )
        self.last_layer.weight_g.data.fill_(1.0)
        self.last_layer.weight_g.requires_grad = False

    def _mlp(self, x_flat: torch.Tensor) -> torch.Tensor:
        z = self.act1(self.fc1(x_flat))
        z = self.act2(self.fc2(z))
        z = F.normalize(z, dim=-1)
        return z

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 4:
            x = x.mean(dim=[2, 3])
        z = self._mlp(x)
        out = self.last_layer(z)
        return out

    @torch.no_grad()
    def forward_map(self, x: torch.Tensor) -> torch.Tensor:
        assert x.dim() == 4, "forward_map expects (B,C,H,W)"
        B, C, H, W = x.shape
        x_flat = x.permute(0, 2, 3, 1).contiguous().view(B * H * W, C)
        z = self._mlp(x_flat)
        logits = self.last_layer(z)
        logits = logits.view(B, H, W, self.out_dim).permute(0, 3, 1, 2).contiguous()
        return logits

class LayerProjector(nn.Module):
    def __init__(self, in_dim: int, proj_dim: int):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(in_dim, proj_dim),
            nn.GELU(),
            nn.Linear(proj_dim, proj_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class TokenPredictor(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, num_tokens: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_tokens),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class OnlineCodebook(nn.Module):

    def __init__(self, num_tokens: int, token_dim: int, ema_momentum: float = 0.95):
        super().__init__()
        self.num_tokens = num_tokens
        self.token_dim = token_dim
        self.ema_momentum = ema_momentum

        self.register_buffer(
            "prototypes",
            F.normalize(torch.randn(num_tokens, token_dim), dim=-1)
        )
        self.register_buffer("cluster_size", torch.zeros(num_tokens))

    @torch.no_grad()
    def assign(self, x: torch.Tensor) -> torch.Tensor:
        x = F.normalize(x, dim=-1)
        p = F.normalize(self.prototypes, dim=-1)
        sim = x @ p.t()
        token_ids = sim.argmax(dim=-1)
        return token_ids

    @torch.no_grad()
    def update(self, x: torch.Tensor, token_ids: torch.Tensor):
        x = F.normalize(x, dim=-1)
        K = self.num_tokens
        D = self.token_dim
        device = x.device
        dtype = x.dtype

        proto_sum = torch.zeros(K, D, device=device, dtype=dtype)
        count = torch.zeros(K, device=device, dtype=dtype)

        proto_sum.index_add_(0, token_ids, x)
        ones = torch.ones_like(token_ids, dtype=dtype)
        count.index_add_(0, token_ids, ones)

        m = self.ema_momentum
        self.cluster_size.mul_(m).add_(count, alpha=1.0 - m)

        used = count > 0
        if used.any():
            new_proto = proto_sum[used] / count[used].unsqueeze(-1).clamp_min(1.0)
            old_proto = self.prototypes[used]
            updated = F.normalize(old_proto * m + new_proto * (1.0 - m), dim=-1)
            self.prototypes[used] = updated



class DINOStudentTeacher(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        feat_dim: int = 64,
        use_diff_attn: bool = False,
        head_hidden: int = 512,
        head_bottleneck: int = 256,
        out_dim: int = 256,
        teacher_momentum: float = 0.996,
        token_dim: int = 128,
        token_hidden: int = 256,
        num_layer_tokens: int = 64,
        token_codebook_m: float = 0.95,
    ):
        super().__init__()

        self.student_backbone = Encoder(
            in_channels=in_channels,
            out_channels=feat_dim,
            patch_size=9,
            use_diff_attn=use_diff_attn,
            attn_num_heads=8,
            attn_scale=0.5,
            attn_chunk_size=1,
        )
        self.student_head = DINOHead(
            in_dim=feat_dim,
            hidden_dim=head_hidden,
            bottleneck_dim=head_bottleneck,
            out_dim=out_dim,
        )

        self.teacher_backbone = Encoder(
            in_channels=in_channels,
            out_channels=feat_dim,
            patch_size=9,
            use_diff_attn=use_diff_attn,
            attn_num_heads=8,
            attn_scale=0.5,
            attn_chunk_size=1,
        )
        self.teacher_head = DINOHead(
            in_dim=feat_dim,
            hidden_dim=head_hidden,
            bottleneck_dim=head_bottleneck,
            out_dim=out_dim,
        )

        layer_channels = self.student_backbone.out_channels_list
        self.num_layers_for_token = len(layer_channels)

        self.student_layer_projectors = nn.ModuleList([
            LayerProjector(ch, token_dim) for ch in layer_channels
        ])
        self.teacher_layer_projectors = nn.ModuleList([
            LayerProjector(ch, token_dim) for ch in layer_channels
        ])
        self.layer_token_predictors = nn.ModuleList([
            TokenPredictor(token_dim, token_hidden, num_layer_tokens)
            for _ in layer_channels
        ])
        self.layer_codebooks = nn.ModuleList([
            OnlineCodebook(num_layer_tokens, token_dim, ema_momentum=token_codebook_m)
            for _ in layer_channels
        ])

        self._init_teacher()

        for p in self.teacher_backbone.parameters():
            p.requires_grad = False
        for p in self.teacher_head.parameters():
            p.requires_grad = False
        for p in self.teacher_layer_projectors.parameters():
            p.requires_grad = False

        self.teacher_momentum = teacher_momentum
        self.register_buffer("center", torch.zeros(1, out_dim))

    @torch.no_grad()
    def _init_teacher(self):
        for s, t in zip(self.student_backbone.parameters(), self.teacher_backbone.parameters()):
            t.data.copy_(s.data)
        for s, t in zip(self.student_head.parameters(), self.teacher_head.parameters()):
            t.data.copy_(s.data)
        for s, t in zip(self.student_layer_projectors.parameters(), self.teacher_layer_projectors.parameters()):
            t.data.copy_(s.data)

    @torch.no_grad()
    def update_teacher(self, m: Optional[float] = None):
        if m is None:
            m = self.teacher_momentum

        for s, t in zip(self.student_backbone.parameters(), self.teacher_backbone.parameters()):
            t.data.mul_(m).add_(s.data, alpha=1.0 - m)

        for s, t in zip(self.student_head.parameters(), self.teacher_head.parameters()):
            t.data.mul_(m).add_(s.data, alpha=1.0 - m)

        for s, t in zip(self.student_layer_projectors.parameters(), self.teacher_layer_projectors.parameters()):
            t.data.mul_(m).add_(s.data, alpha=1.0 - m)

    @torch.no_grad()
    def update_center_logits(self, teacher_logits: torch.Tensor, momentum: float = 0.9):
        batch_center = teacher_logits.detach().mean(dim=0, keepdim=True)
        self.center.mul_(momentum).add_(batch_center, alpha=1.0 - momentum)

    def forward_backbone(self, x: torch.Tensor, use_teacher: bool = False, return_features: bool = False):
        if use_teacher:
            return self.teacher_backbone(x, return_features=return_features)
        return self.student_backbone(x, return_features=return_features)

    def forward_head(self, feat: torch.Tensor, use_teacher: bool = False):
        if use_teacher:
            return self.teacher_head(feat)
        return self.student_head(feat)

    @torch.no_grad()
    def teacher_forward(
        self,
        x: torch.Tensor,
        T: float = 0.04,
        return_logits: bool = False,
        return_features: bool = False,
        return_layer_tokens: bool = False,
        update_codebook: bool = False,
    ):
        feat, feats = self.teacher_backbone(x, return_features=True)
        logits = self.teacher_head(feat)
        centered_logits = (logits - self.center) / T
        probs = F.softmax(centered_logits, dim=-1)

        out = {"probs": probs}

        if return_logits:
            out["logits"] = logits
        if return_features:
            out["feat"] = feat
            out["feats"] = feats

        if return_layer_tokens:
            token_ids_all = []
            token_feats_all = []

            for i, f in enumerate(feats):
                pooled = f.mean(dim=[2, 3])
                z = self.teacher_layer_projectors[i](pooled)
                z = F.normalize(z, dim=-1)
                token_ids = self.layer_codebooks[i].assign(z)
                if update_codebook:
                    self.layer_codebooks[i].update(z, token_ids)
                token_ids_all.append(token_ids)
                token_feats_all.append(z)

            out["layer_token_ids"] = token_ids_all
            out["layer_token_feats"] = token_feats_all

        return out

    def student_forward_logits(
        self,
        x: torch.Tensor,
        T: float = 0.1,
        return_feat: bool = False,
        return_layer_token_logits: bool = False,
    ):
        feat, feats = self.student_backbone(x, return_features=True)
        logits = self.student_head(feat) / T
        log_probs = F.log_softmax(logits, dim=-1)

        out = {"log_probs": log_probs}
        if return_feat:
            out["feat"] = feat
            out["feats"] = feats

        if return_layer_token_logits:
            token_logits_all = []
            token_embs_all = []
            token_probs_all = []
            token_ids_all = []

            for i, f in enumerate(feats):
                pooled = f.mean(dim=[2, 3])
                z = self.student_layer_projectors[i](pooled)
                z = F.normalize(z, dim=-1)
                token_logits = self.layer_token_predictors[i](z)
                token_probs = F.softmax(token_logits, dim=-1)
                token_ids = token_probs.argmax(dim=-1)

                token_logits_all.append(token_logits)
                token_embs_all.append(z)
                token_probs_all.append(token_probs)
                token_ids_all.append(token_ids)

            out["layer_token_logits"] = token_logits_all
            out["layer_token_embs"] = token_embs_all
            out["layer_token_probs"] = token_probs_all
            out["layer_token_ids"] = token_ids_all

        return out

    @torch.no_grad()
    def inference_forward(
        self,
        x: torch.Tensor,
        T_student: float = 0.1,
    ) -> Dict[str, object]:
        feat, feats = self.student_backbone(x, return_features=True)
        proto_logits = self.student_head(feat) / T_student
        proto_probs = F.softmax(proto_logits, dim=-1)

        token_probs_all = []
        token_ids_all = []
        token_embs_all = []

        for i, f in enumerate(feats):
            pooled = f.mean(dim=[2, 3])
            z = self.student_layer_projectors[i](pooled)
            z = F.normalize(z, dim=-1)
            token_logits = self.layer_token_predictors[i](z)
            token_probs = F.softmax(token_logits, dim=-1)
            token_ids = token_probs.argmax(dim=-1)

            token_embs_all.append(z)
            token_probs_all.append(token_probs)
            token_ids_all.append(token_ids)

        return {
            "feat": feat,
            "feats": feats,
            "proto_logits": proto_logits,
            "proto_probs": proto_probs,
            "layer_token_embs": token_embs_all,
            "layer_token_probs": token_probs_all,
            "layer_token_ids": token_ids_all,
        }

    @staticmethod
    @torch.no_grad()
    def batch_entropy(probs: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        ent = -(probs * torch.log(probs.clamp_min(eps))).sum(dim=-1)
        return ent

    @staticmethod
    @torch.no_grad()
    def batch_confidence_from_probs(probs: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        K = probs.shape[-1]
        ent = DINOStudentTeacher.batch_entropy(probs, eps=eps)
        max_ent = math.log(K + 1e-12)
        conf = 1.0 - ent / max_ent
        return conf.clamp(0.0, 1.0)


def dino_loss(student_logp: torch.Tensor, teacher_p: torch.Tensor) -> torch.Tensor:
    return torch.sum(-teacher_p.detach() * student_logp, dim=-1).mean()


def weighted_dino_loss(
    student_logp: torch.Tensor,
    teacher_p: torch.Tensor,
    sample_weight: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    per_sample = torch.sum(-teacher_p.detach() * student_logp, dim=-1)
    if sample_weight is None:
        return per_sample.mean()
    w = sample_weight.detach()
    w = w / (w.sum() + 1e-6)
    return torch.sum(per_sample * w)


def layer_token_ce_loss(
    student_token_logits: List[torch.Tensor],
    teacher_token_ids: List[torch.Tensor],
    layer_weights: Optional[List[float]] = None,
) -> torch.Tensor:
    assert len(student_token_logits) == len(teacher_token_ids)
    n = len(student_token_logits)
    if layer_weights is None:
        layer_weights = [1.0] * n

    total = 0.0
    weight_sum = 0.0
    for i in range(n):
        w = float(layer_weights[i])
        ce = F.cross_entropy(student_token_logits[i], teacher_token_ids[i].detach())
        total = total + w * ce
        weight_sum += w

    return total / max(weight_sum, 1e-6)