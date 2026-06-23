import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple


class NumericFeatureTokenizer(nn.Module):
    def __init__(self, input_dim: int, d_model: int):
        super().__init__()
        self.input_dim = input_dim
        self.d_model = d_model
        self.weight = nn.Parameter(torch.randn(input_dim, d_model) * 0.02)
        self.bias = nn.Parameter(torch.zeros(input_dim, d_model))
        self.cls = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.pos = nn.Parameter(torch.randn(1, input_dim + 1, d_model) * 0.02)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 2:
            raise ValueError(f"Expected [B, F], got {tuple(x.shape)}")
        if x.size(1) != self.input_dim:
            raise ValueError(f"Feature mismatch: expected {self.input_dim}, got {x.size(1)}")
        feat_tokens = x.unsqueeze(-1) * self.weight.unsqueeze(0) + self.bias.unsqueeze(0)
        cls = self.cls.expand(x.size(0), -1, -1)
        tokens = torch.cat([cls, feat_tokens], dim=1)
        return self.norm(tokens + self.pos[:, :tokens.size(1), :])


class FiLMAdapter(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model * 2),
        )

    def forward(self, h: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        gamma, beta = self.net(context).chunk(2, dim=-1)
        return h * (1.0 + torch.tanh(gamma)) + beta


class MTLMetaIRLTransformer(nn.Module):
    def __init__(
        self,
        input_dim: int,
        task_label_class_counts: Dict[str, Dict[str, int]],
        d_model: int = 128,
        nhead: int = 8,
        num_layers: int = 4,
        dropout: float = 0.15,
        proto_alpha: float = 0.35,
        proto_temperature: float = 1.0,
        detach_probs_for_reward: bool = True,
        enable_meta: bool = True,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.task_label_class_counts = task_label_class_counts
        self.d_model = d_model
        self.proto_alpha = proto_alpha
        self.proto_temperature = proto_temperature
        self.detach_probs_for_reward = detach_probs_for_reward
        self.enable_meta = enable_meta

        self.tokenizer = NumericFeatureTokenizer(input_dim, d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.shared_norm = nn.LayerNorm(d_model)
        self.shared_dropout = nn.Dropout(dropout)

        self.task_embeddings = nn.ParameterDict({
            task_name: nn.Parameter(torch.randn(1, d_model) * 0.02)
            for task_name in task_label_class_counts.keys()
        })
        self.adapter = FiLMAdapter(d_model, dropout=dropout)

        self.heads = nn.ModuleDict()
        self.reward_heads = nn.ModuleDict()
        for task_name, label_counts in task_label_class_counts.items():
            self.heads[task_name] = nn.ModuleDict()
            self.reward_heads[task_name] = nn.ModuleDict()
            for lbl, n_cls in label_counts.items():
                self.heads[task_name][lbl] = nn.Sequential(
                    nn.LayerNorm(d_model),
                    nn.Linear(d_model, d_model),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(d_model, n_cls),
                )
                reward_in_dim = d_model + n_cls
                self.reward_heads[task_name][lbl] = nn.Sequential(
                    nn.LayerNorm(reward_in_dim),
                    nn.Linear(reward_in_dim, d_model),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(d_model, 1),
                )

        self.log_vars = nn.ParameterDict({
            task_name: nn.Parameter(torch.zeros(1))
            for task_name in task_label_class_counts.keys()
        })

    def encode(self, x: torch.Tensor, task_name: str, context: Optional[torch.Tensor] = None) -> torch.Tensor:
        tokens = self.tokenizer(x)
        h = self.encoder(tokens)
        h = self.shared_norm(h[:, 0, :])
        h = self.shared_dropout(h)

        task_emb = self.task_embeddings[task_name].expand(h.size(0), -1)
        if context is None:
            merged_context = task_emb
        else:
            if context.dim() == 1:
                context = context.unsqueeze(0)
            if context.size(0) == 1 and h.size(0) > 1:
                context = context.expand(h.size(0), -1)
            merged_context = context + task_emb
        h = self.adapter(h, merged_context)
        return h

    def build_episode_context(self, support_h: torch.Tensor, support_y: torch.Tensor, n_classes: int) -> torch.Tensor:
        if support_h is None or support_h.numel() == 0:
            return torch.zeros(1, self.d_model, device=support_y.device if support_y is not None else "cpu")
        if support_y is None or support_y.numel() == 0:
            return support_h.mean(0, keepdim=True)
        protos = []
        for c in range(n_classes):
            mask = support_y == c
            if mask.any():
                protos.append(support_h[mask].mean(0, keepdim=True))
        if len(protos) == 0:
            return support_h.mean(0, keepdim=True)
        return torch.cat(protos, dim=0).mean(0, keepdim=True)

    def classify_with_prototypes(
        self,
        query_h: torch.Tensor,
        head_logits: torch.Tensor,
        support_h: Optional[torch.Tensor],
        support_y: Optional[torch.Tensor],
        n_classes: int,
        alpha: Optional[float] = None,
    ) -> torch.Tensor:
        if alpha is None:
            alpha = self.proto_alpha
        if (not self.enable_meta) or support_h is None or support_y is None or support_h.numel() == 0 or support_y.numel() == 0:
            return head_logits
        support_y = support_y.long()
        protos = []
        valid = []
        for c in range(n_classes):
            mask = support_y == c
            if mask.any():
                protos.append(support_h[mask].mean(0))
                valid.append(c)
        if len(protos) == 0:
            return head_logits
        proto_bank = torch.stack(protos, dim=0)
        sim = F.cosine_similarity(query_h.unsqueeze(1), proto_bank.unsqueeze(0), dim=-1) / max(self.proto_temperature, 1e-6)
        # NOTE 2026-05-20 P2-12:
        # 問題：support/query 分離後，support 不一定含有 query 中所有類別；若缺席類別被填成極低 logits，rare class 會被錯誤壓低。
        # 使用步驟：forward() 有 support 時會呼叫此函式，僅對 support 中有 prototype 的類別混入 cosine similarity。
        # 功能例子：support 只有 WNL/SNHL，query 可能含 CHL；CHL 會保留原 classifier logits，不會因沒有 prototype 被設成 -1e4。
        proto_logits = head_logits.clone()
        proto_logits[:, valid] = sim
        return (1.0 - alpha) * head_logits + alpha * proto_logits

    def reward_score(self, task_name: str, label_col: str, h: torch.Tensor, class_probs: torch.Tensor) -> torch.Tensor:
        if self.detach_probs_for_reward:
            class_probs = class_probs.detach()
        reward_in = torch.cat([h, class_probs], dim=-1)
        return self.reward_heads[task_name][label_col](reward_in).squeeze(-1)

    def forward(self, x: torch.Tensor, task_name: str, support: Optional[Dict[str, Tuple[torch.Tensor, torch.Tensor]]] = None):
        device = x.device
        context = None
        if self.enable_meta and support:
            for _, (sx, sy) in support.items():
                if sx is None or sy is None or sx.numel() == 0 or sy.numel() == 0:
                    continue
                sx = sx.to(device)
                sy = sy.to(device).long()
                sh0 = self.encode(sx, task_name, None)
                n_classes_ctx = int(sy.max().item()) + 1 if sy.numel() > 0 else 1
                context = self.build_episode_context(sh0, sy, n_classes_ctx)
                break

        h = self.encode(x, task_name, context)
        logits_dict = {}
        reward_dict = {}
        for label_col, head in self.heads[task_name].items():
            logits = head(h)
            n_classes = logits.size(-1)
            if self.enable_meta and support and label_col in support:
                sx, sy = support[label_col]
                if sx is not None and sy is not None and sx.numel() > 0 and sy.numel() > 0:
                    sx = sx.to(device)
                    sy = sy.to(device).long()
                    sh = self.encode(sx, task_name, context)
                    logits = self.classify_with_prototypes(h, logits, sh, sy, n_classes=n_classes)
            probs = torch.softmax(logits, dim=-1)
            reward = self.reward_score(task_name, label_col, h, probs)
            logits_dict[label_col] = logits
            reward_dict[label_col] = reward
        return logits_dict, reward_dict
