"""Model definitions: OCTNet (from-scratch classifier) + GradCAM++, and the
SmallEncoder retrieval transformer + WordTokenizer."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import config


# ═══════════════════════════════ OCTNet ═══════════════════════════════════

class DSConv(nn.Module):
    """Depthwise separable conv: dw → pw → BN → SiLU."""
    def __init__(self, in_ch, out_ch, k=3, p=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, k, padding=p, groups=in_ch, bias=False),
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.SiLU(inplace=True),
        )
    def forward(self, x):
        return self.net(x)


class AnisoBranch(nn.Module):
    """
    1×7 + 7×1 depthwise separable convolutions.
    Captures horizontal retinal layer continuity (1×7) and vertical
    cross-section disruptions (7×1). Outputs summed to keep channels stable.
    """
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.h = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, (1, 7), padding=(0, 3), groups=in_ch, bias=False),
            nn.Conv2d(in_ch, out_ch, 1, bias=False))
        self.v = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, (7, 1), padding=(3, 0), groups=in_ch, bias=False),
            nn.Conv2d(in_ch, out_ch, 1, bias=False))
        self.bn  = nn.BatchNorm2d(out_ch)
        self.act = nn.SiLU(inplace=True)
    def forward(self, x):
        return self.act(self.bn(self.h(x) + self.v(x)))


class CBAM(nn.Module):
    """Channel + Spatial attention (Woo et al. 2018)."""
    def __init__(self, ch, r=16):
        super().__init__()
        mid = max(ch // r, 4)
        self.ca_mlp  = nn.Sequential(nn.Linear(ch, mid, bias=False), nn.ReLU(inplace=True), nn.Linear(mid, ch, bias=False))
        self.sa_conv = nn.Conv2d(2, 1, 7, padding=3, bias=False)
    def forward(self, x):
        avg = x.mean([2, 3]); mx = x.amax([2, 3])
        ca  = torch.sigmoid(self.ca_mlp(avg) + self.ca_mlp(mx))
        x   = x * ca.unsqueeze(-1).unsqueeze(-1)
        sa  = torch.sigmoid(self.sa_conv(torch.cat([x.mean(1, keepdim=True), x.amax(1, keepdim=True)], dim=1)))
        return x * sa


class MSBlock(nn.Module):
    """Multi-scale block: 3 branches → concat → CBAM → residual."""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        bc = out_ch // 3
        ex = out_ch - 3 * bc
        self.b3 = DSConv(in_ch, bc + ex, k=3, p=1)
        self.b5 = DSConv(in_ch, bc,      k=5, p=2)
        self.ba = AnisoBranch(in_ch, bc)
        self.bn   = nn.BatchNorm2d(out_ch)
        self.cbam = CBAM(out_ch)
        self.res  = nn.Sequential(nn.Conv2d(in_ch, out_ch, 1, bias=False), nn.BatchNorm2d(out_ch)) \
                    if in_ch != out_ch else nn.Identity()
        self.act  = nn.SiLU(inplace=True)
    def forward(self, x):
        out = self.cbam(self.bn(torch.cat([self.b3(x), self.b5(x), self.ba(x)], dim=1)))
        return self.act(out + self.res(x))


class OCTNet(nn.Module):
    def __init__(self, num_classes=4, drop=0.4):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1, bias=False), nn.BatchNorm2d(32), nn.SiLU(inplace=True),  # 112
            nn.Conv2d(32, 64, 3, padding=1, bias=False),           nn.BatchNorm2d(64), nn.SiLU(inplace=True),
        )
        self.s1 = nn.Sequential(MSBlock(64, 96),   nn.MaxPool2d(2), nn.Dropout2d(0.10))   # 56
        self.s2 = nn.Sequential(MSBlock(96, 192), MSBlock(192, 192), nn.MaxPool2d(2), nn.Dropout2d(0.15))  # 28
        self.s3 = nn.Sequential(MSBlock(192, 384), MSBlock(384, 384), nn.MaxPool2d(2), nn.Dropout2d(0.20))  # 14
        self.s4 = nn.Sequential(MSBlock(384, 512), nn.MaxPool2d(2))  # 7
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(512, 256), nn.SiLU(inplace=True),
            nn.Dropout(drop), nn.Linear(256, num_classes),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight); nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear) and m.weight is not None:
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.head(self.s4(self.s3(self.s2(self.s1(self.stem(x))))))

    @property
    def cam_layer(self):
        """Last depthwise conv — target for GradCAM++."""
        return self.s4[0].b3.net[0]


def load_octnet(ckpt_path=None, num_classes=None, device=None) -> OCTNet:
    ckpt_path = ckpt_path or config.CKPT_PATH
    num_classes = num_classes or config.NUM_CLASSES
    device = device or config.DEVICE
    model = OCTNet(num_classes).to(device)
    state = torch.load(ckpt_path, map_location=device, weights_only=True)
    if "state_dict" in state:  # tolerate wrapped checkpoints
        state = state["state_dict"]
    model.load_state_dict(state)
    model.eval()
    return model


# ═══════════════════════════ GradCAM++ ═══════════════════════════════════

class GradCAMpp:
    """GradCAM++ — no external library required."""
    def __init__(self, model, layer, device=None):
        self.device = device or config.DEVICE
        self.model = model
        self.grads = None
        self.acts = None
        layer.register_forward_hook(lambda m, i, o: setattr(self, "acts", o.detach()))
        layer.register_full_backward_hook(lambda m, gi, go: setattr(self, "grads", go[0].detach()))

    def __call__(self, img_t, cls=None):
        self.model.eval()
        x = img_t.unsqueeze(0).to(self.device).requires_grad_(True)
        logits = self.model(x)
        cls = cls if cls is not None else logits.argmax(1).item()
        self.model.zero_grad()
        logits[0, cls].backward()
        g, a = self.grads[0], self.acts[0]                # (C, H, W)
        g2, g3 = g**2, g**3
        alpha = g2 / (2 * g2 + (a * g3).sum([1, 2], keepdim=True) + 1e-7)
        w = (alpha * F.relu(g)).sum([1, 2])               # (C,)
        cam = F.relu((w[:, None, None] * a).sum(0))
        cam = (cam - cam.min()) / (cam.max() + 1e-7)
        conf = torch.softmax(logits, 1)[0].detach().cpu().numpy()
        return cam.cpu().numpy(), cls, conf


# ═══════════════════════ Retrieval (from scratch) ═══════════════════════

class WordTokenizer:
    PAD, UNK = "<PAD>", "<UNK>"

    def __init__(self, texts: list[str], max_len: int = 64):
        self.max_len = max_len
        words = [w.lower().strip('.,();:"\'-') for t in texts for w in t.split()]
        vocab = [self.PAD, self.UNK] + sorted(set(words))
        self.w2i = {w: i for i, w in enumerate(vocab)}
        self.vocab_size = len(vocab)

    def encode(self, text: str) -> torch.Tensor:
        tokens = [w.lower().strip('.,();:"\'-') for w in text.split()]
        ids = [self.w2i.get(t, 1) for t in tokens]        # 1 = UNK
        ids = ids[:self.max_len] + [0] * max(0, self.max_len - len(ids))
        return torch.tensor(ids, dtype=torch.long)


class SmallEncoder(nn.Module):
    """
    Shared encoder tower used for both queries and documents.
    token embedding → learned positional → N× TransformerEncoderLayer
    → mean-pool over non-padding → linear proj → L2 normalize.
    """
    def __init__(self, vocab_size, d_model=128, nhead=4, num_layers=4,
                 dim_ff=256, embed_dim=64, max_len=64, dropout=0.1):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_emb = nn.Embedding(max_len, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_ff,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.proj = nn.Linear(d_model, embed_dim, bias=False)
        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.tok_emb.weight, std=0.02)
        nn.init.normal_(self.pos_emb.weight, std=0.02)
        nn.init.xavier_uniform_(self.proj.weight)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        B, L = input_ids.shape
        positions = torch.arange(L, device=input_ids.device).unsqueeze(0).expand(B, -1)
        padding_mask = (input_ids == 0)
        x = self.tok_emb(input_ids) + self.pos_emb(positions)
        x = self.transformer(x, src_key_padding_mask=padding_mask)
        mask = (~padding_mask).float().unsqueeze(-1)
        x = (x * mask).sum(1) / mask.sum(1).clamp(min=1)
        return F.normalize(self.proj(x), dim=-1)