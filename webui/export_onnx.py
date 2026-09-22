#!/usr/bin/env python
"""Export the trained octnet pipeline to ONNX assets for in-browser inference.

The browser SPA (webui/static) runs the real pipeline with onnxruntime-web.
This script produces, under webui/static/models/:

  classify.onnx   OCTNet image -> logits (1, 4)
  cam.onnx        (image, class) -> GradCAM++ heatmap (1, 1, 14, 14), 0..1
  encoder.onnx    token ids (1, 64) -> L2-normalized embedding (1, 64)
  corpus.json     corpus entries + precomputed document embeddings
  tokenizer.json  WordTokenizer vocab (so JS tokenizes identically)
  jet.json        matplotlib jet colormap LUT (256 x 3) for heatmaps
  preprocess.json constants the JS needs (sizes, thresholds, zoom config)

Design notes
------------
* GradCAM++ normally needs autograd. Browsers cannot run PyTorch autograd, so
  ``cam.onnx`` contains an *analytic* backward pass built from the true chain
  rule (transposed convs, affine BatchNorm gradients, CBAM attention terms,
  exact MaxPool unpool via argmax masks). It is validated here against a real
  GradCAMpp hook instance and against onnxruntime on the same inputs.
* The retrieval encoder (transformer) is reconstructed with explicit ops so
  the exported graph only uses core ONNX ops supported by onnxruntime-web.
* Everything is deterministic and CPU-only.

Run:  oct_env/bin/python webui/export_onnx.py
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from octnet import config
from octnet.models import GradCAMpp, OCTNet, SmallEncoder, load_octnet
from octnet.retrieval import build_tokenizer_encoder, load_corpus

OUT = ROOT / "webui" / "static" / "models"
OPSET = 15
NUM_CLASSES = config.NUM_CLASSES


# ── math helpers (mirror PyTorch autograd exactly) ──────────────────────────

def silu_prime(z: torch.Tensor) -> torch.Tensor:
    s = torch.sigmoid(z)
    return s * (1 + z * (1 - s))


class CamGraph(nn.Module):
    """Analytic GradCAM++ graph: (image, cls) -> normalized 14x14 heatmap.

    Reproduces octnet/models.py GradCAMpp.__call__ step by step, except the
    gradient is computed analytically instead of via autograd. Holds the
    classifier as a submodule so the ONNX exporter registers its parameters.
    """
    def __init__(self, model: OCTNet, num_classes: int = NUM_CLASSES):
        super().__init__()
        self.model = model
        self.num_classes = num_classes

    @staticmethod
    def _bn_scale(bn):
        return bn.weight / torch.sqrt(bn.running_var + bn.eps)

    def forward(self, image, cls):
        m = self.model
        s4 = m.s4[0]
        b3 = s4.b3
        fc1, fc2 = m.head[2], m.head[5]
        ca_mlp = s4.cbam.ca_mlp
        sa_conv = s4.cbam.sa_conv
        bn_cat = s4.bn
        # ── forward pass (eval semantics, non-inplace where we keep z) ──
        h1 = m.stem(image)
        h2 = m.s1(h1)
        h3 = m.s2(h2)
        h4 = m.s3(h3)

        A = b3.net[0](h4)                     # CAM target: dw conv output
        z_pw = b3.net[1](A)
        z_b3bn = b3.net[2](z_pw)
        b3_out = F.silu(z_b3bn)
        b5_out = s4.b5(h4)
        ba_out = s4.ba(h4)
        cat = torch.cat([b3_out, b5_out, ba_out], dim=1)
        z_cat = bn_cat(cat)

        # CBAM
        avg = z_cat.mean([2, 3])
        mx = z_cat.amax([2, 3])
        h_avg = F.relu(ca_mlp[0](avg))
        h_mx = F.relu(ca_mlp[0](mx))
        z_ca = ca_mlp[2](h_avg) + ca_mlp[2](h_mx)
        ca = torch.sigmoid(z_ca)
        x_ca = z_cat * ca.unsqueeze(-1).unsqueeze(-1)
        sa_in = torch.cat([x_ca.mean(1, keepdim=True),
                           x_ca.amax(1, keepdim=True)], dim=1)
        z_sa = sa_conv(sa_in)
        sa = torch.sigmoid(z_sa)
        x_sa = x_ca * sa

        z_act = x_sa + s4.res(h4)
        act_out = F.silu(z_act)
        pooled = F.max_pool2d(act_out, 2)
        z_head = fc1(m.head[0](pooled).flatten(1))

        # ── analytic backward: d loss_c / d A ──
        # head: logit_c = fc2(silu(z_head));  one-hot at cls
        gh = F.one_hot(cls, num_classes=self.num_classes).float()      # (1,4)
        gh = torch.matmul(gh, fc2.weight)                        # (1,256)
        gh = gh * silu_prime(z_head)
        gh = torch.matmul(gh, fc1.weight)                        # (1,512)
        gp = gh.view(1, 512, 1, 1) / 49.0                        # avgpool(7x7)->1x1

        # MaxPool2d(2) unpool: exact first-max-mask via argmax on 2x2 windows
        a0 = act_out[0].reshape(512, 7, 2, 7, 2).permute(0, 1, 3, 2, 4).reshape(512, 7, 7, 4)
        mi = a0.argmax(-1)
        mm = F.one_hot(mi, 4).float().reshape(512, 7, 7, 2, 2)
        G_act = (mm * gp.view(512, 1, 1, 1, 1))
        G_act = G_act.permute(0, 1, 3, 2, 4).reshape(512, 14, 14)

        # s4.act (SiLU) backward
        G_z = G_act * silu_prime(z_act[0])                        # -> x_sa grad

        # CBAM backward, x_sa = x_ca * sa  (sa depends on x_ca)
        g_xca = G_z * sa[0]                                       # direct term
        g_sa = (G_z * x_ca[0]).sum(0, keepdim=True)              # (1,14,14)
        g_zs = g_sa * sa[0] * (1 - sa[0])                        # sigmoid'
        g_sain = F.conv_transpose2d(g_zs.unsqueeze(0), sa_conv.weight,
                                    padding=3)[0]
        g_xca = g_xca + g_sain[0:1] / 512.0                      # mean over ch
        xmx = x_ca[0].amax(0, keepdim=True)
        g_xca = g_xca + (x_ca[0] == xmx).float() * g_sain[1:2]   # max over ch

        # CBAM backward, x_ca = z_cat * ca  (ca depends on z_cat)
        g_ca = (g_xca * z_cat[0]).sum([1, 2])                     # (512,)
        g_zca = g_ca * ca[0] * (1 - ca[0])                       # sigmoid'

        def mlp_back(u_pre, g_out):
            g_h = torch.matmul(g_out.unsqueeze(0), ca_mlp[2].weight)
            g_h = g_h * (u_pre > 0).float()
            return torch.matmul(g_h, ca_mlp[0].weight)[0]

        g_u_avg = mlp_back(h_avg, g_zca)                        # (512,)
        g_u_mx = mlp_back(h_mx, g_zca)
        g_zcat = g_xca * ca[0].unsqueeze(-1).unsqueeze(-1)       # direct term
        g_zcat = g_zcat + g_u_avg.view(1, 512, 1, 1) / 196.0     # mean spatial
        zcat_mx = z_cat.amax([2, 3], keepdim=True)
        g_zcat = g_zcat + (z_cat == zcat_mx).float() * g_u_mx.view(1, 512, 1, 1)

        # MSBlock bn (eval affine) backward -> cat grad -> slice b3 branch
        g_cat = g_zcat[0] * self._bn_scale(bn_cat).view(512, 1, 1)
        g_b3 = g_cat[:172]

        # b3 branch backward: silu' -> BN affine -> pw(1x1) transpose
        g_zbn = (g_b3 * silu_prime(z_b3bn[0])).unsqueeze(0)
        g_zpw = g_zbn * self._bn_scale(b3.net[2]).view(1, 172, 1, 1)
        g_A = F.conv_transpose2d(g_zpw, b3.net[1].weight, stride=1, padding=0)

        # ── GradCAM++ combination (exact port of models.py) ──
        g, a = g_A[0], A[0]
        g2 = g * g
        g3 = g2 * g
        alpha = g2 / (2 * g2 + (a * g3).sum([1, 2], keepdim=True) + 1e-7)
        w = (alpha * F.relu(g)).sum([1, 2])
        cam = F.relu((w[:, None, None] * a).sum(0))
        cam = (cam - cam.min()) / (cam.max() + 1e-7)
        return cam.unsqueeze(0).unsqueeze(0)                     # (1,1,14,14)


class EncGraph(nn.Module):
    """SmallEncoder forward reconstructed with explicit core ops (no MHA),
    so the exported graph only uses ops onnxruntime-web supports."""
    def __init__(self, encoder: SmallEncoder):
        super().__init__()
        self.encoder = encoder

    def forward(self, ids):
        enc = self.encoder
        B, Ln = ids.shape
        pos = torch.arange(Ln, dtype=torch.long).unsqueeze(0).expand(B, -1)
        pad = ids == 0
        x = enc.tok_emb(ids) + enc.pos_emb(pos)

        for layer in enc.transformer.layers:
            nx = layer.norm1(x)
            qkv = F.linear(nx, layer.self_attn.in_proj_weight,
                           layer.self_attn.in_proj_bias)
            q, k, v = qkv.chunk(3, dim=-1)
            Bn, Lt, E = q.shape
            H = layer.self_attn.num_heads
            Dh = E // H
            q = q.view(Bn, Lt, H, Dh).transpose(1, 2)
            k = k.view(Bn, Lt, H, Dh).transpose(1, 2)
            v = v.view(Bn, Lt, H, Dh).transpose(1, 2)
            s = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(Dh)
            s = s + torch.where(pad.unsqueeze(1).unsqueeze(2).expand_as(s),
                                torch.full_like(s, -1e9), torch.zeros_like(s))
            att = torch.softmax(s, dim=-1)
            o = torch.matmul(att, v).transpose(1, 2).reshape(Bn, Lt, E)
            o = F.linear(o, layer.self_attn.out_proj.weight,
                         layer.self_attn.out_proj.bias)
            x = x + o
            nx2 = layer.norm2(x)
            x = x + layer.linear2(layer.activation(layer.linear1(nx2)))

        m = (~pad).float().unsqueeze(-1)
        x = (x * m).sum(1) / m.sum(1).clamp(min=1)
        return F.normalize(enc.proj(x), dim=-1)


# ── validation helpers ──────────────────────────────────────────────────────

class _Fn(nn.Module):
    """Wrap a plain function so torch.onnx.export accepts it (legacy exporter)."""
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
    def forward(self, *args):
        return self.fn(*args)


def max_diff(name, a, b):
    d = float((a - b).abs().max()) if isinstance(a, torch.Tensor) else \
        float(np.abs(a - b).max())
    print(f"  [{'PASS' if d < 1e-4 else 'FAIL'}] {name:34s} max abs diff = {d:.3e}")
    return d < 1e-4


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    print(f"=> exporting to {OUT}")

    ok = True
    model = load_octnet(device="cpu")
    model.eval()

    # ── a real scan for numerical validation ──
    from octnet.data import build_infer_tf
    scan_dir = ROOT / "OCT2017" / "test" / "CNV"
    scan = sorted(scan_dir.glob("*.jpeg"))[0]
    pil = Image.open(scan).convert("RGB")
    img_t = build_infer_tf(224)(pil)
    x = img_t.unsqueeze(0)

    # 1) classify.onnx
    print("\n[1/5] classify.onnx")
    torch.onnx.export(model, x, OUT / "classify.onnx",
                      input_names=["image"], output_names=["logits"],
                      opset_version=OPSET)
    with torch.inference_mode():
        ref_logits = model(x)
    import onnxruntime as ort
    so = ort.SessionOptions()
    sess = ort.InferenceSession(str(OUT / "classify.onnx"), so,
                                providers=["CPUExecutionProvider"])
    ort_logits = sess.run(["logits"], {"image": x.numpy()})[0]
    ok &= max_diff("logits (torch vs onnxruntime)", ref_logits,
                   torch.from_numpy(ort_logits))

    # 2) cam.onnx : analytic graph vs real GradCAMpp (hooks) vs onnxruntime
    print("\n[2/5] cam.onnx (analytic GradCAM++)")
    cam_fn = GradCAMpp(model, model.cam_layer, device="cpu")
    cam_ref, cls_ref, conf_ref = cam_fn(img_t)
    cls = torch.tensor([cls_ref], dtype=torch.long)
    cam_graph = CamGraph(model)
    cam_analytic = cam_graph(x, cls)
    ok &= max_diff("analytic vs GradCAMpp hooks", cam_analytic[0],
                   torch.from_numpy(cam_ref))
    torch.onnx.export(cam_graph, (x, cls), OUT / "cam.onnx",
                      input_names=["image", "cls"],
                      output_names=["cam"], opset_version=OPSET)
    ort_cam = sess_safe(OUT / "cam.onnx", {"image": x.numpy(),
                                           "cls": np.array([cls_ref], dtype=np.int64)})["cam"][0]
    ok &= max_diff("onnxruntime vs GradCAMpp hooks", torch.from_numpy(ort_cam),
                   torch.from_numpy(cam_ref))

    # 3) encoder.onnx (explicit transformer graph) + corpus + tokenizer
    print("\n[3/5] encoder.onnx + corpus embeddings + tokenizer")
    corpus = load_corpus()
    tokenizer, encoder = build_tokenizer_encoder(corpus, "cpu")
    state = torch.load(config.CKPT_EMB, map_location="cpu", weights_only=True)
    if "state_dict" in state:
        state = state["state_dict"]
    encoder.load_state_dict(state)
    encoder.eval()

    query = "CNV"
    ids = tokenizer.encode(query).unsqueeze(0)
    with torch.inference_mode():
        emb_ref = encoder(ids)
    enc_graph = EncGraph(encoder)
    emb_analytic = enc_graph(ids)
    ok &= max_diff("encoder analytic vs module", emb_analytic, emb_ref)
    torch.onnx.export(enc_graph, ids, OUT / "encoder.onnx",
                      input_names=["ids"], output_names=["emb"],
                      opset_version=OPSET)
    ort_emb = sess_safe(OUT / "encoder.onnx", {"ids": ids.numpy()})["emb"][0]
    ok &= max_diff("encoder onnxruntime vs module",
                   torch.from_numpy(ort_emb), emb_ref)

    with torch.inference_mode():
        emb_all = encoder(torch.stack([tokenizer.encode(t) for t, _ in corpus]))
    payload = [{"label": l, "text": t, "emb": emb_all[i].tolist()}
               for i, (l, t) in enumerate(corpus)]
    (OUT / "corpus.json").write_text(json.dumps(payload))
    tokenizer_json = {"max_len": tokenizer.max_len, "vocab": tokenizer.w2i}
    (OUT / "tokenizer.json").write_text(json.dumps(tokenizer_json))
    print(f"  wrote corpus.json ({len(payload)} entries), tokenizer.json "
          f"(vocab {tokenizer.vocab_size})")

    # 4) jet LUT + preprocessing constants
    print("\n[4/5] jet.json + preprocess.json")
    from matplotlib import cm
    jet = (cm.jet(np.linspace(0, 1, 256)) * 255).astype(np.uint8)[:, :3]
    (OUT / "jet.json").write_text(json.dumps(jet.tolist()))
    preprocess = {
        "img_size": 224, "num_classes": 4,
        "class_names": config.CLASS_NAMES,
        "mean": 0.5, "std": 0.5,
        "max_side": 512, "top_k": 3,
        "zoom": {"confidence_gate": 0.75, "min_crop_area_frac": 0.15,
                 "blend_alpha": 0.6, "cam_threshold": 0.4,
                 "padding_frac": 0.10},
        "overlay": {"alpha": 0.55, "rect_rgb": [255, 255, 0], "rect_width": 3},
    }
    (OUT / "preprocess.json").write_text(json.dumps(preprocess, indent=1))

    # 5) end-to-end numeric snapshot (torch reference for the browser)
    print("\n[5/5] reference pipeline snapshot")
    from octnet import predict
    res = predict.analyze_image(scan)
    print(f"  prediction={res['prediction']} conf={res['confidence']}")
    print(f"  probs={res['probabilities']}")
    print(f"  bbox={res['localization']['bbox']} region={res['localization']['region']}")
    print(f"  zoom.performed={res['zoom']['performed']} "
          f"crop_pred={res['zoom']['crop_pred']} final_pred={res['zoom']['final_pred']}")

    print("\n" + ("ALL VALIDATIONS PASSED" if ok else "!! SOME CHECKS FAILED"))
    for p in sorted(OUT.glob("*")):
        print(f"  {p.name:22s} {p.stat().st_size / 1e6:.2f} MB")
    sys.exit(0 if ok else 1)


def sess_safe(path: Path, feeds: dict):
    import onnxruntime as ort
    so = ort.SessionOptions()
    s = ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])
    names = [o.name for o in s.get_outputs()]
    return dict(zip(names, s.run(names, feeds)))


if __name__ == "__main__":
    main()