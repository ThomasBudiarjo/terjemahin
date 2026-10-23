"""Network of manga-image-translator's 48px autoregressive OCR (ocr_ar_48px.ckpt).

ConvNeXt backbone -> 4 pre-norm transformer encoders -> 5 pre-norm decoders with XPOS rotary
attention, a tied character head over a 46k alphabet, and per-character colour heads (glyph
RGB, outline RGB and their presence). Parameter names match the checkpoint, so
`load_state_dict` works directly. Rewritten from the reference for inference only: plain torch
(no einops), a real KV cache, and a vectorised beam search that keeps every beam's own history
(the reference's tensor beam search forgets to reorder its cache when beams are re-ranked).

Imported lazily by `mit48px.py` because it needs torch.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn

EMBD = 320
HEADS = 4
HEAD_DIM = EMBD // HEADS
PAD, BOS, EOS = 0, 1, 2


class ConvNeXtBlock(nn.Module):
    def __init__(self, dim: int, ks: int = 7, padding: int = 3) -> None:
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, ks, padding=padding, groups=dim)
        self.norm = nn.BatchNorm2d(dim, eps=1e-6)
        self.pwconv1 = nn.Conv2d(dim, 4 * dim, 1)
        self.act = nn.GELU()
        self.pwconv2 = nn.Conv2d(4 * dim, dim, 1)
        self.gamma = nn.Parameter(torch.ones(1, dim, 1, 1))

    def forward(self, x: Tensor) -> Tensor:
        return x + self.gamma * self.pwconv2(self.act(self.pwconv1(self.norm(self.dwconv(x)))))


def _conv_bn(cin: int, cout: int, k, s, p=0) -> list[nn.Module]:
    return [nn.Conv2d(cin, cout, k, s, p), nn.BatchNorm2d(cout), nn.ReLU()]


class Backbone(nn.Module):
    """(N, 3, 48, W) -> (N, 320, 1, W // 4)."""

    def __init__(self, dim: int = EMBD) -> None:
        super().__init__()
        b = dim // 8
        self.stem = nn.Sequential(
            *_conv_bn(3, b, 7, 1, 3), *_conv_bn(b, b * 2, 2, 2), *_conv_bn(b * 2, b * 2, 3, 1, 1)
        )
        self.block1 = self._blocks(b * 2, 4)
        self.down1 = nn.Sequential(*_conv_bn(b * 2, b * 4, 2, 2))
        self.block2 = self._blocks(b * 4, 12)
        self.down2 = nn.Sequential(*_conv_bn(b * 4, b * 8, (2, 1), (2, 1)))
        self.block3 = self._blocks(b * 8, 10, 5, 2)
        self.down3 = nn.Sequential(*_conv_bn(b * 8, b * 8, (2, 1), (2, 1)))
        self.block4 = self._blocks(b * 8, 8, 3, 1)
        self.down4 = nn.Sequential(*_conv_bn(b * 8, b * 8, (3, 1), (1, 1)))

    @staticmethod
    def _blocks(dim: int, n: int, ks: int = 7, padding: int = 3) -> nn.Sequential:
        return nn.Sequential(*[ConvNeXtBlock(dim, ks, padding) for _ in range(n)])

    def forward(self, x: Tensor) -> Tensor:
        x = self.block1(self.stem(x))
        x = self.block2(self.down1(x))
        x = self.block3(self.down2(x))
        x = self.block4(self.down3(x))
        return self.down4(x)


class XPOS(nn.Module):
    """Rotary embedding with exponential decay (XPOS), evaluated at explicit positions.

    The reference derives positions from a centred range over the sequence; rotation uses the
    0-based index and the decay uses the centred position. Taking both as arguments lets the
    decoder use a KV cache while reproducing the reference numbers exactly.
    """

    def __init__(self, head_dim: int = HEAD_DIM, scale_base: int = EMBD) -> None:
        super().__init__()
        self.scale_base = scale_base
        self.register_buffer(
            "scale", (torch.arange(0, head_dim, 2) + 0.4 * head_dim) / (1.4 * head_dim)
        )
        self.register_buffer(
            "inv_freq",
            1.0 / (10000 ** (torch.arange(0, head_dim // 2) / (head_dim // 2))),
            persistent=False,
        )

    def forward(self, x: Tensor, index: Tensor, pos: Tensor, downscale: bool = False) -> Tensor:
        """x (..., L, D); index, pos broadcastable to (..., L) (rotation index, decay position)."""
        scale = self.scale ** (pos.to(x.dtype).unsqueeze(-1) / self.scale_base)
        if downscale:
            scale = 1 / scale
        ang = index.to(x.dtype).unsqueeze(-1) * self.inv_freq
        sin = (ang.sin() * scale).repeat_interleave(2, -1)
        cos = (ang.cos() * scale).repeat_interleave(2, -1)
        rot = torch.stack((-x[..., 1::2], x[..., ::2]), -1).flatten(-2)
        return x * cos + rot * sin


def centred(index: Tensor, length: Tensor | int) -> Tensor:
    """Reference decay position of `index` in a sequence of `length`: index - ceil(length / 2)."""
    return index + torch.div(-torch.as_tensor(length), 2, rounding_mode="floor")


class Attention(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.q_proj = nn.Linear(EMBD, EMBD)
        self.k_proj = nn.Linear(EMBD, EMBD)
        self.v_proj = nn.Linear(EMBD, EMBD)
        self.out_proj = nn.Linear(EMBD, EMBD)
        self.xpos = XPOS()

    @staticmethod
    def heads(x: Tensor) -> Tensor:
        return x.view(x.shape[0], x.shape[1], HEADS, HEAD_DIM).transpose(1, 2)

    def q(self, x: Tensor) -> Tensor:
        return self.heads(self.q_proj(x) * HEAD_DIM**-0.5)

    def kv(self, x: Tensor) -> tuple[Tensor, Tensor]:
        return self.heads(self.k_proj(x)), self.heads(self.v_proj(x))

    def attend(self, q: Tensor, k: Tensor, v: Tensor, pad: Tensor | None = None) -> Tensor:
        """q (B,H,Lq,D), k/v (B,H,Lk,D), pad (B,Lk) True = ignore -> (B, Lq, E)."""
        w = q @ k.transpose(-1, -2)
        if pad is not None:
            w = w.masked_fill(pad[:, None, None, :], float("-inf"))
        out = w.softmax(-1) @ v
        return self.out_proj(out.transpose(1, 2).reshape(q.shape[0], q.shape[2], EMBD))


class EncoderLayer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.self_attn = Attention()
        self.linear1 = nn.Linear(EMBD, 2048)
        self.linear2 = nn.Linear(2048, EMBD)
        self.norm1 = nn.LayerNorm(EMBD)
        self.norm2 = nn.LayerNorm(EMBD)

    def forward(self, x: Tensor, pad: Tensor) -> Tensor:
        a = self.self_attn
        h = self.norm1(x)
        idx = torch.arange(x.shape[1], device=x.device)
        q = a.xpos(a.q(h), idx, idx)
        k, v = a.kv(h)
        x = x + a.attend(q, a.xpos(k, idx, idx, downscale=True), v, pad)
        return x + self.linear2(F.relu(self.linear1(self.norm2(x))))


class DecoderLayer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.self_attn = Attention()
        self.multihead_attn = Attention()
        self.linear1 = nn.Linear(EMBD, 2048)
        self.linear2 = nn.Linear(2048, EMBD)
        self.norm1 = nn.LayerNorm(EMBD)
        self.norm2 = nn.LayerNorm(EMBD)
        self.norm3 = nn.LayerNorm(EMBD)


@dataclass
class Decoded:
    tokens: list[int]  # without BOS / EOS
    logprob: float  # mean log-probability per emitted token (EOS included)
    colors: Tensor  # (len(tokens), 10): fg rgb, bg rgb, fg logits(2), bg logits(2)


class OCR48px(nn.Module):
    def __init__(self, dict_size: int) -> None:
        super().__init__()
        self.backbone = Backbone()
        self.encoders = nn.ModuleList(EncoderLayer() for _ in range(4))
        self.decoders = nn.ModuleList(DecoderLayer() for _ in range(5))
        self.embd = nn.Embedding(dict_size, EMBD)
        self.pred1 = nn.Sequential(nn.Linear(EMBD, EMBD), nn.GELU(), nn.Dropout(0.15))
        self.pred = nn.Linear(EMBD, dict_size)
        self.pred.weight = self.embd.weight
        self.color_pred1 = nn.Sequential(nn.Linear(EMBD, 64), nn.ReLU())
        self.color_pred_fg = nn.Linear(64, 3)
        self.color_pred_bg = nn.Linear(64, 3)
        self.color_pred_fg_ind = nn.Linear(64, 2)
        self.color_pred_bg_ind = nn.Linear(64, 2)

    def encode(self, img: Tensor, widths: list[int]) -> tuple[Tensor, Tensor, Tensor]:
        """img (N,3,48,W) in [-1,1] -> memory (N,L,E), pad mask (N,L), per-sample memory length.

        Memory length and mask are what each line would get on its own (padded by 7px, as the
        reference does), so masking and cross-attention decay — the one place XPOS is not purely
        relative — does not depend on which other lines share the batch.
        """
        mem = self.backbone(img).squeeze(2).transpose(1, 2)
        L = mem.shape[1]
        alone = torch.tensor([(w + 7) // 4 for w in widths], device=img.device)
        pad = torch.arange(L, device=img.device)[None] >= alone[:, None]
        for layer in self.encoders:
            mem = layer(mem, pad)
        return mem, pad, alone

    def colors(self, h: Tensor) -> Tensor:
        c = self.color_pred1(h)
        return torch.cat(
            [
                self.color_pred_fg(c),
                self.color_pred_bg(c),
                self.color_pred_fg_ind(c),
                self.color_pred_bg_ind(c),
            ],
            -1,
        )

    @torch.inference_mode()
    def decode(
        self, img: Tensor, widths: list[int], beams: int = 5, max_len: int = 255
    ) -> list[Decoded]:
        """Beam search (beams=1 is greedy) over a batch of 48px-high line images."""
        mem, pad, alone = self.encode(img, widths)
        N, Lm, _ = mem.shape
        dev = mem.device
        k = max(1, beams)
        # a 48px line holds about one glyph per 24-48px; cap the search well above that
        max_len = min(max_len, max(widths) // 12 + 16)
        mpos = torch.arange(Lm, device=dev)
        mem_kv = []
        for layer in self.decoders:
            a = layer.multihead_attn
            km, vm = a.kv(mem)
            km = a.xpos(km, mpos, centred(mpos[None], alone[:, None])[:, None], downscale=True)
            mem_kv.append((km, vm))

        # beams flattened as (sample, beam); each sample starts from one live beam
        score = torch.full((N, k), float("-inf"), device=dev)
        score[:, 0] = 0
        B = N * k
        toks = torch.full((B, 1), BOS, dtype=torch.long, device=dev)
        cols = torch.zeros(B, 0, 10, device=dev)
        cache: list[tuple[Tensor, Tensor]] = [
            (torch.zeros(B, HEADS, 0, HEAD_DIM, device=dev),) * 2 for _ in self.decoders
        ]
        done = torch.zeros(B, dtype=torch.bool, device=dev)
        length = torch.zeros(B, dtype=torch.long, device=dev)
        alive = torch.arange(N, device=dev)
        results: dict[int, Decoded] = {}

        for step in range(max_len):
            n = len(alive)
            x = self.embd(toks[:, -1:])
            t = torch.tensor([step], device=dev)
            new_cache = []
            for layer, (kc, vc), (km, vm) in zip(self.decoders, cache, mem_kv):
                sa, ca = layer.self_attn, layer.multihead_attn
                h = layer.norm1(x)
                q = sa.xpos(sa.q(h), t, t)
                kn, vn = sa.kv(h)
                kc = torch.cat([kc, sa.xpos(kn, t, t, downscale=True)], 2)
                vc = torch.cat([vc, vn], 2)
                new_cache.append((kc, vc))
                x = x + sa.attend(q, kc, vc)
                q = ca.xpos(ca.q(layer.norm2(x)), t, centred(t, step + 1))
                x = x + _cross(ca, q, km, vm, pad, n, k)
                x = x + layer.linear2(F.relu(layer.linear1(layer.norm3(x))))
            h = x[:, 0]
            cols = torch.cat([cols, self.colors(h)[:, None]], 1)
            logp = self.pred(self.pred1(h)).log_softmax(-1)

            cand_v, cand_i = logp.topk(k, -1)  # (B, k)
            # a finished beam carries itself forward once, unchanged
            cand_v = cand_v.masked_fill(done[:, None], float("-inf"))
            cand_v[:, 0] = torch.where(done, 0.0, cand_v[:, 0])
            cand_i[:, 0] = torch.where(done, EOS, cand_i[:, 0])
            score, best = (score.view(B, 1) + cand_v).view(n, k * k).topk(k, -1)
            parent = (torch.arange(n, device=dev)[:, None] * k + best // k).view(-1)
            new_tok = cand_i.view(n, k * k).gather(1, best).view(-1)
            was_done = done[parent]
            toks = torch.cat([toks[parent], new_tok[:, None]], 1)
            cols, length = cols[parent], length[parent] + (~was_done).long()
            cache = [(kc[parent], vc[parent]) for kc, vc in new_cache]
            done = was_done | (new_tok == EOS)

            # summed log-probs only fall, so once the best beam has ended it is final
            top_done = done.view(n, k)[:, 0] | (step + 1 == max_len)
            if not bool(top_done.any()):
                continue
            for j in top_done.nonzero().flatten().tolist():
                r, ln = j * k, int(length[j * k])
                ended = bool(done[r])
                n_out = ln - 1 if ended else ln
                results[int(alive[j])] = Decoded(
                    tokens=toks[r, 1 : n_out + 1].tolist(),
                    logprob=float(score[j, 0]) / max(ln, 1),
                    colors=cols[r, :n_out].clone(),
                )
            keep = (~top_done).nonzero().flatten()
            if len(keep) == 0:
                break
            rows = (keep[:, None] * k + torch.arange(k, device=dev)).view(-1)
            alive, score, pad = alive[keep], score[keep], pad[keep]
            mem_kv = [(km[keep], vm[keep]) for km, vm in mem_kv]
            toks, cols, length, done = toks[rows], cols[rows], length[rows], done[rows]
            cache = [(kc[rows], vc[rows]) for kc, vc in cache]
            B = len(rows)
        return [results[i] for i in range(N)]


def _cross(a: Attention, q: Tensor, km: Tensor, vm: Tensor, pad: Tensor, n: int, k: int) -> Tensor:
    """Cross-attention of n*k beams (B,H,1,D) onto their sample's memory (n,H,L,D) without
    copying the memory per beam."""
    q = q.view(n, k, HEADS, HEAD_DIM)
    w = torch.einsum("nkhd,nhld->nhkl", q, km).masked_fill(pad[:, None, None, :], float("-inf"))
    out = torch.einsum("nhkl,nhld->nkhd", w.softmax(-1), vm)
    return a.out_proj(out.reshape(n * k, 1, EMBD))
