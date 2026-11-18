"""A TopK sparse autoencoder on layer 12 of a trained or a re-initialised Gemma-2-2B.

- **Models.** ``--model trained`` loads ``google/gemma-2-2b``; ``--model random`` builds the
  same configuration with its default initialisation under a fixed seed. Both are cut to
  decoder blocks 0-12; the read-out is the output of block 12, with eager attention, in bf16.
- **Data.** The shared shuffled FineWeb stream (``random_model_control/token_stream.py``);
  the BOS position of every chunk is excluded.
- **Normalisation.** One scalar per model, c = sqrt(d / E||x||^2), estimated on the
  normalisation chunks.
- **Autoencoder.** TopK (k = 92) with 16,384 latents:
  - z = TopK(W_enc (x - b_pre) + b_enc), x_hat = W_dec z + b_pre;
  - unit-norm decoder rows (the parallel gradient component is removed);
  - W_enc = W_dec^T at initialisation, and b_pre = the mean of the first buffer;
  - AuxK (k_aux = 512, coefficient 1/32) over latents not fired in the last 10 M tokens;
  - Adam (0.9, 0.999, 6.25e-10), learning rate 2e-4 with 1,000 warm-up steps;
  - batches of 4,096 from a 262,144-token shuffling buffer, half refreshed per round.
- **Checkpoints.** At 0 (the initialisation), 25 M, 50 M and 100 M tokens: weights, the
  fraction of variance unexplained on the held-out chunks, and dead fractions.

``--smoke`` runs a few thousand tokens on the CPU to check the code path.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

SEED = 20260901
D_SAE, K, K_AUX, AUX_COEF = 16384, 92, 512, 1 / 32
LR, WARMUP, BATCH, BUFFER = 2e-4, 1000, 4096, 262_144
DEAD_TOKENS = 10_000_000
CHECKPOINTS = (25_000_000, 50_000_000, 100_000_000)
LAYER = 12
P1 = Path(__file__).resolve().parents[1]


def load_model(which: str, device: str, dtype):
    from transformers import AutoConfig, AutoModelForCausalLM
    if which == "trained":
        m = AutoModelForCausalLM.from_pretrained("google/gemma-2-2b", torch_dtype=torch.float32,
                                                 attn_implementation="eager")
    else:
        cfg = AutoConfig.from_pretrained("google/gemma-2-2b")
        cfg._attn_implementation = "eager"
        torch.manual_seed(SEED)
        m = AutoModelForCausalLM.from_config(cfg, attn_implementation="eager")
    base = m.model
    base.layers = base.layers[: LAYER + 1]
    base.config.num_hidden_layers = LAYER + 1
    del m
    return base.to(device=device, dtype=dtype).eval()


class Reader:
    """Layer-12 activations (positions 1..1023) for batches of token chunks."""

    def __init__(self, base, device):
        self.base, self.device = base, device
        self.h = {}
        base.layers[LAYER].register_forward_hook(
            lambda _m, _i, out: self.h.__setitem__("x", out[0] if isinstance(out, tuple) else out))

    @torch.no_grad()
    def __call__(self, chunks: np.ndarray) -> torch.Tensor:
        x = torch.from_numpy(np.array(chunks, dtype=np.int64)).to(self.device)
        self.base(input_ids=x, use_cache=False)
        return self.h["x"][:, 1:, :].reshape(-1, self.h["x"].shape[-1])


class TopKSAE(torch.nn.Module):
    def __init__(self, d: int, n: int, k: int, gen: torch.Generator):
        super().__init__()
        W = torch.randn(n, d, generator=gen)
        W = W / W.norm(dim=1, keepdim=True)
        self.W_dec = torch.nn.Parameter(W.clone())
        self.W_enc = torch.nn.Parameter(W.T.clone().contiguous())
        self.b_enc = torch.nn.Parameter(torch.zeros(n))
        self.b_pre = torch.nn.Parameter(torch.zeros(d))
        self.k = k

    def pre(self, x):
        return (x - self.b_pre) @ self.W_enc + self.b_enc

    def encode(self, x):
        p = self.pre(x)
        v, i = p.topk(self.k, dim=-1)
        z = torch.zeros_like(p).scatter_(-1, i, torch.relu(v))
        return z, p

    def decode(self, z):
        return z @ self.W_dec + self.b_pre

    @torch.no_grad()
    def project_grad_and_norm(self, after_step: bool):
        W = self.W_dec
        if not after_step and W.grad is not None:
            W.grad -= (W.grad * W).sum(1, keepdim=True) * W
        if after_step:
            W.data /= W.data.norm(dim=1, keepdim=True)


def fvu(sae, X, bs=8192):
    """Fraction of variance unexplained over a set of activations (on the SAE's device)."""
    num, s1, s2, n = 0.0, 0.0, 0.0, 0
    active = torch.zeros(sae.b_enc.shape[0], dtype=torch.bool, device=sae.b_enc.device)
    with torch.no_grad():
        for i in range(0, len(X), bs):
            x = X[i:i + bs].to(sae.b_enc.device, torch.float32)
            z, _ = sae.encode(x)
            num += float(((sae.decode(z) - x) ** 2).sum())
            s1 = s1 + x.sum(0)
            s2 += float((x ** 2).sum())
            n += len(x)
            active |= (z > 0).any(0)
    den = s2 - float((s1 ** 2).sum()) / n
    return num / den, float((~active).float().mean())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=("trained", "random"), required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out-root", type=Path, default=P1 / "results" / "controls" / "random_model")
    args = ap.parse_args()
    out = args.out_root / args.model
    out.mkdir(parents=True, exist_ok=True)
    man = json.load(open(P1 / "data" / "fineweb" / "a3_chunks.json"))
    chunks = np.load(P1 / "data" / "fineweb" / "a3_chunks.npy", mmap_mode="r")
    dev = torch.device(args.device)
    dtype = torch.bfloat16 if dev.type == "cuda" else torch.float32
    tr0, tr1 = man["train"]
    ev0, ev1 = man["eval"]
    nm0, nm1 = man["norm"]
    rows_per_fwd = 16
    buffer, batch, ckpts = BUFFER, BATCH, CHECKPOINTS
    if args.smoke:
        tr1, ev1, nm1 = tr0 + 12, ev0 + 2, nm0 + 2
        rows_per_fwd, buffer, batch = 2, 4096, 512
        ckpts = (4_092, 8_184, 12_276)
    t0 = time.time()
    base = load_model(args.model, dev, dtype)
    read = Reader(base, dev)
    print(f"{args.model} model on {dev} ({dtype}); loaded in {time.time() - t0:.0f} s", flush=True)

    # ---- normalisation constant and held-out evaluation set ---------------------------
    sq, n = 0.0, 0
    for i in range(nm0, nm1, rows_per_fwd):
        x = read(np.asarray(chunks[i:min(i + rows_per_fwd, nm1)])).float()
        sq += float((x ** 2).sum())
        n += len(x)
    d = base.config.hidden_size
    c = float(np.sqrt(d / (sq / n)))
    EV = torch.cat([(read(np.asarray(chunks[i:min(i + rows_per_fwd, ev1)])).float() * c)
                    .to(torch.bfloat16).cpu() for i in range(ev0, ev1, rows_per_fwd)])
    print(f"normalisation c = {c:.5g}; evaluation set {len(EV):,} tokens", flush=True)

    # ---- buffer -----------------------------------------------------------------------
    gen = torch.Generator().manual_seed(SEED)
    nxt = [tr0]

    def produce(m):
        """m fresh normalised activations from the training stream (in order)."""
        parts, got = [], 0
        while got < m and nxt[0] < tr1:
            j = nxt[0]
            x = read(np.asarray(chunks[j:min(j + rows_per_fwd, tr1)])) * c
            nxt[0] = min(j + rows_per_fwd, tr1)
            parts.append(x.to(torch.bfloat16))
            got += len(x)
        return torch.cat(parts) if parts else torch.empty(0, d, dtype=torch.bfloat16, device=dev)

    leftover = torch.empty(0, d, dtype=torch.bfloat16, device=dev)

    def fill(m):
        nonlocal leftover
        while len(leftover) < m and nxt[0] < tr1:
            leftover = torch.cat([leftover, produce(m - len(leftover))])
        take, leftover = leftover[:m], leftover[m:]
        return take

    B = fill(buffer)
    sae = TopKSAE(d, D_SAE, K, gen).to(dev)
    with torch.no_grad():
        sae.b_pre.copy_(B.float().mean(0))
    opt = torch.optim.Adam(sae.parameters(), lr=LR, betas=(0.9, 0.999), eps=6.25e-10)
    last_fired = torch.zeros(D_SAE, dtype=torch.long, device=dev)   # token count at last fire
    seen, step, log = 0, 0, []
    total = (tr1 - tr0) * (chunks.shape[1] - 1)
    ck = list(ckpts)
    labels = ["25", "50", "100"]
    # step 0: the initialisation dictionary, saved before the first optimiser step
    f0, never0 = fvu(sae, EV)
    rec0 = {"tokens": 0, "step": 0, "fvu_eval": f0, "dead_frac_10M_window": float("nan"),
            "never_active_on_eval_frac": never0, "train_loss": float("nan"),
            "minutes": (time.time() - t0) / 60}
    log.append(rec0)
    torch.save({"W_enc": sae.W_enc.detach().cpu(), "b_enc": sae.b_enc.detach().cpu(),
                "W_dec": sae.W_dec.detach().cpu(), "b_pre": sae.b_pre.detach().cpu(),
                "k": K, "c": c, **rec0}, out / "sae_0M.pt")
    print(f"  checkpoint 0 (initialisation): FVU {f0:.4f}, never active on eval {never0:.4f}",
          flush=True)
    tgen = torch.Generator(device=dev).manual_seed(SEED)
    while len(B):
        perm = torch.randperm(len(B), generator=tgen, device=dev)
        B = B[perm]
        half = len(B) // 2 if nxt[0] < tr1 or len(leftover) else len(B)
        for i in range(0, half, batch):
            x = B[i:min(i + batch, half)].float()
            z, p = sae.encode(x)
            xh = sae.decode(z)
            e = x - xh
            loss = (e ** 2).sum(-1).mean() / (x - x.mean(0)).pow(2).sum(-1).mean()
            dead = (seen - last_fired) > DEAD_TOKENS
            aux = torch.zeros((), device=dev)
            if dead.any():
                pa = p.masked_fill(~dead, float("-inf"))
                ka = int(min(K_AUX, int(dead.sum())))
                va, ia = pa.topk(ka, dim=-1)
                za = torch.zeros_like(p).scatter_(-1, ia, torch.relu(va))
                eh = za @ sae.W_dec
                aux = (e.detach() - eh).pow(2).sum(-1).mean() / e.detach().pow(2).sum(-1).mean().clamp_min(1e-12)
            lr = LR * min(1.0, (step + 1) / WARMUP)
            for g in opt.param_groups:
                g["lr"] = lr
            opt.zero_grad(set_to_none=True)
            (loss + AUX_COEF * aux).backward()
            sae.project_grad_and_norm(after_step=False)
            opt.step()
            sae.project_grad_and_norm(after_step=True)
            seen += len(x)
            fired = (z > 0).any(0)
            last_fired[fired] = seen
            step += 1
            if step % 500 == 0:
                print(f"  step {step:,}: {seen / 1e6:.1f} M tokens, loss {float(loss):.4f}, "
                      f"aux {float(aux):.4f}, dead {float(dead.float().mean()):.3f}, "
                      f"{(time.time() - t0) / 60:.1f} min", flush=True)
            if ck and seen >= ck[0]:
                f_ev, never = fvu(sae, EV)
                rec = {"tokens": seen, "step": step, "fvu_eval": f_ev,
                       "dead_frac_10M_window": float(((seen - last_fired) > DEAD_TOKENS).float().mean()),
                       "never_active_on_eval_frac": never, "train_loss": float(loss),
                       "minutes": (time.time() - t0) / 60}
                log.append(rec)
                torch.save({"W_enc": sae.W_enc.detach().cpu(), "b_enc": sae.b_enc.detach().cpu(),
                            "W_dec": sae.W_dec.detach().cpu(), "b_pre": sae.b_pre.detach().cpu(),
                            "k": K, "c": c, **rec}, out / f"sae_{labels[0]}M.pt")
                print(f"  checkpoint {ck[0]:,}: FVU {f_ev:.4f}, dead {rec['dead_frac_10M_window']:.4f}, "
                      f"never active on eval {never:.4f}", flush=True)
                ck.pop(0)
                labels.pop(0)
        B = torch.cat([B[half:], fill(half)]) if half < len(B) else fill(buffer)
    json.dump({"model": args.model, "c": c, "seen_tokens": seen, "steps": step,
               "total_train_tokens": total, "checkpoints": log, "smoke": args.smoke,
               "minutes": (time.time() - t0) / 60}, open(out / "train_log.json", "w"), indent=1)
    print(f"done: {seen:,} tokens, {step:,} steps, {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
