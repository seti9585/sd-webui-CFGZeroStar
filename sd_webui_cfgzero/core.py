"""
sd_webui_cfgzero/core.py
========================
CFG-Zero* core — optimized-scale (stage 1) + zero-init (stage 2).

Paper : arXiv:2503.18886
Port  : ComfyUI built-in ``CFGZeroStar`` (optimized scale) + KJNodes
        ``CFGZeroStarAndInit`` (zero-init), unified into one post-CFG hook.

Optimized scale
---------------
    alpha   = optimized_scale(x - cond_d, x - uncond_d)
            = <v_cond, v_uncond> / (||v_uncond||^2 + 1e-8)
    out_new = out + uncond_d*(alpha-1) + cond_scale*uncond_d*(1-alpha)
Additive ``out + delta`` form stacks on top of earlier post-CFG hooks. Faithful
noise-space (needs x_t = args["input"]; x0-space fallback otherwise).

Measured behaviour (reForge / amanatsuIllustrious_v11 / CFG7): alpha is ~0.999
at the highest sigma and drifts down to ~0.964 at the low-sigma tail. The
per-step correction magnitude is |1-alpha|*(cond_scale-1)*||uncond|| — small
early (~1%) and largest late (~20% near sigma~0.07) — so the effect concentrates
in the late steps (tone / contrast), leaving composition intact.

Zero-init
---------
Zero the model prediction for the first N steps (paper default ~4% of total).
In x0/denoised space the hook returns ``0`` for those steps; the k-diffusion
sampler then steps ``x_next = x + (x - 0)/sigma * dsigma = x*(sigma_next/sigma)``,
so x rescales to track sigma (the early, unreliable prediction is skipped while
the latent stays magnitude-consistent with the noise level).

  NOTE: returning the *input* x_t here instead (a "freeze x / zero the ODE
  update" reading) is WRONG for variance-exploding k-diffusion: x would stay at
  the sigma_max magnitude while sigma drops, and the magnitude/sigma mismatch
  destroys the image (worse with more zeroed steps). Returning 0 is the correct,
  ComfyUI-faithful behaviour.

Locating "the first N steps":
  * Primary  — read the full descending sigma schedule from model_options /
               transformer_options (keys sample_sigmas / sigmas) and zero while
               ``sigma > schedule[N]``. Exact; robust to multi-stage RK and the
               hires pass.
  * Fallback — reForge does NOT expose the schedule in post-CFG
               (transformer_options == ['patches']). When absent, approximate
               the first N of T steps as the top ``N/T`` of the LOG-sigma range:
               capture sigma_max per pass and zero while
               ``log(sigma) > log(sigma_max) - (N/T)*(log(sigma_max)-log(smin))``.
               log-sigma is ~linear in step index for AYS/Karras, so for small N
               this reproduces the exact step boundary (verified on AYS-32).

Backend status (confirmed 2026-06, reForge)
-------------------------------------------
post-CFG arg keys: denoised, cond, uncond, cond_scale, model, uncond_denoised,
cond_denoised, sigma, model_options, input  (so x_t present -> faithful scale).
transformer_options keys: ['patches'] only -> zero-init uses the fallback above.

Robustness / diagnostics
------------------------
The hook is wrapped; any failure returns the standard CFG result. CFGZERO_DEBUG=1
logs arg/transformer_options keys, the sigma-throttled alpha trajectory, and the
resolved zero-init mode/threshold once.

Public surface
--------------
    apply_cfgzero(unet, *, zero_init=False, zero_steps=0, total_steps=0)
    remove_cfgzero_patches(unet)
    CFGZERO_HOOK_QUALNAME
"""

from __future__ import annotations

import logging
import math
import os
import sys

import torch

logger = logging.getLogger(__name__)


def _emit(fmt, *args):
    """Emit a diagnostic line via BOTH the logger and a stderr print, so it
    shows regardless of a backend's logging configuration (reForge surfaces
    module warnings; some forks, e.g. Forge Neo, may not)."""
    try:
        msg = (fmt % args) if args else fmt
    except Exception:
        msg = str(fmt)
    logger.log(logging.WARNING, msg)
    try:
        print(msg, file=sys.stderr, flush=True)
    except Exception:
        pass


CFGZERO_HOOK_QUALNAME = "cfgzero_hook"

CFGZERO_DEBUG = os.environ.get("CFGZERO_DEBUG", "0") not in ("0", "", "false", "False")

# Floor for the fallback's log-sigma range (typical SDXL sigma_min ~0.03).
ZERO_INIT_SIGMA_MIN_FLOOR = 0.02

# One-shot latches / per-pass state (process-wide; harmless across generations).
_WARNED_NO_INPUT = False
_WARNED_HOOK_FAIL = False
_WARNED_ZI_FALLBACK = False
_DEBUG_KEYS_DUMPED = False
_DEBUG_ZI_DUMPED = False
_LAST_LOG_SIGMA = None
_ZI_SIGMA_MAX = None     # captured max sigma of the current pass (fallback)
_ZI_LAST_SIGMA = None    # for new-pass detection (sigma rising = new pass)


# --------------------------------------------------------------------------- #
# Core math                                                                   #
# --------------------------------------------------------------------------- #

def optimized_scale(positive: torch.Tensor, negative: torch.Tensor) -> torch.Tensor:
    """Per-sample least-squares scale ``s* = <pos, neg> / ||neg||^2`` -> (B,1,1,1)."""
    pos_flat = positive.reshape(positive.shape[0], -1).float()
    neg_flat = negative.reshape(negative.shape[0], -1).float()
    dot     = torch.sum(pos_flat * neg_flat, dim=1, keepdim=True)
    sq_norm = torch.sum(neg_flat ** 2, dim=1, keepdim=True) + 1e-8
    st_star = dot / sq_norm
    return st_star.reshape([positive.shape[0]] + [1] * (positive.ndim - 1))


# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #

def _sigma_of(args: dict) -> float:
    s = args.get("sigma")
    if s is None:
        return float("nan")
    try:
        return float(torch.as_tensor(s).flatten()[0].item())
    except Exception:
        return float("nan")


def _sigma_schedule(args: dict):
    """Best-effort full (descending) sigma schedule, or None. Probes several
    fork-specific locations (reForge currently exposes none of them)."""
    candidates = []
    mo = args.get("model_options")
    if isinstance(mo, dict):
        candidates.append(mo)
        to = mo.get("transformer_options")
        if isinstance(to, dict):
            candidates.append(to)
    candidates.append(args)  # some forks stash it at the top level
    for d in candidates:
        for k in ("sample_sigmas", "sigmas"):
            v = d.get(k)
            if v is None:
                continue
            try:
                t = torch.as_tensor(v).flatten().float()
                if t.numel() >= 2:
                    return t
            except Exception:
                pass
    return None


def _in_zero_init_region(args: dict, zero_steps: int, total_steps: int) -> bool:
    """True if the current eval is within the first ``zero_steps`` steps (so the
    model prediction should be zeroed). Uses the exact schedule when available,
    else an approximate log-sigma fraction fallback."""
    global _WARNED_ZI_FALLBACK, _DEBUG_ZI_DUMPED, _ZI_SIGMA_MAX, _ZI_LAST_SIGMA

    cur = _sigma_of(args)
    if cur != cur:  # NaN -> cannot locate; do not zero
        return False

    # --- per-pass sigma_max capture (new pass = sigma rose vs last call) ---
    if _ZI_LAST_SIGMA is None or cur > _ZI_LAST_SIGMA * 1.0001:
        _ZI_SIGMA_MAX = cur
    _ZI_LAST_SIGMA = cur

    # --- primary: exact schedule ------------------------------------------
    sched = _sigma_schedule(args)
    if sched is not None:
        n = int(min(max(zero_steps, 1), sched.numel() - 1))
        thr = float(sched[n])
        if CFGZERO_DEBUG and not _DEBUG_ZI_DUMPED:
            _emit("[CFG-Zero*] zero-init: mode=schedule steps=%d "
                  "threshold_sigma=%.4f sched_len=%d", n, thr, int(sched.numel()))
            _DEBUG_ZI_DUMPED = True
        return cur > thr + 1e-4

    # --- fallback: log-sigma fraction -------------------------------------
    if not _WARNED_ZI_FALLBACK:
        _emit("[CFG-Zero*] zero-init: sigma schedule not exposed by backend; "
              "using approximate log-sigma fraction fallback.")
        _WARNED_ZI_FALLBACK = True

    smax = _ZI_SIGMA_MAX if (_ZI_SIGMA_MAX and _ZI_SIGMA_MAX > 0) else cur
    smin = ZERO_INIT_SIGMA_MIN_FLOOR
    if smax <= smin:
        return False
    t = int(total_steps) if total_steps and total_steps > 0 else 0
    frac = (zero_steps / t) if t > 0 else 0.04
    frac = max(0.0, min(frac, 0.99))
    thr_log = math.log(smax) - frac * (math.log(smax) - math.log(smin))
    thr = math.exp(thr_log)
    if CFGZERO_DEBUG and not _DEBUG_ZI_DUMPED:
        _emit("[CFG-Zero*] zero-init: mode=fallback steps=%d frac=%.4f "
              "sigma_max=%.4f threshold_sigma=%.4f",
              int(zero_steps), frac, smax, thr)
        _DEBUG_ZI_DUMPED = True
    return cur > thr


# --------------------------------------------------------------------------- #
# Hook factory                                                                #
# --------------------------------------------------------------------------- #

def _make_hook(zero_init: bool, zero_steps: int, total_steps: int):
    @torch.no_grad()
    def cfgzero_hook(args: dict) -> torch.Tensor:
        global _WARNED_NO_INPUT, _WARNED_HOOK_FAIL, _DEBUG_KEYS_DUMPED, _LAST_LOG_SIGMA

        out = args.get("denoised")
        try:
            uncond_d   = args.get("uncond_denoised")
            cond_d     = args.get("cond_denoised")
            cond_scale = args.get("cond_scale")

            if out is None or uncond_d is None or cond_d is None:
                return out
            if not torch.any(uncond_d):
                return out
            try:
                cfg_is_one = (float(cond_scale) == 1.0)
            except (TypeError, ValueError):
                cfg_is_one = False

            x = args.get("input")

            if CFGZERO_DEBUG and not _DEBUG_KEYS_DUMPED:
                _emit("[CFG-Zero*] post-CFG keys=%s", list(args.keys()))
                mo = args.get("model_options") or {}
                _emit("[CFG-Zero*] model_options keys=%s",
                               list(mo.keys()) if isinstance(mo, dict) else type(mo).__name__)
                to = mo.get("transformer_options") if isinstance(mo, dict) else None
                _emit("[CFG-Zero*] transformer_options keys=%s",
                               list(to.keys()) if isinstance(to, dict) else type(to).__name__)
                _DEBUG_KEYS_DUMPED = True

            # stage 2: zero-init. Zero the model prediction (denoised := 0) for
            # the first N steps. In k-diffusion the sampler then steps
            # x_next = x + (x - 0)/sigma * dsigma = x * (sigma_next/sigma), i.e.
            # x rescales to track sigma (the early uncertain prediction is
            # skipped) instead of being frozen at the wrong magnitude.
            if zero_init and zero_steps >= 1 and _in_zero_init_region(args, zero_steps, total_steps):
                return torch.zeros_like(out)

            if cfg_is_one:
                return out

            # stage 1: optimized scale.
            use_noise_space = x is not None
            if use_noise_space:
                pos, neg = x - cond_d, x - uncond_d
            else:
                if not _WARNED_NO_INPUT:
                    _emit("[CFG-Zero*] post-CFG args has no 'input' (x_t); "
                                   "using x0-space scale fallback (NOT faithful).")
                    _WARNED_NO_INPUT = True
                pos, neg = cond_d, uncond_d

            alpha = optimized_scale(pos, neg).to(out.dtype)

            if CFGZERO_DEBUG:
                sig = _sigma_of(args)
                changed = (_LAST_LOG_SIGMA is None or _LAST_LOG_SIGMA <= 0 or sig != sig
                           or abs(sig - _LAST_LOG_SIGMA) / max(_LAST_LOG_SIGMA, 1e-6) > 0.03)
                if changed:
                    _emit("[CFG-Zero*] path=%s sigma=%.4f alpha.mean=%.4f "
                                   "alpha.min=%.4f alpha.max=%.4f",
                                   "noise" if use_noise_space else "x0", sig,
                                   float(alpha.float().mean()),
                                   float(alpha.float().min()), float(alpha.float().max()))
                    _LAST_LOG_SIGMA = sig

            return out + uncond_d * (alpha - 1.0) + cond_scale * uncond_d * (1.0 - alpha)

        except Exception as exc:
            if not _WARNED_HOOK_FAIL:
                _emit("[CFG-Zero*] hook skipped (returning standard CFG): %r", exc)
                _WARNED_HOOK_FAIL = True
            return out

    cfgzero_hook.__qualname__ = CFGZERO_HOOK_QUALNAME
    return cfgzero_hook


# --------------------------------------------------------------------------- #
# Fail-safe + registration                                                    #
# --------------------------------------------------------------------------- #

def remove_cfgzero_patches(unet) -> None:
    """Strip this extension's own post-CFG hook from *unet*, in place."""
    opts = getattr(unet, "model_options", None)
    if not isinstance(opts, dict):
        return
    fns = opts.get("sampler_post_cfg_function")
    if not fns:
        return
    opts["sampler_post_cfg_function"] = [
        fn for fn in fns if getattr(fn, "__qualname__", None) != CFGZERO_HOOK_QUALNAME
    ]


def apply_cfgzero(unet, *, zero_init: bool = False, zero_steps: int = 0,
                  total_steps: int = 0):
    """Register the CFG-Zero* post-CFG hook on *unet* (deduplicates first).

    zero_init / zero_steps / total_steps are resolved by the script layer
    (total_steps feeds the schedule-free zero-init fallback). Pure post-CFG,
    appended (additive form is order-robust). Returns the same unet object.
    """
    remove_cfgzero_patches(unet)
    hook = _make_hook(bool(zero_init), int(zero_steps), int(total_steps))
    unet.set_model_sampler_post_cfg_function(hook)
    return unet
