# sd-webui-CFGZeroStar

**CFG-Zero\*** for Forge-derived WebUIs — both the *optimized-scale* term and
*zero-init*, in a single post-CFG hook.

A port of the ComfyUI built-in [`CFGZeroStar`](https://github.com/comfyanonymous/ComfyUI)
node (optimized scale) and the KJNodes `CFGZeroStarAndInit` behaviour
(zero-init), based on the paper
[**CFG-Zero\*: Improved Classifier-Free Guidance for Flow Matching Models**](https://arxiv.org/abs/2503.18886)
(Fan et al., 2025).

---

## What it does

CFG-Zero\* has two parts; this extension implements both.

**1. Optimized scale.** A per-sample scalar `s*` — the least-squares optimal
scale of the unconditional prediction relative to the conditional one — is
applied to the unconditional branch:

```
s*       = <v_cond, v_uncond> / (||v_uncond||^2 + 1e-8)
denoised = s* * uncond + cfg_scale * (cond - s* * uncond)
```

with `v_cond = x - cond_denoised`, `v_uncond = x - uncond_denoised` (x0-space
prediction gaps). This corrects inaccuracies in the estimated velocity/score.

**2. Zero-init.** The model prediction is zeroed for the first few steps (the
paper's default is ~4% of total), where the flow/score estimate is least
reliable. In x0/denoised space the hook returns `0` for those steps; the
k-diffusion sampler then steps `x_next = x * (sigma_next / sigma)`, so the latent
rescales to track sigma (the early prediction is skipped while x stays
magnitude-consistent with the noise level). The "first N steps" are located from
the sampler's sigma schedule (no step counting), which keeps it correct under
multi-stage solvers (RK) and the hires.fix second pass.

### Architecture-agnostic by construction

Both parts run in **x0 (denoised) space** in a **post-CFG** hook. By the time
the hook fires, the model output (epsilon / velocity / flow-matching) has been
converted to an x0 estimate, so the behaviour does not depend on the model's
prediction space. Verified to take effect on SDXL-class UNet models (Illustrious
etc.) as well as flow-matching DiT models (Anima).

---

## Install

WebUI → **Extensions** → **Install from URL**, paste
`https://github.com/seti9585/sd-webui-CFGZeroStar`, **Apply and restart UI**. Or
from your WebUI root:

```bash
cd extensions
git clone https://github.com/seti9585/sd-webui-CFGZeroStar
```

Or extract the release `.zip` into `extensions/` so that
`extensions/sd-webui-CFGZeroStar/` exists, and restart. No extra dependencies.

## Compatibility

| Backend                      | Status |
|------------------------------|:------:|
| reForge                      |   ✅   |
| Forge Classic                |   ✅   |
| Forge (original, lllyasviel) |   ✅   |
| Forge Neo                    |   ✅   |
| A1111 (AUTOMATIC1111)        |   ❌   |

A1111 lacks the Forge backend's `set_model_sampler_post_cfg_function`. Verified
on SDXL (UNet) and Anima (flow-matching DiT) under Forge Neo, and on SDXL under
reForge.

## Usage

Open the **CFG-Zero\*** panel (txt2img / img2img):

- **Enable CFG-Zero\*** — turns on the optimized-scale correction.
- **Enable zero-init** — additionally zeroes the first few ODE steps.
- **Zero-init steps (0 = auto 4%)** — leading steps to zero; `0` resolves to
  ~4% of the total step count automatically.

When zero-init is active, the effective step count is written to the image
metadata as `cfgzero_zero_init_steps` for reproducibility. XYZ Grid axes are
registered for **Enabled**, **Zero-init**, and **Zero-init steps** so you can do
same-seed comparison grids directly.

**Recommended CFG:** ~7–10. As with the other CFG-axis extensions in this family
(TCFG / FreSca / MaHiRo / SkimmedCFG), stacking many post-CFG corrections at very
high CFG (≈20+) can let numerical error accumulate.

**Note on zero-init:** it locates the first N steps from the sampler's sigma
schedule when the backend exposes it (`model_options.transformer_options`).
Neither reForge nor Forge Neo currently exposes it in post-CFG, so on those
backends zero-init automatically uses an approximate log-sigma fraction fallback
(verified to reproduce the exact step boundary for small N on AYS schedules). A
one-time log line reports which mode is used (`mode=schedule` / `mode=fallback`).

**zero-init is not a quality feature here.** Dose tests (same seed, N = 0/1/2/4/8)
on both an SDXL/UNet model and a flow-matching (Anima) model, run through the
sampler's sigma/ODE interface, show that zero-init re-rolls early composition and
*reduces* contrast/saturation at higher N — with no quality gain on either
architecture. The paper's benefit is tied to native flow-matching pipelines,
which Forge's unified σ samplers do not reproduce. Keep it low (the 4% auto
default ≈ 1 step at 32 steps); treat it as a composition-variation knob, not an
enhancement. The optimized-scale term is the validated, useful part of this
extension.

## How it composes with other extensions

Registered via `set_model_sampler_post_cfg_function`. The optimized-scale
correction uses an **additive** (`out + delta`) form, so it stacks on top of
earlier post-CFG hooks (FreSca, MaHiRo, TCFG) rather than discarding them.

> **Forge Neo ordering:** post-CFG hooks run in registration order, not by
> `sorting_priority`. The additive form keeps this hook order-robust; if you
> need a strict order, control it via extension load order.

## Layout

```
sd-webui-CFGZeroStar/
├── scripts/
│   └── sd_webui_cfgzero.py     # Gradio UI + hook registration
└── sd_webui_cfgzero/
    ├── __init__.py             # public surface re-export
    └── core.py                 # optimized_scale + zero-init + apply/remove
```

## Debugging

Run the WebUI with the env var `CFGZERO_DEBUG=1` (or set `CFGZERO_DEBUG = True`
in `core.py`) to log, once per process: the post-CFG arg keys, the
`transformer_options` keys, the (sigma-throttled) alpha trajectory, and the
resolved zero-init threshold. Useful for confirming the faithful noise-space
path and whether the sigma schedule is available for zero-init.

---

## Credits & licensing

- **Algorithm / original code:** [WeichenFan/CFG-Zero-star](https://github.com/WeichenFan/CFG-Zero-star),
  licensed **Apache-2.0**. `optimized_scale` and the zero-init concept are
  reimplementations of that work.
- **Port references:** the ComfyUI built-in `CFGZeroStar` node
  (`comfy_extras/nodes_cfg.py`) for the optimized-scale post-CFG adaptation, and
  KJNodes `CFGZeroStarAndInit` for the zero-init behaviour.
- This extension's own code is released under the **MIT License** (see
  `LICENSE`). The upstream method is Apache-2.0 (permissive, MIT-compatible);
  redistribution under MIT with the attribution above is fine, but confirm you
  are comfortable with it before publishing. *(General note, not legal advice.)*

### Citation

```bibtex
@misc{fan2025cfgzerostar,
  title         = {CFG-Zero*: Improved Classifier-Free Guidance for Flow Matching Models},
  author        = {Weichen Fan and Amber Yijia Zheng and Raymond A. Yeh and Ziwei Liu},
  year          = {2025},
  eprint        = {2503.18886},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CV},
  url           = {https://arxiv.org/abs/2503.18886}
}
```
