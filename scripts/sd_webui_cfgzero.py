"""
sd-webui-CFGZeroStar — CFG-Zero* for Forge-derived WebUIs
========================================================
Location: extensions/sd-webui-CFGZeroStar/scripts/sd_webui_cfgzero.py

Paper : arXiv:2503.18886
Port  : ComfyUI built-in ``CFGZeroStar`` (optimized scale) + KJNodes
        ``CFGZeroStarAndInit`` (zero-init), as one post-CFG hook.

Hook  : set_model_sampler_post_cfg_function   (pure post-CFG, x0 space)

Compatibility:
    OK   reForge / Forge Classic / Forge (lllyasviel) / Forge Neo
    NO   A1111 — no Forge backend

UI (kept deliberately minimal — WebUI's value is "tick and go"):
    * Enable CFG-Zero*          — optimized-scale correction
    * Enable zero-init          — zero the first few ODE steps
    * Zero-init steps (0 = auto) — auto resolves to ~4% of total steps

Ordering note (Forge Neo): post-CFG hooks run in registration order, not by
sorting_priority. The additive form makes this hook order-robust.
"""

import logging
import os
import sys
import traceback
from functools import partial
from typing import Any

import gradio as gr
from modules import scripts, script_callbacks

# --------------------------------------------------------------------------- #
# sys.path — ensure the extension root is importable
# --------------------------------------------------------------------------- #
_EXT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _EXT_ROOT not in sys.path:
    sys.path.insert(0, _EXT_ROOT)
# --------------------------------------------------------------------------- #

from sd_webui_cfgzero import apply_cfgzero, remove_cfgzero_patches

logger = logging.getLogger(__name__)

ZERO_INIT_AUTO_FRACTION = 0.04  # paper default: ~4% of total steps


# --------------------------------------------------------------------------- #
# Backend detection
# --------------------------------------------------------------------------- #

def _has_forge_backend(p) -> bool:
    return hasattr(p, "sd_model") and hasattr(p.sd_model, "forge_objects")


def _warn_no_forge() -> None:
    msg = (
        "[sd-webui-CFGZeroStar] Requires Forge backend "
        "(reForge / Forge Classic / Forge Neo / Forge). A1111 is not supported."
    )
    logger.warning(msg)
    print(msg, file=sys.stderr)


def _resolve_zero_steps(p, zero_init: bool, zero_steps_ui: int) -> int:
    """Effective number of leading steps to zero. 0 from UI -> auto (~4%)."""
    if not zero_init:
        return 0
    if zero_steps_ui and zero_steps_ui > 0:
        return int(zero_steps_ui)
    total = int(getattr(p, "steps", 0) or 0)
    if total <= 0:
        return 1
    return max(1, round(ZERO_INIT_AUTO_FRACTION * total))


# --------------------------------------------------------------------------- #
# Script
# --------------------------------------------------------------------------- #

class CFGZeroStarScript(scripts.Script):
    """CFG-Zero* — optimized scale (+ optional zero-init)."""

    sorting_priority = 15.0

    def __init__(self):
        self.enabled = False

    def title(self) -> str:
        return "CFG-Zero*"

    def show(self, is_img2img: bool):
        return scripts.AlwaysVisible

    def ui(self, is_img2img: bool):
        with gr.Accordion(open=False, label=self.title()):
            gr.HTML(
                "<p><i>"
                "<b>Post-CFG</b>: CFG-Zero*. Rescales the unconditional branch "
                "by the least-squares optimal factor s* (optimized scale), and "
                "optionally zeroes the first few ODE steps (zero-init). Operates "
                "in x0 space — prediction-space agnostic, so it works on SDXL "
                "and flow-matching (Anima) models alike. Requires Forge backend."
                "</i></p>"
            )
            enabled = gr.Checkbox(label="Enable CFG-Zero*", value=False)
            with gr.Row():
                zero_init = gr.Checkbox(label="Enable zero-init", value=False)
                zero_steps = gr.Number(
                    label="Zero-init steps (0 = auto 4%)",
                    value=0, minimum=0, maximum=100, step=1, precision=0,
                    info="Leading steps to zero. 0 = auto (~4% of total steps).",
                )
            gr.HTML(
                "<p style='color:gray;font-size:0.9em;'>"
                "💡 Keep CFG ~7-10 when stacking CFG-axis extensions. zero-init "
                "needs the sampler's sigma schedule; if a backend doesn't expose "
                "it, zero-init is skipped and optimized-scale still applies."
                "</p>"
            )

        enabled.change(fn=lambda x: self._update_enabled(x), inputs=[enabled])
        return [enabled, zero_init, zero_steps]

    def _update_enabled(self, value: bool) -> None:
        self.enabled = value

    def process_before_every_sampling(self, p, *args, **kwargs):
        enabled       = bool(args[0]) if len(args) >= 1 else False
        zero_init     = bool(args[1]) if len(args) >= 2 else False
        zero_steps_ui = int(args[2])  if len(args) >= 3 else 0

        # XYZ Grid overrides
        xyz = getattr(p, "_cfgzero_xyz", {})
        if "enabled" in xyz:
            enabled = (xyz["enabled"] == "True")
        if "zero_init" in xyz:
            zero_init = (xyz["zero_init"] == "True")
        if "zero_steps" in xyz:
            try:
                zero_steps_ui = int(xyz["zero_steps"])
            except (TypeError, ValueError):
                pass

        self.enabled = enabled
        if not enabled:
            return

        if not _has_forge_backend(p):
            _warn_no_forge()
            return

        zero_steps = _resolve_zero_steps(p, zero_init, zero_steps_ui)
        total_steps = int(getattr(p, "steps", 0) or 0)

        unet = p.sd_model.forge_objects.unet.clone()
        remove_cfgzero_patches(unet)   # fail-safe: never stack two copies
        apply_cfgzero(unet, zero_init=zero_init, zero_steps=zero_steps,
                      total_steps=total_steps)
        p.sd_model.forge_objects.unet = unet

        gen_params = {"cfgzero": "enabled"}
        if zero_init:
            gen_params["cfgzero_zero_init_steps"] = zero_steps
        p.extra_generation_params.update(gen_params)
        logger.debug("[CFG-Zero*] applied (zero_init=%s, zero_steps=%d)",
                     zero_init, zero_steps)


# --------------------------------------------------------------------------- #
# XYZ Grid support
# --------------------------------------------------------------------------- #

def _set_xyz_value(p, x: Any, xs: Any, *, field: str) -> None:
    if not hasattr(p, "_cfgzero_xyz"):
        p._cfgzero_xyz = {}
    p._cfgzero_xyz[field] = x


def _register_xyz_axes() -> None:
    xyz_grid = None
    for script in scripts.scripts_data:
        if script.script_class.__module__ == "xyz_grid.py":
            xyz_grid = script.module
            break

    if xyz_grid is None:
        return

    new_axes = [
        xyz_grid.AxisOption(
            "(CFG-Zero*) Enabled", str,
            partial(_set_xyz_value, field="enabled"),
            choices=lambda: ["True", "False"],
        ),
        xyz_grid.AxisOption(
            "(CFG-Zero*) Zero-init", str,
            partial(_set_xyz_value, field="zero_init"),
            choices=lambda: ["True", "False"],
        ),
        xyz_grid.AxisOption(
            "(CFG-Zero*) Zero-init steps", str,
            partial(_set_xyz_value, field="zero_steps"),
        ),
    ]

    if not any(x.label.startswith("(CFG-Zero*)") for x in xyz_grid.axis_options):
        xyz_grid.axis_options.extend(new_axes)


def _on_before_ui() -> None:
    try:
        _register_xyz_axes()
    except Exception:
        print(
            f"[sd-webui-CFGZeroStar] XYZ Grid registration failed:\n{traceback.format_exc()}",
            file=sys.stderr,
        )


script_callbacks.on_before_ui(_on_before_ui)
