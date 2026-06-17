"""sd_webui_cfgzero — CFG-Zero* (optimized-scale) for Forge-derived WebUIs.

Re-exports the public surface from :mod:`sd_webui_cfgzero.core` so the script
layer can ``from sd_webui_cfgzero import apply_cfgzero, remove_cfgzero_patches``
(same convention as sd_webui_tcfg / sd_webui_mahiro).
"""

from sd_webui_cfgzero.core import (
    CFGZERO_HOOK_QUALNAME,
    apply_cfgzero,
    optimized_scale,
    remove_cfgzero_patches,
)

__all__ = [
    "CFGZERO_HOOK_QUALNAME",
    "apply_cfgzero",
    "optimized_scale",
    "remove_cfgzero_patches",
]
