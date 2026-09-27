"""Temporal Identity Anchoring (TIA).

TIA splats the local appearance of an instance across time through the backbone's
own patch correspondence, which keeps identity stable without adding any
per-instance parameters at inference time.
"""

from .adapter import TIAAdapter, TIAConfig, attach_adapter, build_adapter
from .layer_probe import ProbeResult, layer_epe, probe_layers, run_probe
from .loss import contrastive_loss, lambda_schedule, noise_gate
from .matcher import (
    correlation_matrix,
    local_window_indices,
    normalize_features,
    retrieve,
    select_layer,
)
from .transport import first_frame_unchanged, transport, transport_weights

__all__ = [
    "ProbeResult",
    "TIAAdapter",
    "TIAConfig",
    "attach_adapter",
    "build_adapter",
    "contrastive_loss",
    "correlation_matrix",
    "first_frame_unchanged",
    "lambda_schedule",
    "layer_epe",
    "local_window_indices",
    "noise_gate",
    "normalize_features",
    "probe_layers",
    "retrieve",
    "run_probe",
    "select_layer",
    "transport",
    "transport_weights",
]
