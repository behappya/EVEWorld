#!/usr/bin/env python3
"""Reuse the established multi-seed dispatcher with an isolated Transport probe."""

from __future__ import annotations

import os
from pathlib import Path

from eveworld.evaluation import eval175_multiseed_dispatch


def main() -> None:
    model = os.environ.get("CIC_TRANSPORT_MODEL_NAME", "")
    model_dir_raw = os.environ.get("CIC_TRANSPORT_MODEL_DIR", "")
    if not model or not model_dir_raw:
        raise RuntimeError(
            "CIC_TRANSPORT_MODEL_NAME and CIC_TRANSPORT_MODEL_DIR are required"
        )
    model_dir = Path(model_dir_raw).resolve()
    if not model_dir.is_dir():
        raise FileNotFoundError(model_dir)

    # Injection is process-local; the shared dispatcher and its existing model map
    # remain unchanged for all other campaigns.
    eval175_multiseed_dispatch.MODELS[model] = str(model_dir)
    eval175_multiseed_dispatch.main()


if __name__ == "__main__":
    main()
