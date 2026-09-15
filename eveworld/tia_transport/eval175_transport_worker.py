#!/usr/bin/env python3
"""Run the established EVAL-175 worker with the CIC-Transport loader."""

from __future__ import annotations

from eveworld.method import pipeline_eag
from eveworld.tia_transport.cic_transport_pipeline import (
    CICTransportEAGGigaWorld0Pipeline,
)
from eveworld.evaluation.eval175_multiseed_worker import main


# The established worker imports this symbol lazily inside main(). Replacing
# only that loader keeps preprocessing, prompts, seeds and video writing exact.
pipeline_eag.EAGGigaWorld0Pipeline = CICTransportEAGGigaWorld0Pipeline


if __name__ == "__main__":
    main()
