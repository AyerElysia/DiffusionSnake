"""
DiT Flow Matching V3.6 for Diffusion Snake.

V3.6 reuses the V3.0 global semantic path:
  - PerceiverCompressor with learnable query tokens
  - Alternating global/local context injection
  - Separate point embedding and adaLN-zero output head

The denoiser itself stays close to V3.0 so the surrounding
Flow Matching wrapper can focus on iterative refinement.
"""

from .dit_denoiser_v3 import DiTDenoiserV3


class DiTFlowMatchingV3_6(DiTDenoiserV3):
    """Flow Matching variant that keeps the V3.0 learnable-query backbone."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
