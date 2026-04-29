"""
DiT Flow Matching V3.4 adapter.

Keeps the original V3 backbone and accepts the FM wrapper's extra
arguments so V3.4 can be trained with Flow Matching without changing
the core backbone behavior.
"""

from .dit_denoiser_v3 import DiTDenoiserV3


class DiTFlowMatchingV3_4(DiTDenoiserV3):
    """V3.4 FM adapter that ignores FM-only side inputs."""

    def forward(
        self,
        cnn_feature,
        sampled_feat,
        x_t,
        t,
        adj=None,
        polys=None,
        py_ind=None,
        contour_scale=None,
        detail_feat=None,
        x_self_cond=None,
    ):
        return super().forward(
            cnn_feature,
            sampled_feat,
            x_t,
            t,
            adj=adj,
            polys=polys,
            py_ind=py_ind,
        )
