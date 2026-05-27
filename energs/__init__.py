"""
EnerGS: Energy-Based Gaussian Splatting

Geometric energy field for Gaussian relocation guidance:
  E_geom(x) = E_occ(x) + E_unk(x) + λ * E_free(x)

This module provides:
  - GeometricEnergyField: Compute and query E_geom and ∇E_geom
  - CoverageTracker: Track voxel coverage by Gaussians
  - GaussianModelEnerGS: Extended GaussianModel with energy-guided relax

Design principle: Minimal invasion to original 3DGS codebase.
"""

from .geometric_energy import GeometricEnergyField
from .coverage import CoverageTracker
from .gaussian_model import GaussianModelEnerGS

__all__ = [
    'GeometricEnergyField',
    'CoverageTracker', 
    'GaussianModelEnerGS',
]
