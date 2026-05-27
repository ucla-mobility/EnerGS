"""
GaussianModelEnerGS: Extended GaussianModel with geometric energy-guided relax.

EnerGS (Energy-Based Gaussian Splatting) key additions:
  - GeometricEnergyField: E_geom = E_occ + E_unk + λ*E_free
  - Relax step: xyz updated by Δμ = -η∇E_geom
  - Coverage modulation (paper extension)
  - FREE pruning (safety mechanism)

Design: Inherit from GaussianModel, minimal override.
"""

import torch
import numpy as np
from torch import nn
from typing import Optional, Tuple

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scene.gaussian_model import GaussianModel
from energs.geometric_energy import GeometricEnergyField
from energs.coverage import CoverageTracker, ViewConsistentCoverage


class GaussianModelEnerGS(GaussianModel):
    """
    Extended GaussianModel with Geometric Energy Field guidance (EnerGS).
    
    Energy formulation:
      E_geom(x) = E_occ(x) + E_unk(x) + λ*E_free(x)
      F = -∇E_geom  (force for Gaussian mean update)
    
    Training flow:
      - Photometric gradient → appearance params (SH, opacity, scale, rotation)
      - Photometric gradient → densification decisions
      - Geometric energy → xyz position (via relax step)
      
    Key insight:
      - Photometric decides WHAT (densify, appearance)
      - E_geom decides WHERE (xyz migration)
    """
    
    def __init__(self, sh_degree: int, optimizer_type: str = "default"):
        super().__init__(sh_degree, optimizer_type)
        
        # EnerGS components (initialized via setup_energs)
        self.energy_field: Optional[GeometricEnergyField] = None
        self.coverage_tracker: Optional[CoverageTracker] = None
        self.view_coverage: Optional[ViewConsistentCoverage] = None
        
        # Alias for backward compatibility
        self.potential_field = None
        
        # Relax parameters
        self.energs_enabled = False
        self.relax_lr = 0.001           # η: step size for Δμ = -η∇E_geom
        self.relax_force_scale = 1.0    # additional scaling
        self.use_trap = False           # soft trap for stability
        self.trap_lambda = 0.1
        self.xyz_freeze_in_relax = True # block photometric gradient on xyz
        
        # Statistics
        self.relax_step_count = 0
        self.free_intrusion_count = 0
        
    def setup_energs(
        self,
        field_npz_path: str,
        # E_occ parameters
        w_occ: float = 1.0,
        sigma_occ: float = 1.0,
        r_occ_max: float = 3.0,
        # E_unk parameters
        w_unk: float = 0.25,
        sigma_unk: float = 2.0,
        band_occ_max: float = 8.0,
        # E_free parameters (FREE barrier)
        barrier_lambda: float = 1.0,    # λ for E_free
        barrier_delta: float = 0.5,     # δ margin
        barrier_tau: float = 0.5,       # τ temperature
        barrier_type: str = "softplus", # [Ablation] barrier function type
        # Coverage modulation
        coverage_epsilon: float = 1.0,  # 1.0 = disabled
        coverage_gamma: float = 2.0,
        # Mode selection
        use_paper_energy: bool = False, # True = paper E_geom, False = legacy
        free_escape_k: float = 0.2,     # legacy escape force
        grad_smooth_sigma: float = 0.0,
        # Coverage tracker
        saturation_count: int = 3,
        soft_coverage: bool = True,
        use_hysteresis: bool = True,
        # Relax parameters
        relax_lr: float = 0.001,
        relax_force_scale: float = 1.0,
        use_trap: bool = False,
        trap_lambda: float = 0.1,
        xyz_freeze_in_relax: bool = True,
    ):
        """
        Initialize EnerGS geometric energy field.
        
        Paper mode (use_paper_energy=True):
          F = -∇E_occ - ∇E_unk - λ∇E_free
          
        Legacy mode (use_paper_energy=False):
          F = -∇U + F_escape
        """
        # Initialize geometric energy field
        self.energy_field = GeometricEnergyField(
            field_npz_path=field_npz_path,
            # E_occ params
            w_occ=w_occ,
            sigma_occ=sigma_occ,
            r_occ_max=r_occ_max,
            # E_unk params
            w_unk=w_unk,
            sigma_unk=sigma_unk,
            band_occ_max=band_occ_max,
            # E_free params
            lambda_free=barrier_lambda,
            delta=barrier_delta,
            tau=barrier_tau,
            barrier_type=barrier_type,
            # Coverage
            coverage_epsilon=coverage_epsilon,
            coverage_gamma=coverage_gamma,
            # Mode
            use_legacy_mode=not use_paper_energy,
            legacy_escape_k=free_escape_k,
            grad_smooth_sigma=grad_smooth_sigma,
        )
        
        # Backward compatibility alias
        self.potential_field = self.energy_field
        
        # Initialize coverage tracker
        grid_shape = (self.energy_field.X, self.energy_field.Y, self.energy_field.Z)
        self.coverage_tracker = CoverageTracker(
            grid_shape=grid_shape,
            voxel_size=self.energy_field.voxel_size,
            grid_origin=self.energy_field.grid_origin,
            saturation_count=saturation_count,
            soft_coverage=soft_coverage,
            use_hysteresis=use_hysteresis,
        )
        
        # Store relax params
        self.relax_lr = relax_lr
        self.relax_force_scale = relax_force_scale
        self.use_trap = use_trap
        self.trap_lambda = trap_lambda
        self.xyz_freeze_in_relax = xyz_freeze_in_relax
        
        # Initialize view coverage
        n_gaussians = self.get_xyz.shape[0]
        self.view_coverage = ViewConsistentCoverage(
            n_gaussians=n_gaussians,
            decay=0.99,
            saturation=10.0,
        )
        
        self.energs_enabled = True
        print(f"[EnerGS] GaussianModelEnerGS initialized")
        print(f"         relax_lr={relax_lr}, force_scale={relax_force_scale}")
    
    
    def update_coverage(self):
        """Update voxel coverage based on current Gaussian positions."""
        if not self.energs_enabled or self.coverage_tracker is None:
            return
        
        xyz = self.get_xyz.detach()
        opacity = self.get_opacity.detach()
        self.coverage_tracker.update_coverage(xyz, opacity)
    
    def update_view_coverage(
        self,
        visibility_filter: torch.Tensor,
        radii: torch.Tensor,
    ):
        """Update view-consistent coverage after rendering."""
        if not self.energs_enabled or self.view_coverage is None:
            return
        
        opacity = self.get_opacity.detach()
        self.view_coverage.update(visibility_filter, radii, opacity)
    
    def relax_step(
        self,
        compute_stats: bool = False,
        photometric_gate: Optional[torch.Tensor] = None,
        relax_lr_override: Optional[float] = None,
    ) -> dict:
        """
        Perform relax step: update xyz using F = -∇E_geom.
        
        Δμ = η * F = -η * ∇E_geom
        
        With step clipping: ||Δμ|| ≤ 0.5 * voxel_size
        
        Args:
            compute_stats: compute detailed statistics
            photometric_gate: (N,) per-Gaussian gate [0,1] for force scaling
            relax_lr_override: override η for decay schedule
        Returns:
            stats: dict with relax statistics
        """
        if not self.energs_enabled:
            return {}
        
        with torch.no_grad():
            xyz = self.get_xyz
            
            # Update coverage periodically
            if self.relax_step_count % 10 == 0:
                opacity = self.get_opacity
                self.coverage_tracker.update_coverage(xyz.detach(), opacity.detach())
            
            # Get coverage field for modulation
            coverage_field = self.coverage_tracker.get_normalized_coverage_field()
            
            # Compute force: F = -∇E_geom
            force = self.energy_field.compute_force(xyz, coverage_field)
            
            # Optional trap force
            if self.use_trap:
                trap_force = self.coverage_tracker.get_trap_force(xyz, self.trap_lambda)
                force = force + trap_force
            
            # Scale force
            force = force * self.relax_force_scale
            
            # Apply photometric gate
            if photometric_gate is not None:
                gate = photometric_gate.unsqueeze(-1).clamp(0, 1)
                force = force * gate
            
            # Compute Δμ = η * F
            lr = relax_lr_override if relax_lr_override is not None else self.relax_lr
            delta = lr * force
            
            # Step clipping: ||Δμ|| ≤ 0.5 * voxel_size
            max_step = self.energy_field.voxel_size * 0.5
            delta_norm = delta.norm(dim=-1, keepdim=True)
            delta = torch.where(
                delta_norm > max_step,
                delta * (max_step / (delta_norm + 1e-8)),
                delta
            )
            
            # Apply update: μ ← μ + Δμ
            self._xyz.data.add_(delta)
            
            self.relax_step_count += 1
            
            # Compute stats if requested
            if compute_stats:
                delta_norms = delta.norm(dim=-1)
                force_norms = force.norm(dim=-1)
                
                stats = {
                    'mean_force_mag': force_norms.mean().item(),
                    'mean_delta': delta_norms.mean().item(),
                    'max_delta': delta_norms.max().item(),
                    'occ_coverage': self.coverage_tracker.get_occ_coverage_ratio(
                        self.energy_field.occ_mask_t
                    ),
                    'unk_coverage': self.coverage_tracker.get_unk_coverage_ratio(
                        self.energy_field.unk_mask_t
                    ),
                    'free_intrusion': self.coverage_tracker.get_free_intrusion_count(
                        self.energy_field.free_mask_t
                    ),
                }
                self.free_intrusion_count = stats['free_intrusion']
                
                # Per-region delta statistics
                regions = self.energy_field.query_region(xyz)
                occ_mask = (regions == 1)
                free_mask = (regions == 2)
                unk_mask = (regions == 3)
                
                if occ_mask.any():
                    stats['delta_occ_mean'] = delta_norms[occ_mask].mean().item()
                    stats['delta_occ_max'] = delta_norms[occ_mask].max().item()
                    stats['force_occ_mean'] = force_norms[occ_mask].mean().item()
                else:
                    stats['delta_occ_mean'] = 0.0
                    stats['delta_occ_max'] = 0.0
                    stats['force_occ_mean'] = 0.0
                
                if free_mask.any():
                    stats['delta_free_mean'] = delta_norms[free_mask].mean().item()
                    stats['delta_free_max'] = delta_norms[free_mask].max().item()
                    stats['force_free_mean'] = force_norms[free_mask].mean().item()
                else:
                    stats['delta_free_mean'] = 0.0
                    stats['delta_free_max'] = 0.0
                    stats['force_free_mean'] = 0.0
                
                if unk_mask.any():
                    stats['delta_unk_mean'] = delta_norms[unk_mask].mean().item()
                    stats['delta_unk_max'] = delta_norms[unk_mask].max().item()
                    stats['force_unk_mean'] = force_norms[unk_mask].mean().item()
                else:
                    stats['delta_unk_mean'] = 0.0
                    stats['delta_unk_max'] = 0.0
                    stats['force_unk_mean'] = 0.0
                
                # Add energy component stats
                energy_stats = self.energy_field.compute_energy_stats(xyz)
                stats.update(energy_stats)
                
                return stats
            
            return {}
    
    def clamp_scaling(self, max_scale: float = 0.5):
        """
        Clamp Gaussian scales to a maximum value.
        
        This prevents Gaussians from growing too large, which can cause:
        - Blurry reconstructions
        - Memory issues
        - Poor geometry
        
        Args:
            max_scale: Maximum allowed scale in meters (default 0.5m)
        """
        with torch.no_grad():
            # _scaling is stored as log(scale), so we need to clamp log(scale) ≤ log(max_scale)
            max_log_scale = np.log(max_scale)
            
            # Count how many will be clamped
            current_scales = self.get_scaling  # exp(_scaling)
            n_clamped = (current_scales > max_scale).any(dim=1).sum().item()
            
            # Clamp
            self._scaling.data.clamp_(max=max_log_scale)
            
            return n_clamped
    
    def compute_photometric_gate(
        self,
        threshold: float = 0.0002,
        temperature: float = 0.0001,
    ) -> torch.Tensor:
        """
        Compute photometric gate based on gradient accumulator.
        
        gate = sigmoid((grad_accum - threshold) / temperature)
        
        High error → high gate → strong geometric guidance
        Low error → low gate → stable position
        """
        if not hasattr(self, 'xyz_gradient_accum') or self.xyz_gradient_accum is None:
            return torch.ones(self.get_xyz.shape[0], device="cuda")
        
        grad_accum = self.xyz_gradient_accum.squeeze(-1)
        denom = self.denom.squeeze(-1).clamp(min=1)
        avg_grad = grad_accum / denom
        
        return torch.sigmoid((avg_grad - threshold) / temperature)
    
    def compute_combined_gate(
        self,
        use_photometric: bool = True,
        use_view_coverage: bool = True,
        photometric_threshold: float = 0.0002,
        photometric_temperature: float = 0.0001,
    ) -> torch.Tensor:
        """
        Compute combined gate from photometric error and view coverage.
        
        gate = max(photometric_gate, 1 - view_coverage)
        """
        n = self.get_xyz.shape[0]
        gate = torch.zeros(n, device="cuda")
        
        if use_photometric:
            photo_gate = self.compute_photometric_gate(
                threshold=photometric_threshold,
                temperature=photometric_temperature,
            )
            gate = torch.max(gate, photo_gate)
        
        if use_view_coverage and self.view_coverage is not None:
            view_gate = self.view_coverage.get_field_gate()
            gate = torch.max(gate, view_gate)
        
        if not use_photometric and not use_view_coverage:
            gate = torch.ones(n, device="cuda")
        
        return gate
    
    def get_xyz_grad_mask_for_relax(self) -> torch.Tensor:
        """Get mask to zero out xyz gradient (photometric decoupling)."""
        if self.xyz_freeze_in_relax and self.energs_enabled:
            return torch.zeros_like(self._xyz)
        return torch.ones_like(self._xyz)
    
    def filter_by_region(self, keep_occ: bool = True, keep_unk: bool = True, keep_free: bool = False) -> torch.Tensor:
        """Get mask of Gaussians to keep based on region."""
        if not self.energs_enabled:
            return torch.ones(self.get_xyz.shape[0], dtype=torch.bool, device="cuda")
        
        regions = self.energy_field.query_region(self.get_xyz)
        
        keep_mask = torch.zeros_like(regions, dtype=torch.bool)
        keep_mask |= (regions == 0)  # OOB
        if keep_occ:
            keep_mask |= (regions == 1)
        if keep_unk:
            keep_mask |= (regions == 3)
        if keep_free:
            keep_mask |= (regions == 2)
        
        return keep_mask
    
    def prune_free_gaussians(self, min_iterations: int = 1000):
        """Prune Gaussians in FREE space (hard FREE pruning)."""
        if not self.energs_enabled or self.relax_step_count < min_iterations:
            return 0
        
        regions = self.energy_field.query_region(self.get_xyz)
        free_mask = (regions == 2)
        prune_count = free_mask.sum().item()
        
        if prune_count > 0:
            self.tmp_radii = torch.zeros(self.get_xyz.shape[0], device="cuda")
            self.prune_points(free_mask)
            self.tmp_radii = None
            print(f"[EnerGS] Pruned {prune_count} Gaussians from FREE")
        
        return prune_count
    
    def get_region_stats(self) -> dict:
        """Get Gaussian distribution across regions."""
        if not self.energs_enabled:
            return {}
        
        regions = self.energy_field.query_region(self.get_xyz)
        return {
            'n_oob': (regions == 0).sum().item(),
            'n_occ': (regions == 1).sum().item(),
            'n_free': (regions == 2).sum().item(),
            'n_unk': (regions == 3).sum().item(),
            'total': regions.shape[0],
        }
    
    def get_theory_verification_stats(self, epsilon: float = 0.1) -> dict:
        """
        Compute comprehensive statistics for verifying Theorem 1 claims.
        
        Outputs detailed quantiles and distributions for paper figures.
        """
        if not self.energs_enabled:
            return {}
        
        import math
        
        xyz = self.get_xyz.detach()
        field = self.energy_field
        n_total = xyz.shape[0]
        
        # === Basic field parameters ===
        delta = field.delta
        tau = field.tau
        lambda_free = field.lambda_free
        w_occ = field.w_occ
        sigma_occ = field.sigma_occ
        w_unk = field.w_unk
        sigma_unk = field.sigma_unk
        
        # === Claim 1: Energy and gradient ===
        energy_stats = field.compute_total_energy(xyz)
        
        # === Claim 2: Deep FREE exclusion ===
        # s_0 = δ + τ * log((1-ε)/ε)  [Paper Eq. 21]
        s_0 = delta + tau * math.log((1 - epsilon) / epsilon)
        
        # m_0 = (1-ε)/τ  [Paper Eq. 22]
        m_0 = (1 - epsilon) / tau
        
        # B(s_0) = B_occ(s_0) + B_unk(s_0)  [Paper Eq. 23-24]
        def compute_B_j(w_j, sigma_j, s_0):
            s_bar = max(s_0, sigma_j)
            return (w_j / (sigma_j**2)) * s_bar * math.exp(-(s_bar**2) / (2 * sigma_j**2))
        
        B_occ = compute_B_j(w_occ, sigma_occ, s_0)
        B_unk = compute_B_j(w_unk, sigma_unk, s_0)
        B_s0 = B_occ + B_unk
        
        # Dominance condition: λ * m_0 > B(s_0)  [Paper Eq. 27]
        dominance_lhs = lambda_free * m_0
        dominance_satisfied = dominance_lhs > B_s0
        
        # === Distance and region queries ===
        d_trust = field.query_d_trust(xyz)
        d_occ = field.query_d_occ(xyz)
        regions = field.query_region(xyz)
        
        is_occ = (regions == 1)
        is_free = (regions == 2)
        is_unk = (regions == 3)
        is_deep_free = is_free & (d_trust >= s_0)
        
        n_occ = is_occ.sum().item()
        n_free = is_free.sum().item()
        n_unk = is_unk.sum().item()
        n_deep_free = is_deep_free.sum().item()
        
        # === Per-Gaussian energy for quantiles ===
        E_per_gaussian = field.compute_per_gaussian_energy(xyz)
        
        # === d_trust quantiles (全局 + 分区域) ===
        def compute_quantiles(tensor, mask=None):
            if mask is not None:
                tensor = tensor[mask]
            if len(tensor) == 0:
                return {'mean': 0, 'p10': 0, 'p50': 0, 'p90': 0, 'max': 0}
            return {
                'mean': tensor.mean().item(),
                'p10': tensor.quantile(0.1).item(),
                'p50': tensor.quantile(0.5).item(),
                'p90': tensor.quantile(0.9).item(),
                'max': tensor.max().item(),
            }
        
        d_trust_all = compute_quantiles(d_trust)
        d_trust_free = compute_quantiles(d_trust, is_free)
        d_trust_deep_free = compute_quantiles(d_trust, is_deep_free)
        
        E_all = compute_quantiles(E_per_gaussian)
        E_free = compute_quantiles(E_per_gaussian, is_free)
        E_occ_region = compute_quantiles(E_per_gaussian, is_occ)
        
        # === Directional descent in deep FREE: ⟨∇E_geom, ∇d_trust⟩ ===
        # This should be > 0 (= β) for Proposition 2
        directional_descent_stats = self._compute_directional_descent_stats(
            xyz, is_deep_free, s_0, lambda_free
        )
        
        # === Claim 4: Geometry metrics ===
        if self.coverage_tracker is not None:
            self.coverage_tracker.update_coverage(xyz, self.get_opacity.detach())
            coverage_count = self.coverage_tracker.coverage_count
            occ_mask = field.occ_mask_t
            occ_voxels_covered = (coverage_count[occ_mask] >= 1).sum().item()
            occ_voxels_total = occ_mask.sum().item()
            occ_coverage = occ_voxels_covered / max(1, occ_voxels_total)
        else:
            occ_coverage = 0.0
        
        free_intrusion_ratio = n_free / max(1, n_total)
        
        return {
            # === Basic counts ===
            'n_total': n_total,
            'n_occ': n_occ,
            'n_free': n_free,
            'n_unk': n_unk,
            'n_deep_free': n_deep_free,
            
            # === Claim 1: Energy descent + convergence ===
            'E_total': energy_stats['E_total'],
            'E_occ_total': energy_stats['E_occ_total'],
            'E_unk_total': energy_stats['E_unk_total'],
            'E_free_total': energy_stats['E_free_total'],
            'grad_norm_sq_total': energy_stats['grad_norm_sq_total'],
            
            # E_geom quantiles (for Figure)
            'E_mean': E_all['mean'],
            'E_p50': E_all['p50'],
            'E_p90': E_all['p90'],
            'E_free_mean': E_free['mean'],
            'E_free_p90': E_free['p90'],
            
            # === Claim 2: Deep FREE exclusion ===
            's_0': s_0,
            'm_0': m_0,
            'B_s0': B_s0,
            'B_occ': B_occ,
            'B_unk': B_unk,
            'dominance_lhs': dominance_lhs,  # λ * m_0
            'dominance_rhs': B_s0,           # B(s_0)
            'dominance_margin': dominance_lhs - B_s0,  # β lower bound
            'dominance_satisfied': dominance_satisfied,
            'lambda_over_tau': lambda_free / tau,  # λ/τ for grid search
            
            # d_trust quantiles (for Figure)
            'd_trust_mean': d_trust_all['mean'],
            'd_trust_p50': d_trust_all['p50'],
            'd_trust_p90': d_trust_all['p90'],
            'd_trust_free_mean': d_trust_free['mean'],
            'd_trust_free_p50': d_trust_free['p50'],
            'd_trust_free_p90': d_trust_free['p90'],
            'd_trust_deep_free_mean': d_trust_deep_free['mean'],
            
            # Directional descent (Proposition 2 verification)
            **directional_descent_stats,
            
            # === Claim 4: Geometry metrics ===
            'occ_coverage': occ_coverage,
            'free_intrusion_ratio': free_intrusion_ratio,
        }
    
    def _compute_directional_descent_stats(
        self,
        xyz: torch.Tensor,
        is_deep_free: torch.Tensor,
        s_0: float,
        lambda_free: float,
    ) -> dict:
        """
        Compute ⟨∇E_geom, ∇d_trust⟩ statistics for deep FREE points.
        
        Proposition 2 claims this should be ≥ β > 0 in deep FREE.
        """
        field = self.energy_field
        
        if not is_deep_free.any():
            return {
                'dir_descent_mean': 0.0,
                'dir_descent_p10': 0.0,
                'dir_descent_p50': 0.0,
                'dir_descent_positive_ratio': 1.0,  # No points = trivially satisfied
                'beta_empirical': 0.0,
            }
        
        # Get deep FREE points
        xyz_deep = xyz[is_deep_free]
        
        # Query indices
        idx = field.world_to_voxel_idx(xyz_deep)
        in_bounds = field.is_in_bounds(idx)
        
        if not in_bounds.any():
            return {
                'dir_descent_mean': 0.0,
                'dir_descent_p10': 0.0,
                'dir_descent_p50': 0.0,
                'dir_descent_positive_ratio': 1.0,
                'beta_empirical': 0.0,
            }
        
        valid_idx = idx[in_bounds]
        i, j, k = valid_idx[:, 0], valid_idx[:, 1], valid_idx[:, 2]
        
        # Get distances and gradients
        d_occ = field.d_occ[i, j, k]
        d_unk = field.d_unk[i, j, k]
        d_trust = field.d_trust[i, j, k]
        
        grad_d_occ = field.grad_d_occ[i, j, k]
        grad_d_unk = field.grad_d_unk[i, j, k]
        grad_d_trust = field.grad_d_trust[i, j, k]
        
        # Compute ∇E_occ
        exp_occ = torch.exp(-(d_occ**2) / (2 * field.sigma_occ**2))
        coef_occ = (field.w_occ / (field.sigma_occ**2)) * d_occ * exp_occ
        grad_E_occ = coef_occ.unsqueeze(-1) * grad_d_occ
        
        # Compute ∇E_unk
        exp_unk = torch.exp(-(d_unk**2) / (2 * field.sigma_unk**2))
        coef_unk = (field.w_unk / (field.sigma_unk**2)) * d_unk * exp_unk
        grad_E_unk = coef_unk.unsqueeze(-1) * grad_d_unk
        
        # Compute ∇E_free = φ'(d) * ∇d
        phi_prime = (1.0 / field.tau) * torch.sigmoid((d_trust - field.delta) / field.tau)
        grad_E_free = phi_prime.unsqueeze(-1) * grad_d_trust
        
        # ∇E_geom = ∇E_occ + ∇E_unk + λ∇E_free
        grad_E_geom = grad_E_occ + grad_E_unk + lambda_free * grad_E_free
        
        # ⟨∇E_geom, ∇d_trust⟩
        dir_descent = (grad_E_geom * grad_d_trust).sum(dim=-1)
        
        # Statistics
        positive_ratio = (dir_descent > 0).float().mean().item()
        
        return {
            'dir_descent_mean': dir_descent.mean().item(),
            'dir_descent_p10': dir_descent.quantile(0.1).item(),
            'dir_descent_p50': dir_descent.quantile(0.5).item(),
            'dir_descent_positive_ratio': positive_ratio,
            'beta_empirical': dir_descent.min().item(),  # Worst case
        }
    
    def get_free_barrier_loss(
        self,
        delta: float = 0.5,
        tau: float = 0.5,
    ) -> torch.Tensor:
        """
        Compute E_free loss (optional soft constraint).
        
        E_free(x) = softplus((d_trust - δ)/τ)
        """
        if not self.energs_enabled:
            return torch.tensor(0.0, device="cuda")
        
        xyz = self.get_xyz
        E = self.energy_field.query_E_free(xyz, delta, tau)
        return E.mean()
    
    def get_voxel_occupancy_stats(self) -> dict:
        """Get detailed voxel occupancy statistics."""
        if not self.energs_enabled:
            return {}
        
        with torch.no_grad():
            xyz = self.get_xyz.detach()
            opacity = self.get_opacity.detach()
            self.coverage_tracker.update_coverage(xyz, opacity)
            
            coverage_count = self.coverage_tracker.coverage_count
            
            occ_mask = self.energy_field.occ_mask_t
            free_mask = self.energy_field.free_mask_t
            unk_mask = self.energy_field.unk_mask_t
            
            stats = {}
            
            for name, mask in [('occ', occ_mask), ('free', free_mask), ('unk', unk_mask)]:
                total = mask.sum().item()
                if total == 0:
                    continue
                    
                counts = coverage_count[mask]
                
                stats[f'{name}_total'] = total
                stats[f'{name}_0'] = (counts == 0).sum().item()
                stats[f'{name}_1'] = ((counts >= 1) & (counts < 2)).sum().item()
                stats[f'{name}_2'] = ((counts >= 2) & (counts < 3)).sum().item()
                stats[f'{name}_3+'] = (counts >= 3).sum().item()
                
                stats[f'{name}_0_pct'] = 100.0 * stats[f'{name}_0'] / total
                stats[f'{name}_1_pct'] = 100.0 * stats[f'{name}_1'] / total
                stats[f'{name}_2_pct'] = 100.0 * stats[f'{name}_2'] / total
                stats[f'{name}_3+_pct'] = 100.0 * stats[f'{name}_3+'] / total
            
            return stats
    
    def print_voxel_occupancy_stats(self):
        """Print formatted voxel occupancy statistics."""
        stats = self.get_voxel_occupancy_stats()
        if not stats:
            return
        
        print("\n" + "="*70)
        print("EnerGS Voxel Occupancy Statistics")
        print("="*70)
        
        for region in ['occ', 'free', 'unk']:
            total_key = f'{region}_total'
            if total_key not in stats:
                continue
            
            total = stats[total_key]
            print(f"\n{region.upper()} ({total} voxels):")
            print(f"  0 Gaussians: {stats[f'{region}_0']:6d} ({stats[f'{region}_0_pct']:5.1f}%)")
            print(f"  1 Gaussian:  {stats[f'{region}_1']:6d} ({stats[f'{region}_1_pct']:5.1f}%)")
            print(f"  2 Gaussians: {stats[f'{region}_2']:6d} ({stats[f'{region}_2_pct']:5.1f}%)")
            print(f"  3+ Gaussians:{stats[f'{region}_3+']:6d} ({stats[f'{region}_3+_pct']:5.1f}%)")
        
        print("="*70 + "\n")
    
    def densification_postfix(self, new_xyz, new_features_dc, new_features_rest, 
                              new_opacities, new_scaling, new_rotation, new_tmp_radii):
        """Override to resize coverage trackers."""
        super().densification_postfix(
            new_xyz, new_features_dc, new_features_rest,
            new_opacities, new_scaling, new_rotation, new_tmp_radii
        )
        
        n = self.get_xyz.shape[0]
        if self.energs_enabled:
            if self.coverage_tracker is not None:
                self.coverage_tracker.resize(n)
            if self.view_coverage is not None:
                self.view_coverage.resize(n)
    
    def prune_points(self, mask):
        """Override to resize coverage trackers."""
        super().prune_points(mask)
        
        n = self.get_xyz.shape[0]
        if self.energs_enabled:
            if self.coverage_tracker is not None:
                self.coverage_tracker.resize(n)
            if self.view_coverage is not None:
                self.view_coverage.resize(n)
    
    def save_energs_state(self, path: str):
        """Save EnerGS state."""
        if not self.energs_enabled:
            return
        
        state = {
            'relax_step_count': self.relax_step_count,
            'relax_lr': self.relax_lr,
            'relax_force_scale': self.relax_force_scale,
            'use_trap': self.use_trap,
            'trap_lambda': self.trap_lambda,
        }
        torch.save(state, path)
    
    def load_energs_state(self, path: str):
        """Load EnerGS state."""
        if os.path.exists(path):
            state = torch.load(path)
            self.relax_step_count = state.get('relax_step_count', 0)


class RelaxScheduler:
    """Schedule relax steps (alternating/phase/triggered strategies)."""
    
    def __init__(
        self,
        strategy: str = "alternating",
        photo_steps: int = 10,
        relax_steps: int = 5,
        relax_start_iter: int = 500,
        relax_end_iter: int = 15000,
        relax_interval: int = 100,
        relax_duration: int = 20,
    ):
        self.strategy = strategy
        self.photo_steps = photo_steps
        self.relax_steps = relax_steps
        self.relax_start_iter = relax_start_iter
        self.relax_end_iter = relax_end_iter
        self.relax_interval = relax_interval
        self.relax_duration = relax_duration
        
        self._step_counter = 0
        self._relax_remaining = 0
    
    def should_relax(self, iteration: int) -> bool:
        if self.strategy == "alternating":
            cycle = self.photo_steps + self.relax_steps
            pos = self._step_counter % cycle
            self._step_counter += 1
            return pos >= self.photo_steps
        
        elif self.strategy == "phase":
            if iteration < self.relax_start_iter or iteration > self.relax_end_iter:
                return False
            rel_iter = iteration - self.relax_start_iter
            return rel_iter % self.relax_interval < self.relax_duration
        
        elif self.strategy == "triggered":
            if self._relax_remaining > 0:
                self._relax_remaining -= 1
                return True
            return False
        
        return False
    
    def trigger_relax(self, n_steps: int):
        self._relax_remaining = n_steps
    
    def reset(self):
        self._step_counter = 0
        self._relax_remaining = 0
