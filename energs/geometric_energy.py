"""
GeometricEnergyField: Compute and query geometric energy field for Gaussian guidance.

=== EnerGS Energy Formulation ===

E_geom(x) = E_occ(x) + E_unk(x) + λ * E_free(x)

Where:
  E_occ(x) = -w_occ * exp(-d_occ²/(2σ_occ²))       # OCC attraction (negative)
  E_unk(x) = -w_unk * exp(-d_unk²/(2σ_unk²))       # UNK attraction (negative)
  E_free(x) = softplus((d_trust - δ)/τ)            # FREE barrier (positive)

Gradients:
  ∇E_occ = (w_occ/σ_occ²) * d_occ * exp(...) * ∇d_occ
  ∇E_unk = (w_unk/σ_unk²) * d_unk * exp(...) * ∇d_unk
  ∇E_free = φ'(d_trust) * ∇d_trust,  where φ'(s) = (1/τ) * sigmoid((s-δ)/τ)

Force (for Gaussian mean update):
  F = -∇E_geom = -∇E_occ - ∇E_unk - λ∇E_free

=== Paper Extensions ===
  1. Coverage modulation: g(c) = ε + (1-ε)*(1-c)^γ
  2. OCC cutoff radius (r_occ_max)
  3. UNK band gate (band_occ_max)
"""

import os
import numpy as np
import torch
from typing import Tuple, Optional
from scipy.ndimage import distance_transform_edt, gaussian_filter


def _unpack_bool(packed: np.ndarray, shape: Tuple[int, int, int]) -> np.ndarray:
    """Unpack bit-packed boolean array."""
    n = int(np.prod(shape))
    flat = np.unpackbits(packed, bitorder="little")[:n].astype(np.bool_)
    return flat.reshape(shape)


class GeometricEnergyField:
    """
    Geometric Energy Field for Gaussian relocation guidance (EnerGS).
    
    Computes F = -∇E_geom for Gaussian mean migration.
    Photometric optimization handles densification separately.
    """
    
    def __init__(
        self,
        field_npz_path: str,
        # E_occ parameters
        w_occ: float = 1.0,          # weight for OCC attraction
        sigma_occ: float = 1.0,      # σ_occ (meters)
        r_occ_max: float = 3.0,      # OCC cutoff radius (0 = disabled)
        # E_unk parameters
        w_unk: float = 0.25,         # weight for UNK attraction
        sigma_unk: float = 2.0,      # σ_unk (meters)
        band_occ_max: float = 8.0,   # UNK band gate distance (0 = disabled)
        # E_free parameters (FREE barrier)
        lambda_free: float = 1.0,    # λ: barrier weight in E_geom
        delta: float = 0.5,          # δ: margin distance (meters)
        tau: float = 0.5,            # τ: softplus temperature
        barrier_type: str = "softplus",  # [Ablation] barrier type: softplus, hinge, log
        # Coverage modulation (paper extension)
        coverage_epsilon: float = 1.0,  # ε: floor (1.0 = disabled)
        coverage_gamma: float = 2.0,    # γ: decay exponent
        # Legacy mode (for backward compatibility with exp6)
        use_legacy_mode: bool = False,  # False = paper E_geom, True = legacy U+escape
        legacy_escape_k: float = 0.2,   # k for legacy F_escape = k*d*dir
        # Other
        grad_smooth_sigma: float = 0.0,
        device: str = "cuda",
    ):
        self.device = device
        
        # E_occ parameters
        self.w_occ = w_occ
        self.sigma_occ = max(1e-6, sigma_occ)
        self.r_occ_max = r_occ_max
        
        # E_unk parameters
        self.w_unk = w_unk
        self.sigma_unk = max(1e-6, sigma_unk)
        self.band_occ_max = band_occ_max
        
        # E_free parameters
        self.lambda_free = lambda_free
        self.delta = delta
        self.tau = tau
        self.barrier_type = barrier_type  # softplus, hinge, log
        
        # Coverage modulation
        self.coverage_epsilon = coverage_epsilon
        self.coverage_gamma = coverage_gamma
        
        # Legacy mode
        self.use_legacy_mode = use_legacy_mode
        self.legacy_escape_k = legacy_escape_k
        self.grad_smooth_sigma = grad_smooth_sigma
        
        # Load occupancy field
        self._load_field(field_npz_path)
        
        # Compute distance fields and gradients
        self._compute_fields()
        
        print(f"[EnerGS] GeometricEnergyField initialized:")
        print(f"         voxel_size={self.voxel_size:.3f}m, grid={self.dims}")
        print(f"         E_occ: w={w_occ}, σ={sigma_occ}m, r_max={r_occ_max}m")
        print(f"         E_unk: w={w_unk}, σ={sigma_unk}m, band={band_occ_max}m")
        print(f"         E_free: λ={lambda_free}, δ={delta}m, τ={tau}, type={barrier_type}")
        print(f"         Mode: {'LEGACY (U + escape)' if use_legacy_mode else 'PAPER (E_geom)'}")
    
    def _load_field(self, path: str):
        """Load cached OCC/FREE/UNK voxel masks."""
        if not os.path.exists(path):
            raise FileNotFoundError(f"Field cache not found: {path}")
        
        data = np.load(path, allow_pickle=True)
        shape = tuple(data["shape"].tolist())
        
        self.occ_mask = _unpack_bool(data["occ_p"], shape)
        self.free_mask = _unpack_bool(data["free_p"], shape)
        self.unk_mask = _unpack_bool(data["unk_p"], shape)
        
        self.voxel_size = float(data["voxel_size"])
        self.grid_origin = np.asarray(data["grid_origin"], dtype=np.float64)
        self.dims = np.asarray(data["dims"], dtype=np.int32)
        
        self.X, self.Y, self.Z = shape
        
        print(f"[EnerGS] Loaded: OCC={self.occ_mask.sum()}, FREE={self.free_mask.sum()}, UNK={self.unk_mask.sum()}")
    
    def _compute_fields(self):
        """
        Compute distance fields and gradients for energy computation.
        
        Distance fields:
          d_occ(x): distance to nearest OCC voxel
          d_unk(x): distance to nearest UNK voxel
          d_trust(x): distance to nearest trust region (OCC ∪ UNK)
        
        Gradient fields (unit vectors):
          ∇d_occ, ∇d_unk, ∇d_trust
        """
        # ========== Distance transforms (meters) ==========
        d_occ = distance_transform_edt(~self.occ_mask).astype(np.float32) * self.voxel_size
        d_unk = distance_transform_edt(~self.unk_mask).astype(np.float32) * self.voxel_size
        
        # d_trust: min(d_occ, d_unk) with slight OCC preference
        occ_preference = 0.01  # meters
        prefer_occ = (d_occ <= d_unk + occ_preference)
        d_trust = np.where(prefer_occ, d_occ, d_unk)
        
        # ========== Compute ∇d fields ==========
        def compute_grad_d(d_field):
            """Compute gradient using central difference."""
            gx = np.zeros_like(d_field)
            gy = np.zeros_like(d_field)
            gz = np.zeros_like(d_field)
            # Central difference (interior)
            gx[1:-1, :, :] = (d_field[2:, :, :] - d_field[:-2, :, :]) / (2 * self.voxel_size)
            gy[:, 1:-1, :] = (d_field[:, 2:, :] - d_field[:, :-2, :]) / (2 * self.voxel_size)
            gz[:, :, 1:-1] = (d_field[:, :, 2:] - d_field[:, :, :-2]) / (2 * self.voxel_size)
            # Boundary
            gx[0, :, :] = (d_field[1, :, :] - d_field[0, :, :]) / self.voxel_size
            gx[-1, :, :] = (d_field[-1, :, :] - d_field[-2, :, :]) / self.voxel_size
            gy[:, 0, :] = (d_field[:, 1, :] - d_field[:, 0, :]) / self.voxel_size
            gy[:, -1, :] = (d_field[:, -1, :] - d_field[:, -2, :]) / self.voxel_size
            gz[:, :, 0] = (d_field[:, :, 1] - d_field[:, :, 0]) / self.voxel_size
            gz[:, :, -1] = (d_field[:, :, -1] - d_field[:, :, -2]) / self.voxel_size
            return np.stack([gx, gy, gz], axis=-1)
        
        grad_d_occ = compute_grad_d(d_occ)
        grad_d_unk = compute_grad_d(d_unk)
        grad_d_trust = np.where(prefer_occ[..., np.newaxis], grad_d_occ, grad_d_unk)
        
        # Normalize to unit vectors
        def normalize(g):
            return g / (np.linalg.norm(g, axis=-1, keepdims=True) + 1e-8)
        
        grad_d_occ = normalize(grad_d_occ)
        grad_d_unk = normalize(grad_d_unk)
        grad_d_trust = normalize(grad_d_trust)
        
        # ========== Store as tensors ==========
        self.d_occ = torch.from_numpy(d_occ).to(self.device)
        self.d_unk = torch.from_numpy(d_unk).to(self.device)
        self.d_trust = torch.from_numpy(d_trust).to(self.device)
        
        self.grad_d_occ = torch.from_numpy(grad_d_occ.astype(np.float32)).to(self.device)
        self.grad_d_unk = torch.from_numpy(grad_d_unk.astype(np.float32)).to(self.device)
        self.grad_d_trust = torch.from_numpy(grad_d_trust.astype(np.float32)).to(self.device)
        
        # ========== Legacy mode: precompute U and escape force ==========
        if self.use_legacy_mode:
            # U_occ, U_unk (positive potential for legacy -∇U force)
            U_occ = self.w_occ * np.exp(-(d_occ**2) / (2 * self.sigma_occ**2))
            if self.r_occ_max > 0:
                U_occ = U_occ * (d_occ <= self.r_occ_max).astype(np.float32)
            
            U_unk = self.w_unk * np.exp(-(d_unk**2) / (2 * self.sigma_unk**2))
            if self.band_occ_max > 0:
                U_unk = U_unk * (d_occ <= self.band_occ_max).astype(np.float32)
            
            U_total = U_occ + U_unk
            if self.grad_smooth_sigma > 0:
                U_total = gaussian_filter(U_total, sigma=self.grad_smooth_sigma).astype(np.float32)
            
            self.U_total = torch.from_numpy(U_total).to(self.device)
            
            # Legacy escape force: F_escape = k * d_trust * dir_to_trust (only in FREE)
            if self.legacy_escape_k > 0:
                dir_to_trust = -grad_d_trust  # points toward trust
                F_escape = self.legacy_escape_k * d_trust[..., np.newaxis] * dir_to_trust
                F_escape = F_escape * self.free_mask[..., np.newaxis].astype(np.float32)
                self.F_escape = torch.from_numpy(F_escape.astype(np.float32)).to(self.device)
            else:
                self.F_escape = torch.zeros((self.X, self.Y, self.Z, 3), dtype=torch.float32, device=self.device)
        
        # Store masks as tensors
        self.occ_mask_t = torch.from_numpy(self.occ_mask).to(self.device)
        self.free_mask_t = torch.from_numpy(self.free_mask).to(self.device)
        self.unk_mask_t = torch.from_numpy(self.unk_mask).to(self.device)
        
        print(f"[EnerGS] Distance fields: d_occ=[{d_occ.min():.2f}, {d_occ.max():.2f}]m, "
              f"d_trust=[{d_trust.min():.2f}, {d_trust.max():.2f}]m")
    
    def world_to_voxel_idx(self, xyz: torch.Tensor) -> torch.Tensor:
        """Convert world coordinates to voxel indices."""
        origin = torch.tensor(self.grid_origin, dtype=xyz.dtype, device=xyz.device)
        return torch.floor((xyz - origin) / self.voxel_size).long()
    
    def is_in_bounds(self, idx: torch.Tensor) -> torch.Tensor:
        """Check if voxel indices are within grid bounds."""
        return (
            (idx[:, 0] >= 0) & (idx[:, 0] < self.X) &
            (idx[:, 1] >= 0) & (idx[:, 1] < self.Y) &
            (idx[:, 2] >= 0) & (idx[:, 2] < self.Z)
        )
    
    def compute_geometric_force(
        self,
        xyz: torch.Tensor,
        coverage_field: Optional[torch.Tensor] = None,
        lambda_free: Optional[float] = None,
        delta: Optional[float] = None,
        tau: Optional[float] = None,
    ) -> torch.Tensor:
        """
        Compute force from geometric energy: F = -∇E_geom
        
        E_geom = E_occ + E_unk + λ*E_free
        
        Where:
          ∇E_occ = (w_occ/σ²) * d_occ * exp(-d²/(2σ²)) * ∇d_occ
          ∇E_unk = (w_unk/σ²) * d_unk * exp(-d²/(2σ²)) * ∇d_unk
          ∇E_free = φ'(d_trust) * ∇d_trust,  φ'(s) = (1/τ) * sigmoid((s-δ)/τ)
        
        Args:
            xyz: (N, 3) Gaussian positions
            coverage_field: (X, Y, Z) optional coverage for modulation
            lambda_free/delta/tau: optional parameter overrides
        Returns:
            F: (N, 3) force vectors
        """
        # Use instance params if not overridden
        lam = lambda_free if lambda_free is not None else self.lambda_free
        d = delta if delta is not None else self.delta
        t = tau if tau is not None else self.tau
        
        idx = self.world_to_voxel_idx(xyz)
        in_bounds = self.is_in_bounds(idx)
        
        F = torch.zeros(xyz.shape[0], 3, dtype=torch.float32, device=self.device)
        
        if not in_bounds.any():
            return F
        
        valid_idx = idx[in_bounds]
        i, j, k = valid_idx[:, 0], valid_idx[:, 1], valid_idx[:, 2]
        
        # Lookup distance fields
        d_occ = self.d_occ[i, j, k]
        d_unk = self.d_unk[i, j, k]
        d_trust = self.d_trust[i, j, k]
        
        grad_d_occ = self.grad_d_occ[i, j, k]
        grad_d_unk = self.grad_d_unk[i, j, k]
        grad_d_trust = self.grad_d_trust[i, j, k]
        
        is_free = self.free_mask_t[i, j, k]
        
        # ∇E_occ = (w_occ/σ²) * d * exp(-d²/(2σ²)) * ∇d
        exp_occ = torch.exp(-(d_occ**2) / (2 * self.sigma_occ**2))
        coef_occ = (self.w_occ / (self.sigma_occ**2)) * d_occ * exp_occ
        grad_E_occ = coef_occ.unsqueeze(-1) * grad_d_occ
        
        if self.r_occ_max > 0:
            grad_E_occ = grad_E_occ * (d_occ <= self.r_occ_max).float().unsqueeze(-1)
        
        # ∇E_unk = (w_unk/σ²) * d * exp(-d²/(2σ²)) * ∇d
        exp_unk = torch.exp(-(d_unk**2) / (2 * self.sigma_unk**2))
        coef_unk = (self.w_unk / (self.sigma_unk**2)) * d_unk * exp_unk
        grad_E_unk = coef_unk.unsqueeze(-1) * grad_d_unk
        
        if self.band_occ_max > 0:
            grad_E_unk = grad_E_unk * (d_occ <= self.band_occ_max).float().unsqueeze(-1)
        
        # ∇E_free = φ'(d_trust) * ∇d_trust, only in FREE
        # Support different barrier types for ablation study
        if self.barrier_type == "softplus":
            # softplus: φ(s) = softplus((s-δ)/τ), φ'(s) = (1/τ) * sigmoid((s-δ)/τ)
            phi_prime = (1.0 / t) * torch.sigmoid((d_trust - d) / t)
        elif self.barrier_type == "hinge":
            # hinge: φ(s) = max(0, s-δ), φ'(s) = 1 if s > δ else 0
            phi_prime = (d_trust > d).float()
        elif self.barrier_type == "log":
            # log barrier: φ(s) = -log(max(ε, δ-s)), φ'(s) = 1/(δ-s+ε)
            eps = 0.01
            phi_prime = 1.0 / (torch.clamp(d - d_trust, min=eps) + eps)
            phi_prime = torch.clamp(phi_prime, max=100.0)  # Clamp to avoid explosion
        else:
            # Default to softplus
            phi_prime = (1.0 / t) * torch.sigmoid((d_trust - d) / t)
        
        grad_E_free = phi_prime.unsqueeze(-1) * grad_d_trust
        grad_E_free = grad_E_free * is_free.float().unsqueeze(-1)
        
        # F = -∇E_occ - ∇E_unk - λ∇E_free
        force = -grad_E_occ - grad_E_unk - lam * grad_E_free
        
        # Coverage modulation (optional)
        if coverage_field is not None:
            coverage = coverage_field[i, j, k]
            modulation = self.coverage_epsilon + (1 - self.coverage_epsilon) * torch.pow(1 - coverage, self.coverage_gamma)
            force = force * modulation.unsqueeze(-1)
        
        F[in_bounds] = force
        return F
    
    def compute_energy_stats(
        self,
        xyz: torch.Tensor,
    ) -> dict:
        """
        Compute energy statistics for logging.
        
        Returns mean values of each energy component:
          - E_occ: OCC attraction energy
          - E_unk: UNK attraction energy  
          - E_free: FREE repulsion energy
          - grad_E_occ/unk/free: gradient magnitudes (force magnitudes)
        """
        idx = self.world_to_voxel_idx(xyz)
        in_bounds = self.is_in_bounds(idx)
        
        stats = {
            'E_occ_mean': 0.0,
            'E_unk_mean': 0.0,
            'E_free_mean': 0.0,
            'grad_E_occ_mean': 0.0,
            'grad_E_unk_mean': 0.0,
            'grad_E_free_mean': 0.0,
        }
        
        if not in_bounds.any():
            return stats
        
        valid_idx = idx[in_bounds]
        i, j, k = valid_idx[:, 0], valid_idx[:, 1], valid_idx[:, 2]
        
        # Distance fields
        d_occ = self.d_occ[i, j, k]
        d_unk = self.d_unk[i, j, k]
        d_trust = self.d_trust[i, j, k]
        is_free = self.free_mask_t[i, j, k]
        
        # E_occ = -w_occ * exp(-d²/(2σ²))  [Paper Eq. 14]
        # Negative energy: minimum at d=0 (on OCC surface)
        E_occ = -self.w_occ * torch.exp(-(d_occ**2) / (2 * self.sigma_occ**2))
        if self.r_occ_max > 0:
            E_occ = E_occ * (d_occ <= self.r_occ_max).float()
        
        # E_unk = -w_unk * exp(-d²/(2σ²))  [Paper Eq. 15]
        exp_unk = torch.exp(-(d_unk**2) / (2 * self.sigma_unk**2))
        E_unk = -self.w_unk * exp_unk
        
        if self.band_occ_max > 0:
            E_unk = E_unk * (d_occ <= self.band_occ_max).float()
        
        # E_free = softplus((d_trust - δ) / τ)  [Paper Eq. 16]
        # Positive energy: increases with distance from trust region
        t = self.tau
        d = self.delta
        E_free = torch.nn.functional.softplus((d_trust - d) / t)
        E_free = E_free * is_free.float()
        
        # Gradient magnitudes (from force computation)
        grad_d_occ = self.grad_d_occ[i, j, k]
        grad_d_unk = self.grad_d_unk[i, j, k]
        
        exp_occ = torch.exp(-(d_occ**2) / (2 * self.sigma_occ**2))
        coef_occ = (self.w_occ / (self.sigma_occ**2)) * d_occ * exp_occ
        grad_E_occ = coef_occ.unsqueeze(-1) * grad_d_occ
        
        coef_unk = self.w_unk * (1.0 / (self.sigma_unk**2)) * d_unk * exp_unk
        grad_E_unk = coef_unk.unsqueeze(-1) * grad_d_unk
        
        stats['E_occ_mean'] = E_occ.mean().item()
        stats['E_unk_mean'] = E_unk.mean().item()
        stats['E_free_mean'] = E_free.mean().item()
        stats['grad_E_occ_mean'] = grad_E_occ.norm(dim=-1).mean().item()
        stats['grad_E_unk_mean'] = grad_E_unk.norm(dim=-1).mean().item()
        # grad_E_free_mean would need d_trust gradient, skip for simplicity
        
        return stats
    
    def compute_total_energy(
        self,
        xyz: torch.Tensor,
    ) -> dict:
        """
        Compute total geometric energy for theoretical verification.
        
        E_geom(x) = E_occ(x) + E_unk(x) + λ * E_free(x)  [Paper Eq. 13]
        
        This function computes the exact energy values as defined in the paper,
        suitable for verifying Proposition 1 (monotone energy decrease).
        
        Returns:
            dict with:
              - E_total: total E_geom summed over all Gaussians
              - E_occ_total, E_unk_total, E_free_total: component totals
              - grad_norm_sq_total: Σ||∇E_geom||² for Corollary 1
        """
        idx = self.world_to_voxel_idx(xyz)
        in_bounds = self.is_in_bounds(idx)
        
        result = {
            'E_total': 0.0,
            'E_occ_total': 0.0,
            'E_unk_total': 0.0,
            'E_free_total': 0.0,
            'grad_norm_sq_total': 0.0,
        }
        
        if not in_bounds.any():
            return result
        
        valid_idx = idx[in_bounds]
        i, j, k = valid_idx[:, 0], valid_idx[:, 1], valid_idx[:, 2]
        
        # Distance fields
        d_occ = self.d_occ[i, j, k]
        d_unk = self.d_unk[i, j, k]
        d_trust = self.d_trust[i, j, k]
        is_free = self.free_mask_t[i, j, k]
        
        # === Energy components (Paper Eq. 14-16) ===
        # E_occ = -w_occ * exp(-d²/(2σ²))
        exp_occ = torch.exp(-(d_occ**2) / (2 * self.sigma_occ**2))
        E_occ = -self.w_occ * exp_occ
        if self.r_occ_max > 0:
            E_occ = E_occ * (d_occ <= self.r_occ_max).float()
        
        # E_unk = -w_unk * exp(-d²/(2σ²))
        exp_unk = torch.exp(-(d_unk**2) / (2 * self.sigma_unk**2))
        E_unk = -self.w_unk * exp_unk
        if self.band_occ_max > 0:
            E_unk = E_unk * (d_occ <= self.band_occ_max).float()
        
        # E_free = softplus((d_trust - δ) / τ)
        E_free = torch.nn.functional.softplus((d_trust - self.delta) / self.tau)
        E_free = E_free * is_free.float()
        
        # E_geom = E_occ + E_unk + λ * E_free
        E_geom = E_occ + E_unk + self.lambda_free * E_free
        
        # === Gradient computation (Paper Eq. 17-20) ===
        grad_d_occ = self.grad_d_occ[i, j, k]
        grad_d_unk = self.grad_d_unk[i, j, k]
        grad_d_trust = self.grad_d_trust[i, j, k]
        
        # ∇E_occ
        coef_occ = (self.w_occ / (self.sigma_occ**2)) * d_occ * exp_occ
        grad_E_occ = coef_occ.unsqueeze(-1) * grad_d_occ
        if self.r_occ_max > 0:
            grad_E_occ = grad_E_occ * (d_occ <= self.r_occ_max).float().unsqueeze(-1)
        
        # ∇E_unk
        coef_unk = (self.w_unk / (self.sigma_unk**2)) * d_unk * exp_unk
        grad_E_unk = coef_unk.unsqueeze(-1) * grad_d_unk
        if self.band_occ_max > 0:
            grad_E_unk = grad_E_unk * (d_occ <= self.band_occ_max).float().unsqueeze(-1)
        
        # ∇E_free = φ'(d) * ∇d, φ'(s) = (1/τ) * sigmoid((s-δ)/τ)
        phi_prime = (1.0 / self.tau) * torch.sigmoid((d_trust - self.delta) / self.tau)
        grad_E_free = phi_prime.unsqueeze(-1) * grad_d_trust
        grad_E_free = grad_E_free * is_free.float().unsqueeze(-1)
        
        # ∇E_geom = ∇E_occ + ∇E_unk + λ * ∇E_free
        grad_E_geom = grad_E_occ + grad_E_unk + self.lambda_free * grad_E_free
        
        # Compute totals
        result['E_occ_total'] = E_occ.sum().item()
        result['E_unk_total'] = E_unk.sum().item()
        result['E_free_total'] = (self.lambda_free * E_free).sum().item()
        result['E_total'] = E_geom.sum().item()
        result['grad_norm_sq_total'] = (grad_E_geom.norm(dim=-1)**2).sum().item()
        
        return result
    
    def compute_per_gaussian_energy(
        self,
        xyz: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute E_geom for each Gaussian (for quantile analysis).
        
        Returns:
            E: (N,) energy per Gaussian
        """
        idx = self.world_to_voxel_idx(xyz)
        in_bounds = self.is_in_bounds(idx)
        
        E = torch.zeros(xyz.shape[0], dtype=torch.float32, device=self.device)
        
        if not in_bounds.any():
            return E
        
        valid_idx = idx[in_bounds]
        i, j, k = valid_idx[:, 0], valid_idx[:, 1], valid_idx[:, 2]
        
        # Distance fields
        d_occ = self.d_occ[i, j, k]
        d_unk = self.d_unk[i, j, k]
        d_trust = self.d_trust[i, j, k]
        is_free = self.free_mask_t[i, j, k]
        
        # E_occ = -w_occ * exp(-d²/(2σ²))
        E_occ = -self.w_occ * torch.exp(-(d_occ**2) / (2 * self.sigma_occ**2))
        if self.r_occ_max > 0:
            E_occ = E_occ * (d_occ <= self.r_occ_max).float()
        
        # E_unk = -w_unk * exp(-d²/(2σ²))
        E_unk = -self.w_unk * torch.exp(-(d_unk**2) / (2 * self.sigma_unk**2))
        if self.band_occ_max > 0:
            E_unk = E_unk * (d_occ <= self.band_occ_max).float()
        
        # E_free = softplus((d_trust - δ) / τ)
        E_free = torch.nn.functional.softplus((d_trust - self.delta) / self.tau)
        E_free = E_free * is_free.float()
        
        # E_geom = E_occ + E_unk + λ * E_free
        E[in_bounds] = E_occ + E_unk + self.lambda_free * E_free
        
        return E
    
    def compute_force(
        self,
        xyz: torch.Tensor,
        coverage_field: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute force for Gaussian mean update.
        
        Dispatches to:
          - compute_geometric_force() for paper mode (E_geom)
          - Legacy U + escape force for legacy mode
        
        Args:
            xyz: (N, 3) Gaussian positions
            coverage_field: (X, Y, Z) optional coverage
        Returns:
            F: (N, 3) force vectors
        """
        if not self.use_legacy_mode:
            return self.compute_geometric_force(xyz, coverage_field)
        
        # ========== Legacy mode ==========
        if coverage_field is not None:
            modulation = self.coverage_epsilon + (1 - self.coverage_epsilon) * torch.pow(1 - coverage_field, self.coverage_gamma)
            U_eff = self.U_total * modulation
        else:
            U_eff = self.U_total
        
        # Compute -∇U using 3D convolution
        U_5d = U_eff.unsqueeze(0).unsqueeze(0)
        
        import torch.nn.functional as F_nn
        kernel_x = torch.tensor([[[-1, 0, 1]]], device=self.device, dtype=torch.float32).view(1, 1, 3, 1, 1)
        kernel_y = torch.tensor([[[-1, 0, 1]]], device=self.device, dtype=torch.float32).view(1, 1, 1, 3, 1)
        kernel_z = torch.tensor([[[-1, 0, 1]]], device=self.device, dtype=torch.float32).view(1, 1, 1, 1, 3)
        
        F_x = F_nn.conv3d(U_5d, kernel_x, padding=(1, 0, 0)).squeeze()
        F_y = F_nn.conv3d(U_5d, kernel_y, padding=(0, 1, 0)).squeeze()
        F_z = F_nn.conv3d(U_5d, kernel_z, padding=(0, 0, 1)).squeeze()
        
        force_field = -torch.stack([F_x, F_y, F_z], dim=-1) / 2.0
        
        idx = self.world_to_voxel_idx(xyz)
        in_bounds = self.is_in_bounds(idx)
        
        F = torch.zeros(xyz.shape[0], 3, dtype=torch.float32, device=self.device)
        
        if in_bounds.any():
            valid_idx = idx[in_bounds]
            i, j, k = valid_idx[:, 0], valid_idx[:, 1], valid_idx[:, 2]
            F[in_bounds] = force_field[i, j, k]
            
            if self.legacy_escape_k > 0:
                F[in_bounds] = F[in_bounds] + self.F_escape[i, j, k]
        
        return F
    
    def query_region(self, xyz: torch.Tensor) -> torch.Tensor:
        """
        Query region type: 0=OOB, 1=OCC, 2=FREE, 3=UNK
        """
        idx = self.world_to_voxel_idx(xyz)
        in_bounds = self.is_in_bounds(idx)
        
        region = torch.zeros(xyz.shape[0], dtype=torch.int32, device=self.device)
        
        if in_bounds.any():
            valid_idx = idx[in_bounds]
            i, j, k = valid_idx[:, 0], valid_idx[:, 1], valid_idx[:, 2]
            
            valid_region = torch.zeros(valid_idx.shape[0], dtype=torch.int32, device=self.device)
            valid_region[self.unk_mask_t[i, j, k]] = 3
            valid_region[self.free_mask_t[i, j, k]] = 2
            valid_region[self.occ_mask_t[i, j, k]] = 1
            
            region[in_bounds] = valid_region
        
        return region
    
    def query_E_free(
        self,
        xyz: torch.Tensor,
        delta: Optional[float] = None,
        tau: Optional[float] = None,
    ) -> torch.Tensor:
        """
        Query FREE barrier energy: E_free(x) = softplus((d_trust - δ)/τ)
        
        Only non-zero in FREE space.
        """
        d = delta if delta is not None else self.delta
        t = tau if tau is not None else self.tau
        
        idx = self.world_to_voxel_idx(xyz)
        in_bounds = self.is_in_bounds(idx)
        
        E = torch.zeros(xyz.shape[0], dtype=torch.float32, device=self.device)
        
        if in_bounds.any():
            valid_idx = idx[in_bounds]
            i, j, k = valid_idx[:, 0], valid_idx[:, 1], valid_idx[:, 2]
            
            d_trust = self.d_trust[i, j, k]
            is_free = self.free_mask_t[i, j, k]
            
            barrier = torch.nn.functional.softplus((d_trust - d) / t)
            E[in_bounds] = torch.where(is_free, barrier, torch.zeros_like(barrier))
        
        return E
    
    def query_d_occ(self, xyz: torch.Tensor) -> torch.Tensor:
        """Query d_occ(x): distance to nearest OCC."""
        idx = self.world_to_voxel_idx(xyz)
        in_bounds = self.is_in_bounds(idx)
        
        dist = torch.full((xyz.shape[0],), float('inf'), dtype=torch.float32, device=self.device)
        if in_bounds.any():
            valid_idx = idx[in_bounds]
            dist[in_bounds] = self.d_occ[valid_idx[:, 0], valid_idx[:, 1], valid_idx[:, 2]]
        return dist
    
    def query_d_trust(self, xyz: torch.Tensor) -> torch.Tensor:
        """Query d_trust(x): distance to nearest trust region (OCC ∪ UNK)."""
        idx = self.world_to_voxel_idx(xyz)
        in_bounds = self.is_in_bounds(idx)
        
        dist = torch.zeros(xyz.shape[0], dtype=torch.float32, device=self.device)
        if in_bounds.any():
            valid_idx = idx[in_bounds]
            dist[in_bounds] = self.d_trust[valid_idx[:, 0], valid_idx[:, 1], valid_idx[:, 2]]
        return dist
    
    def get_occ_voxel_count(self) -> int:
        """Return total OCC voxels."""
        return int(self.occ_mask.sum())
    
    def get_unk_voxel_count(self) -> int:
        """Return total UNK voxels."""
        return int(self.unk_mask.sum())


# Backward compatibility alias
PotentialField = GeometricEnergyField
