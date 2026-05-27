"""
CoverageTracker: Track voxel coverage by Gaussians.

Coverage is used to:
  1. Modulate attraction force (prevent clustering)
  2. Monitor how well Gaussians fill geometric trust regions
  3. Provide soft stability via hysteresis

Design:
  - Maintain per-voxel coverage count
  - Support soft coverage (weighted by Gaussian opacity/size)
  - Support hysteresis for stability
"""

import torch
import numpy as np
from typing import Optional, Tuple


class CoverageTracker:
    """
    Track voxel coverage by Gaussians.
    
    Coverage c(x) represents how "occupied" a voxel is by Gaussians:
      - c = 0: no Gaussian in this voxel
      - c = 1: fully covered (saturated)
    
    This is used to modulate the attraction force:
      F_modulated = F * g(c)
      g(c) = epsilon + (1 - epsilon) * (1 - c)^gamma
    
    Preventing clustering: high coverage -> low attraction
    """
    
    def __init__(
        self,
        grid_shape: Tuple[int, int, int],
        voxel_size: float,
        grid_origin: np.ndarray,
        # Coverage parameters
        saturation_count: int = 3,       # how many Gaussians to saturate a voxel
        soft_coverage: bool = True,      # weight by opacity
        # Hysteresis (for stability)
        use_hysteresis: bool = True,
        tau_enter: float = 0.3,          # coverage threshold to "enter" a voxel
        tau_leave: float = 0.7,          # coverage threshold to "leave" a voxel
        device: str = "cuda",
    ):
        self.device = device
        self.X, self.Y, self.Z = grid_shape
        self.voxel_size = voxel_size
        self.grid_origin = torch.tensor(grid_origin, dtype=torch.float32, device=device)
        
        self.saturation_count = max(1, saturation_count)
        self.soft_coverage = soft_coverage
        self.use_hysteresis = use_hysteresis
        self.tau_enter = tau_enter
        self.tau_leave = tau_leave
        
        # Per-voxel coverage count (soft or hard)
        self.coverage_count = torch.zeros(
            (self.X, self.Y, self.Z),
            dtype=torch.float32,
            device=device
        )
        
        # Per-Gaussian assignment (for hysteresis)
        # Maps gaussian_idx -> voxel_flat_idx (or -1 if not assigned)
        self.gaussian_assignment = None  # Will be initialized when we know N
        
        print(f"[GTPF] CoverageTracker initialized: grid={grid_shape}, saturation={saturation_count}")
    
    def world_to_voxel_idx(self, xyz: torch.Tensor) -> torch.Tensor:
        """Convert world coordinates to voxel indices."""
        idx = torch.floor((xyz - self.grid_origin) / self.voxel_size).long()
        return idx
    
    def is_in_bounds(self, idx: torch.Tensor) -> torch.Tensor:
        """Check if voxel indices are within grid bounds."""
        return (
            (idx[:, 0] >= 0) & (idx[:, 0] < self.X) &
            (idx[:, 1] >= 0) & (idx[:, 1] < self.Y) &
            (idx[:, 2] >= 0) & (idx[:, 2] < self.Z)
        )
    
    def idx_to_flat(self, idx: torch.Tensor) -> torch.Tensor:
        """Convert 3D voxel index to flat index."""
        return idx[:, 0] * (self.Y * self.Z) + idx[:, 1] * self.Z + idx[:, 2]
    
    def flat_to_idx(self, flat: torch.Tensor) -> torch.Tensor:
        """Convert flat index to 3D voxel index."""
        x = flat // (self.Y * self.Z)
        y = (flat % (self.Y * self.Z)) // self.Z
        z = flat % self.Z
        return torch.stack([x, y, z], dim=-1)
    
    def update_coverage(
        self,
        xyz: torch.Tensor,
        opacity: Optional[torch.Tensor] = None,
    ):
        """
        Update coverage based on current Gaussian positions.
        Fast vectorized version - no Python loops.
        
        Args:
            xyz: (N, 3) Gaussian means
            opacity: (N, 1) optional opacity for soft coverage
        """
        N = xyz.shape[0]
        
        # Reset coverage count
        self.coverage_count.zero_()
        
        # Get voxel indices
        idx = self.world_to_voxel_idx(xyz)
        in_bounds = self.is_in_bounds(idx)
        
        if not in_bounds.any():
            return
        
        valid_idx = idx[in_bounds]
        
        # Compute weights
        if self.soft_coverage and opacity is not None:
            weights = opacity[in_bounds].squeeze(-1).clamp(0, 1)
        else:
            weights = torch.ones(valid_idx.shape[0], device=self.device)
        
        # Fast scatter add using flat indexing
        flat_idx = self.idx_to_flat(valid_idx)
        
        # Use scatter_add on flattened coverage
        flat_coverage = self.coverage_count.view(-1)
        flat_coverage.scatter_add_(0, flat_idx, weights)
        
        # Note: hysteresis disabled for performance (was using Python for-loop)
    
    def _update_assignments(self, idx: torch.Tensor, in_bounds: torch.Tensor):
        """Update Gaussian-to-voxel assignments with hysteresis."""
        N = idx.shape[0]
        flat_idx = self.idx_to_flat(idx)
        
        for i in range(N):
            if not in_bounds[i]:
                self.gaussian_assignment[i] = -1
                continue
            
            current_voxel = flat_idx[i].item()
            prev_voxel = self.gaussian_assignment[i].item()
            
            if prev_voxel == -1:
                # Not assigned - check entry threshold
                voxel_coverage = self.get_coverage_at_flat(current_voxel)
                if voxel_coverage < self.tau_enter:
                    self.gaussian_assignment[i] = current_voxel
            elif prev_voxel == current_voxel:
                # Same voxel - stay assigned
                pass
            else:
                # Different voxel - check if should leave old / enter new
                old_coverage = self.get_coverage_at_flat(prev_voxel)
                new_coverage = self.get_coverage_at_flat(current_voxel)
                
                if old_coverage > self.tau_leave and new_coverage < self.tau_enter:
                    self.gaussian_assignment[i] = current_voxel
                # else: stay in old assignment (hysteresis)
    
    def get_coverage_at_flat(self, flat_idx: int) -> float:
        """Get normalized coverage at a flat voxel index."""
        if flat_idx < 0:
            return 0.0
        idx_3d = self.flat_to_idx(torch.tensor([flat_idx], device=self.device))[0]
        count = self.coverage_count[idx_3d[0], idx_3d[1], idx_3d[2]].item()
        return min(1.0, count / self.saturation_count)
    
    def get_coverage(self, xyz: torch.Tensor) -> torch.Tensor:
        """
        Get normalized coverage at given positions.
        
        Args:
            xyz: (N, 3) world coordinates
        Returns:
            coverage: (N,) in [0, 1]
        """
        idx = self.world_to_voxel_idx(xyz)
        in_bounds = self.is_in_bounds(idx)
        
        coverage = torch.zeros(xyz.shape[0], dtype=torch.float32, device=self.device)
        
        if in_bounds.any():
            valid_idx = idx[in_bounds]
            counts = self.coverage_count[valid_idx[:, 0], valid_idx[:, 1], valid_idx[:, 2]]
            coverage[in_bounds] = (counts / self.saturation_count).clamp(0, 1)
        
        return coverage
    
    def get_trap_force(
        self,
        xyz: torch.Tensor,
        lambda_trap: float = 0.1,
    ) -> torch.Tensor:
        """
        Compute soft trap force pulling Gaussians toward voxel centers.
        
        F_trap = -lambda_trap * (x - x_center)
        
        This provides soft stability without hard locking.
        
        Args:
            xyz: (N, 3) Gaussian means
            lambda_trap: trap strength
        Returns:
            F_trap: (N, 3) trap force
        """
        idx = self.world_to_voxel_idx(xyz)
        in_bounds = self.is_in_bounds(idx)
        
        F_trap = torch.zeros_like(xyz)
        
        if in_bounds.any():
            valid_xyz = xyz[in_bounds]
            valid_idx = idx[in_bounds]
            
            # Voxel center
            center = self.grid_origin + (valid_idx.float() + 0.5) * self.voxel_size
            
            # Trap force toward center
            F_trap[in_bounds] = -lambda_trap * (valid_xyz - center)
        
        return F_trap
    
    def get_assigned_voxel_center(self, gaussian_idx: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get the assigned voxel center for given Gaussians (for hysteresis).
        
        Returns:
            centers: (N, 3) voxel centers (world coords)
            is_assigned: (N,) bool mask
        """
        N = gaussian_idx.shape[0]
        centers = torch.zeros(N, 3, dtype=torch.float32, device=self.device)
        is_assigned = torch.zeros(N, dtype=torch.bool, device=self.device)
        
        if self.gaussian_assignment is None:
            return centers, is_assigned
        
        for i, gidx in enumerate(gaussian_idx):
            flat = self.gaussian_assignment[gidx].item()
            if flat >= 0:
                idx_3d = self.flat_to_idx(torch.tensor([flat], device=self.device))[0]
                centers[i] = self.grid_origin + (idx_3d.float() + 0.5) * self.voxel_size
                is_assigned[i] = True
        
        return centers, is_assigned
    
    def get_occ_coverage_ratio(self, occ_mask: torch.Tensor) -> float:
        """
        Compute what fraction of OCC voxels have coverage > 0.
        
        Args:
            occ_mask: (X, Y, Z) boolean tensor
        Returns:
            ratio: fraction of OCC voxels covered
        """
        occ_coverage = self.coverage_count[occ_mask]
        covered = (occ_coverage > 0).sum().item()
        total = occ_mask.sum().item()
        return covered / max(1, total)
    
    def get_unk_coverage_ratio(self, unk_mask: torch.Tensor) -> float:
        """Compute what fraction of UNK voxels have coverage > 0."""
        unk_coverage = self.coverage_count[unk_mask]
        covered = (unk_coverage > 0).sum().item()
        total = unk_mask.sum().item()
        return covered / max(1, total)
    
    def get_free_intrusion_count(self, free_mask: torch.Tensor) -> int:
        """Count how many FREE voxels have Gaussians (should be low)."""
        free_coverage = self.coverage_count[free_mask]
        return (free_coverage > 0).sum().item()
    
    def get_normalized_coverage_field(self) -> torch.Tensor:
        """
        Get the normalized coverage field (3D tensor, range [0, 1]).
        
        Returns:
            coverage: (X, Y, Z) normalized coverage
        """
        return (self.coverage_count / self.saturation_count).clamp(0, 1)
    
    def reset(self):
        """Reset all coverage tracking."""
        self.coverage_count.zero_()
        if self.gaussian_assignment is not None:
            self.gaussian_assignment.fill_(-1)
    
    def resize(self, new_n: int):
        """Resize assignment buffer for new number of Gaussians."""
        if self.gaussian_assignment is None:
            self.gaussian_assignment = torch.full(
                (new_n,), -1, dtype=torch.long, device=self.device
            )
        elif self.gaussian_assignment.shape[0] != new_n:
            old_n = self.gaussian_assignment.shape[0]
            new_assignment = torch.full(
                (new_n,), -1, dtype=torch.long, device=self.device
            )
            # Copy existing assignments
            copy_n = min(old_n, new_n)
            new_assignment[:copy_n] = self.gaussian_assignment[:copy_n]
            self.gaussian_assignment = new_assignment


class ViewConsistentCoverage:
    """
    Track per-Gaussian view-consistent coverage.
    
    Unlike voxel-based coverage, this tracks how well each Gaussian
    is "seen" across training views. This is useful for:
    
    1. Densify priority: Gaussians with low view coverage need more capacity
    2. Field gating: Under-observed regions should be guided more strongly
    3. Quality estimation: High coverage + low error = well-reconstructed
    
    Coverage is computed as:
        c_i = 1 - exp(-sum_v w_v * alpha_i_v)
    
    where alpha_i_v is Gaussian i's contribution to view v.
    """
    
    def __init__(
        self,
        n_gaussians: int,
        decay: float = 0.99,      # EMA decay for incremental updates
        saturation: float = 10.0, # sum of alpha contributions to saturate
        device: str = "cuda",
    ):
        self.device = device
        self.decay = decay
        self.saturation = saturation
        
        # Per-Gaussian accumulated alpha contribution
        self.alpha_accum = torch.zeros(n_gaussians, dtype=torch.float32, device=device)
        
        # View count for normalization
        self.view_count = 0
        
        print(f"[GTPF] ViewConsistentCoverage initialized: n={n_gaussians}, decay={decay}")
    
    def update(
        self,
        visibility_filter: torch.Tensor,
        radii: torch.Tensor,
        opacity: Optional[torch.Tensor] = None,
    ):
        """
        Update coverage after rendering a view.
        
        Args:
            visibility_filter: (N,) bool - which Gaussians were visible
            radii: (N,) - projected radii (proxy for contribution)
            opacity: (N, 1) optional - Gaussian opacity
        """
        self.view_count += 1
        
        # Simple approximation: visible Gaussians with large radii contribute more
        # More accurate would be to use actual alpha from rasterizer, but this
        # is a reasonable proxy that doesn't require rasterizer changes.
        
        contribution = torch.zeros(self.alpha_accum.shape[0], device=self.device)
        
        if visibility_filter.any():
            # Use radii as proxy for screen contribution
            # Normalize by max to get [0, 1] range
            visible_radii = radii[visibility_filter].float()
            if visible_radii.max() > 0:
                # Larger radii = more contribution
                contrib = (visible_radii / (visible_radii.max() + 1e-6)).clamp(0, 1)
                
                # Weight by opacity if available
                if opacity is not None:
                    contrib = contrib * opacity[visibility_filter].squeeze(-1).clamp(0, 1)
                
                contribution[visibility_filter] = contrib
        
        # EMA update
        self.alpha_accum = self.decay * self.alpha_accum + (1 - self.decay) * contribution
    
    def get_coverage(self) -> torch.Tensor:
        """
        Get normalized view coverage [0, 1] for each Gaussian.
        
        Returns:
            coverage: (N,) in [0, 1]
                0 = never seen / very weak contribution
                1 = saturated (well covered by many views)
        """
        # c = 1 - exp(-alpha_accum / saturation_factor)
        # Adjusted for EMA accumulation
        scaled = self.alpha_accum * self.view_count / self.saturation
        return 1 - torch.exp(-scaled)
    
    def get_densify_priority(
        self,
        grad_accum: torch.Tensor,
        denom: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute densify priority: (1 - coverage) * residual_gradient.
        
        Low coverage + high gradient = high priority (need more Gaussians here)
        High coverage + low gradient = low priority (already well reconstructed)
        
        Args:
            grad_accum: (N, 1) accumulated gradient magnitude
            denom: (N, 1) accumulation count
        Returns:
            priority: (N,) densify priority score
        """
        coverage = self.get_coverage()
        
        # Average gradient
        avg_grad = (grad_accum.squeeze(-1) / denom.squeeze(-1).clamp(min=1))
        
        # Priority = (1 - coverage) * gradient
        priority = (1 - coverage) * avg_grad
        
        return priority
    
    def get_field_gate(
        self,
        base_gate: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Get field strength gate based on view coverage.
        
        Under-observed Gaussians should be guided more by the field.
        Well-observed Gaussians should stabilize.
        
        Args:
            base_gate: (N,) optional base gate (e.g., from photometric error)
        Returns:
            gate: (N,) in [0, 1]
        """
        coverage = self.get_coverage()
        
        # Low coverage -> high gate (need more guidance)
        coverage_gate = 1 - coverage
        
        if base_gate is not None:
            # Combine: either high error OR low coverage -> high gate
            gate = torch.max(base_gate, coverage_gate)
        else:
            gate = coverage_gate
        
        return gate
    
    def resize(self, new_n: int):
        """Resize for new number of Gaussians (after densify/prune)."""
        old_n = self.alpha_accum.shape[0]
        if new_n == old_n:
            return
        
        new_accum = torch.zeros(new_n, dtype=torch.float32, device=self.device)
        copy_n = min(old_n, new_n)
        new_accum[:copy_n] = self.alpha_accum[:copy_n]
        self.alpha_accum = new_accum
    
    def reset(self):
        """Reset all tracking."""
        self.alpha_accum.zero_()
        self.view_count = 0
