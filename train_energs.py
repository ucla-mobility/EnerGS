#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_energs.py - EnerGS: Energy-Based Gaussian Splatting Training

Training script with geometric energy field guidance:
  E_geom(x) = E_occ(x) + E_unk(x) + λ * E_free(x)

Key features:
  1. Geometry-guided relax: Δμ = -η∇E_geom
  2. Photometric decoupling: ∇_μ L_photo = 0
  3. Hard FREE pruning (safety mechanism)
  4. Coverage modulation (paper extension)

Training flow:
  - Photometric gradient → appearance params + densification
  - Geometric energy → xyz positions (via relax step)

Usage:
  # First generate field cache:
  python field_gen_fast.py --scene <path> --save_field --field_out field_cache.npz
  
  # Then train:
  python train_energs.py -s <scene_path> --field_npz field_cache.npz --eval
"""

import os
import sys
import uuid
import torch
from random import randint
from tqdm import tqdm
from argparse import ArgumentParser, Namespace

from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render, network_gui
from scene import Scene
from utils.general_utils import safe_state, get_expon_lr_func
from utils.image_utils import psnr
from arguments import ModelParams, PipelineParams, OptimizationParams

# EnerGS imports
from energs import GaussianModelEnerGS
from energs.snapshot_visualizer import SnapshotVisualizer
from energs.density_visualizer import DensityVisualizer
from energs.view_corruption import ViewCorruptor, get_corruption_config, create_sparse_view_subset
from energs.trajectory_tracker import TrajectoryTracker
from energs.region_evaluator import RegionEvaluator, GeometryMetrics

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

try:
    from fused_ssim import fused_ssim
    FUSED_SSIM_AVAILABLE = True
except:
    FUSED_SSIM_AVAILABLE = False

try:
    from diff_gaussian_rasterization import SparseGaussianAdam
    SPARSE_ADAM_AVAILABLE = True
except:
    SPARSE_ADAM_AVAILABLE = False


class EnerGSParams:
    """EnerGS (Energy-Based Gaussian Splatting) parameters."""
    def __init__(self, parser: ArgumentParser):
        # Field file
        parser.add_argument("--field_npz", type=str, default="",
                          help="Path to pre-computed OCC/FREE/UNK field cache (.npz)")
        
        # === E_occ parameters ===
        parser.add_argument("--energs_w_occ", type=float, default=1.0,
                          help="w_occ: OCC attraction weight in E_occ")
        parser.add_argument("--energs_sigma_occ", type=float, default=1.0,
                          help="σ_occ: OCC attraction sigma (meters)")
        parser.add_argument("--energs_r_occ_max", type=float, default=3.0,
                          help="OCC cutoff radius (meters, 0=disabled)")
        
        # === E_unk parameters ===
        parser.add_argument("--energs_w_unk", type=float, default=0.25,
                          help="w_unk: UNK attraction weight in E_unk")
        parser.add_argument("--energs_sigma_unk", type=float, default=2.0,
                          help="σ_unk: UNK attraction sigma (meters)")
        parser.add_argument("--energs_band_occ_max", type=float, default=8.0,
                          help="UNK band gate distance (meters, 0=disabled)")
        
        # === Legacy mode: escape force (exp6 best: 0.2) ===
        parser.add_argument("--energs_free_escape_k", type=float, default=0.2,
                          help="Legacy: F_escape = k * d_trust (set 0 for paper mode)")
        
        # Coverage parameters (exp6 best: epsilon=1.0 to disable modulation)
        parser.add_argument("--energs_coverage_epsilon", type=float, default=1.0,
                          help="Coverage modulation floor (1.0 = disabled, exp6 best)")
        parser.add_argument("--energs_coverage_gamma", type=float, default=2.0,
                          help="Coverage modulation decay exponent")
        parser.add_argument("--energs_saturation_count", type=int, default=3,
                          help="Gaussians needed to saturate a voxel")
        
        # Relax parameters
        parser.add_argument("--energs_relax_lr", type=float, default=0.005,
                          help="Learning rate for relax step xyz update")
        parser.add_argument("--energs_relax_force_scale", type=float, default=1.0,
                          help="Scale factor for potential force")
        parser.add_argument("--energs_use_trap", action="store_true",
                          help="Use soft trap for stability")
        parser.add_argument("--energs_trap_lambda", type=float, default=0.1,
                          help="Trap strength")
        
        # Scheduling
        parser.add_argument("--energs_strategy", type=str, default="phase",
                          choices=["alternating", "phase", "triggered"],
                          help="Relax scheduling strategy")
        parser.add_argument("--energs_photo_steps", type=int, default=10,
                          help="Photometric steps per cycle (alternating)")
        parser.add_argument("--energs_relax_steps", type=int, default=5,
                          help="Relax steps per cycle (alternating)")
        parser.add_argument("--energs_relax_start", type=int, default=500,
                          help="Start relax at this iteration (phase)")
        parser.add_argument("--energs_relax_end", type=int, default=15000,
                          help="Stop relax at this iteration (phase)")
        parser.add_argument("--energs_relax_interval", type=int, default=100,
                          help="Relax every N iterations (phase)")
        parser.add_argument("--energs_relax_duration", type=int, default=20,
                          help="Relax for M iterations per interval (phase)")
        
        # Pruning (exp6 best: enabled, interval=3000)
        parser.add_argument("--energs_prune_free", action="store_true", default=True,
                          help="Periodically prune Gaussians in FREE space (default: enabled)")
        parser.add_argument("--energs_no_prune_free", action="store_true",
                          help="Disable FREE pruning")
        parser.add_argument("--energs_prune_free_interval", type=int, default=3000,
                          help="Prune FREE Gaussians every N iterations (exp6 best: 3000)")
        
        # Gradient control
        parser.add_argument("--energs_freeze_xyz_in_relax", action="store_true", default=True,
                          help="Freeze xyz from photometric gradient during relax phase")
        parser.add_argument("--energs_no_freeze_xyz", action="store_true",
                          help="[Ablation] Allow photometric gradient to update xyz")
        
        # === FREE barrier parameters ===
        # For paper mode: λ in E_geom = E_occ + E_unk + λ*E_free
        parser.add_argument("--energs_barrier_lambda", type=float, default=2.0,
                          help="Paper mode: λ for E_free in relax")
        parser.add_argument("--energs_barrier_delta", type=float, default=0.5,
                          help="δ margin distance (meters) for softplus barrier")
        parser.add_argument("--energs_barrier_tau", type=float, default=0.5,
                          help="τ temperature for softplus barrier (smaller = sharper)")
        parser.add_argument("--energs_barrier_type", type=str, default="softplus",
                          choices=["softplus", "hinge", "log"],
                          help="[Ablation] Barrier function type: softplus (default), hinge, log")
        
        # === Mode selection ===
        # Default: paper mode (E_geom formulation)
        parser.add_argument("--energs_use_paper_energy", action="store_true", default=True,
                          help="Use paper E_geom formulation (F = -∇E_occ - ∇E_unk - λ∇E_free)")
        parser.add_argument("--energs_use_legacy", action="store_true", default=False,
                          help="Use legacy U + escape force")
        parser.add_argument("--energs_no_legacy", action="store_true", default=True,
                          help="Disable legacy mode, use paper mode instead")
        
        # === Barrier as additional loss ===
        parser.add_argument("--energs_barrier_loss_weight", type=float, default=0.1,
                          help="Barrier loss weight")
        
        # === Photometric gate (exp6 best: enabled) ===
        parser.add_argument("--energs_use_photometric_gate", action="store_true", default=True,
                          help="Gate field strength by photometric error (default: enabled)")
        parser.add_argument("--energs_no_photometric_gate", action="store_true",
                          help="Disable photometric gate")
        parser.add_argument("--energs_gate_threshold", type=float, default=0.0002,
                          help="Photometric gate threshold")
        parser.add_argument("--energs_gate_temperature", type=float, default=0.0001,
                          help="Photometric gate temperature")
        
        # === View-consistent coverage gate (exp6 best: enabled) ===
        parser.add_argument("--energs_use_view_coverage_gate", action="store_true", default=True,
                          help="Gate field strength by view coverage (default: enabled)")
        parser.add_argument("--energs_no_view_coverage_gate", action="store_true",
                          help="Disable view coverage gate")
        
        # === Relax decay schedule ===
        parser.add_argument("--energs_relax_lr_final", type=float, default=0.0001,
                          help="Final relax learning rate")
        parser.add_argument("--energs_relax_decay_start", type=int, default=15000,
                          help="Start relax lr decay at this iteration")
        parser.add_argument("--energs_relax_decay_end", type=int, default=15000,
                          help="End relax lr decay at this iteration (exp6 best: 15000)")
        
        # Snapshot visualization
        parser.add_argument("--energs_save_snapshots", action="store_true", default=True,
                          help="Save BEV snapshots after each densify (default: enabled)")
        parser.add_argument("--energs_no_save_snapshots", action="store_true",
                          help="Disable BEV snapshot saving")
        parser.add_argument("--energs_snapshot_interval", type=int, default=500,
                          help="Save snapshot every N iterations (in addition to densify)")
        parser.add_argument("--energs_snapshot_dir", type=str, default="snapshots",
                          help="Directory for snapshot images (relative to model_path)")
        
        # === Density visualization (for paper figures) ===
        parser.add_argument("--save_density", action="store_true", default=True,
                          help="Save Gaussian density heatmaps (default: enabled)")
        parser.add_argument("--no_save_density", action="store_true",
                          help="Disable density heatmap saving")
        parser.add_argument("--density_interval", type=int, default=1000,
                          help="Save density heatmap every N iterations")
        parser.add_argument("--density_resolution", type=int, default=512,
                          help="Resolution for density heatmap (default: 512)")
        parser.add_argument("--density_dpi", type=int, default=300,
                          help="DPI for density figures (default: 300 for paper)")
        parser.add_argument("--density_dir", type=str, default="density",
                          help="Directory for density images (relative to model_path)")
        
        # === Stats and scale cap ===
        parser.add_argument("--energs_stats_interval", type=int, default=500,
                          help="Interval for logging detailed stats to JSON (e.g. 100, 500)")
        parser.add_argument("--energs_psnr_interval", type=int, default=2000,
                          help="Interval for evaluating train/test PSNR (e.g. 2000)")
        parser.add_argument("--energs_max_scale", type=float, default=0.0,
                          help="Max Gaussian scale in meters (0=disabled, e.g. 0.5 for 0.5m cap)")
        
        # === Theory verification (Theorem 1) ===
        parser.add_argument("--energs_theory_verify", action="store_true", default=False,
                          help="Enable Theorem 1 verification logging (energy descent, deep FREE, etc.)")
        parser.add_argument("--energs_theory_interval", type=int, default=100,
                          help="Interval for theory verification stats (default: 100)")
        parser.add_argument("--energs_theory_epsilon", type=float, default=0.1,
                          help="ε for computing s_0 threshold (default: 0.1)")
        
        # === Robustness testing: View corruption ===
        parser.add_argument("--view_corruption", type=str, default="",
                          help="View corruption config: clean, mild_noise, moderate_noise, severe_noise, occlusion_test, blur_test")
        parser.add_argument("--corruption_severity", type=float, default=0.5,
                          help="Corruption severity (0-1)")
        parser.add_argument("--corruption_prob", type=float, default=0.3,
                          help="Probability of corrupting each view")
        
        # === Robustness testing: Sparse views ===
        parser.add_argument("--sparse_view_ratio", type=float, default=1.0,
                          help="Fraction of training views to use (1.0=all, 0.5=half)")
        parser.add_argument("--sparse_view_strategy", type=str, default="uniform",
                          choices=["uniform", "random", "front_only", "side_only"],
                          help="Sparse view selection strategy")
        
        # === Trajectory tracking (for stability analysis) ===
        parser.add_argument("--track_trajectories", action="store_true", default=False,
                          help="Track Gaussian position trajectories during training")
        parser.add_argument("--track_n_gaussians", type=int, default=1000,
                          help="Number of Gaussians to track")
        parser.add_argument("--track_interval", type=int, default=100,
                          help="Trajectory snapshot interval")


def get_relax_lr_decay(
    iteration: int,
    lr_init: float,
    lr_final: float,
    decay_start: int,
    decay_end: int,
) -> float:
    """
    Compute decayed relax learning rate.
    
    - Before decay_start: lr_init
    - Between decay_start and decay_end: linear decay
    - After decay_end: lr_final
    """
    if iteration < decay_start:
        return lr_init
    elif iteration >= decay_end:
        return lr_final
    else:
        # Linear decay
        t = (iteration - decay_start) / max(1, decay_end - decay_start)
        return lr_init + t * (lr_final - lr_init)


def training_energs(
    dataset, opt, pipe, energs_args,
    testing_iterations, saving_iterations, checkpoint_iterations,
    checkpoint, debug_from
):
    """Training loop with EnerGS integration."""
    
    # Handle --energs_no_xxx flags (override defaults)
    if hasattr(energs_args, 'energs_no_prune_free') and energs_args.energs_no_prune_free:
        energs_args.energs_prune_free = False
    if hasattr(energs_args, 'energs_no_photometric_gate') and energs_args.energs_no_photometric_gate:
        energs_args.energs_use_photometric_gate = False
    if hasattr(energs_args, 'energs_no_view_coverage_gate') and energs_args.energs_no_view_coverage_gate:
        energs_args.energs_use_view_coverage_gate = False
    if hasattr(energs_args, 'energs_no_legacy') and energs_args.energs_no_legacy:
        energs_args.energs_use_legacy = False
    if hasattr(energs_args, 'energs_no_save_snapshots') and energs_args.energs_no_save_snapshots:
        energs_args.energs_save_snapshots = False
    if hasattr(energs_args, 'no_save_density') and energs_args.no_save_density:
        energs_args.save_density = False
    if hasattr(energs_args, 'energs_no_freeze_xyz') and energs_args.energs_no_freeze_xyz:
        energs_args.energs_freeze_xyz_in_relax = False
    
    if not SPARSE_ADAM_AVAILABLE and opt.optimizer_type == "sparse_adam":
        sys.exit(f"Sparse adam not available, please install the correct rasterizer.")
    
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    
    # Use GaussianModelEnerGS instead of GaussianModel
    gaussians = GaussianModelEnerGS(dataset.sh_degree, opt.optimizer_type)
    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)
    
    # Setup EnerGS if field file provided
    energs_enabled = bool(energs_args.field_npz.strip())
    if energs_enabled:
        field_path = energs_args.field_npz
        if not os.path.isabs(field_path):
            field_path = os.path.join(dataset.source_path, field_path)
        
        if not os.path.exists(field_path):
            print(f"[EnerGS] WARNING: Field file not found: {field_path}")
            print(f"[EnerGS] Training without EnerGS guidance")
            energs_enabled = False
        else:
            # Determine mode: paper energy vs legacy
            use_paper_energy = energs_args.energs_use_paper_energy and not energs_args.energs_use_legacy
            
            gaussians.setup_energs(
                field_npz_path=field_path,
                # Paper energy params (E_occ, E_unk)
                w_occ=energs_args.energs_w_occ,
                sigma_occ=energs_args.energs_sigma_occ,
                r_occ_max=energs_args.energs_r_occ_max,
                w_unk=energs_args.energs_w_unk,
                sigma_unk=energs_args.energs_sigma_unk,
                band_occ_max=energs_args.energs_band_occ_max,
                # Paper barrier params (E_free)
                barrier_lambda=energs_args.energs_barrier_lambda,
                barrier_delta=energs_args.energs_barrier_delta,
                barrier_tau=energs_args.energs_barrier_tau,
                barrier_type=energs_args.energs_barrier_type,
                # Coverage modulation (paper extension)
                coverage_epsilon=energs_args.energs_coverage_epsilon,
                coverage_gamma=energs_args.energs_coverage_gamma,
                saturation_count=energs_args.energs_saturation_count,
                # Mode
                use_paper_energy=use_paper_energy,
                free_escape_k=energs_args.energs_free_escape_k,  # should be 0 for paper
                # Relax params
                relax_lr=energs_args.energs_relax_lr,
                relax_force_scale=energs_args.energs_relax_force_scale,
                use_trap=energs_args.energs_use_trap,
                trap_lambda=energs_args.energs_trap_lambda,
                xyz_freeze_in_relax=energs_args.energs_freeze_xyz_in_relax,
            )
            
    
    # Note: RelaxScheduler removed - now xyz is ALWAYS updated by potential field
    # Photometric only affects: densify decisions + other params (opacity, scale, rot, SH)
    
    # Setup snapshot visualizer
    snapshot_vis = None
    if energs_enabled and energs_args.energs_save_snapshots:
        snapshot_dir = os.path.join(dataset.model_path, energs_args.energs_snapshot_dir)
        snapshot_vis = SnapshotVisualizer(
            output_dir=snapshot_dir,
            potential_field=gaussians.potential_field,
        )
        # Save initial snapshot
        snapshot_vis.save_snapshot(gaussians.get_xyz, iteration=0, tag="init")
    
    # Setup density visualizer (works for both EnerGS and baseline)
    density_vis = None
    if energs_args.save_density:
        density_dir = os.path.join(dataset.model_path, energs_args.density_dir)
        density_vis = DensityVisualizer(
            output_dir=density_dir,
            potential_field=gaussians.potential_field if energs_enabled else None,
            dpi=energs_args.density_dpi,
        )
        # Save energy field slices (only for EnerGS)
        if energs_enabled:
            density_vis.save_energy_field_slices()
        # Save initial density (energy field already saved separately in save_energy_field_slices)
        density_vis.save_density_heatmap(gaussians.get_xyz, iteration=0, tag="init",
                                         show_field_overlay=energs_enabled)
    
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)
    
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    
    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)
    
    use_sparse_adam = opt.optimizer_type == "sparse_adam" and SPARSE_ADAM_AVAILABLE
    depth_l1_weight = get_expon_lr_func(opt.depth_l1_weight_init, opt.depth_l1_weight_final, max_steps=opt.iterations)
    
    viewpoint_stack = scene.getTrainCameras().copy()
    viewpoint_indices = list(range(len(viewpoint_stack)))
    ema_loss_for_log = 0.0
    ema_Ll1depth_for_log = 0.0
    
    # EnerGS logging
    energs_log_interval = 100  # For tensorboard
    stats_log_interval = energs_args.energs_stats_interval if energs_enabled else 500
    last_region_stats = {}
    last_relax_stats = {}
    energs_stats_log = []  # Store all stats for JSON export
    
    # Track individual loss components for JSON export
    last_Ll1 = 0.0
    last_ssim = 0.0
    last_Ll1depth = 0.0
    last_barrier_loss = 0.0
    last_total_loss = 0.0
    
    # Theory verification (Theorem 1)
    theory_verify_enabled = energs_enabled and energs_args.energs_theory_verify
    theory_interval = energs_args.energs_theory_interval
    theory_epsilon = energs_args.energs_theory_epsilon
    theory_log = []  # Separate log for theory verification
    prev_E_total = None  # For checking monotone descent
    
    # Save initial config to JSON (first entry) - works for both baseline and EnerGS
    import json
    config_entry = {
        'type': 'config',
        'iteration': 0,
        'energs_enabled': energs_enabled,
        # Training info
        'iterations': opt.iterations,
        'stats_interval': stats_log_interval,
        'psnr_interval': energs_args.energs_psnr_interval,
        # Densification params
        'densify_from_iter': opt.densify_from_iter,
        'densify_until_iter': opt.densify_until_iter,
        'densify_grad_threshold': opt.densify_grad_threshold,
        'densification_interval': opt.densification_interval,
    }
    
    # Add EnerGS-specific config
    if energs_enabled:
        config_entry.update({
            # Energy field parameters
            'w_occ': energs_args.energs_w_occ,
            'sigma_occ': energs_args.energs_sigma_occ,
            'r_occ_max': energs_args.energs_r_occ_max,
            'w_unk': energs_args.energs_w_unk,
            'sigma_unk': energs_args.energs_sigma_unk,
            'band_occ_max': energs_args.energs_band_occ_max,
            # Barrier parameters
            'barrier_lambda': energs_args.energs_barrier_lambda,
            'barrier_delta': energs_args.energs_barrier_delta,
            'barrier_tau': energs_args.energs_barrier_tau,
            'barrier_loss_weight': energs_args.energs_barrier_loss_weight,
            # Coverage parameters
            'coverage_epsilon': energs_args.energs_coverage_epsilon,
            'coverage_gamma': energs_args.energs_coverage_gamma,
            'saturation_count': energs_args.energs_saturation_count,
            # Relax parameters
            'relax_lr': energs_args.energs_relax_lr,
            'relax_lr_final': energs_args.energs_relax_lr_final,
            'relax_force_scale': energs_args.energs_relax_force_scale,
            'relax_decay_start': energs_args.energs_relax_decay_start,
            'relax_decay_end': energs_args.energs_relax_decay_end,
            # Mode
            'use_paper_energy': energs_args.energs_use_paper_energy,
            'use_legacy': energs_args.energs_use_legacy,
            'free_escape_k': energs_args.energs_free_escape_k,
            # Pruning
            'prune_free': energs_args.energs_prune_free,
            'prune_free_interval': energs_args.energs_prune_free_interval,
            # Gates
            'use_photometric_gate': energs_args.energs_use_photometric_gate,
            'use_view_coverage_gate': energs_args.energs_use_view_coverage_gate,
            'field_npz': energs_args.field_npz,
        })
    
    energs_stats_log.append(config_entry)
    
    # Initial stats entry
    init_stats_entry = {
        'type': 'init',
        'iteration': 0,
        'n_gaussians': gaussians.get_xyz.shape[0],
    }
    
    # Add EnerGS-specific init stats
    if energs_enabled:
        initial_region_stats = gaussians.get_region_stats()
        init_stats_entry.update({
            'n_occ': initial_region_stats.get('n_occ', 0),
            'n_free': initial_region_stats.get('n_free', 0),
            'n_unk': initial_region_stats.get('n_unk', 0),
            'n_oob': initial_region_stats.get('n_oob', 0),
        })
    
    energs_stats_log.append(init_stats_entry)
    
    # Save initial config
    stats_filename = "training_stats.json"
    with open(os.path.join(dataset.model_path, stats_filename), 'w') as f:
        json.dump(energs_stats_log, f, indent=2)
    print(f"[Training] Config saved to {stats_filename} (energs_enabled={energs_enabled})")
    
    # === View corruption for robustness testing ===
    view_corruptor = None
    if hasattr(energs_args, 'view_corruption') and energs_args.view_corruption:
        corruption_config = get_corruption_config(energs_args.view_corruption)
        if energs_args.corruption_severity > 0:
            corruption_config['severity'] = energs_args.corruption_severity
        if energs_args.corruption_prob > 0:
            corruption_config['corruption_prob'] = energs_args.corruption_prob
        view_corruptor = ViewCorruptor(**corruption_config)
        print(f"[View Corruption] Enabled: {energs_args.view_corruption}, severity={corruption_config['severity']}")
    
    # === Sparse view subset ===
    train_cameras_full = scene.getTrainCameras()
    if hasattr(energs_args, 'sparse_view_ratio') and energs_args.sparse_view_ratio < 1.0:
        sparse_strategy = getattr(energs_args, 'sparse_view_strategy', 'uniform')
        train_cameras_sparse = create_sparse_view_subset(
            train_cameras_full, 
            keep_ratio=energs_args.sparse_view_ratio,
            strategy=sparse_strategy,
        )
        print(f"[Sparse Views] Using {len(train_cameras_sparse)}/{len(train_cameras_full)} views ({energs_args.sparse_view_ratio*100:.0f}%)")
        # Replace scene's train cameras
        scene._train_cameras = {1.0: train_cameras_sparse}
    
    # === Trajectory tracker for stability analysis ===
    trajectory_tracker = None
    if hasattr(energs_args, 'track_trajectories') and energs_args.track_trajectories:
        track_dir = os.path.join(dataset.model_path, "trajectories")
        os.makedirs(track_dir, exist_ok=True)
        trajectory_tracker = TrajectoryTracker(
            n_track=energs_args.track_n_gaussians,
            track_interval=energs_args.track_interval,
            save_dir=track_dir,
        )
        # Initialize with current Gaussians
        xyz = gaussians.get_xyz.detach()
        regions = None
        if energs_enabled:
            regions = gaussians.energy_field.query_region(xyz)
        trajectory_tracker.initialize(xyz, regions, iteration=0)
        print(f"[Trajectory Tracking] Tracking {energs_args.track_n_gaussians} Gaussians")
    
    # === Geometry metrics tracker ===
    geometry_metrics = None
    if energs_enabled:
        geometry_metrics = GeometryMetrics(gaussians.energy_field)
    
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    
    for iteration in range(first_iter, opt.iterations + 1):
        # Network GUI handling
        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            try:
                net_image_bytes = None
                custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifier = network_gui.receive()
                if custom_cam != None:
                    net_image = render(custom_cam, gaussians, pipe, background, scaling_modifier=scaling_modifier, 
                                      use_trained_exp=dataset.train_test_exp, separate_sh=SPARSE_ADAM_AVAILABLE)["render"]
                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                network_gui.send(net_image_bytes, dataset.source_path)
                if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                    break
            except Exception as e:
                network_gui.conn = None
        
        iter_start.record()
        
        gaussians.update_learning_rate(iteration)
        
        # EnerGS: Always do relax step (xyz updated by E_geom every iteration)
        if energs_enabled:
            # Compute decayed relax learning rate
            relax_lr = get_relax_lr_decay(
                iteration,
                lr_init=energs_args.energs_relax_lr,
                lr_final=energs_args.energs_relax_lr_final,
                decay_start=energs_args.energs_relax_decay_start,
                decay_end=energs_args.energs_relax_decay_end,
            )
            
            # Compute combined gate (photometric + view coverage)
            combined_gate = None
            if energs_args.energs_use_photometric_gate or energs_args.energs_use_view_coverage_gate:
                combined_gate = gaussians.compute_combined_gate(
                    use_photometric=energs_args.energs_use_photometric_gate,
                    use_view_coverage=energs_args.energs_use_view_coverage_gate,
                    photometric_threshold=energs_args.energs_gate_threshold,
                    photometric_temperature=energs_args.energs_gate_temperature,
                )
            
            # Only compute stats when logging (expensive)
            compute_stats = (iteration % stats_log_interval == 0)
            relax_stats = gaussians.relax_step(
                compute_stats=compute_stats,
                photometric_gate=combined_gate,
                relax_lr_override=relax_lr,
            )
            
            # Log relax stats to tensorboard
            if tb_writer and compute_stats and relax_stats:
                for key, val in relax_stats.items():
                    tb_writer.add_scalar(f'energs/{key}', val, iteration)
                tb_writer.add_scalar('energs/relax_lr', relax_lr, iteration)
            
            # Store for JSON
            if compute_stats:
                last_relax_stats = relax_stats
        
        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()
        
        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
            viewpoint_indices = list(range(len(viewpoint_stack)))
        rand_idx = randint(0, len(viewpoint_indices) - 1)
        viewpoint_cam = viewpoint_stack.pop(rand_idx)
        vind = viewpoint_indices.pop(rand_idx)
        
        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True
        
        bg = torch.rand((3), device="cuda") if opt.random_background else background
        
        render_pkg = render(viewpoint_cam, gaussians, pipe, bg, 
                          use_trained_exp=dataset.train_test_exp, separate_sh=SPARSE_ADAM_AVAILABLE)
        image, viewspace_point_tensor, visibility_filter, radii = \
            render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
        
        if viewpoint_cam.alpha_mask is not None:
            alpha_mask = viewpoint_cam.alpha_mask.cuda()
            image *= alpha_mask
        
        
        # Loss
        gt_image = viewpoint_cam.original_image.cuda()
        
        # Apply view corruption for robustness testing
        gt_corrupted = False
        if view_corruptor is not None:
            gt_image, gt_corrupted = view_corruptor(gt_image)
        
        Ll1 = l1_loss(image, gt_image)
        if FUSED_SSIM_AVAILABLE:
            ssim_value = fused_ssim(image.unsqueeze(0), gt_image.unsqueeze(0))
        else:
            ssim_value = ssim(image, gt_image)
        
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim_value)
        
        # Depth regularization
        Ll1depth_pure = 0.0
        if depth_l1_weight(iteration) > 0 and viewpoint_cam.depth_reliable:
            invDepth = render_pkg["depth"]
            mono_invdepth = viewpoint_cam.invdepthmap.cuda()
            depth_mask = viewpoint_cam.depth_mask.cuda()
            
            Ll1depth_pure = torch.abs((invDepth - mono_invdepth) * depth_mask).mean()
            Ll1depth = depth_l1_weight(iteration) * Ll1depth_pure
            loss += Ll1depth
            Ll1depth = Ll1depth.item()
        else:
            Ll1depth = 0
        
        # EnerGS: E_free barrier loss (optional soft constraint)
        barrier_loss = 0.0
        if energs_enabled and energs_args.energs_barrier_loss_weight > 0:
            barrier_loss = gaussians.get_free_barrier_loss(
                delta=energs_args.energs_barrier_delta,
                tau=energs_args.energs_barrier_tau,
            )
            loss = loss + energs_args.energs_barrier_loss_weight * barrier_loss
            barrier_loss = barrier_loss.item()
        
        loss.backward()
        
        # Track individual loss components for JSON export
        last_Ll1 = Ll1.item()
        last_ssim = ssim_value.item() if torch.is_tensor(ssim_value) else ssim_value
        last_Ll1depth = Ll1depth if isinstance(Ll1depth, float) else Ll1depth
        last_barrier_loss = barrier_loss if isinstance(barrier_loss, float) else barrier_loss
        last_total_loss = loss.item()
        
        # EnerGS: Update view-consistent coverage after render
        if energs_enabled:
            gaussians.update_view_coverage(visibility_filter, radii)
        
        # EnerGS: Zero out xyz gradient (xyz is controlled by E_geom only)
        # Photometric gradient only affects: opacity, scaling, rotation, SH features
        # Photometric gradient also drives densify decisions via viewspace_point_tensor
        if energs_enabled:
            gaussians._xyz.grad = None
        
        iter_end.record()
        
        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            ema_Ll1depth_for_log = 0.4 * Ll1depth + 0.6 * ema_Ll1depth_for_log
            
            # EnerGS region statistics (compute every iter for display)
            if energs_enabled and iteration % 1 == 0:
                last_region_stats = gaussians.get_region_stats()
            
            if iteration % 10 == 0:
                postfix = {"Loss": f"{ema_loss_for_log:.{7}f}"}
                if energs_enabled and last_region_stats:
                    n_occ = last_region_stats.get('n_occ', 0)
                    n_free = last_region_stats.get('n_free', 0)
                    n_unk = last_region_stats.get('n_unk', 0)
                    total = last_region_stats.get('total', 1)
                    # Compact display: O=occ F=free U=unk
                    postfix["O"] = f"{n_occ}"
                    postfix["F"] = f"{n_free}"
                    postfix["U"] = f"{n_unk}"
                    postfix["Tot"] = f"{total}"
                progress_bar.set_postfix(postfix)
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()
            
            # EnerGS region statistics logging to tensorboard
            if energs_enabled and iteration % energs_log_interval == 0:
                if tb_writer and last_region_stats:
                    for key, val in last_region_stats.items():
                        tb_writer.add_scalar(f'energs_region/{key}', val, iteration)
                if tb_writer and barrier_loss > 0:
                    tb_writer.add_scalar('energs/barrier_loss', barrier_loss, iteration)
            
            # Collect all stats at configured interval (for JSON export)
            # Works for both baseline and EnerGS modes
            if iteration % stats_log_interval == 0:
                # Build comprehensive stats dict
                stats_entry = {
                    'type': 'stats',
                    'iteration': iteration,
                    'n_gaussians': gaussians.get_xyz.shape[0],
                    'energs_enabled': energs_enabled,
                }
                
                # EnerGS-specific stats (only when enabled)
                if energs_enabled:
                    stats_entry['relax_lr'] = relax_lr
                    
                    # Region stats
                    region_stats = gaussians.get_region_stats()
                    stats_entry.update({
                        'n_occ': region_stats.get('n_occ', 0),
                        'n_free': region_stats.get('n_free', 0),
                        'n_unk': region_stats.get('n_unk', 0),
                        'n_oob': region_stats.get('n_oob', 0),
                    })
                    
                    # Compute region ratios for easy visualization
                    total = region_stats.get('total', 1)
                    stats_entry.update({
                        'ratio_occ': region_stats.get('n_occ', 0) / max(1, total),
                        'ratio_free': region_stats.get('n_free', 0) / max(1, total),
                        'ratio_unk': region_stats.get('n_unk', 0) / max(1, total),
                    })
                    
                    # Relax stats (force, delta, coverage)
                    if last_relax_stats:
                        stats_entry.update({
                            'mean_force_mag': last_relax_stats.get('mean_force_mag', 0),
                            'mean_delta': last_relax_stats.get('mean_delta', 0),
                            'max_delta': last_relax_stats.get('max_delta', 0),
                            'occ_coverage': last_relax_stats.get('occ_coverage', 0),
                            'unk_coverage': last_relax_stats.get('unk_coverage', 0),
                            'free_intrusion': last_relax_stats.get('free_intrusion', 0),
                            # Per-region delta (key insight!)
                            'delta_occ_mean': last_relax_stats.get('delta_occ_mean', 0),
                            'delta_occ_max': last_relax_stats.get('delta_occ_max', 0),
                            'delta_free_mean': last_relax_stats.get('delta_free_mean', 0),
                            'delta_free_max': last_relax_stats.get('delta_free_max', 0),
                            'delta_unk_mean': last_relax_stats.get('delta_unk_mean', 0),
                            'delta_unk_max': last_relax_stats.get('delta_unk_max', 0),
                            # Per-region force
                            'force_occ_mean': last_relax_stats.get('force_occ_mean', 0),
                            'force_free_mean': last_relax_stats.get('force_free_mean', 0),
                            'force_unk_mean': last_relax_stats.get('force_unk_mean', 0),
                            # Energy components
                            'E_occ_mean': last_relax_stats.get('E_occ_mean', 0),
                            'E_unk_mean': last_relax_stats.get('E_unk_mean', 0),
                            'E_free_mean': last_relax_stats.get('E_free_mean', 0),
                            'grad_E_occ_mean': last_relax_stats.get('grad_E_occ_mean', 0),
                            'grad_E_unk_mean': last_relax_stats.get('grad_E_unk_mean', 0),
                        })
                    
                    # Geometry loss (E_free barrier) - only for EnerGS
                    stats_entry['loss_barrier'] = last_barrier_loss
                
                # Add loss components for analysis (both baseline and EnerGS)
                stats_entry.update({
                    # Photometric losses
                    'loss_total': last_total_loss,
                    'loss_l1': last_Ll1,
                    'loss_ssim': 1.0 - last_ssim,  # Convert to loss (lower is better)
                    'ssim': last_ssim,  # Raw SSIM value (higher is better)
                    # Depth loss
                    'loss_depth': last_Ll1depth,
                    # EMA loss for smoothed view
                    'loss_ema': ema_loss_for_log,
                    'loss_depth_ema': ema_Ll1depth_for_log,
                })
                
                # Evaluate train/test PSNR at psnr_interval (both baseline and EnerGS)
                psnr_interval = energs_args.energs_psnr_interval
                if iteration > 0 and iteration % psnr_interval == 0:
                    with torch.no_grad():
                        # Test PSNR (all test cameras)
                        test_cameras = scene.getTestCameras()
                        if test_cameras and len(test_cameras) > 0:
                            psnr_test_total = 0.0
                            for viewpoint in test_cameras:
                                image = torch.clamp(render(viewpoint, gaussians, pipe, background, 
                                                          1., SPARSE_ADAM_AVAILABLE, None, dataset.train_test_exp)["render"], 0.0, 1.0)
                                gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                                if dataset.train_test_exp:
                                    image = image[..., image.shape[-1] // 2:]
                                    gt_image = gt_image[..., gt_image.shape[-1] // 2:]
                                psnr_test_total += psnr(image, gt_image).mean().double()
                            stats_entry['psnr_test'] = (psnr_test_total / len(test_cameras)).item()
                        
                        # Train PSNR (sample 5 cameras evenly)
                        train_cameras = scene.getTrainCameras()
                        if train_cameras and len(train_cameras) > 0:
                            sample_indices = [idx % len(train_cameras) for idx in range(5, 30, 5)]
                            psnr_train_total = 0.0
                            for idx in sample_indices:
                                viewpoint = train_cameras[idx]
                                image = torch.clamp(render(viewpoint, gaussians, pipe, background,
                                                          1., SPARSE_ADAM_AVAILABLE, None, dataset.train_test_exp)["render"], 0.0, 1.0)
                                gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                                if dataset.train_test_exp:
                                    image = image[..., image.shape[-1] // 2:]
                                    gt_image = gt_image[..., gt_image.shape[-1] // 2:]
                                psnr_train_total += psnr(image, gt_image).mean().double()
                            stats_entry['psnr_train'] = (psnr_train_total / len(sample_indices)).item()
                        
                        print(f"[ITER {iteration}] PSNR - train: {stats_entry.get('psnr_train', 'N/A'):.2f}, test: {stats_entry.get('psnr_test', 'N/A'):.2f}")
                
                # === Trajectory tracking: update and add stats ===
                if trajectory_tracker is not None:
                    xyz = gaussians.get_xyz.detach()
                    regions = None
                    if energs_enabled:
                        regions = gaussians.energy_field.query_region(xyz)
                    drift_stats = trajectory_tracker.update(xyz, regions, iteration)
                    if drift_stats:
                        stats_entry.update({
                            'drift_mean': drift_stats.get('drift_mean', 0),
                            'drift_p90': drift_stats.get('drift_p90', 0),
                            'drift_max': drift_stats.get('drift_max', 0),
                            'drift_occ_mean': drift_stats.get('drift_occ_mean', 0),
                            'drift_free_mean': drift_stats.get('drift_free_mean', 0),
                            'drift_unk_mean': drift_stats.get('drift_unk_mean', 0),
                            'trans_free_to_trusted': drift_stats.get('trans_free_to_trusted', 0),
                            'trans_trusted_to_free': drift_stats.get('trans_trusted_to_free', 0),
                        })
                
                # === Geometry metrics: free-space leakage ===
                if geometry_metrics is not None:
                    xyz = gaussians.get_xyz.detach()
                    opacity = gaussians.get_opacity.detach()
                    leakage_stats = geometry_metrics.compute_free_space_leakage(xyz, opacity)
                    stats_entry.update({
                        'leakage_count_ratio': leakage_stats.get('leakage_count_ratio', 0),
                        'leakage_opacity_ratio': leakage_stats.get('leakage_opacity_ratio', 0),
                    })
                
                energs_stats_log.append(stats_entry)
                
                # Real-time save
                import json
                stats_filename = "training_stats.json"
                with open(os.path.join(dataset.model_path, stats_filename), 'w') as f:
                    json.dump(energs_stats_log, f, indent=2)
            
            # === Theory Verification (Theorem 1) - Save to JSON for plotting ===
            if theory_verify_enabled and iteration > 0 and iteration % theory_interval == 0:
                theory_stats = gaussians.get_theory_verification_stats(epsilon=theory_epsilon)
                region_stats = gaussians.get_region_stats()
                
                # Check Claim 1: Energy monotone descent
                E_total = theory_stats['E_total']
                energy_decreased = True
                energy_delta = 0.0
                if prev_E_total is not None:
                    energy_delta = E_total - prev_E_total
                    energy_decreased = energy_delta <= 1e-6
                prev_E_total = E_total
                
                # Save all data needed for plotting
                theory_entry = {
                    'iteration': iteration,
                    'n_gaussians': theory_stats.get('n_total', 0),
                    
                    # === Claim 1: Energy descent + convergence ===
                    'E_total': E_total,
                    'E_occ_total': theory_stats['E_occ_total'],
                    'E_unk_total': theory_stats['E_unk_total'],
                    'E_free_total': theory_stats['E_free_total'],
                    'grad_norm_sq_total': theory_stats['grad_norm_sq_total'],
                    'energy_decreased': energy_decreased,
                    'energy_delta': energy_delta,
                    # E_geom quantiles (for Figure 3.1)
                    'E_mean': theory_stats.get('E_mean', 0),
                    'E_p50': theory_stats.get('E_p50', 0),
                    'E_p90': theory_stats.get('E_p90', 0),
                    'E_free_mean': theory_stats.get('E_free_mean', 0),
                    'E_free_p90': theory_stats.get('E_free_p90', 0),
                    
                    # === Claim 2: Deep FREE exclusion ===
                    's_0': theory_stats['s_0'],
                    'm_0': theory_stats['m_0'],
                    'B_s0': theory_stats['B_s0'],
                    'B_occ': theory_stats.get('B_occ', 0),
                    'B_unk': theory_stats.get('B_unk', 0),
                    'dominance_lhs': theory_stats['dominance_lhs'],
                    'dominance_rhs': theory_stats['dominance_rhs'],
                    'dominance_margin': theory_stats.get('dominance_margin', 0),
                    'dominance_satisfied': theory_stats['dominance_satisfied'],
                    'lambda_over_tau': theory_stats.get('lambda_over_tau', 0),
                    'n_deep_free': theory_stats['n_deep_free'],
                    # d_trust quantiles (for Figure 3.1)
                    'd_trust_mean': theory_stats.get('d_trust_mean', 0),
                    'd_trust_p50': theory_stats.get('d_trust_p50', 0),
                    'd_trust_p90': theory_stats.get('d_trust_p90', 0),
                    'd_trust_free_mean': theory_stats.get('d_trust_free_mean', 0),
                    'd_trust_free_p50': theory_stats.get('d_trust_free_p50', 0),
                    'd_trust_free_p90': theory_stats.get('d_trust_free_p90', 0),
                    'd_trust_deep_free_mean': theory_stats.get('d_trust_deep_free_mean', 0),
                    # Directional descent ⟨∇E, ∇d⟩ (Proposition 2)
                    'dir_descent_mean': theory_stats.get('dir_descent_mean', 0),
                    'dir_descent_p10': theory_stats.get('dir_descent_p10', 0),
                    'dir_descent_p50': theory_stats.get('dir_descent_p50', 0),
                    'dir_descent_positive_ratio': theory_stats.get('dir_descent_positive_ratio', 1),
                    'beta_empirical': theory_stats.get('beta_empirical', 0),
                    
                    # === Claim 4: Geometry metrics ===
                    'n_occ': theory_stats.get('n_occ', 0),
                    'n_free': theory_stats.get('n_free', 0),
                    'n_unk': theory_stats.get('n_unk', 0),
                    'occ_coverage': theory_stats['occ_coverage'],
                    'free_intrusion_ratio': theory_stats['free_intrusion_ratio'],
                }
                theory_log.append(theory_entry)
                
                # Save to JSON (overwrite each time for real-time access)
                theory_filename = "theory_verification.json"
                with open(os.path.join(dataset.model_path, theory_filename), 'w') as f:
                    json.dump(theory_log, f, indent=2)
            
            # Log and save
            training_report(tb_writer, iteration, Ll1, loss, l1_loss, 
                          iter_start.elapsed_time(iter_end), testing_iterations, scene, render,
                          (pipe, background, 1., SPARSE_ADAM_AVAILABLE, None, dataset.train_test_exp),
                          dataset.train_test_exp)
            
            if iteration in saving_iterations:
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)
                if energs_enabled:
                    energs_state_path = os.path.join(scene.model_path, f"energs_state_{iteration}.pth")
                    gaussians.save_energs_state(energs_state_path)
            
            # Densification
            if iteration < opt.densify_until_iter:
                gaussians.max_radii2D[visibility_filter] = torch.max(
                    gaussians.max_radii2D[visibility_filter], radii[visibility_filter]
                )
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)
                
                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, 
                                               scene.cameras_extent, size_threshold, radii)
                    
                    # Clamp scale if enabled
                    if energs_enabled and energs_args.energs_max_scale > 0:
                        n_clamped = gaussians.clamp_scaling(energs_args.energs_max_scale)
                        if n_clamped > 0 and iteration % 1000 == 0:
                            print(f"[EnerGS] Clamped {n_clamped} Gaussians to max_scale={energs_args.energs_max_scale}m")
                    
                    # Save snapshot after densify
                    if snapshot_vis is not None:
                        snapshot_vis.save_snapshot(gaussians.get_xyz, iteration, tag="densify")
                
                if iteration % opt.opacity_reset_interval == 0 or \
                   (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()
            
            # EnerGS: Prune FREE Gaussians periodically
            if energs_enabled and energs_args.energs_prune_free and \
               iteration > 0 and iteration % energs_args.energs_prune_free_interval == 0:
                pruned = gaussians.prune_free_gaussians(min_iterations=energs_args.energs_prune_free_interval)
                if tb_writer and pruned > 0:
                    tb_writer.add_scalar('energs/pruned_free', pruned, iteration)
            
            # EnerGS: Save periodic snapshots
            if snapshot_vis is not None and iteration > 0 and \
               iteration % energs_args.energs_snapshot_interval == 0:
                snapshot_vis.save_snapshot(gaussians.get_xyz, iteration, tag="periodic")
            
            # Save density heatmaps (both EnerGS and baseline)
            if density_vis is not None and iteration > 0 and \
               iteration % energs_args.density_interval == 0:
                density_vis.save_density_heatmap(
                    gaussians.get_xyz, iteration,
                    resolution=energs_args.density_resolution,
                    show_field_overlay=energs_enabled,
                )
            
            # Optimizer step
            if iteration < opt.iterations:
                gaussians.exposure_optimizer.step()
                gaussians.exposure_optimizer.zero_grad(set_to_none=True)
                if use_sparse_adam:
                    visible = radii > 0
                    gaussians.optimizer.step(visible, radii.shape[0])
                    gaussians.optimizer.zero_grad(set_to_none=True)
                else:
                    gaussians.optimizer.step()
                    gaussians.optimizer.zero_grad(set_to_none=True)
            
            if iteration in checkpoint_iterations:
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), 
                          scene.model_path + "/chkpnt" + str(iteration) + ".pth")
    
    # Final EnerGS statistics
    if energs_enabled:
        print("\n[EnerGS] Final statistics:")
        final_stats = gaussians.get_region_stats()
        for key, val in final_stats.items():
            print(f"  {key}: {val}")
    
    # Save final density heatmap
    if density_vis is not None:
        density_vis.save_density_heatmap(
            gaussians.get_xyz, opt.iterations, tag="final",
            resolution=energs_args.density_resolution,
            show_field_overlay=energs_enabled,
        )
        print(f"\n[Density] Saved final density to {energs_args.density_dir}/")
    
    # Theory verification summary
    if theory_verify_enabled and len(theory_log) > 0:
        theory_filename = os.path.join(dataset.model_path, "theory_verification.json")
        print(f"\n[Theory] Saved {len(theory_log)} entries to {theory_filename}")
    
    # === Trajectory tracking: save final data and visualizations ===
    if trajectory_tracker is not None:
        track_dir = os.path.join(dataset.model_path, "trajectories")
        
        # Save trajectory data
        trajectory_tracker.save("trajectories.npz")
        
        # Generate visualizations
        trajectory_tracker.visualize_trajectories(
            os.path.join(track_dir, "trajectory_paths.png"),
            n_show=50,
        )
        trajectory_tracker.visualize_drift_over_time(
            os.path.join(track_dir, "drift_over_time.png"),
        )
        
        # Save stability metrics to JSON
        stability_metrics = trajectory_tracker.get_stability_metrics()
        import json
        with open(os.path.join(track_dir, "stability_metrics.json"), 'w') as f:
            json.dump(stability_metrics, f, indent=2)
        
        print(f"\n[Trajectories] Saved tracking data to {track_dir}/")


def prepare_output_and_logger(args):
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str = os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
    
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok=True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))
    
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer


def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, 
                   testing_iterations, scene, renderFunc, renderArgs, train_test_exp):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)
    
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = (
            {'name': 'test', 'cameras': scene.getTestCameras()},
            {'name': 'train', 'cameras': [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] 
                                          for idx in range(5, 30, 5)]}
        )
        
        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if train_test_exp:
                        image = image[..., image.shape[-1] // 2:]
                        gt_image = gt_image[..., gt_image.shape[-1] // 2:]
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), 
                                           image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), 
                                               gt_image[None], global_step=iteration)
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)
        
        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        
        # EnerGS: Print voxel occupancy statistics
        if hasattr(scene.gaussians, 'print_voxel_occupancy_stats'):
            scene.gaussians.print_voxel_occupancy_stats()
        
        torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = ArgumentParser(description="EnerGS Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    energs_p = EnerGSParams(parser)  # EnerGS parameters
    
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[1000, 5000,7_000, 10000, 15000,30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[1000, 5000,7_000, 10000, 15000,30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument('--disable_viewer', action='store_true', default=False)
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default=None)
    
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    
    print("Optimizing " + args.model_path)
    print("EnerGS enabled: " + ("Yes" if args.field_npz else "No"))
    
    safe_state(args.quiet)
    
    if not args.disable_viewer:
        network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    
    training_energs(
        lp.extract(args), op.extract(args), pp.extract(args), args,
        args.test_iterations, args.save_iterations, args.checkpoint_iterations,
        args.start_checkpoint, args.debug_from
    )
    
    print("\nTraining complete.")

