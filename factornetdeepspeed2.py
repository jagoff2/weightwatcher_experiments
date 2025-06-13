"""
factor_resonance_experiments_512bit_patched_deepspeed.py - DeepSpeed ZeRO-3 Optimized 512-bit Factorization Network
Implements all critical patches (P0-P5) plus 8-bit kickstart with DeepSpeed ZeRO-3 offload for memory efficiency
"""
import os
os.environ["DS_BUILD_OPS"] = "OFF"
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import json
from dataclasses import dataclass, asdict, field
from typing import List, Tuple, Dict, Optional, Union
import logging
import warnings
import math
import random
from sympy import factorint, isprime, nextprime, legendre_symbol
import gmpy2
from collections import defaultdict
from functools import lru_cache
import time
import argparse
import torch.cuda.amp as amp
import deepspeed
from deepspeed import DeepSpeedConfig
from deepspeed.ops.adam import DeepSpeedCPUAdam, FusedAdam
from torch.utils.data import Dataset, DataLoader, DistributedSampler
import torch.distributed as dist
print("DS_BUILD_SHM =", os.getenv("DS_BUILD_SHM"))

# Handle weightwatcher import gracefully
try:
    import weightwatcher as ww
    WW_AVAILABLE = True
except ImportError:
    WW_AVAILABLE = False
    logging.warning("WeightWatcher not available. Install with: pip install weightwatcher")

warnings.filterwarnings('ignore')

# --------------------------------------------------------------------
# ►  Enable TF32 kernels
# --------------------------------------------------------------------
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_default_dtype(torch.float32)

# Set up logging with more detailed format
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - [%(name)s] %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

# Ensure reproducibility
torch.manual_seed(42)
np.random.seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
random.seed(42)

# Create results directory
Path("results").mkdir(exist_ok=True)

# Generate first 4096 primes for expanded modular arithmetic
def generate_first_n_primes(n):
    """Generate first n prime numbers"""
    primes = [2]
    candidate = 3
    while len(primes) < n:
        is_prime = True
        sqrt_candidate = int(math.sqrt(candidate))
        for p in primes:
            if p > sqrt_candidate:
                break
            if candidate % p == 0:
                is_prime = False
                break
        if is_prime:
            primes.append(candidate)
        candidate += 2
    return primes

FIRST_4096_PRIMES = generate_first_n_primes(4096)
logger.info(f"Generated {len(FIRST_4096_PRIMES)} primes, largest: {FIRST_4096_PRIMES[-1]}")

def compute_gcd(a: int, b: int) -> int:
    """Compute greatest common divisor using Euclidean algorithm"""
    while b:
        a, b = b, a % b
    return a

def heavy_tail_alpha(model: nn.Module) -> Tuple[List[float], float]:
    """
    Compute layer-wise alpha values using WeightWatcher
    Returns: (layer_alphas, mean_alpha)
    """
    if not WW_AVAILABLE:
        logger.warning("WeightWatcher not available, returning default alpha values")
        return [3.0], 3.0
        
    try:
        # PATCH P0: Fix alpha logging with proper parameters
        watcher = ww.WeightWatcher(model=model)
        details = watcher.analyze()
        
        # Extract alpha values for each layer
        alphas = []
        layer_names = []
        
        for idx, row in details.iterrows():
            if 'alpha' in row and not pd.isna(row['alpha']) and row['alpha'] > 0:
                alphas.append(float(row['alpha']))
                layer_names.append(row.get('layer_id', f'layer_{idx}'))
        
        if not alphas:
            logger.warning("No valid alpha values found")
            return [3.0], 3.0
            
        mean_alpha = np.mean(alphas)
        
        # Log layer-wise alphas for debugging
        logger.debug(f"Layer-wise alphas: {dict(zip(layer_names, alphas))}")
        
        return alphas, float(mean_alpha)
        
    except Exception as e:
        logger.warning(f"WeightWatcher analysis failed: {e}")
        return [3.0], 3.0

def number_to_nybble_tensor(n, max_bits=4096, device=None):
    """
    Convert number to nybble (4-bit) tensor representation
    Returns tensor of shape (seq_len,) where each entry ∈ {0-15}
    """
    if device is None:
        device = torch.cuda.current_device() if torch.cuda.is_available() else 'cpu'
    
    seq_len = max_bits // 4
    arr = torch.zeros(seq_len, dtype=torch.long, device=device)
    
    # Convert number to nybbles (4-bit groups)
    n = int(n)
    for i in range(min(seq_len, (n.bit_length() + 3) // 4)):
        nybble = (n >> (4 * i)) & 0xF
        arr[i] = nybble
        
    return arr

def nybble_tensor_to_number(tensor):
    """Convert nybble tensor back to number"""
    # Move to CPU for conversion
    tensor = tensor.detach().cpu()
    
    # Convert to Python int
    number = 0
    seq_len = len(tensor)
    
    for i in range(seq_len):
        nybble = int(tensor[i].item())
        if nybble > 0:
            number |= (nybble << (4 * i))
            
    return number

def compute_modular_labels(n, primes):
    """Compute modular arithmetic labels for auxiliary tasks"""
    mod_zero = []  # Binary: is n ≡ 0 (mod p)?
    legendre_symbols = []  # Quadratic residue indicators
    
    for p in primes:
        # Check if n is divisible by p
        mod_zero.append(1 if n % p == 0 else 0)
        
        # Compute Legendre symbol (quadratic residue)
        if p == 2:
            legendre_symbols.append(1)  # All odd numbers are QR mod 2
        else:
            # Use Euler's criterion: a^((p-1)/2) ≡ 1 (mod p) if a is QR
            leg = pow(n % p, (p - 1) // 2, p)
            legendre_symbols.append(1 if leg == 1 else 0)
    
    return mod_zero, legendre_symbols

class FactorizationDataset(Dataset):
    """Dataset for factorization problems - compatible with DeepSpeed DataLoader"""
    def __init__(self, config, bit_size, size=10000):
        self.config = config
        self.bit_size = bit_size
        self.size = size
        self.data_gen = RSA512DataGenerator(config)
        
    def __len__(self):
        return self.size
        
    def __getitem__(self, idx):
        # Generate a single sample
        batch_data = self.data_gen.generate_batch(1, self.bit_size)
        
        n = batch_data['numbers'][0]
        f1 = batch_data['factors1'][0]
        f2 = batch_data['factors2'][0]
        
        # Convert to tensors
        x = number_to_nybble_tensor(n, device='cpu')
        y1 = number_to_nybble_tensor(f1, device='cpu')
        y2 = number_to_nybble_tensor(f2, device='cpu')
        
        # Auxiliary labels
        mod_zero = torch.tensor(batch_data['mod_zero_labels'][0], dtype=torch.float32)
        legendre = torch.tensor(batch_data['legendre_labels'][0], dtype=torch.float32)
        
        return {
            'input': x,
            'factor1': y1,
            'factor2': y2,
            'number': n,
            'mod_zero': mod_zero,
            'legendre': legendre
        }

class UniversalBlock(nn.Module):
    """
    Universal Transformer block with shared weights across iterations
    Provides O(seq_len^2) compute that scales with problem size
    """
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        
        # Multi-head self-attention
        self.self_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        
        # Feed-forward network
        self.ff = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * d_model, d_model)
        )
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # Dropout
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        """
        x: (batch, seq_len, d_model)
        mask: optional attention mask
        """
        # Self-attention with residual
        attn_out, _ = self.self_attn(x, x, x, attn_mask=mask)
        x = x + self.dropout1(attn_out)
        x = self.norm1(x)
        
        # Feed-forward with residual
        ff_out = self.ff(x)
        x = x + self.dropout2(ff_out)
        x = self.norm2(x)
        
        return x

class ExtendedModularArithmeticLayer(nn.Module):
    """
    PATCH P1: Vectorized modular arithmetic layer with 4096 primes
    Computes residues mod p using efficient torch operations
    """
    def __init__(self, hidden_dim, num_moduli=4096):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_moduli = num_moduli
        
        # Use pre-computed first 4096 primes
        primes_tensor = torch.tensor(FIRST_4096_PRIMES[:num_moduli], dtype=torch.float32)
        self.register_buffer('moduli', primes_tensor)
        
        # Learned projections for modular features
        # Split into smaller groups for memory efficiency
        self.group_size = 256
        self.num_groups = num_moduli // self.group_size
        
        self.mod_embeds = nn.ModuleList([
            nn.Linear(self.group_size, hidden_dim // self.num_groups)
            for _ in range(self.num_groups)
        ])
        
        self.combine_groups = nn.Linear(hidden_dim, hidden_dim)
        
        # Gating mechanism
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Sigmoid()
        )
        
        self.output_proj = nn.Linear(hidden_dim * 2, hidden_dim)
        
    def forward(self, x, number_representation):
        """
        x: (batch, seq_len, hidden_dim) - current hidden states
        number_representation: (batch, max_nybbles) - nybble representation
        Returns: (batch, seq_len, hidden_dim)
        """
        device = x.device
        batch_size, seq_len, hidden_dim = x.shape
        
        # PATCH P1: Vectorized residue computation
        # Pre-compute powers of 16 modulo each prime
        max_nybbles_to_process = min(number_representation.size(1), 128)
        positions = torch.arange(max_nybbles_to_process, device=device)
        
        # Compute all group features at once
        all_group_features = []
        
        for group_idx in range(self.num_groups):
            start_idx = group_idx * self.group_size
            end_idx = start_idx + self.group_size
            group_moduli = self.moduli[start_idx:end_idx]  # (group_size,)
            
            # Vectorized power computation: 16^i mod p for all positions and moduli
            # Shape: (max_nybbles, group_size)
            pow16 = torch.zeros(max_nybbles_to_process, self.group_size, device=device)
            pow16[0, :] = 1  # 16^0 = 1
            
            for i in range(1, max_nybbles_to_process):
                pow16[i] = torch.remainder(pow16[i-1] * 16, group_moduli)
            
            # Get nybble values for batch
            x_nybbles = number_representation[:, :max_nybbles_to_process].float()  # (batch, max_nybbles)
            
            # Compute residues: sum(nybble_i * 16^i) mod p
            # (batch, max_nybbles) @ (max_nybbles, group_size) -> (batch, group_size)
            residues = torch.remainder(x_nybbles @ pow16, group_moduli)
            
            # Normalize to [0, 1)
            # Normalize to [0,1) and match parameter dtype (bf16 when ZeRO converts)
            normalized_residues = (residues / group_moduli).to(x.dtype)
            normalized_residues = torch.clamp(normalized_residues, 0, 1) * 0.25
            # Apply learned embedding
            group_features = self.mod_embeds[group_idx](normalized_residues)  # (batch, hidden_dim/num_groups)
            all_group_features.append(group_features)
        
        # Combine all group features
        combined_features = torch.cat(all_group_features, dim=-1)  # (batch, hidden_dim)
        mod_features = self.combine_groups(combined_features)  # (batch, hidden_dim)
        
        # Expand to match sequence length
        mod_features = mod_features.unsqueeze(1).expand(-1, seq_len, -1)  # (batch, seq_len, hidden_dim)
        
        # Gated combination with input
        combined = torch.cat([x, mod_features], dim=-1)  # (batch, seq_len, hidden_dim * 2)
        gate = self.gate(combined)  # (batch, seq_len, hidden_dim)
        
        # Output with residual connection
        output = self.output_proj(combined)  # (batch, seq_len, hidden_dim)
        return x + gate * output

class Factor512Net(nn.Module):
    """
    Main network architecture for 512-bit factorization with patches P2, P3, P5
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.max_bits = 4096  # Support up to 4096 bits
        self.max_nybbles = self.max_bits // 4  # 1024 nybbles
        
        # Nybble embedding (4-bit groups)
        self.nybble_embed = nn.Embedding(16, config.hidden_dim)  # 16 possible nybble values
        
        # Positional encoding
        self.position_embed = nn.Parameter(torch.randn(self.max_nybbles, config.hidden_dim) * 0.02)
        
        # Initial projection
        self.input_proj = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # PATCH P2: Single modular arithmetic layer OUTSIDE the Universal loop
        self.mod_layer = ExtendedModularArithmeticLayer(config.hidden_dim, num_moduli=4096)
        
        # Universal Transformer block (shared weights across iterations)
        self.universal_block = UniversalBlock(
            d_model=config.hidden_dim,
            n_heads=config.num_heads,
            dropout=0.1
        )
        

        # ────────────────────────────────────────────────
        #  ACT halting: a 1-unit linear head on the mean
        #  hidden state.  σ( … ) → halting prob p∈(0,1).
        # ────────────────────────────────────────────────
        self.halt_proj = nn.Linear(config.hidden_dim, 1)

        # Initialize halting bias to encourage earlier stopping
        # sigmoid(2.0) ≈ 0.88, so most samples will halt in 1-2 steps initially
        nn.init.constant_(self.halt_proj.bias, 2.0)
        # Keep weight initialization small to allow learning
        nn.init.normal_(self.halt_proj.weight, mean=0.0, std=0.01)

        # Layer normalization for stability
        self.layer_norm_input = nn.LayerNorm(config.hidden_dim)
        self.layer_norm_mod = nn.LayerNorm(config.hidden_dim)
        
        # PATCH P3: Binary residue heads instead of 4096-class CE
        self.residue_head = nn.Linear(config.hidden_dim, 4096)  # Binary prediction per prime
        self.legendre_head = nn.Linear(config.hidden_dim, 4096)  # Binary Legendre symbols
        
        # Factor decoder heads
        self.factor1_decoder = self._make_decoder()
        self.factor2_decoder = self._make_decoder()
        
        # Initialize weights
        self._init_weights()
        
    def _make_decoder(self):
        """Create a decoder head for factor prediction"""
        return nn.Sequential(
            nn.Linear(self.config.hidden_dim, self.config.hidden_dim * 2),
            nn.LayerNorm(self.config.hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.config.hidden_dim * 2, self.config.hidden_dim),
            nn.GELU(),
            nn.Linear(self.config.hidden_dim, self.max_nybbles)
        )
        
    def _init_weights(self):
        """Initialize weights for stable training"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # Special handling for halt projection
                if module == self.halt_proj:
                    nn.init.normal_(module.weight, mean=0.0, std=0.01)
                    nn.init.constant_(module.bias, 2.0)
                else:
                    nn.init.xavier_uniform_(module.weight, gain=0.67)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
                
    def forward(self, x_nybbles, current_bit_size=None):
        """
        x_nybbles: (batch, max_nybbles) nybble representation of numbers
        current_bit_size: current curriculum bit size for compute scaling
        Returns: (factor1_nybbles, factor2_nybbles, aux_outputs)
        """
        batch_size = x_nybbles.size(0)
        device = x_nybbles.device
        
        # Determine sequence length based on current bit size
        if current_bit_size is None:
            current_bit_size = self.config.bit_curriculum[-1]  # Default to max
        
        seq_len = min((current_bit_size + 3) // 4, self.max_nybbles)  # Round up to nybbles
        
        # Truncate input to relevant length
        x_nybbles_truncated = x_nybbles[:, :seq_len]
        
        # Embed nybbles
        nybble_embeds = self.nybble_embed(x_nybbles_truncated)  # (batch, seq_len, hidden_dim)
        
        # Add positional encoding
        pos_embeds = self.position_embed[:seq_len].unsqueeze(0).expand(batch_size, -1, -1)
        h = nybble_embeds + pos_embeds  # (batch, seq_len, hidden_dim)
        
        # Initial projection and normalization
        h = self.input_proj(h)
        h = self.layer_norm_input(h)
        
        # PATCH P2: Apply modular arithmetic layer ONCE before Universal iterations
        h = self.mod_layer(h, x_nybbles)
        h = self.layer_norm_mod(h)
        
# In Factor512Net.forward(), replace the entire ACT loop with:

        # ────────────────────────────────────────────────
        #  ACT-style halting (Graves 2016) - CORRECTED   
        # ────────────────────────────────────────────────
        T_max = seq_len

        # Running masks and accumulators
        batch_size = h.size(0)
        still_running = torch.ones(batch_size, 1, 1, device=h.device, dtype=h.dtype)
        halting_acc = torch.zeros_like(still_running)
        n_updates = torch.zeros_like(still_running)

        # Debug tracking (optional but recommended)
        halt_probs_debug = []

        for t in range(T_max):
            # Apply Universal Transformer block
            if self.training and T_max > 16:
                h_new = checkpoint(self.universal_block, h)
            else:
                h_new = self.universal_block(h)

            # Compute halting probability for this step
            p_t_raw = self.halt_proj(h_new.mean(dim=1, keepdim=True))  # (B, 1, 1)
            p_t = torch.sigmoid(p_t_raw) * still_running  # Only for samples still running
            
            # Track for debugging
            if self.training and t == 0:
                halt_probs_debug.append(p_t.mean().item())

            # Will adding p_t make us halt?
            new_halting_acc = halting_acc + p_t
            
            # Compute the actual weight for this update
            # If this step would push us over 1.0, use only the remainder
            will_halt = new_halting_acc >= 1.0
            remainder_weight = 1.0 - halting_acc  # How much weight is left to assign
            
            # The weight for this step is either p_t or the remainder (if halting)
            update_weight = torch.where(
                will_halt,
                remainder_weight,  # Use exactly the remainder to reach 1.0
                p_t               # Otherwise use the full p_t
            )
            
            # Update hidden state with proper convex combination
            h = h * (1.0 - update_weight) + h_new * update_weight
            
            # Update accumulators
            halting_acc = halting_acc + update_weight
            n_updates = n_updates + still_running
            
            # Mark samples that have halted
            still_running = still_running * (~will_halt).float()
            
            # Early exit if all samples have halted
            if still_running.sum() == 0:
                break

        # Store debug info for monitoring
        if self.training and halt_probs_debug:
            self._last_halt_prob = np.mean(halt_probs_debug)
            self._last_n_iters = n_updates.mean().item()
        
        # Global pooling for decoder context
        context = h.mean(dim=1)  # (batch, hidden_dim)
        
        # Auxiliary task predictions (binary outputs)
        residue_logits = self.residue_head(context)  # (batch, 4096)
        legendre_logits = self.legendre_head(context)  # (batch, 4096)
        
        # Factor predictions
        factor1_logits = self.factor1_decoder(context)  # (batch, max_nybbles)
        factor2_logits = self.factor2_decoder(context)  # (batch, max_nybbles)
        
        # Apply sigmoid for nybble probabilities (soft selection)
        factor1_probs = torch.sigmoid(factor1_logits) * 15.0  # Scale to [0, 15]
        factor2_probs = torch.sigmoid(factor2_logits) * 15.0
        
        aux_outputs = {
            'residue_logits': residue_logits,
            'legendre_logits': legendre_logits
        }
        
        return factor1_probs, factor2_probs, aux_outputs

@dataclass
class Factor512Config:
    """Configuration for 512-bit factorization experiments with 8-bit kickstart"""
    # Model architecture
    hidden_dim: int = 2048
    num_layers: int = 8  # For initialization only; Universal block is shared
    num_heads: int = 64
    
    # Training hyperparameters
    batch_size: int = 2
    gradient_accumulation_steps: int = 1   # ← one optimiser step per batch
    lr: float = 6e-4
    weight_decay: float = 0.01
    max_steps: int = 200000
    
    # Learning rate schedule
    warmup_steps: int = 2000
    plateau_steps: int = 12000
    
    # 8-bit kickstart settings (new)
    kickstart_bits: int = 8  # Pre-train on 8-bit numbers
    kickstart_steps: int = 500  # Steps for kick-start phase (increased from 200)
    
    # Curriculum settings
    bit_curriculum: List[int] = field(default_factory=lambda: [
        12, 16, 24, 32, 40, 48, 56, 64, 80, 96, 112, 128,  # Up to 128 bits
        160, 192, 224, 256, 288, 320, 384, 448, 512     # Extended to 512 bits
    ])
    advancement_window: int = 3  # Require success for 3 consecutive evals
    success_threshold: float = 0.9
    product_error_threshold: float = 0.02  # Tighter than before
    
    # Evaluation settings
    eval_interval_small: int = 10   # For ≤192 bits
    eval_interval_large: int = 100  # For >192 bits
    
    # Loss weights (updated product_error_weight_initial)
    factor_loss_weight: float = 1.0
    residue_loss_weight: float = 0.15
    legendre_loss_weight: float = 0.05
    product_error_weight_initial: float = 0.3  # Increased from 0.2
    product_error_weight_high: float = 0.5
    
    # System settings
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    alpha_interval: int = 50
    checkpoint_interval: int = 2000
    experiment_type: str = "512bit_kickstart_deepspeed"
    eval_only: bool = False
    eval_bits: int = 512
    resume_from: Optional[str] = None  # Path to checkpoint to resume from
    
    # DeepSpeed specific settings
    use_deepspeed: bool = True
    local_rank: int = -1
    deepspeed_config: Optional[Dict] = None
    zero_stage: int = 3  # ZeRO-3 for maximum memory efficiency
    offload_optimizer: bool = True
    offload_param: bool = True
    pin_memory: bool = True
    
    def get_deepspeed_config(self):
        """Get DeepSpeed configuration dictionary with custom scheduler"""
        if self.deepspeed_config is not None:
            return self.deepspeed_config
            
        config = {
            "train_batch_size": self.batch_size * self.gradient_accumulation_steps,
            "train_micro_batch_size_per_gpu": self.batch_size,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "steps_per_print": 100,
            "wall_clock_breakdown": False,
            
            "optimizer": {
                "type": "Adam",  # Use DeepSpeed's CPU Adam for ZeRO-3 offload
                "params": {
                    "lr": self.lr,
                    "betas": [0.9, 0.98],
                    "eps": 1e-9,
                    "weight_decay": self.weight_decay
                }
            },
            
            # Custom scheduler with warmup, plateau, and cosine decay
            "scheduler": {
                "type": "WarmupDecayLR",
                "params": {
                    "warmup_min_lr": 0.0,
                    "warmup_max_lr": self.lr,
                    "warmup_num_steps": self.warmup_steps,
                    "total_num_steps": self.max_steps,
                    "warmup_type": "linear"
                }
            },
            
            "fp16": {
                "enabled": False
            },
            
            "bf16": {
                "enabled": False
            },
            
            "gradient_clipping": 1.0,
            
            "zero_optimization": {
                "stage": self.zero_stage,
                "offload_optimizer": {
                    "device": "cpu" if self.offload_optimizer else "none",
                    "pin_memory": self.pin_memory,
                    "buffer_count": 4,
                    "fast_init": False
                },
                "offload_param": {
                    "device": "cpu" if self.offload_param else "none",
                    "pin_memory": self.pin_memory,
                    "buffer_count": 5,
                    "buffer_size": 1e8,
                    "max_in_cpu": 1e9
                },
                "overlap_comm": True,
                "contiguous_gradients": True,
                "sub_group_size": 1e8,
                "reduce_bucket_size": 1e8,
                "stage3_prefetch_bucket_size": 1e7,
                "stage3_param_persistence_threshold": 1e5,
                "stage3_max_live_parameters": 1e8,
                "stage3_max_reuse_distance": 1e8,
                "stage3_gather_16bit_weights_on_model_save": True
            },
            
            "activation_checkpointing": {
                "partition_activations": True,
                "cpu_checkpointing": True,
                "contiguous_memory_optimization": False,
                "number_checkpoints": None,
                "synchronize_checkpoint_boundary": False,
                "profile": False
            },
            
            "flops_profiler": {
                "enabled": False,
                "profile_step": 1,
                "module_depth": -1,
                "top_modules": 1,
                "detailed": True,
                "output_file": None
            },
            
            "tensorboard": {
                "enabled": False
            },
            
            "comms_logger": {
                "enabled": False,
                "verbose": False,
                "prof_all": False,
                "debug": False
            }
        }
        
        return config

class ExtendedFactorizationLoss(nn.Module):
    """
    Multi-objective loss for 512-bit factorization with patches P3 and P4
    """
    def __init__(self, config, num_primes=4096):
        super().__init__()
        self.config = config
        self.num_primes = num_primes
        self.primes = FIRST_4096_PRIMES[:num_primes]
        
        # Main factorization loss
        self.mse = nn.MSELoss(reduction='none')
        
        # PATCH P3: Binary losses for auxiliary tasks
        self.bce = nn.BCEWithLogitsLoss()
        
        # Dynamic weight for product error
        self.current_product_weight = config.product_error_weight_initial
        
    def forward(self, pred_factor1, pred_factor2, true_factor1, true_factor2, 
                original_numbers, aux_outputs, aux_labels):
        """
        Compute multi-component loss with auxiliary tasks
        """
        device = pred_factor1.device
        batch_size = pred_factor1.size(0)
        
        # 1. Factor prediction loss (MSE on nybbles)
        true_factor1 = true_factor1.to(pred_factor1.dtype)
        true_factor2 = true_factor2.to(pred_factor2.dtype)

        # ── PATCH S1: mask inactive nybbles ──────────────────────────
        # active nybble count per sample: (½ bit-length + pad) // 4
        bit_lens   = torch.tensor([int(n).bit_length() for n in original_numbers],
                                  device=device)
        active_len = ((bit_lens // 2) + 3) // 4  # at least 1

        idx  = torch.arange(pred_factor1.size(1), device=device)
        mask = idx.unsqueeze(0) < active_len.unsqueeze(1)
        mask = mask.to(pred_factor1.dtype)

        factor_loss1 = (self.mse(pred_factor1, true_factor1) * mask).sum() / mask.sum()
        factor_loss2 = (self.mse(pred_factor2, true_factor2) * mask).sum() / mask.sum()
        factor_loss = (factor_loss1 + factor_loss2) / 2
        
        # 2. PATCH P4: Vectorized product consistency loss
        # Round predictions to nearest integers
        pred_nyb1_rounded = torch.round(pred_factor1).clamp(0, 15).long()
        pred_nyb2_rounded = torch.round(pred_factor2).clamp(0, 15).long()

        # ── PATCH L0: Zero-out nybbles beyond the true factor length ──
        # avoids huge dummy digits that blow up the numeric range
        for i in range(batch_size):
            seq_len = max(1, (int(original_numbers[i]).bit_length() // 2 + 3) // 4)
            pred_nyb1_rounded[i, seq_len:] = 0
            pred_nyb2_rounded[i, seq_len:] = 0
        
        # Convert to numbers in batched fashion
        pred_nums1 = torch.zeros(batch_size, dtype=torch.float64, device=device)
        pred_nums2 = torch.zeros(batch_size, dtype=torch.float64, device=device)
        true_nums = torch.tensor(original_numbers, dtype=torch.float64, device=device)
        
        # --- PATCH P6: robust int→float cast ---------------------------------
        FLOAT_MAX = 1.7976931348623157e308  # max IEEE-754 double
        def _safe_float(x: int) -> float:
            try:
                return float(x)
            except OverflowError:
                return FLOAT_MAX          # clamp instead of raising

        for i in range(batch_size):
            pred_nums1[i] = _safe_float(nybble_tensor_to_number(pred_nyb1_rounded[i]))
            pred_nums2[i] = _safe_float(nybble_tensor_to_number(pred_nyb2_rounded[i]))
        
        # ── PATCH L1: log-space product error (no overflow) ──
        log_pred1 = torch.log(pred_nums1.clamp_min(1))
        log_pred2 = torch.log(pred_nums2.clamp_min(1))
        log_true  = torch.log(true_nums.clamp_min(1))

        product_errors = torch.abs((log_pred1 + log_pred2) - log_true) / log_true
        product_error  = product_errors.mean()
        
        # 3. PATCH P3: Binary auxiliary task losses
        # mod_zero_labels is now a binary matrix (batch, 4096)
        residue_loss = self.bce(aux_outputs['residue_logits'], aux_labels['mod_zero_labels'].float())
        legendre_loss = self.bce(aux_outputs['legendre_logits'], aux_labels['legendre_labels'].float())
        
        # Update product weight based on factor loss
        if factor_loss.item() < 1.0:  # When factor prediction improves
            self.current_product_weight = self.config.product_error_weight_high
        else:
            self.current_product_weight = self.config.product_error_weight_initial
        
        # Combine losses
        total_loss = (
            self.config.factor_loss_weight * factor_loss +
            self.current_product_weight * product_error +
            self.config.residue_loss_weight * residue_loss +
            self.config.legendre_loss_weight * legendre_loss
        )
        
        # Compute accuracy metrics
        correct_factors = 0
        partial_correct = 0
        gcd_successes = []
        
        for i in range(batch_size):
            pred_n1 = int(pred_nums1[i].item())
            pred_n2 = int(pred_nums2[i].item())
            true_n = int(original_numbers[i])
            
            # Check exact factorization
            if pred_n1 > 1 and pred_n2 > 1:
                if pred_n1 * pred_n2 == true_n:
                    # Verify they are the actual prime factors
                    true_f1 = nybble_tensor_to_number(true_factor1[i])
                    true_f2 = nybble_tensor_to_number(true_factor2[i])
                    
                    if set([pred_n1, pred_n2]) == set([true_f1, true_f2]):
                        correct_factors += 1
            
            # Check partial progress via GCD
            for pred_f in [pred_n1, pred_n2]:
                if pred_f > 1:
                    gcd = compute_gcd(true_n, pred_f)
                    if gcd > 1 and gcd < true_n:
                        partial_correct += 1
                        gcd_successes.append((true_n, pred_f, gcd))
                        break
        
        success_rate = correct_factors / batch_size
        partial_rate = partial_correct / batch_size
        
        loss_dict = {
            'factor_loss': factor_loss.item(),
            'product_error': product_error.item(),
            'residue_loss': residue_loss.item(),
            'legendre_loss': legendre_loss.item(),
            'total_loss': total_loss.item(),
            'success_rate': success_rate,
            'partial_rate': partial_rate,
            'product_weight': self.current_product_weight,
            'gcd_successes': len(gcd_successes)
        }
        
        return total_loss, loss_dict, gcd_successes

class RSA512DataGenerator:
    """Generate RSA-style semiprime factorization problems up to 512 bits"""
    def __init__(self, config: Factor512Config):
        self.config = config
        self.prime_cache = defaultdict(list)
        self.prime_cache_size = 100  # Keep more primes cached
        
    @lru_cache(maxsize=100000)
    def is_prime_cached(self, n: int) -> bool:
        """Cached primality test"""
        return bool(gmpy2.is_prime(n))
        
    def generate_prime(self, bits):
        """Generate a prime number with specified bit length"""
        # Validate input
        if bits < 2:
            raise ValueError(f"Cannot generate prime with {bits} bits, minimum is 2")
        
        # Special handling for very small bit sizes
        if bits == 2:
            return random.choice([2, 3])
        elif bits == 3:
            return random.choice([5, 7])
        
        # Use cache if available
        if len(self.prime_cache[bits]) >= 10:
            return random.choice(self.prime_cache[bits])
        
        min_val = 2**(bits-1)
        max_val = 2**bits - 1
        
        # For small bit sizes, ensure we have room to find primes
        if bits <= 8:
            # For small ranges, just iterate to find primes
            primes_in_range = []
            for n in range(min_val, min(max_val + 1, min_val + 1000)):
                if gmpy2.is_prime(n):
                    primes_in_range.append(n)
            
            if primes_in_range:
                prime = random.choice(primes_in_range)
                if len(self.prime_cache[bits]) < self.prime_cache_size:
                    self.prime_cache[bits].extend(primes_in_range[:10])
                return prime
        
        # Generate new prime for larger bit sizes
        max_attempts = 1000
        for attempt in range(max_attempts):
            # Use a better distribution for candidate selection
            if attempt < max_attempts // 2:
                # First half: random in range
                candidate = random.randint(min_val, max_val)
            else:
                # Second half: try near the middle of the range
                mid = (min_val + max_val) // 2
                spread = (max_val - min_val) // 4
                candidate = mid + random.randint(-spread, spread)
            
            # Ensure candidate is odd (except for 2)
            if candidate > 2 and candidate % 2 == 0:
                candidate += 1
                
            prime = int(gmpy2.next_prime(candidate))
            
            if prime <= max_val:
                # Cache the prime
                if len(self.prime_cache[bits]) < self.prime_cache_size:
                    self.prime_cache[bits].append(prime)
                return prime
            
            # If we overshot, try finding a prime going backwards
            if attempt > max_attempts // 3:
                candidate = max_val
                while candidate >= min_val:
                    if gmpy2.is_prime(candidate):
                        if len(self.prime_cache[bits]) < self.prime_cache_size:
                            self.prime_cache[bits].append(candidate)
                        return candidate
                    candidate -= 2 if candidate > 2 else 1
        
        # Ultimate fallback: use a known prime in range
        logger.warning(f"Failed to generate {bits}-bit prime efficiently, using fallback")
        # Start from a random point and search
        start = random.randint(min_val, max_val)
        for offset in range(0, max_val - min_val, 2):
            for direction in [1, -1]:
                candidate = start + direction * offset
                if min_val <= candidate <= max_val and gmpy2.is_prime(candidate):
                    return candidate
        
        # If all else fails, just find the first prime >= min_val
        prime = int(gmpy2.next_prime(min_val))
        if prime > max_val:
            # This should be very rare - means no prime exists in the range
            raise ValueError(f"No prime found in {bits}-bit range [{min_val}, {max_val}]")
        return prime
   
    def generate_semiprime(self, total_bits, retry_count=0, max_retries=10):
        """Generate a semiprime with approximately total_bits bits"""
        # Prevent infinite recursion
        if retry_count >= max_retries:
            logger.warning(f"Failed to generate exact {total_bits}-bit semiprime after {max_retries} attempts")
            # Fall back to a more relaxed approach
            bits1 = total_bits // 2
            bits2 = total_bits - bits1
            bits1 = max(4, min(bits1, total_bits - 4))
            bits2 = max(4, min(bits2, total_bits - 4))
            
            p = self.generate_prime(bits1)
            q = self.generate_prime(bits2)
            if p > q:
                p, q = q, p
            return p * q, p, q
        
        # For very large bit sizes, ensure balanced factors
        if total_bits >= 256:
            # Keep factors within 2 bits of each other for balance
            bits1 = total_bits // 2 + random.randint(-1, 1)
            bits2 = total_bits - bits1
        else:
            # More variation for smaller numbers
            bits1 = total_bits // 2 + random.randint(-2, 2)
            bits2 = total_bits - bits1
        
        # Ensure valid bit ranges
        bits1 = max(4, min(bits1, total_bits - 4))  # Changed from 8 to 4 for more flexibility
        bits2 = max(4, min(bits2, total_bits - 4))
        

        
        # Additional validation to prevent edge cases
        if bits1 <= 0 or bits2 <= 0 or bits1 + bits2 > total_bits + 4:
            # Force reasonable values
            bits1 = max(4, total_bits // 2)
            bits2 = max(4, total_bits - bits1)
        
        try:
            p = self.generate_prime(bits1)
            q = self.generate_prime(bits2)
        except Exception as e:
            logger.warning(f"Prime generation failed for {bits1}, {bits2} bits: {e}")
            # Fallback to simpler bit distribution
            bits1 = bits2 = total_bits // 2
            p = self.generate_prime(bits1)
            q = self.generate_prime(bits2)
        
        # Ensure p <= q for consistency
        if p > q:
            p, q = q, p
            
        n = p * q
        
        # Verify bit length is approximately correct
        actual_bits = n.bit_length()
        
        # Adjust tolerance based on bit size
        tolerance = 3 if total_bits < 64 else 5 if total_bits < 256 else 8
        
        if abs(actual_bits - total_bits) > tolerance:
            # Log the mismatch for debugging
            if retry_count == 0:
                logger.debug(f"Bit length mismatch: target={total_bits}, actual={actual_bits}, "
                            f"p_bits={p.bit_length()}, q_bits={q.bit_length()}")
            
            # Retry with incremented counter
            return self.generate_semiprime(total_bits, retry_count + 1, max_retries)
            
        return n, p, q
   
    def generate_batch(self, batch_size, bit_size):
        """Generate a batch of factorization problems with auxiliary labels"""
        # ────────────────────────────────────────────────────────────────
        #  PATCH S-8bit  |  Exhaus­tive 8-bit semiprime sampler
        # ────────────────────────────────────────────────────────────────
        if bit_size == 8:
            # build the *complete* 8-bit semiprime table once, cache it
            if not hasattr(self, "_all_8bit_semiprimes"):
                full_list = []
                for p in range(3, 256, 2):
                    if not gmpy2.is_prime(p):
                        continue
                    for q in range(p, 256 // p + 1, 2):
                        if gmpy2.is_prime(q):
                            full_list.append((p * q, p, q))
                random.shuffle(full_list)
                self._all_8bit_semiprimes = full_list
                self._cursor_8bit = 0

            out = {'numbers': [], 'factors1': [], 'factors2': [],
                   'mod_zero_labels': [], 'legendre_labels': []}

            for _ in range(batch_size):
                if self._cursor_8bit >= len(self._all_8bit_semiprimes):
                    random.shuffle(self._all_8bit_semiprimes)
                    self._cursor_8bit = 0
                n, p, q = self._all_8bit_semiprimes[self._cursor_8bit]
                self._cursor_8bit += 1

                out['numbers'].append(n)
                out['factors1'].append(p)
                out['factors2'].append(q)

                mod_zero, legendre = compute_modular_labels(n, FIRST_4096_PRIMES)
                out['mod_zero_labels'].append(mod_zero)
                out['legendre_labels'].append(legendre)
            return out

        # ────────────────────────────────────────────────────────────────
        #  …all other bit-sizes keep the original stochastic generator
        # ────────────────────────────────────────────────────────────────
        return super().generate_batch(batch_size, bit_size)

class Factor512TrainerDeepSpeed:
    """Training loop for 512-bit factorization with DeepSpeed ZeRO-3 optimization"""
    def __init__(self, model: Factor512Net, config: Factor512Config):
        """Initialize DeepSpeed trainer with proper optimizer and scheduler configuration"""
        self.config = config
        self.device = config.device
        
        # --- PATCH D0 ─ Windows / single-GPU friendly distributed init ----
        import tempfile, pathlib
        if not dist.is_initialized():
            tmp_path = pathlib.Path(tempfile.gettempdir()) / "ds_rendezvous"
            store = dist.FileStore(str(tmp_path), 1)
            dist.init_process_group(
                backend="gloo",
                store=store,
                rank=0,
                world_size=1,
            )

        self.local_rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        # --- PATCH D2 ─ satisfy DeepSpeed's args check -----------------
        import os
        os.environ.setdefault("LOCAL_RANK", str(self.local_rank))
        
        # Data generator
        self.data_gen = RSA512DataGenerator(config)
        
        # Loss function
        self.criterion = ExtendedFactorizationLoss(config)
        
        # Initialize DeepSpeed with config-based optimizer and scheduler
        ds_config = config.get_deepspeed_config()
        
        # Create a dummy dataset for DeepSpeed initialization
        dummy_dataset = FactorizationDataset(config, config.kickstart_bits if config.kickstart_steps > 0 else config.bit_curriculum[0])
        
        # Initialize DeepSpeed engine - let it create optimizer and scheduler
        self.model_engine, self.optimizer, _, self.lr_scheduler = deepspeed.initialize(
            model=model,
            config=ds_config,
            model_parameters=model.parameters()
        )
        
        # Store the current step for our custom scheduler logic
        self.global_step = 0
        
        # Tracking variables with kickstart support
        self.step = 0
        self.current_bit_size = config.kickstart_bits if config.kickstart_steps > 0 else config.bit_curriculum[0]
        self.bit_level = -1 if config.kickstart_steps > 0 else 0
        self.in_kickstart = config.kickstart_steps > 0
        self.train_losses = []
        self.val_losses = []
        self.alphas = []
        self.layer_alphas = []
        self.success_rates = []
        self.partial_success_rates = []
        self.product_errors = []
        self.loss_components = defaultdict(list)
        self.advancement_history = []
        self.grokking_detected = False
        self.collapse_detected = False
        self.best_success_rate = 0.0
        
        # Timing
        self.step_times = []
        
        # Load checkpoint if specified
        if config.resume_from:
            self.load_checkpoint(config.resume_from)

    def plot_act_efficiency(self, ax=None):
        """Plot ACT efficiency metrics over training"""
        if not hasattr(self, 'act_efficiency_history') or not self.act_efficiency_history:
            return
            
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))
        
        history = self.act_efficiency_history
        steps = [h['step'] for h in history]
        avg_iters = [h['avg_iters'] for h in history]
        max_iters = [h['max_iters'] for h in history]
        efficiency = [h['efficiency'] for h in history]
        
        # Plot average iterations vs max
        ax2 = ax.twinx()
        
        # Left axis: iterations
        ax.plot(steps, avg_iters, 'b-', label='Avg Iterations', linewidth=2)
        ax.plot(steps, max_iters, 'b--', label='Max Iterations', alpha=0.5)
        ax.fill_between(steps, avg_iters, max_iters, alpha=0.2, color='blue')
        
        # Right axis: efficiency
        ax2.plot(steps, efficiency, 'g-', label='ACT Efficiency', linewidth=2)
        
        # Add bit size transitions
        current_bits = history[0]['bit_size']
        for i, h in enumerate(history):
            if h['bit_size'] != current_bits:
                ax.axvline(x=h['step'], color='red', linestyle=':', alpha=0.5)
                ax.text(h['step'], ax.get_ylim()[1] * 0.95, f"{h['bit_size']}b", 
                    rotation=90, va='top', fontsize=8)
                current_bits = h['bit_size']
        
        ax.set_xlabel('Training Step')
        ax.set_ylabel('Iterations', color='blue')
        ax2.set_ylabel('ACT Efficiency (1 - avg/max)', color='green')
        ax.tick_params(axis='y', labelcolor='blue')
        ax2.tick_params(axis='y', labelcolor='green')
        
        ax.set_title('ACT (Adaptive Computation Time) Performance')
        ax.grid(True, alpha=0.3)
        
        # Add legend
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc='best')
        
        return ax

    def _create_scheduler_lambda(self):
        """Create scheduler lambda function for LambdaLR"""
        def lr_lambda(step):
            # Adjust for kickstart phase
            if self.config.kickstart_steps > 0:
                if step < self.config.kickstart_steps:
                    # During kickstart, use normal warmup
                    effective_step = step
                else:
                    # After kickstart, restart the schedule
                    effective_step = step - self.config.kickstart_steps
            else:
                effective_step = step
            
            # Warmup phase
            if effective_step < self.config.warmup_steps:
                # Smooth quadratic ramp
                progress = effective_step / self.config.warmup_steps
                return progress * progress
            # Plateau phase
            elif effective_step < self.config.warmup_steps + self.config.plateau_steps:
                return 1.0
            # Cosine decay phase
            else:
                decay_steps = effective_step - self.config.warmup_steps - self.config.plateau_steps
                total_decay_steps = max(1, self.config.max_steps - self.config.warmup_steps - 
                                    self.config.plateau_steps)
                if self.config.kickstart_steps > 0:
                    total_decay_steps -= self.config.kickstart_steps
                return 0.5 * (1.0 + math.cos(math.pi * min(decay_steps / total_decay_steps, 1.0)))
        
        return lr_lambda
   
    def load_checkpoint(self, checkpoint_path: str):
        """Load DeepSpeed checkpoint and restore training state"""
        if not Path(checkpoint_path).exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
            
        logger.info(f"Loading checkpoint from: {checkpoint_path}")
        
        # DeepSpeed handles model, optimizer, and scheduler loading
        _, client_state = self.model_engine.load_checkpoint(checkpoint_path)
        
        if client_state:
            # Load training state
            self.step = client_state.get('step', 0)
            self.current_bit_size = client_state.get('bit_size', self.config.bit_curriculum[0])
            self.bit_level = client_state.get('bit_level', 0)
            self.in_kickstart = client_state.get('in_kickstart', False)
            
            # Load tracking variables
            self.alphas = client_state.get('alphas', [])
            self.success_rates = client_state.get('success_rates', [])
            self.partial_success_rates = client_state.get('partial_success_rates', [])
            self.product_errors = client_state.get('product_errors', [])
            
            # Load additional state if available
            if 'train_losses' in client_state:
                self.train_losses = client_state['train_losses']
            if 'val_losses' in client_state:
                self.val_losses = client_state['val_losses']
            if 'layer_alphas' in client_state:
                self.layer_alphas = client_state['layer_alphas']
            if 'loss_components' in client_state:
                self.loss_components = defaultdict(list, client_state['loss_components'])
            if 'advancement_history' in client_state:
                self.advancement_history = client_state['advancement_history']
            if 'grokking_detected' in client_state:
                self.grokking_detected = client_state['grokking_detected']
            if 'collapse_detected' in client_state:
                self.collapse_detected = client_state['collapse_detected']
            if 'best_success_rate' in client_state:
                self.best_success_rate = client_state['best_success_rate']
            
        logger.info(f"✅ Checkpoint loaded successfully!")
        logger.info(f"  Resuming from step: {self.step}")
        logger.info(f"  Current bit size: {self.current_bit_size}")
        logger.info(f"  Bit level: {self.bit_level}")
        logger.info(f"  In kickstart: {self.in_kickstart}")
        logger.info(f"  Best success rate: {self.best_success_rate:.1%}")
        
    def get_eval_interval(self):
        """Get evaluation interval based on current bit size"""
        if self.current_bit_size <= 192:
            return self.config.eval_interval_small
        else:
            return self.config.eval_interval_large
            
    def train_step(self):
        """Single training step with DeepSpeed and custom LR adjustment"""
        self.model_engine.train()
        start_time = time.time()
        
        # Generate batch with auxiliary labels
        batch_data = self.data_gen.generate_batch(
            self.config.batch_size, self.current_bit_size
        )
        
        # Convert to tensors
        numbers = batch_data['numbers']
        x = torch.stack([number_to_nybble_tensor(n, device=self.device) for n in numbers])
        y1 = torch.stack([number_to_nybble_tensor(f, device=self.device) for f in batch_data['factors1']])
        y2 = torch.stack([number_to_nybble_tensor(f, device=self.device) for f in batch_data['factors2']])
        
        # PATCH P3: Prepare binary auxiliary labels
        mod_zero_labels = torch.tensor(
            batch_data['mod_zero_labels'], 
            device=self.device,
            dtype=torch.float32
        )
        legendre_labels = torch.tensor(
            batch_data['legendre_labels'], 
            device=self.device,
            dtype=torch.float32
        )
        
        aux_labels = {
            'mod_zero_labels': mod_zero_labels,
            'legendre_labels': legendre_labels
        }
        
        # Forward pass
        pred1, pred2, aux_outputs = self.model_engine(x, self.current_bit_size)
        
        # ===== ACT Monitoring Block (FIXED) =====
        # Calculate expected T_max based on current bit size
        # This matches the logic in Factor512Net.forward()
        expected_seq_len = min((self.current_bit_size + 3) // 4, self.model_engine.module.max_nybbles)
        expected_T_max = expected_seq_len  # T_max equals sequence length in the model
        
        # Monitor ACT behavior every 100 steps
        if self.step % 100 == 0 and hasattr(self.model_engine.module, '_last_n_iters'):
            avg_iters = self.model_engine.module._last_n_iters
            avg_halt_prob = getattr(self.model_engine.module, '_last_halt_prob', 0.0)
            
            # Log ACT statistics
            logger.debug(f"ACT stats @ step {self.step}: avg_iters={avg_iters:.2f}/{expected_T_max}, "
                        f"first_halt_prob={avg_halt_prob:.3f}, bit_size={self.current_bit_size}")
            
            # Warn if ACT is not halting efficiently
            if avg_iters > expected_T_max * 0.9:  # If using >90% of max iterations
                logger.warning(f"⚠️ ACT not halting efficiently: {avg_iters:.2f}/{expected_T_max} iterations "
                            f"(bit_size={self.current_bit_size})")
                
            # Track ACT efficiency over time
            if not hasattr(self, 'act_efficiency_history'):
                self.act_efficiency_history = []
            
            self.act_efficiency_history.append({
                'step': self.step,
                'avg_iters': avg_iters,
                'max_iters': expected_T_max,
                'efficiency': 1.0 - (avg_iters / expected_T_max),  # Higher is better
                'halt_prob': avg_halt_prob,
                'bit_size': self.current_bit_size
            })
            
            # Keep only recent history
            if len(self.act_efficiency_history) > 100:
                self.act_efficiency_history = self.act_efficiency_history[-100:]
        # ===== End ACT Monitoring Block =====
        
        # Compute loss
        loss, loss_components, gcd_successes = self.criterion(
            pred1, pred2, y1, y2, numbers, aux_outputs, aux_labels
        )
        
        # Backward pass with DeepSpeed
        self.model_engine.backward(loss)
        
        # Step only when gradient accumulation is complete
        if self.model_engine.is_gradient_accumulation_boundary():
            self.model_engine.step()
            
            # Custom learning rate adjustment on top of DeepSpeed's scheduler
            self._adjust_learning_rate()
            
            self.global_step += 1
        
        # Track timing
        step_time = time.time() - start_time
        self.step_times.append(step_time)
        
        return loss.item(), loss_components, step_time
   
    def _adjust_learning_rate(self):
        """
        Apply custom learning rate schedule on top of DeepSpeed's base scheduler.
        This implements our warmup, plateau, and cosine decay phases.
        """
        # Calculate the effective step considering kickstart
        if self.config.kickstart_steps > 0:
            if self.global_step < self.config.kickstart_steps:
                effective_step = self.global_step
            else:
                effective_step = self.global_step - self.config.kickstart_steps
        else:
            effective_step = self.global_step
        
        # Calculate the learning rate multiplier based on our custom schedule
        if effective_step < self.config.warmup_steps:
            # Quadratic warmup
            progress = effective_step / self.config.warmup_steps
            lr_mult = progress * progress
        elif effective_step < self.config.warmup_steps + self.config.plateau_steps:
            # Plateau at full LR
            lr_mult = 1.0
        else:
            # Cosine decay
            decay_steps = effective_step - self.config.warmup_steps - self.config.plateau_steps
            total_decay_steps = max(1, self.config.max_steps - self.config.warmup_steps - 
                                self.config.plateau_steps)
            if self.config.kickstart_steps > 0:
                total_decay_steps -= self.config.kickstart_steps
            lr_mult = 0.5 * (1.0 + math.cos(math.pi * min(decay_steps / total_decay_steps, 1.0)))
        
        # Apply the multiplier to all parameter groups
        base_lr = self.config.lr
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = base_lr * lr_mult

      
    def evaluate_with_gcd(self, num_samples=50):
        """
        Enhanced evaluation with GCD tracking and auxiliary metrics.
        
        Args:
            num_samples: Number of validation samples to evaluate
            
        Returns:
            Dictionary containing evaluation metrics including loss, success rates,
            product errors, and inference timing information
        """
        self.model_engine.eval()
        
        total_loss = 0
        all_components = defaultdict(float)
        all_success_rates = []
        all_partial_rates = []
        all_product_errors = []
        gcd_successes_all = []
        
        # Timing
        inference_times = []
        
        with torch.no_grad():
            num_batches = max(1, num_samples // self.config.batch_size)
            
            for _ in range(num_batches):
                start_time = time.time()
                
                # Generate validation batch
                batch_data = self.data_gen.generate_batch(
                    self.config.batch_size, self.current_bit_size
                )
                
                numbers = batch_data['numbers']
                x = torch.stack([number_to_nybble_tensor(n, device=self.device) for n in numbers])
                y1 = torch.stack([number_to_nybble_tensor(f, device=self.device) for f in batch_data['factors1']])
                y2 = torch.stack([number_to_nybble_tensor(f, device=self.device) for f in batch_data['factors2']])
                
                # Prepare binary auxiliary labels
                mod_zero_labels = torch.tensor(
                    batch_data['mod_zero_labels'], 
                    device=self.device,
                    dtype=torch.float32
                )
                legendre_labels = torch.tensor(
                    batch_data['legendre_labels'], 
                    device=self.device,
                    dtype=torch.float32
                )
                
                aux_labels = {
                    'mod_zero_labels': mod_zero_labels,
                    'legendre_labels': legendre_labels
                }
                
                # Forward pass
                pred1, pred2, aux_outputs = self.model_engine(x, self.current_bit_size)
                
                # Compute loss and metrics
                loss, loss_components, gcd_successes = self.criterion(
                    pred1, pred2, y1, y2, numbers, aux_outputs, aux_labels
                )
                
                inference_time = time.time() - start_time
                inference_times.append(inference_time)
                
                total_loss += loss.item()
                for k, v in loss_components.items():
                    all_components[k] += v
                
                all_success_rates.append(loss_components['success_rate'])
                all_partial_rates.append(loss_components['partial_rate'])
                all_product_errors.append(loss_components['product_error'])
                gcd_successes_all.extend(gcd_successes)
        
        # Compute averages
        avg_loss = total_loss / num_batches
        avg_success = np.mean(all_success_rates)
        avg_partial = np.mean(all_partial_rates)
        avg_product_error = np.mean(all_product_errors)
        avg_inference_time = np.mean(inference_times)
        
        for k in all_components:
            all_components[k] /= num_batches
            
        # Log GCD successes for smaller bit sizes
        if gcd_successes_all and self.current_bit_size <= 64:
            logger.info(f"  GCD successes: {len(gcd_successes_all)} partial factorizations")
            for n, pred, gcd in gcd_successes_all[:3]:
                logger.info(f"    n={n}, pred={pred}, GCD={gcd}")
                
        return {
            'loss': avg_loss,
            'success_rate': avg_success,
            'partial_rate': avg_partial,
            'product_error': avg_product_error,
            'components': dict(all_components),
            'inference_time': avg_inference_time
        }
      
    def check_curriculum_advancement(self):
        """Check if ready to advance to next bit size, handling kickstart transition"""
        # Handle kick-start to main curriculum transition
        if self.in_kickstart:
            # Stay on 8-bit until **full grokking**
            if self.advancement_history and self.alphas:
                grokking_ready = (
                    self.advancement_history[-1]['success_rate'] >= 1.0
                )
                if grokking_ready:
                    self.in_kickstart = False
                    self.bit_level = 0
                    self.current_bit_size = self.config.bit_curriculum[0]
                    logger.info(
                        f"🚀 8-bit grokking reached (α={self.alphas[-1]:.3f}); "
                        f"advancing to {self.current_bit_size}-bit numbers"
                    )
                    self.advancement_history.clear()
                    return False  # Prevent immediate advancement
            return False
        
        # Regular curriculum advancement
        if len(self.advancement_history) < self.config.advancement_window:
            return False
            
        # Check last N evaluations
        recent_success = [h['success_rate'] for h in self.advancement_history[-self.config.advancement_window:]]
        recent_product_error = [h['product_error'] for h in self.advancement_history[-self.config.advancement_window:]]
        
        # Advancement criteria
        success_criterion = all(s >= self.config.success_threshold for s in recent_success)
        product_criterion = all(e < self.config.product_error_threshold for e in recent_product_error)
        
        return success_criterion and product_criterion
        
    def advance_curriculum(self):
        """Advance to next bit size in curriculum"""
        if self.bit_level < len(self.config.bit_curriculum) - 1:
            self.bit_level += 1
            old_size = self.current_bit_size
            self.current_bit_size = self.config.bit_curriculum[self.bit_level]
            logger.info(f"🎯 Advancing curriculum: {old_size} → {self.current_bit_size} bits")
            
            # Reset advancement history
            self.advancement_history = []
            self.best_success_rate = 0.0
            
            # Log memory usage
            if torch.cuda.is_available():
                logger.info(f"  GPU memory: {torch.cuda.memory_allocated() / 1e9:.2f} GB allocated")
                
    def detect_grokking(self):
        """Detect grokking behavior"""
        if len(self.success_rates) < 50:
            return
            
        # Look for sudden improvement
        recent_avg = np.mean(self.success_rates[-10:])
        past_avg = np.mean(self.success_rates[-50:-40])
        
        # Also check partial success
        if len(self.partial_success_rates) >= 50:
            recent_partial = np.mean(self.partial_success_rates[-10:])
            past_partial = np.mean(self.partial_success_rates[-50:-40])
            
            if recent_partial > 0.5 and past_partial < 0.1:
                logger.info(f"📈 Partial grokking detected! GCD success: {past_partial:.2%} → {recent_partial:.2%}")
        
        if recent_avg > 0.8 and past_avg < 0.2:
            if not self.grokking_detected:
                self.grokking_detected = True
                logger.info(f"🎉 FULL GROKKING DETECTED at step {self.step}!")
                logger.info(f"   Success rate: {past_avg:.2%} → {recent_avg:.2%}")
                
    def run(self):
        """Main training loop for 512-bit factorization with DeepSpeed"""
        logger.info(f"Starting 512-bit Factorization Network training with DeepSpeed ZeRO-{self.config.zero_stage}")
        logger.info(f"World size: {self.world_size}, Local rank: {self.local_rank}")
        logger.info(f"Model parameters: {sum(p.numel() for p in self.model_engine.parameters()):,}")
        logger.info(f"DeepSpeed offload: optimizer={self.config.offload_optimizer}, param={self.config.offload_param}")
        
        if self.config.resume_from:
            logger.info(f"📂 Resumed from checkpoint: {self.config.resume_from}")
            logger.info(f"   Starting from step {self.step}")
        
        if self.in_kickstart:
            logger.info(f"🚀 Starting with {self.config.kickstart_steps}-step kick-start on {self.config.kickstart_bits}-bit numbers")
        else:
            logger.info(f"Initial bit size: {self.current_bit_size}")
            
        logger.info(f"🔧 Configuration: product_error_weight_initial={self.config.product_error_weight_initial}")
        logger.info(f"📦 All patches applied: P0-P5 + 8-bit kickstart + DeepSpeed ZeRO-3")
        
        # Log GPU info
        if torch.cuda.is_available():
            logger.info(f"GPU: {torch.cuda.get_device_name()}")
            logger.info(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        
        try:
            for step in range(self.step, self.config.max_steps):
                self.step = step
                
                # Training step
                loss, loss_components, step_time = self.train_step()
                self.train_losses.append(loss)
                
                # Track components
                for k, v in loss_components.items():
                    self.loss_components[k].append(v)
                
                # Evaluation
                eval_interval = self.get_eval_interval()
                if step % eval_interval == 0 and self.local_rank == 0:  # Only evaluate on rank 0
                    eval_results = self.evaluate_with_gcd()
                    
                    self.val_losses.append(eval_results['loss'])
                    self.success_rates.append(eval_results['success_rate'])
                    self.partial_success_rates.append(eval_results['partial_rate'])
                    self.product_errors.append(eval_results['product_error'])
                    
                    # Track for advancement
                    self.advancement_history.append({
                        'success_rate': eval_results['success_rate'],
                        'product_error': eval_results['product_error']
                    })
                    
                    # Keep only recent history
                    if len(self.advancement_history) > self.config.advancement_window * 2:
                        self.advancement_history = self.advancement_history[-self.config.advancement_window * 2:]
                    
                    # Update best
                    if not self.in_kickstart:
                        self.best_success_rate = max(self.best_success_rate, eval_results['success_rate'])
                    
                    # Log progress - FIXED: use self.step instead of step
                    current_lr = self.optimizer.param_groups[0]['lr']
                    phase = "kick-start" if self.in_kickstart else f"{self.current_bit_size}-bit"
                    logger.info(
                        f"Step {self.step} [{phase}]: "  # Changed from 'step' to 'self.step'
                        f"loss={loss:.4f}, val_loss={eval_results['loss']:.4f}, "
                        f"success={eval_results['success_rate']:.1%}, "
                        f"partial={eval_results['partial_rate']:.1%}, "
                        f"prod_err={eval_results['product_error']:.3f}, "
                        f"lr={current_lr:.2e}, time={step_time:.2f}s"
                    )
                    
                    # Detailed components every 1000 steps
                    if self.step % 1000 == 0 and self.step > 0:
                        logger.info(f"  Components: {eval_results['components']}")
                        logger.info(f"  Inference time: {eval_results['inference_time']*1000:.1f} ms")
                    
                    # Check curriculum advancement (handles kickstart transition)
                    if self.check_curriculum_advancement() and not self.in_kickstart:
                        self.advance_curriculum()
                    
                    # Detect grokking
                    self.detect_grokking()
                
                # Alpha monitoring
                if self.step % self.config.alpha_interval == 0 and self.step > 0 and self.local_rank == 0:
                    # For DeepSpeed, we need to gather the model first
                    layer_alphas, mean_alpha = heavy_tail_alpha(self.model_engine.module)
                    self.alphas.append(mean_alpha)
                    self.layer_alphas.append(layer_alphas)
                    
                    logger.info(f"📊 Alpha at step {self.step}: {mean_alpha:.3f}")
                    
                    # Check for collapse
                    if mean_alpha < 2.0 and len(self.alphas) > 5:
                        recent_alphas = self.alphas[-5:]
                        if all(a < 2.0 for a in recent_alphas):
                            self.collapse_detected = True
                            logger.warning(f"⚠️  Alpha collapse detected: α={mean_alpha:.3f}")
                    
                    # Save checkpoint in critical range
                    if 2.1 <= mean_alpha <= 2.25 and self.current_bit_size >= 128:
                        self.save_checkpoint(f"alpha_{mean_alpha:.2f}_bits_{self.current_bit_size}_step_{self.step}")
                
                # Regular checkpointing
                if self.step % self.config.checkpoint_interval == 0 and self.step > 0 and self.local_rank == 0:
                    self.save_checkpoint(f"checkpoint_step_{self.step}_bits_{self.current_bit_size}")
                    
                    # Log average step time
                    if self.step_times:
                        avg_step_time = np.mean(self.step_times[-100:])
                        logger.info(f"  Average step time: {avg_step_time:.3f}s")
                        
                        # Estimate time to completion
                        remaining_steps = self.config.max_steps - self.step
                        eta_hours = (remaining_steps * avg_step_time) / 3600
                        logger.info(f"  Estimated time remaining: {eta_hours:.1f} hours")
                
                # Save on grokking
                if self.grokking_detected and not hasattr(self, 'grokking_saved') and self.local_rank == 0:
                    self.save_checkpoint(f"grokking_detected_step_{self.step}_bits_{self.current_bit_size}")
                    self.grokking_saved = True
                    
                # Check if reached 512-bit target
                if self.current_bit_size == 512 and self.success_rates and self.success_rates[-1] >= 0.9:
                    logger.info("🎊 512-bit factorization achieved! Success rate ≥ 90%")
                    if self.local_rank == 0:
                        self.save_checkpoint(f"512bit_success_step_{self.step}")
                    break
                    
        except KeyboardInterrupt:
            logger.info("Training interrupted by user")
        except Exception as e:
            logger.error(f"Training error: {e}", exc_info=True)
        finally:
            if self.local_rank == 0:
                self.save_results()

    def save_checkpoint(self, tag):
        """Save DeepSpeed checkpoint with complete training state"""
        # Prepare client state with all tracking variables
        client_state = {
            'step': self.step,
            'bit_size': self.current_bit_size,
            'bit_level': self.bit_level,
            'in_kickstart': self.in_kickstart,
            'config': asdict(self.config),
            'alphas': self.alphas,
            'success_rates': self.success_rates,
            'partial_success_rates': self.partial_success_rates,
            'product_errors': self.product_errors,
            'train_losses': self.train_losses[-10000:],  # Keep last 10k
            'val_losses': self.val_losses,
            'layer_alphas': self.layer_alphas[-100:],  # Keep last 100
            'loss_components': {k: v[-1000:] for k, v in self.loss_components.items()},  # Keep last 1k
            'advancement_history': self.advancement_history,
            'grokking_detected': self.grokking_detected,
            'collapse_detected': self.collapse_detected,
            'best_success_rate': self.best_success_rate,
            # Add ACT efficiency history
            'act_efficiency_history': getattr(self, 'act_efficiency_history', [])
        }
        
        # DeepSpeed save
        save_path = Path('results') / 'checkpoints' / tag
        save_path.parent.mkdir(exist_ok=True, parents=True)
        
        self.model_engine.save_checkpoint(str(save_path), client_state=client_state)
        logger.info(f"💾 DeepSpeed checkpoint saved: {save_path}")
            
    def save_results(self):
        """Save comprehensive results including 512-bit performance"""
        results = {
            'config': asdict(self.config),
            'final_step': self.step,
            'final_bit_size': self.current_bit_size,
            'best_success_rate': float(self.best_success_rate),
            'grokking_detected': self.grokking_detected,
            'collapse_detected': self.collapse_detected,
            'train_losses': [float(x) for x in self.train_losses[-10000:]],  # Last 10k only
            'val_losses': [float(x) for x in self.val_losses],
            'success_rates': [float(x) for x in self.success_rates],
            'partial_success_rates': [float(x) for x in self.partial_success_rates],
            'product_errors': [float(x) for x in self.product_errors],
            'alphas': [float(x) for x in self.alphas],
            'layer_alphas': [[float(x) for x in layer] for layer in self.layer_alphas[-100:]],  # Last 100
            'loss_components': {k: [float(x) for x in v[-1000:]] for k, v in self.loss_components.items()},
            'bit_size_progression': self._get_bit_size_progression(),
            'performance_by_bits': self._get_performance_by_bits(),
            'kickstart_used': self.config.kickstart_steps > 0,
            'deepspeed_config': self.config.get_deepspeed_config()
        }
        
        # Save JSON
        results_path = Path('results') / f'factor512_{self.config.experiment_type}_results.json'
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Save CSVs
        self._save_csv_data()
        
        # Save 512-bit predictions if reached
        if self.current_bit_size >= 512:
            self._save_512bit_predictions()
        
        # Generate plots
        self.plot_results()
        
        logger.info(f"✅ Results saved to results/factor512_{self.config.experiment_type}_*")
        
    def _get_bit_size_progression(self):
        """Track when each bit size was reached"""
        progression = []
        current_bits = self.config.kickstart_bits if self.config.kickstart_steps > 0 else self.config.bit_curriculum[0]
        
        # Track kickstart phase
        if self.config.kickstart_steps > 0:
            progression.append({
                'step': 0,
                'bits': self.config.kickstart_bits,
                'phase': 'kickstart',
                'success_rate': 0.0
            })
        
        # Track main curriculum
        for i, rate in enumerate(self.success_rates):
            step = i * self.get_eval_interval()
            
            # Check if we transitioned from kickstart
            if self.config.kickstart_steps > 0 and step >= self.config.kickstart_steps and len(progression) == 1:
                progression.append({
                    'step': step,
                    'bits': self.config.bit_curriculum[0],
                    'phase': 'main',
                    'success_rate': rate
                })
                current_bits = self.config.bit_curriculum[0]
            elif len(progression) == 0 or progression[-1]['bits'] != current_bits:
                progression.append({
                    'step': step,
                    'bits': current_bits,
                    'phase': 'main',
                    'success_rate': rate
                })
                
            # Update current bits based on curriculum
            if rate > 0.9 and current_bits < 512:
                if current_bits in self.config.bit_curriculum:
                    idx = self.config.bit_curriculum.index(current_bits)
                    if idx < len(self.config.bit_curriculum) - 1:
                        current_bits = self.config.bit_curriculum[idx + 1]
                    
        return progression
        
    def _get_performance_by_bits(self):
        """Get best performance for each bit size"""
        perf_by_bits = defaultdict(lambda: {'best_success': 0.0, 'steps_to_90': None})
        
        # Track kickstart performance
        if self.config.kickstart_steps > 0:
            kickstart_success = 0.0
            for i, rate in enumerate(self.success_rates):
                step = i * self.get_eval_interval()
                if step < self.config.kickstart_steps:
                    kickstart_success = max(kickstart_success, rate)
                else:
                    break
            perf_by_bits[self.config.kickstart_bits] = {
                'best_success': kickstart_success,
                'steps_to_90': None,
                'phase': 'kickstart'
            }
        
        # Track main curriculum performance
        current_bits = self.config.bit_curriculum[0]
        
        for i, rate in enumerate(self.success_rates):
            step = i * self.get_eval_interval()
            
            # Skip kickstart steps
            if self.config.kickstart_steps > 0 and step < self.config.kickstart_steps:
                continue
            
            # Update best
            if rate > perf_by_bits[current_bits]['best_success']:
                perf_by_bits[current_bits]['best_success'] = rate
                
            # Track steps to 90%
            if rate >= 0.9 and perf_by_bits[current_bits]['steps_to_90'] is None:
                perf_by_bits[current_bits]['steps_to_90'] = step
                
            # Update current bits
            if rate > 0.9 and current_bits < 512:
                idx = self.config.bit_curriculum.index(current_bits)
                if idx < len(self.config.bit_curriculum) - 1:
                    current_bits = self.config.bit_curriculum[idx + 1]
                    
        return dict(perf_by_bits)
        
    def _save_csv_data(self):
        """Save detailed CSV files"""
        # Training data
        df_train = pd.DataFrame({
            'step': range(len(self.train_losses)),
            'train_loss': self.train_losses
        })
        df_train.to_csv(Path('results') / f'factor512_{self.config.experiment_type}_training.csv', index=False)
        
        # Evaluation data
        eval_steps = [i * self.get_eval_interval() for i in range(len(self.val_losses))]
        df_eval = pd.DataFrame({
            'step': eval_steps,
            'val_loss': self.val_losses,
            'success_rate': self.success_rates,
            'partial_success_rate': self.partial_success_rates,
            'product_error': self.product_errors
        })
        df_eval.to_csv(Path('results') / f'factor512_{self.config.experiment_type}_evaluation.csv', index=False)
        
        # Alpha data
        alpha_steps = [i * self.config.alpha_interval for i in range(1, len(self.alphas) + 1)]
        df_alpha = pd.DataFrame({
            'step': alpha_steps,
            'mean_alpha': self.alphas
        })
        df_alpha.to_csv(Path('results') / f'factor512_{self.config.experiment_type}_alphas.csv', index=False)
        
    def _save_512bit_predictions(self):
        """Save detailed predictions for 512-bit test cases"""
        logger.info("Generating 512-bit test predictions...")
        
        self.model_engine.eval()
        predictions = []
        
        with torch.no_grad():
            for i in range(100):  # 100 test cases
                # Generate 512-bit test case
                n, p_true, q_true = self.data_gen.generate_semiprime(512)
                
                # Time inference
                start_time = time.time()
                
                # Convert to tensor and predict
                x = number_to_nybble_tensor(n, device=self.device).unsqueeze(0)
                pred1, pred2, _ = self.model_engine(x, 512)
                
                inference_time = time.time() - start_time
                
                # Convert predictions to numbers
                pred_nyb1 = torch.round(pred1[0]).clamp(0, 15).long()
                pred_nyb2 = torch.round(pred2[0]).clamp(0, 15).long()
                
                p_pred = nybble_tensor_to_number(pred_nyb1)
                q_pred = nybble_tensor_to_number(pred_nyb2)
                
                # Check success
                success = (p_pred * q_pred == n) and p_pred > 1 and q_pred > 1
                if success:
                    success = set([p_pred, q_pred]) == set([p_true, q_true])
                
                predictions.append({
                    'n': str(n),
                    'p_true': str(p_true),
                    'q_true': str(q_true),
                    'p_pred': str(p_pred),
                    'q_pred': str(q_pred),
                    'success': success,
                    'inference_time_ms': inference_time * 1000,
                    'n_bits': n.bit_length()
                })
                
        # Save predictions
        df_pred = pd.DataFrame(predictions)
        df_pred.to_csv(Path('results') / '512bit_predictions_deepspeed.csv', index=False)
        
        # Log summary
        success_rate = df_pred['success'].mean()
        avg_time = df_pred['inference_time_ms'].mean()
        logger.info(f"512-bit test results: {success_rate:.1%} success rate, {avg_time:.1f}ms avg inference")
        
    def plot_results(self):
        """Generate comprehensive plots for 512-bit experiments with DeepSpeed metrics"""
        fig = plt.figure(figsize=(20, 24))
        gs = fig.add_gridspec(5, 3, hspace=0.3, wspace=0.3)
        
        # 1. Loss curves
        ax = fig.add_subplot(gs[0, 0])
        ax.plot(self.train_losses, label='Train Loss', alpha=0.7, linewidth=1)
        eval_steps = [i * self.get_eval_interval() for i in range(len(self.val_losses))]
        ax.plot(eval_steps, self.val_losses, label='Val Loss', marker='o', markersize=2)
        
        # Mark kickstart end
        if self.config.kickstart_steps > 0:
            ax.axvline(x=self.config.kickstart_steps, color='green', linestyle='--', 
                      alpha=0.5, label='Kickstart End')
        
        ax.set_xlabel('Step')
        ax.set_ylabel('Loss')
        ax.set_title('Training and Validation Loss')
        ax.legend()
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)
        
        # 2. Success rates with curriculum markers
        ax = fig.add_subplot(gs[0, 1])
        ax.plot(eval_steps, self.success_rates, label='Full Success', marker='o', markersize=2)
        ax.plot(eval_steps, self.partial_success_rates, label='Partial (GCD)', marker='s', markersize=2, alpha=0.7)
        
        # Mark curriculum transitions
        progression = self._get_bit_size_progression()
        for item in progression:
            color = 'green' if item.get('phase') == 'kickstart' else 'gray'
            ax.axvline(x=item['step'], color=color, linestyle='--', alpha=0.3)
            ax.text(item['step'], 0.95, f"{item['bits']}b", rotation=90, 
                   verticalalignment='bottom', fontsize=8)
        
        ax.set_xlabel('Step')
        ax.set_ylabel('Success Rate')
        ax.set_title('Factorization Success Rates with Curriculum')
        ax.set_ylim([0, 1.05])
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 3. Product error evolution
        ax = fig.add_subplot(gs[0, 2])
        ax.plot(eval_steps, self.product_errors, color='red', marker='o', markersize=2)
        ax.axhline(y=self.config.product_error_threshold, color='green', linestyle='--', 
                  label=f'Threshold ({self.config.product_error_threshold})')
        ax.set_xlabel('Step')
        ax.set_ylabel('Product Error')
        ax.set_title('Product Consistency Error')
        ax.set_yscale('log')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 4. Alpha evolution
        ax = fig.add_subplot(gs[1, 0])
        if self.alphas:
            alpha_steps = [i * self.config.alpha_interval for i in range(1, len(self.alphas) + 1)]
            ax.plot(alpha_steps, self.alphas, marker='o', color='red', markersize=3)
            ax.axhspan(2.8, 3.3, alpha=0.2, color='green', label='Expected 512-bit range')
            ax.axhline(y=2.0, color='k', linestyle='--', alpha=0.5)
        ax.set_xlabel('Step')
        ax.set_ylabel('Mean α')
        ax.set_title('Heavy-tail Exponent Evolution')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 5. Step time analysis
        ax = fig.add_subplot(gs[1, 1])
        if self.step_times:
            # Moving average of step times
            window = 100
            step_times_smooth = np.convolve(self.step_times, np.ones(window)/window, mode='valid')
            ax.plot(step_times_smooth, alpha=0.7)
            ax.axhline(y=0.6, color='red', linestyle='--', label='Target (<0.6s)')
            ax.set_xlabel('Step')
            ax.set_ylabel('Step Time (s)')
            ax.set_title(f'Training Step Time (smoothed, window={window})')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # 6. DeepSpeed Configuration Info
        ax = fig.add_subplot(gs[1, 2])
        ax.text(0.1, 0.9, "DeepSpeed Configuration:", 
               transform=ax.transAxes, fontsize=14, weight='bold')
        
        config_text = f"""
ZeRO Stage: {self.config.zero_stage}
Offload Optimizer: {self.config.offload_optimizer}
Offload Parameters: {self.config.offload_param}
Pin Memory: {self.config.pin_memory}
World Size: {self.world_size if hasattr(self, 'world_size') else 1}
Mixed Precision: bf16

Expected Memory Savings:
- Optimizer states: ~8x reduction
- Gradients: ~{self.world_size if hasattr(self, 'world_size') else 1}x reduction
- Parameters: CPU offload enabled
        """
        ax.text(0.1, 0.1, config_text, transform=ax.transAxes, fontsize=10, 
               fontfamily='monospace', verticalalignment='bottom')
        ax.axis('off')
        
        # 7. Bit size progression timeline
        ax = fig.add_subplot(gs[2, 0])
        progression = self._get_bit_size_progression()
        if progression:
            steps = [item['step'] for item in progression]
            bits = [item['bits'] for item in progression]
            colors = ['green' if item.get('phase') == 'kickstart' else 'purple' for item in progression]
            ax.scatter(steps, bits, marker='o', s=50, c=colors)
            ax.set_xlabel('Step')
            ax.set_ylabel('Bit Size')
            ax.set_title('Curriculum Progression')
            ax.set_yscale('log', base=2)
            ax.grid(True, alpha=0.3)
        
        # 8. Performance by bit size (bar chart)
        ax = fig.add_subplot(gs[2, 1:])
        perf_by_bits = self._get_performance_by_bits()
        if perf_by_bits:
            bit_sizes = sorted(perf_by_bits.keys())
            success_rates = [perf_by_bits[b]['best_success'] for b in bit_sizes]
            steps_to_90 = [perf_by_bits[b]['steps_to_90'] if perf_by_bits[b]['steps_to_90'] else 0 
                          for b in bit_sizes]
            colors = ['lightgreen' if perf_by_bits[b].get('phase') == 'kickstart' else 'green' 
                     for b in bit_sizes]
            
            x = np.arange(len(bit_sizes))
            width = 0.35
            
            ax2 = ax.twinx()
            
            bars1 = ax.bar(x - width/2, success_rates, width, label='Best Success Rate', 
                           color=colors, alpha=0.7)
            bars2 = ax2.bar(x + width/2, steps_to_90, width, label='Steps to 90%', 
                            color='blue', alpha=0.7)
            
            ax.set_xlabel('Bit Size')
            ax.set_ylabel('Best Success Rate', color='green')
            ax2.set_ylabel('Steps to 90% Success', color='blue')
            ax.set_title('Performance by Bit Size')
            ax.set_xticks(x)
            ax.set_xticklabels(bit_sizes)
            ax.tick_params(axis='y', labelcolor='green')
            ax2.tick_params(axis='y', labelcolor='blue')
            ax.set_ylim([0, 1.05])
            ax.grid(True, alpha=0.3)
        
        # 9. Loss component breakdown
        ax = fig.add_subplot(gs[3, 0])
        if self.loss_components:
            components_to_plot = ['factor_loss', 'product_error', 'residue_loss', 'legendre_loss']
            for comp in components_to_plot:
                if comp in self.loss_components:
                    values = self.loss_components[comp]
                    ax.plot(values, label=comp, alpha=0.7)
            ax.set_xlabel('Step')
            ax.set_ylabel('Loss Value')
            ax.set_title('Loss Components Breakdown')
            ax.legend()
            ax.set_yscale('log')
            ax.grid(True, alpha=0.3)
        
        # 10. Memory usage estimate
        ax = fig.add_subplot(gs[3, 1])
        ax.text(0.1, 0.9, "Memory Usage (DeepSpeed ZeRO-3):", 
               transform=ax.transAxes, fontsize=14, weight='bold')
        
        memory_text = f"""
Bit Size    VRAM (Original)    VRAM (ZeRO-3)
--------    --------------     -------------
8 (kick)    ~4-6 GB           ~1-2 GB
16-64       ~6-10 GB          ~2-3 GB
128         ~12-14 GB         ~3-4 GB
256         ~16-18 GB         ~4-5 GB
512         ~18-21 GB         ~5-6 GB

Current: {self.current_bit_size} bits
Model params: {sum(p.numel() for p in self.model_engine.parameters())/1e6:.1f}M
        """
        ax.text(0.1, 0.1, memory_text, transform=ax.transAxes, fontsize=10, 
               fontfamily='monospace', verticalalignment='bottom')
        ax.axis('off')
        
        # 11. Summary statistics
        ax = fig.add_subplot(gs[4, :])
        ax.axis('off')
        

        # 10.5 ACT Efficiency Plot (new)
        ax = fig.add_subplot(gs[3, 2])
        self.plot_act_efficiency(ax)

        # Compute summary
        summary_lines = [
            f"Experiment: {self.config.experiment_type}",
            f"Final step: {self.step:,}",
            f"Final bit size: {self.current_bit_size}",
            f"Best success rate: {self.best_success_rate:.1%}",
            f"Grokking detected: {'Yes' if self.grokking_detected else 'No'}",
            f"Alpha collapse: {'Yes' if self.collapse_detected else 'No'}",
f"Patches applied: P0-P5 + 8-bit kickstart + DeepSpeed ZeRO-3",
            f"Product error weight: {self.config.product_error_weight_initial} → {self.config.product_error_weight_high}",
            f"DeepSpeed: ZeRO-{self.config.zero_stage}, World Size: {self.world_size if hasattr(self, 'world_size') else 1}",
        ]
        
        if self.config.kickstart_steps > 0:
            summary_lines.append(f"Kickstart: {self.config.kickstart_steps} steps on {self.config.kickstart_bits}-bit")
        
        if self.alphas:
            summary_lines.extend([
                f"Final α: {self.alphas[-1]:.3f}",
                f"α range: [{min(self.alphas):.3f}, {max(self.alphas):.3f}]"
            ])
            
        if self.current_bit_size >= 512:
            # Add 512-bit specific stats
            recent_success = np.mean(self.success_rates[-10:]) if len(self.success_rates) >= 10 else 0
            summary_lines.append(f"512-bit success rate: {recent_success:.1%}")
            
        summary_text = "\n".join(summary_lines)
        ax.text(0.5, 0.5, summary_text, transform=ax.transAxes, fontsize=12,
               horizontalalignment='center', verticalalignment='center',
               bbox=dict(boxstyle='round,pad=1', facecolor='lightblue', alpha=0.8))
        
        plt.suptitle(f'512-bit Factorization Experiment (DeepSpeed): {self.config.experiment_type}', 
                    fontsize=16)
        plt.tight_layout()
        plt.savefig(f'results/factor512_{self.config.experiment_type}_analysis.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"📊 Plots saved to results/factor512_{self.config.experiment_type}_analysis.png")

def run_evaluation_only(config: Factor512Config):
    """Run evaluation on a held-out test set with DeepSpeed"""
    logger.info(f"Running evaluation-only mode for {config.eval_bits}-bit numbers")
    
    # Initialize distributed training if needed
    if not dist.is_initialized():
        deepspeed.init_distributed()
    
    # Load model
    model = Factor512Net(config)
    
    # Initialize DeepSpeed for inference
    ds_config = config.get_deepspeed_config()
    ds_config['zero_optimization']['stage'] = 0  # No ZeRO for inference
    
    model_engine, _, _, _ = deepspeed.initialize(
        model=model,
        config=ds_config,
        model_parameters=model.parameters()
    )
    
    # Try to load checkpoint
    checkpoint_path = Path('results') / 'checkpoints'
    checkpoints = list(checkpoint_path.glob('512bit_success_*'))
    
    if not checkpoints:
        checkpoints = list(checkpoint_path.glob('checkpoint_*'))
        
    if checkpoints:
        # Load most recent checkpoint
        checkpoint_dir = sorted(checkpoints)[-1]
        logger.info(f"Loading checkpoint: {checkpoint_dir}")
        
        _, client_state = model_engine.load_checkpoint(str(checkpoint_dir))
        model_engine.eval()
    else:
        logger.error("No checkpoint found!")
        return
    
    # Generate test set
    data_gen = RSA512DataGenerator(config)
    test_size = 10000
    batch_size = 100
    
    logger.info(f"Generating {test_size} test cases...")
    
    all_success = []
    all_times = []
    
    with torch.no_grad():
        for i in range(0, test_size, batch_size):
            # Generate batch
            batch_data = data_gen.generate_batch(batch_size, config.eval_bits)
            
            # Convert to tensors
            numbers = batch_data['numbers']
            x = torch.stack([number_to_nybble_tensor(n, device=config.device) for n in numbers])
            
            # Time inference
            start_time = time.time()
            pred1, pred2, _ = model_engine(x, config.eval_bits)
            inference_time = (time.time() - start_time) / batch_size
            
            # Check success
            for j in range(batch_size):
                pred_nyb1 = torch.round(pred1[j]).clamp(0, 15).long()
                pred_nyb2 = torch.round(pred2[j]).clamp(0, 15).long()
                
                p_pred = nybble_tensor_to_number(pred_nyb1)
                q_pred = nybble_tensor_to_number(pred_nyb2)
                
                success = (p_pred * q_pred == numbers[j]) and p_pred > 1 and q_pred > 1
                if success:
                    # Verify correct factors
                    true_factors = set([batch_data['factors1'][j], batch_data['factors2'][j]])
                    pred_factors = set([p_pred, q_pred])
                    success = true_factors == pred_factors
                    
                all_success.append(success)
                all_times.append(inference_time * 1000)  # Convert to ms
            
            if (i + batch_size) % 1000 == 0:
                logger.info(f"  Processed {i + batch_size}/{test_size} test cases...")
    
    # Compute statistics
    success_rate = np.mean(all_success)
    success_std = np.std(all_success)
    n_success = sum(all_success)
    
    # 95% confidence interval (Wilson score interval)
    from scipy import stats
    ci_low, ci_high = stats.beta.ppf([0.025, 0.975], n_success + 1, test_size - n_success + 1)
    
    avg_time = np.mean(all_times)
    std_time = np.std(all_times)
    
    # Report results
    logger.info("=" * 60)
    logger.info(f"Evaluation Results for {config.eval_bits}-bit Factorization (DeepSpeed)")
    logger.info("=" * 60)
    logger.info(f"Test set size: {test_size:,}")
    logger.info(f"Success rate: {success_rate:.2%} ± {success_std:.2%}")
    logger.info(f"95% CI: [{ci_low:.2%}, {ci_high:.2%}]")
    logger.info(f"Successful factorizations: {n_success:,}")
    logger.info(f"Average inference time: {avg_time:.2f} ± {std_time:.2f} ms")
    logger.info(f"Total GPU-seconds: {sum(all_times) / 1000:.1f}")
    logger.info("=" * 60)
    
    # Save detailed results
    results = {
        'bit_size': config.eval_bits,
        'test_size': test_size,
        'success_rate': float(success_rate),
        'success_std': float(success_std),
        'ci_95_low': float(ci_low),
        'ci_95_high': float(ci_high),
        'n_success': int(n_success),
        'avg_inference_ms': float(avg_time),
        'std_inference_ms': float(std_time),
        'total_gpu_seconds': float(sum(all_times) / 1000),
        'deepspeed_enabled': True
    }
    
    with open(f'results/{config.eval_bits}bit_evaluation_results_deepspeed.json', 'w') as f:
        json.dump(results, f, indent=2)
        
    logger.info(f"Detailed results saved to results/{config.eval_bits}bit_evaluation_results_deepspeed.json")

def main():
    """Main entry point for 512-bit factorization experiments with DeepSpeed"""
    parser = argparse.ArgumentParser(description='512-bit Factorization Network with DeepSpeed ZeRO-3')
    parser.add_argument('--eval-only', action='store_true', help='Run evaluation only')
    parser.add_argument('--bits', type=int, default=512, help='Bit size for evaluation')
    parser.add_argument('--experiment', type=str, default='512bit_kickstart_deepspeed', help='Experiment name')
    parser.add_argument('--no-kickstart', action='store_true', help='Disable kickstart')
    parser.add_argument('--kickstart-steps', type=int, default=500, help='Number of kickstart steps')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume from')
    parser.add_argument('--zero-stage', type=int, default=3, choices=[0, 1, 2, 3], help='DeepSpeed ZeRO stage')
    parser.add_argument('--no-offload', action='store_true', help='Disable CPU offloading')
    parser.add_argument('--local_rank', type=int, default=-1, help='Local rank for distributed training')
    args = parser.parse_args()
    
    # Configuration
    config = Factor512Config(
        experiment_type=args.experiment,
        eval_only=args.eval_only,
        eval_bits=args.bits,
        kickstart_steps=0 if args.no_kickstart else args.kickstart_steps,
        resume_from=args.resume,
        zero_stage=args.zero_stage,
        offload_optimizer=not args.no_offload,
        offload_param=not args.no_offload and args.zero_stage >= 3,
        local_rank=args.local_rank
    )
    
    logger.info("=" * 60)
    logger.info("512-bit Factorization Network Experiment (DeepSpeed ZeRO-3)")
    logger.info("Applied patches: P0-P5 + 8-bit kickstart + DeepSpeed optimization")
    logger.info(f"Product error weight: {config.product_error_weight_initial} → {config.product_error_weight_high}")
    logger.info(f"DeepSpeed ZeRO stage: {config.zero_stage}")
    logger.info(f"CPU offload: optimizer={config.offload_optimizer}, param={config.offload_param}")
    if config.resume_from:
        logger.info(f"Resume from checkpoint: {config.resume_from}")
    logger.info("=" * 60)
    
    if args.eval_only:
        run_evaluation_only(config)
    else:
        # Create model
        model = Factor512Net(config)
        
        # Log model info
        total_params = sum(p.numel() for p in model.parameters())
        logger.info(f"Model parameters: {total_params:,} ({total_params/1e6:.1f}M)")
        
        # Memory estimate
        param_memory = total_params * 4 / 1e9  # FP32
        logger.info(f"Parameter memory (original): {param_memory:.2f} GB")
        
        # DeepSpeed memory savings estimate
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        zero_memory_divisor = {0: 1, 1: world_size, 2: world_size, 3: world_size}[config.zero_stage]
        
        if config.zero_stage == 3:
            logger.info(f"Expected VRAM usage with ZeRO-3: ~{param_memory / zero_memory_divisor:.2f} GB")
            if config.offload_param:
                logger.info("  + Parameters offloaded to CPU")
            if config.offload_optimizer:
                logger.info("  + Optimizer states offloaded to CPU")
        

        # Create trainer and run
        trainer = Factor512TrainerDeepSpeed(model, config)
        trainer.run()
        
    logger.info("Experiment complete!")

if __name__ == "__main__":
    main()