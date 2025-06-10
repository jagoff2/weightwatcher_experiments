"""
tiny_gpt_experiments.py - Heavy-tail experiments on character-level Tiny-GPT
Tests all 5 hypotheses including region counting
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import weightwatcher as ww
from pathlib import Path
import json
from dataclasses import dataclass, asdict
from typing import List, Tuple, Dict, Optional
import logging
import warnings
import requests
import zipfile
import io
from tqdm import tqdm
warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

# Ensure reproducibility
torch.manual_seed(42)
np.random.seed(42)
torch.backends.cudnn.deterministic = True

# Create results directory
Path("results").mkdir(exist_ok=True)

@dataclass
class TinyGPTConfig:
    """Configuration for Tiny-GPT experiments"""
    vocab_size: int = 256  # Character-level
    n_embd: int = 128
    n_head: int = 4
    n_layer: int = 4
    block_size: int = 128
    dropout: float = 0.1
    batch_size: int = 64
    lr: float = 3e-4
    max_steps: int = 20000
    alpha_interval: int = 500
    region_count_interval: int = 2000
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    use_8bit: bool = False
    experiment_type: str = "baseline"

class TinyStoriesDataset:
    """Character-level TinyStories dataset"""
    def __init__(self, max_size_mb: int = 10):
        self.max_size_mb = max_size_mb
        self.data = self._load_or_generate_data()
        self.vocab_size = 256  # ASCII characters
        
        # Split data
        n = len(self.data)
        train_size = int(0.9 * n)
        self.train_data = self.data[:train_size]
        self.val_data = self.data[train_size:]
        
    def _load_or_generate_data(self) -> str:
        """Load TinyStories or generate synthetic data"""
        data_path = Path("data/tinystories.txt")
        
        if data_path.exists():
            logger.info("Loading existing TinyStories data...")
            with open(data_path, 'r', encoding='utf-8') as f:
                data = f.read()
        else:
            logger.info("Generating synthetic story data...")
            # Generate simple synthetic stories
            data_path.parent.mkdir(exist_ok=True)
            
            templates = [
                "Once upon a time, there was a {adj} {noun} who lived in a {place}. ",
                "The {noun} wanted to {verb} but didn't know how. ",
                "Every day, the {noun} would {verb} and feel {emotion}. ",
                "One day, a {adj2} {noun2} came to visit. ",
                "They became best friends and {verb2} together. ",
                "The end. "
            ]
            
            vocab = {
                'adj': ['happy', 'sad', 'big', 'small', 'brave', 'shy', 'clever', 'funny'],
                'noun': ['cat', 'dog', 'bird', 'mouse', 'rabbit', 'bear', 'fox', 'owl'],
                'place': ['forest', 'mountain', 'river', 'meadow', 'cave', 'tree', 'lake'],
                'verb': ['sing', 'dance', 'play', 'jump', 'run', 'swim', 'fly', 'climb'],
                'emotion': ['happy', 'excited', 'proud', 'grateful', 'peaceful', 'joyful'],
                'adj2': ['wise', 'old', 'young', 'magical', 'mysterious', 'friendly'],
                'noun2': ['wizard', 'fairy', 'dragon', 'unicorn', 'elf', 'gnome'],
                'verb2': ['explored', 'laughed', 'played', 'learned', 'discovered']
            }
            
            stories = []
            target_size = self.max_size_mb * 1024 * 1024  # Convert to bytes
            current_size = 0
            
            while current_size < target_size:
                story = ""
                for template in templates:
                    filled = template
                    for key, values in vocab.items():
                        if '{' + key + '}' in filled:
                            filled = filled.replace('{' + key + '}', np.random.choice(values))
                    story += filled
                
                stories.append(story)
                current_size += len(story.encode('utf-8'))
            
            data = '\n'.join(stories)
            
            # Save for future use
            with open(data_path, 'w', encoding='utf-8') as f:
                f.write(data)
        
        # Ensure ASCII only
        data = ''.join(c if ord(c) < 256 else ' ' for c in data)
        return data
    
    def get_batch(self, batch_size: int, block_size: int, split: str = "train"):
        """Get a batch of sequences"""
        data = self.train_data if split == "train" else self.val_data
        ix = torch.randint(len(data) - block_size, (batch_size,))
        x = torch.stack([torch.tensor([ord(c) for c in data[i:i+block_size]], 
                                     dtype=torch.long) for i in ix])
        y = torch.stack([torch.tensor([ord(c) for c in data[i+1:i+block_size+1]], 
                                     dtype=torch.long) for i in ix])
        return x, y

class MultiHeadSelfAttention(nn.Module):
    """Multi-head self attention block"""
    def __init__(self, config: TinyGPTConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head
        self.dropout = config.dropout
        
        self.key = nn.Linear(config.n_embd, config.n_embd)
        self.query = nn.Linear(config.n_embd, config.n_embd)
        self.value = nn.Linear(config.n_embd, config.n_embd)
        self.proj = nn.Linear(config.n_embd, config.n_embd)
        
        self.attn_dropout = nn.Dropout(dropout=config.dropout)
        self.resid_dropout = nn.Dropout(dropout=config.dropout)
        
        # Causal mask
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                            .view(1, 1, config.block_size, config.block_size))
    
    def forward(self, x):
        B, T, C = x.size()
        
        k = self.key(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        q = self.query(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = self.value(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        
        att = (q @ k.transpose(-2, -1)) * (1.0 / np.sqrt(self.head_dim))
        att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.proj(y))
        
        return y

class TransformerBlock(nn.Module):
    """Transformer block with attention and MLP"""
    def __init__(self, config: TinyGPTConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.attn = MultiHeadSelfAttention(config)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.mlp = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.dropout)
        )
    
    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x

class TinyGPT(nn.Module):
    """Small GPT model for character-level modeling"""
    def __init__(self, config: TinyGPTConfig):
        super().__init__()
        self.config = config
        
        self.token_embedding = nn.Embedding(config.vocab_size, config.n_embd)
        self.position_embedding = nn.Embedding(config.block_size, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)
        
        self.blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        
        # Weight tying
        self.token_embedding.weight = self.lm_head.weight
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
    
    def forward(self, idx, targets=None):
        B, T = idx.shape
        
        tok_emb = self.token_embedding(idx)
        pos_emb = self.position_embedding(torch.arange(T, device=idx.device))
        x = self.dropout(tok_emb + pos_emb)
        
        for block in self.blocks:
            x = block(x)
        
        x = self.ln_f(x)
        logits = self.lm_head(x)
        
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        
        return logits, loss

def heavy_tail_alpha(model: nn.Module) -> Tuple[List[float], float]:
    """Compute layer-wise alpha values using WeightWatcher"""
    try:
        watcher = ww.WeightWatcher(model=model)
        details = watcher.analyze(compute_alphas=True, plot=False)
        
        alphas = []
        for _, row in details.iterrows():
            if 'alpha' in row and not pd.isna(row['alpha']):
                alphas.append(float(row['alpha']))
        
        mean_alpha = np.mean(alphas) if alphas else 3.0
        return alphas, mean_alpha
    except Exception as e:
        logger.warning(f"WeightWatcher analysis failed: {e}")
        return [3.0] * 4, 3.0

def region_counter(model: nn.Module, x0: torch.Tensor, v: torch.Tensor, 
                  steps: int = 1024) -> int:
    """
    Count activation sign changes along ray x = x0 + t*v
    Fast implementation without gradients
    """
    model.eval()
    device = next(model.parameters()).device
    x0 = x0.to(device)
    v = v.to(device)
    
    # Normalize direction vector
    v = v / (torch.norm(v) + 1e-8)
    
    # Sample points along ray
    t_values = torch.linspace(0, 10, steps).to(device)
    region_changes = 0
    prev_pattern = None
    
    with torch.no_grad():
        for t in t_values:
            # Current input point
            x = x0 + t * v
            x = torch.clamp(x, 0, 255).long()  # Ensure valid token indices
            x = x.unsqueeze(0)  # Add batch dimension
            
            # Get activations through the model
            activations = []
            
            # Hook to capture intermediate activations
            def hook_fn(module, input, output):
                if isinstance(module, nn.Linear) and module.out_features > 10:
                    activations.append(output.detach())
            
            # Register hooks
            hooks = []
            for module in model.modules():
                if isinstance(module, nn.Linear):
                    hooks.append(module.register_forward_hook(hook_fn))
            
            # Forward pass
            _ = model(x)
            
            # Remove hooks
            for hook in hooks:
                hook.remove()
            
            # Compute activation pattern (signs)
            if activations:
                pattern = torch.cat([torch.sign(act.flatten()) for act in activations])
                
                if prev_pattern is not None:
                    # Count sign changes
                    changes = (pattern != prev_pattern).sum().item()
                    region_changes += changes > 0  # Count as region change if any sign flipped
                
                prev_pattern = pattern
    
    return region_changes

def widen_layer(model: TinyGPT, factor: float = 1.25, layers: str = "mlp") -> TinyGPT:
    """Widen MLP layers in transformer blocks"""
    with torch.no_grad():
        for block in model.blocks:
            if layers == "mlp":
                # Widen first MLP layer
                mlp = block.mlp
                if isinstance(mlp[0], nn.Linear):
                    old_layer = mlp[0]
                    old_in = old_layer.in_features
                    old_out = old_layer.out_features
                    new_out = int(old_out * factor)
                    
                    # Create new layer
                    new_layer = nn.Linear(old_in, new_out).to(old_layer.weight.device)
                    
                    # Copy and duplicate weights
                    indices = torch.randperm(old_out)[:new_out - old_out]
                    new_layer.weight.data[:old_out] = old_layer.weight.data
                    new_layer.weight.data[old_out:] = old_layer.weight.data[indices] + \
                        torch.randn_like(old_layer.weight.data[indices]) * 0.01
                    
                    # Rescale
                    k = new_out / old_out
                    new_layer.weight.data *= 1.0 / np.sqrt(k)
                    
                    if old_layer.bias is not None:
                        new_layer.bias.data[:old_out] = old_layer.bias.data
                        new_layer.bias.data[old_out:] = old_layer.bias.data[indices]
                    
                    mlp[0] = new_layer
                    
                    # Update second MLP layer
                    if isinstance(mlp[2], nn.Linear):
                        old_layer2 = mlp[2]
                        new_layer2 = nn.Linear(new_out, old_layer2.out_features).to(old_layer2.weight.device)
                        
                        new_layer2.weight.data[:, :old_out] = old_layer2.weight.data
                        new_layer2.weight.data[:, old_out:] = old_layer2.weight.data[:, indices]
                        if old_layer2.bias is not None:
                            new_layer2.bias.data = old_layer2.bias.data
                        
                        mlp[2] = new_layer2
    
    return model

class AlphaAwareTrainer:
    """Training loop with comprehensive monitoring"""
    def __init__(self, model: TinyGPT, dataset: TinyStoriesDataset, config: TinyGPTConfig):
        self.model = model.to(config.device)
        self.dataset = dataset
        self.config = config
        self.device = config.device
        
        # Optimizer
        if config.use_8bit:
            try:
                import bitsandbytes as bnb
                self.optimizer = bnb.optim.Adam8bit(model.parameters(), lr=config.lr)
            except ImportError:
                logger.warning("bitsandbytes not available, using standard Adam")
                self.optimizer = optim.Adam(model.parameters(), lr=config.lr)
        else:
            self.optimizer = optim.Adam(model.parameters(), lr=config.lr)
        
        # Tracking
        self.step = 0
        self.train_losses = []
        self.val_losses = []
        self.alphas = []
        self.layer_alphas = []
        self.region_counts = []
        self.lr_history = []
        self.micro_step_mode = False
        self.collapse_detected = False
        self.widened = False
        self.val_loss_increases = 0
        self.last_alpha = 3.0
        self.checkpoint_saved = False
    
    def train_step(self) -> float:
        """Single training step"""
        self.model.train()
        x, y = self.dataset.get_batch(self.config.batch_size, self.config.block_size, "train")
        x, y = x.to(self.device), y.to(self.device)
        
        self.optimizer.zero_grad()
        _, loss = self.model(x, y)
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        
        self.optimizer.step()
        return loss.item()
    
    def evaluate(self) -> float:
        """Evaluate on validation set"""
        self.model.eval()
        total_loss = 0
        n_batches = 10
        
        with torch.no_grad():
            for _ in range(n_batches):
                x, y = self.dataset.get_batch(self.config.batch_size, 
                                            self.config.block_size, "val")
                x, y = x.to(self.device), y.to(self.device)
                _, loss = self.model(x, y)
                total_loss += loss.item()
        
        return total_loss / n_batches
    
    def measure_region_counts(self):
        """Measure linear region counts along random rays"""
        # Generate random start point and direction
        x0 = torch.randint(0, 256, (self.config.block_size,))
        v = torch.randn(self.config.block_size)
        
        count = region_counter(self.model, x0, v)
        self.region_counts.append(count)
        
        return count
    
    def check_alpha_and_adapt(self):
        """Monitor alpha and adapt training"""
        layer_alphas, mean_alpha = heavy_tail_alpha(self.model)
        self.alphas.append(mean_alpha)
        self.layer_alphas.append(layer_alphas)
        
        # Checkpoint in critical range
        if 2.05 <= mean_alpha <= 2.15 and not self.checkpoint_saved:
            torch.save({
                'model_state': self.model.state_dict(),
                'optimizer_state': self.optimizer.state_dict(),
                'step': self.step,
                'alpha': mean_alpha
            }, f'results/tinygpt_checkpoint_alpha_{mean_alpha:.3f}.pt')
            self.checkpoint_saved = True
            logger.info(f"Checkpoint saved at α={mean_alpha:.3f}")
        
        # Micro-stepping
        if self.config.experiment_type == "microstep":
            if mean_alpha < 2.30 and not self.micro_step_mode:
                self.micro_step_mode = True
                logger.info(f"Micro-step mode at α={mean_alpha:.3f}")
            
            if self.micro_step_mode and mean_alpha < self.last_alpha - 0.02:
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] *= 0.75
                logger.info(f"Reduced LR to {param_group['lr']:.6f}")
        
        # Collapse detection
        val_loss = self.evaluate()
        if len(self.val_losses) > 0 and val_loss > self.val_losses[-1]:
            self.val_loss_increases += 1
        else:
            self.val_loss_increases = 0
        
        if mean_alpha < 2.00 and self.val_loss_increases >= 2:
            self.collapse_detected = True
            logger.warning(f"Collapse at step {self.step}: α={mean_alpha:.3f}")
            
            if self.config.experiment_type == "widen" and not self.widened:
                logger.info("Applying widening...")
                self.model = widen_layer(self.model, factor=1.25)
                self.widened = True
                
                # Reinitialize optimizer
                if self.config.use_8bit:
                    try:
                        import bitsandbytes as bnb
                        self.optimizer = bnb.optim.Adam8bit(
                            self.model.parameters(), 
                            lr=self.optimizer.param_groups[0]['lr']
                        )
                    except ImportError:
                        self.optimizer = optim.Adam(
                            self.model.parameters(),
                            lr=self.optimizer.param_groups[0]['lr']
                        )
                else:
                    self.optimizer = optim.Adam(
                        self.model.parameters(),
                        lr=self.optimizer.param_groups[0]['lr']
                    )
        
        self.last_alpha = mean_alpha
        self.val_losses.append(val_loss)
    
    def run(self):
        """Main training loop"""
        logger.info(f"Starting {self.config.experiment_type} experiment on {self.device}")
        
        for step in tqdm(range(self.config.max_steps), desc="Training"):
            self.step = step
            
            # Train step
            train_loss = self.train_step()
            self.train_losses.append(train_loss)
            self.lr_history.append(self.optimizer.param_groups[0]['lr'])
            
            # Alpha monitoring
            if step % self.config.alpha_interval == 0:
                self.check_alpha_and_adapt()
                
                # Region counting
                if step % self.config.region_count_interval == 0:
                    region_count = self.measure_region_counts()
                    logger.info(f"Step {step}: regions={region_count}, α={self.alphas[-1]:.3f}")
                
                if step % 1000 == 0:
                    logger.info(f"Step {step}: train_loss={train_loss:.4f}, "
                              f"val_loss={self.val_losses[-1]:.4f}, α={self.alphas[-1]:.3f}")
                
                # Early stopping
                if self.collapse_detected and self.config.experiment_type != "widen":
                    logger.info("Stopping due to collapse")
                    break
        
        self.save_results()
    
    def save_results(self):
        """Save all results"""
        results = {
            'config': asdict(self.config),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'alphas': self.alphas,
            'layer_alphas': self.layer_alphas,
            'region_counts': self.region_counts,
            'lr_history': self.lr_history,
            'collapse_detected': self.collapse_detected,
            'final_step': self.step
        }
        
        # JSON results
        with open(f'results/tinygpt_{self.config.experiment_type}_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        # CSV files
        df = pd.DataFrame({
            'step': range(len(self.train_losses)),
            'train_loss': self.train_losses,
            'lr': self.lr_history
        })
        df.to_csv(f'results/tinygpt_{self.config.experiment_type}_training.csv', index=False)
        
        # Alpha and region data
        alpha_steps = list(range(0, len(self.train_losses), self.config.alpha_interval))
        alpha_df = pd.DataFrame({
            'step': alpha_steps[:len(self.alphas)],
            'mean_alpha': self.alphas,
            'val_loss': self.val_losses
        })
        alpha_df.to_csv(f'results/tinygpt_{self.config.experiment_type}_alphas.csv', index=False)
        
        # Region counts
        region_steps = list(range(0, len(self.train_losses), self.config.region_count_interval))
        region_df = pd.DataFrame({
            'step': region_steps[:len(self.region_counts)],
            'region_count': self.region_counts,
            'alpha': [self.alphas[i * self.config.region_count_interval // self.config.alpha_interval] 
                     for i in range(len(self.region_counts))]
        })
        region_df.to_csv(f'results/tinygpt_{self.config.experiment_type}_regions.csv', index=False)
        
        self.plot_results()
    
    def plot_results(self):
        """Generate comprehensive plots"""
        fig = plt.figure(figsize=(15, 12))
        
        # 1. Loss curves
        ax1 = plt.subplot(3, 2, 1)
        ax1.plot(self.train_losses, label='Train Loss', alpha=0.7)
        val_steps = list(range(0, len(self.train_losses), self.config.alpha_interval))
        ax1.plot(val_steps[:len(self.val_losses)], self.val_losses, 
                label='Val Loss', marker='o', markersize=3)
        ax1.set_xlabel('Step')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training Dynamics')
        ax1.legend()
        ax1.set_yscale('log')
        
        # 2. Alpha evolution
        ax2 = plt.subplot(3, 2, 2)
        alpha_steps = list(range(0, len(self.train_losses), self.config.alpha_interval))
        ax2.plot(alpha_steps[:len(self.alphas)], self.alphas, 
                marker='o', color='red', markersize=4)
        ax2.axhline(y=2.0, color='k', linestyle='--', alpha=0.5, label='α=2')
        ax2.set_xlabel('Step')
        ax2.set_ylabel('Mean α')
        ax2.set_title('Heavy-tail Exponent Evolution')
        ax2.legend()
        
        # 3. Layer-wise synchrony
        ax3 = plt.subplot(3, 2, 3)
        if self.layer_alphas:
            layer_alphas_array = np.array(self.layer_alphas)
            for i in range(min(4, layer_alphas_array.shape[1])):
                ax3.plot(alpha_steps[:len(self.alphas)], layer_alphas_array[:, i], 
                        alpha=0.7, label=f'Layer {i+1}')
            ax3.set_xlabel('Step')
            ax3.set_ylabel('Layer α')
            ax3.set_title('Layer-wise α Synchronization')
            ax3.legend()
        
        # 4. Region counts
        ax4 = plt.subplot(3, 2, 4)
        if self.region_counts:
            region_steps = list(range(0, len(self.train_losses), self.config.region_count_interval))
            ax4.plot(region_steps[:len(self.region_counts)], self.region_counts, 
                    marker='s', color='green', markersize=4)
            ax4.set_xlabel('Step')
            ax4.set_ylabel('Region Count')
            ax4.set_title('Linear Region Evolution')
            ax4.set_yscale('log')
        
        # 5. Region scaling analysis
        ax5 = plt.subplot(3, 2, 5)
        if len(self.region_counts) > 5 and len(self.alphas) > 5:
            # Get alphas at region count steps
            region_alphas = []
            for i in range(len(self.region_counts)):
                alpha_idx = i * self.config.region_count_interval // self.config.alpha_interval
                if alpha_idx < len(self.alphas):
                    region_alphas.append(self.alphas[alpha_idx])
            
            if len(region_alphas) == len(self.region_counts):
                # Plot log(regions) vs depth*(3-alpha)
                depth = self.config.n_layer
                x_vals = [depth * (3 - alpha) for alpha in region_alphas]
                y_vals = [np.log(rc + 1) for rc in self.region_counts]
                
                ax5.scatter(x_vals, y_vals, color='purple', s=50)
                
                # Fit line if enough points
                if len(x_vals) > 3:
                    from scipy import stats
                    slope, intercept, r_value, _, _ = stats.linregress(x_vals, y_vals)
                    x_fit = np.linspace(min(x_vals), max(x_vals), 100)
                    y_fit = slope * x_fit + intercept
                    ax5.plot(x_fit, y_fit, 'r--', 
                            label=f'R²={r_value**2:.3f}, slope={slope:.2f}')
                    ax5.legend()
                
                ax5.set_xlabel('depth × (3 - α)')
                ax5.set_ylabel('log(region count)')
                ax5.set_title('Region Scaling Analysis')
        
        # 6. Learning rate
        ax6 = plt.subplot(3, 2, 6)
        ax6.plot(self.lr_history)
        ax6.set_xlabel('Step')
        ax6.set_ylabel('Learning Rate')
        ax6.set_title('Learning Rate Schedule')
        ax6.set_yscale('log')
        
        plt.tight_layout()
        plt.savefig(f'results/tinygpt_{self.config.experiment_type}_plots.png', dpi=150)
        plt.close()

def experiment_runner(cfg: Dict):
    """Run experiment with given config"""
    config = TinyGPTConfig(**cfg)
    
    # Create model and dataset
    model = TinyGPT(config)
    dataset = TinyStoriesDataset(max_size_mb=10)
    
    # Run training
    trainer = AlphaAwareTrainer(model, dataset, config)
    trainer.run()
    
    return trainer

def main():
    """Run all Tiny-GPT experiments"""
    experiments = [
        {"experiment_type": "baseline", "max_steps": 15000},
        {"experiment_type": "microstep", "max_steps": 20000},
        {"experiment_type": "widen", "max_steps": 18000}
    ]
    
    results_summary = {}
    
    for exp_config in experiments:
        logger.info(f"\n{'='*50}")
        logger.info(f"Starting {exp_config['experiment_type']} experiment")
        logger.info(f"{'='*50}\n")
        
        trainer = experiment_runner(exp_config)
        
        # Summary statistics
        results_summary[exp_config['experiment_type']] = {
            'final_train_loss': trainer.train_losses[-1] if trainer.train_losses else None,
            'final_val_loss': trainer.val_losses[-1] if trainer.val_losses else None,
            'final_alpha': trainer.alphas[-1] if trainer.alphas else None,
            'collapse_detected': trainer.collapse_detected,
            'final_region_count': trainer.region_counts[-1] if trainer.region_counts else None,
            'region_scaling_verified': verify_region_scaling(trainer)
        }
    
    # Save summary
    with open('results/tinygpt_experiments_summary.json', 'w') as f:
        json.dump(results_summary, f, indent=2)
    
    # Generate comparison plots
    generate_comparison_plots(experiments)
    
    logger.info("\nAll experiments completed! Check 'results/' directory.")

def verify_region_scaling(trainer) -> bool:
    """Check if region scaling hypothesis holds (R² > 0.8)"""
    if len(trainer.region_counts) < 5:
        return False
    
    # Get corresponding alphas
    region_alphas = []
    for i in range(len(trainer.region_counts)):
        alpha_idx = i * trainer.config.region_count_interval // trainer.config.alpha_interval
        if alpha_idx < len(trainer.alphas):
            region_alphas.append(trainer.alphas[alpha_idx])
    
    if len(region_alphas) != len(trainer.region_counts):
        return False
    
    # Check correlation
    depth = trainer.config.n_layer
    x_vals = [depth * (3 - alpha) for alpha in region_alphas]
    y_vals = [np.log(rc + 1) for rc in trainer.region_counts]
    
    from scipy import stats
    _, _, r_value, _, _ = stats.linregress(x_vals, y_vals)
    
    return r_value**2 > 0.8

def generate_comparison_plots(experiments):
    """Generate comparison plots across experiments"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Alpha comparison
    ax = axes[0, 0]
    for exp in experiments:
        exp_type = exp['experiment_type']
        try:
            df = pd.read_csv(f'results/tinygpt_{exp_type}_alphas.csv')
            ax.plot(df['step'], df['mean_alpha'], label=exp_type, marker='o', markersize=3)
        except:
            pass
    ax.axhline(y=2.0, color='k', linestyle='--', alpha=0.5)
    ax.set_xlabel('Step')
    ax.set_ylabel('Mean α')
    ax.set_title('Alpha Evolution Comparison')
    ax.legend()
    
    # Validation loss comparison
    ax = axes[0, 1]
    for exp in experiments:
        exp_type = exp['experiment_type']
        try:
            df = pd.read_csv(f'results/tinygpt_{exp_type}_alphas.csv')
            ax.plot(df['step'], df['val_loss'], label=exp_type, marker='o', markersize=3)
        except:
            pass
    ax.set_xlabel('Step')
    ax.set_ylabel('Validation Loss')
    ax.set_title('Validation Loss Comparison')
    ax.legend()
    ax.set_yscale('log')
    
    # Region count comparison
    ax = axes[1, 0]
    for exp in experiments:
        exp_type = exp['experiment_type']
        try:
            df = pd.read_csv(f'results/tinygpt_{exp_type}_regions.csv')
            ax.plot(df['step'], df['region_count'], label=exp_type, marker='s', markersize=3)
        except:
            pass
    ax.set_xlabel('Step')
    ax.set_ylabel('Region Count')
    ax.set_title('Region Count Evolution')
    ax.legend()
    ax.set_yscale('log')
    
    # Alpha synchrony analysis
    ax = axes[1, 1]
    # Load baseline layer alphas for synchrony analysis
    try:
        with open('results/tinygpt_baseline_results.json', 'r') as f:
            results = json.load(f)
            layer_alphas = np.array(results['layer_alphas'])
            
            if layer_alphas.shape[0] > 0:
                # Calculate synchrony (std dev across layers at each step)
                synchrony = np.std(layer_alphas, axis=1)
                steps = list(range(0, results['final_step'], results['config']['alpha_interval']))
                ax.plot(steps[:len(synchrony)], synchrony)
                ax.set_xlabel('Step')
                ax.set_ylabel('Layer α Std Dev')
                ax.set_title('Layer Synchronization (lower = more synchronized)')
    except:
        pass
    
    plt.tight_layout()
    plt.savefig('results/tinygpt_experiments_comparison.png', dpi=150)
    plt.close()

if __name__ == "__main__":
    main()