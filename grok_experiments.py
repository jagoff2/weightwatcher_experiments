"""
grok_experiments.py - Heavy-tail experiments on modular arithmetic grokking
Tests hypotheses 2-4: anti-grokking collapse, micro-step rescue, and widening rescue
"""

import torch
import torch.nn as nn
import torch.optim as optim
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
class GrokConfig:
    """Configuration for grokking experiments"""
    modulus: int = 97
    hidden_dim: int = 256
    num_layers: int = 4
    batch_size: int = 512
    lr: float = 1e-3
    max_steps: int = 50000
    alpha_interval: int = 250
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    use_8bit: bool = False
    experiment_type: str = "baseline"  # baseline, microstep, widen
    
class ModularArithmeticDataset:
    """Dataset for modular addition task"""
    def __init__(self, modulus: int):
        self.modulus = modulus
        self.data = []
        for a in range(modulus):
            for b in range(modulus):
                self.data.append((a, b, (a + b) % modulus))
        self.data = torch.tensor(self.data)
        
        # Split into train/val (90/10)
        n = len(self.data)
        perm = torch.randperm(n)
        split = int(0.9 * n)
        self.train_data = self.data[perm[:split]]
        self.val_data = self.data[perm[split:]]
    
    def get_batch(self, batch_size: int, split: str = "train"):
        data = self.train_data if split == "train" else self.val_data
        idx = torch.randint(0, len(data), (batch_size,))
        batch = data[idx]
        return batch[:, :2], batch[:, 2]

class GrokNet(nn.Module):
    """MLP for modular arithmetic with configurable depth and width"""
    def __init__(self, modulus: int, hidden_dim: int, num_layers: int):
        super().__init__()
        self.modulus = modulus
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Input embeddings
        self.embed_a = nn.Embedding(modulus, hidden_dim // 2)
        self.embed_b = nn.Embedding(modulus, hidden_dim // 2)
        
        # MLP layers
        layers = []
        for i in range(num_layers):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
        self.mlp = nn.Sequential(*layers)
        
        # Output projection
        self.output = nn.Linear(hidden_dim, modulus)
        
        # Initialize weights with small random values
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.5)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)
    
    def forward(self, x):
        a, b = x[:, 0], x[:, 1]
        a_emb = self.embed_a(a)
        b_emb = self.embed_b(b)
        h = torch.cat([a_emb, b_emb], dim=-1)
        h = self.mlp(h)
        return self.output(h)

def heavy_tail_alpha(model: nn.Module) -> Tuple[List[float], float]:
    """
    Compute layer-wise alpha values using WeightWatcher
    Returns: (layer_alphas, mean_alpha)
    """
    try:
        # WeightWatcher analysis
        watcher = ww.WeightWatcher(model=model)
        details = watcher.analyze()
        
        # Extract alpha values for each layer
        alphas = []
        for _, row in details.iterrows():
            if 'alpha' in row and not pd.isna(row['alpha']):
                alphas.append(float(row['alpha']))
        
        mean_alpha = np.mean(alphas) if alphas else 3.0
        return alphas, mean_alpha
    except Exception as e:
        logger.warning(f"WeightWatcher analysis failed: {e}")
        # Fallback: return dummy values
        return [3.0] * 4, 3.0

def widen_layer(model: GrokNet, factor: float = 1.25, layers: str = "mlp") -> GrokNet:
    """
    Net2Net-style widening of MLP layers
    Duplicates neurons, adds noise, rescales weights
    """
    with torch.no_grad():
        for i, module in enumerate(model.mlp):
            if isinstance(module, nn.Linear) and layers == "mlp":
                old_in = module.in_features
                old_out = module.out_features
                new_out = int(old_out * factor)
                
                # Create new layer
                new_layer = nn.Linear(old_in, new_out).to(module.weight.device)
                
                # Copy and duplicate weights
                old_weight = module.weight.data
                indices = torch.randperm(old_out)[:new_out - old_out]
                
                new_layer.weight.data[:old_out] = old_weight
                new_layer.weight.data[old_out:] = old_weight[indices] + \
                    torch.randn_like(old_weight[indices]) * 0.01
                
                # Rescale by 1/sqrt(k)
                k = new_out / old_out
                new_layer.weight.data *= 1.0 / np.sqrt(k)
                
                if module.bias is not None:
                    new_layer.bias.data[:old_out] = module.bias.data
                    new_layer.bias.data[old_out:] = module.bias.data[indices]
                
                # Replace in model
                model.mlp[i] = new_layer
                
                # Update next layer's input dimension if exists
                if i + 2 < len(model.mlp) and isinstance(model.mlp[i + 2], nn.Linear):
                    next_layer = model.mlp[i + 2]
                    new_next = nn.Linear(new_out, next_layer.out_features).to(next_layer.weight.device)
                    
                    # Duplicate input connections
                    old_weight = next_layer.weight.data
                    new_next.weight.data[:, :old_out] = old_weight
                    new_next.weight.data[:, old_out:] = old_weight[:, indices]
                    new_next.bias.data = next_layer.bias.data
                    
                    model.mlp[i + 2] = new_next
    
    model.hidden_dim = int(model.hidden_dim * factor)
    return model

class AlphaAwareTrainer:
    """Training loop with alpha monitoring and adaptive learning rate"""
    def __init__(self, model: GrokNet, dataset: ModularArithmeticDataset, config: GrokConfig):
        self.model = model.to(config.device)
        self.dataset = dataset
        self.config = config
        self.device = config.device
        
        # Optimizer setup
        if config.use_8bit:
            try:
                import bitsandbytes as bnb
                self.optimizer = bnb.optim.Adam8bit(model.parameters(), lr=config.lr)
            except ImportError:
                logger.warning("bitsandbytes not available, using standard Adam")
                self.optimizer = optim.Adam(model.parameters(), lr=config.lr)
        else:
            self.optimizer = optim.Adam(model.parameters(), lr=config.lr)
        
        self.criterion = nn.CrossEntropyLoss()
        
        # Tracking variables
        self.step = 0
        self.train_losses = []
        self.val_losses = []
        self.alphas = []
        self.layer_alphas = []
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
        x, y = self.dataset.get_batch(self.config.batch_size, "train")
        x, y = x.to(self.device), y.to(self.device)
        
        self.optimizer.zero_grad()
        logits = self.model(x)
        loss = self.criterion(logits, y)
        loss.backward()
        
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        
        self.optimizer.step()
        return loss.item()
    
    def evaluate(self) -> Tuple[float, float]:
        """Evaluate on validation set"""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for _ in range(10):  # Sample validation
                x, y = self.dataset.get_batch(self.config.batch_size, "val")
                x, y = x.to(self.device), y.to(self.device)
                logits = self.model(x)
                loss = self.criterion(logits, y)
                total_loss += loss.item()
                
                pred = logits.argmax(dim=-1)
                correct += (pred == y).sum().item()
                total += y.size(0)
        
        return total_loss / 10, correct / total
    
    def check_alpha_and_adapt(self):
        """Monitor alpha and adapt training strategy"""
        layer_alphas, mean_alpha = heavy_tail_alpha(self.model)
        self.alphas.append(mean_alpha)
        self.layer_alphas.append(layer_alphas)
        
        # Checkpoint if alpha in critical range
        if 2.05 <= mean_alpha <= 2.15 and not self.checkpoint_saved:
            torch.save({
                'model_state': self.model.state_dict(),
                'optimizer_state': self.optimizer.state_dict(),
                'step': self.step,
                'alpha': mean_alpha
            }, f'results/checkpoint_alpha_{mean_alpha:.3f}_step_{self.step}.pt')
            self.checkpoint_saved = True
            logger.info(f"Checkpoint saved at α={mean_alpha:.3f}")
        
        # Micro-stepping logic
        if self.config.experiment_type == "microstep":
            if mean_alpha < 2.30 and not self.micro_step_mode:
                self.micro_step_mode = True
                logger.info(f"Entering micro-step mode at α={mean_alpha:.3f}")
            
            if self.micro_step_mode and mean_alpha < self.last_alpha - 0.02:
                # Reduce LR by 25%
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] *= 0.75
                logger.info(f"Reduced LR to {param_group['lr']:.6f} at α={mean_alpha:.3f}")
        
        # Check for collapse
        val_loss, val_acc = self.evaluate()
        if len(self.val_losses) > 0 and val_loss > self.val_losses[-1]:
            self.val_loss_increases += 1
        else:
            self.val_loss_increases = 0
        
        if mean_alpha < 2.00 and self.val_loss_increases >= 2:
            self.collapse_detected = True
            logger.warning(f"Collapse detected at step {self.step}: α={mean_alpha:.3f}, val_loss={val_loss:.4f}")
            
            # Apply widening if configured
            if self.config.experiment_type == "widen" and not self.widened:
                logger.info("Applying widening rescue...")
                self.model = widen_layer(self.model, factor=1.25)
                self.widened = True
                
                # Reinitialize optimizer for new parameters
                if self.config.use_8bit:
                    try:
                        import bitsandbytes as bnb
                        self.optimizer = bnb.optim.Adam8bit(self.model.parameters(), 
                                                           lr=self.optimizer.param_groups[0]['lr'])
                    except ImportError:
                        self.optimizer = optim.Adam(self.model.parameters(), 
                                                  lr=self.optimizer.param_groups[0]['lr'])
                else:
                    self.optimizer = optim.Adam(self.model.parameters(), 
                                              lr=self.optimizer.param_groups[0]['lr'])
        
        self.last_alpha = mean_alpha
        self.val_losses.append(val_loss)
    
    def run(self):
        """Main training loop"""
        logger.info(f"Starting {self.config.experiment_type} experiment on {self.device}")
        
        for step in range(self.config.max_steps):
            self.step = step
            
            # Training step
            train_loss = self.train_step()
            self.train_losses.append(train_loss)
            self.lr_history.append(self.optimizer.param_groups[0]['lr'])
            
            # Periodic evaluation and alpha monitoring
            if step % self.config.alpha_interval == 0:
                val_loss, val_acc = self.evaluate()
                self.check_alpha_and_adapt()
                
                logger.info(f"Step {step}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, "
                          f"val_acc={val_acc:.3f}, α={self.alphas[-1]:.3f}")
                
                # Early stopping on collapse (unless widening experiment)
                if self.collapse_detected and self.config.experiment_type != "widen":
                    logger.info("Stopping due to collapse")
                    break
        
        # Save final results
        self.save_results()
    
    def save_results(self):
        """Save experiment results to disk"""
        results = {
            'config': asdict(self.config),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'alphas': self.alphas,
            'layer_alphas': self.layer_alphas,
            'lr_history': self.lr_history,
            'collapse_detected': self.collapse_detected,
            'final_step': self.step
        }
        
        # Save JSON results
        with open(f'results/grok_{self.config.experiment_type}_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        # Save CSV for easy analysis
        df = pd.DataFrame({
            'step': range(len(self.train_losses)),
            'train_loss': self.train_losses,
            'lr': self.lr_history
        })
        
        # Add alpha values at corresponding steps
        alpha_steps = list(range(0, len(self.train_losses), self.config.alpha_interval))
        alpha_df = pd.DataFrame({
            'step': alpha_steps[:len(self.alphas)],
            'mean_alpha': self.alphas,
            'val_loss': self.val_losses
        })
        
        df.to_csv(f'results/grok_{self.config.experiment_type}_training.csv', index=False)
        alpha_df.to_csv(f'results/grok_{self.config.experiment_type}_alphas.csv', index=False)
        
        # Generate plots
        self.plot_results()
    
    def plot_results(self):
        """Generate visualization plots"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # 1. Loss curves
        ax = axes[0, 0]
        ax.plot(self.train_losses, label='Train Loss', alpha=0.7)
        val_steps = list(range(0, len(self.train_losses), self.config.alpha_interval))
        ax.plot(val_steps[:len(self.val_losses)], self.val_losses, 
                label='Val Loss', marker='o', markersize=3)
        ax.set_xlabel('Step')
        ax.set_ylabel('Loss')
        ax.set_title('Training Dynamics')
        ax.legend()
        ax.set_yscale('log')
        
        # 2. Alpha evolution
        ax = axes[0, 1]
        alpha_steps = list(range(0, len(self.train_losses), self.config.alpha_interval))
        ax.plot(alpha_steps[:len(self.alphas)], self.alphas, 
                marker='o', color='red', markersize=4)
        ax.axhline(y=2.0, color='k', linestyle='--', alpha=0.5, label='α=2 threshold')
        ax.set_xlabel('Step')
        ax.set_ylabel('Mean α')
        ax.set_title('Heavy-tail Exponent Evolution')
        ax.legend()
        
        # 3. Layer-wise alpha synchrony
        ax = axes[1, 0]
        if self.layer_alphas:
            layer_alphas_array = np.array(self.layer_alphas)
            for i in range(layer_alphas_array.shape[1]):
                ax.plot(alpha_steps[:len(self.alphas)], layer_alphas_array[:, i], 
                       alpha=0.7, label=f'Layer {i+1}')
            ax.set_xlabel('Step')
            ax.set_ylabel('Layer α')
            ax.set_title('Layer-wise α Synchronization')
            ax.legend()
        
        # 4. Learning rate schedule
        ax = axes[1, 1]
        ax.plot(self.lr_history)
        ax.set_xlabel('Step')
        ax.set_ylabel('Learning Rate')
        ax.set_title('Learning Rate Schedule')
        ax.set_yscale('log')
        
        plt.tight_layout()
        plt.savefig(f'results/grok_{self.config.experiment_type}_plots.png', dpi=150)
        plt.close()
        
        # Additional plot for collapse/recovery if applicable
        if self.collapse_detected:
            fig, ax = plt.subplots(figsize=(8, 6))
            
            # Mark collapse point
            collapse_step = self.step
            ax.axvline(x=collapse_step, color='red', linestyle='--', 
                      alpha=0.7, label='Collapse')
            
            if self.widened:
                ax.axvline(x=collapse_step, color='green', linestyle='--', 
                          alpha=0.7, label='Widening Applied')
            
            ax.plot(self.train_losses, label='Train Loss')
            ax.plot(val_steps[:len(self.val_losses)], self.val_losses, 
                    label='Val Loss', marker='o', markersize=3)
            
            ax2 = ax.twinx()
            ax2.plot(alpha_steps[:len(self.alphas)], self.alphas, 
                    color='orange', label='Mean α')
            ax2.set_ylabel('α', color='orange')
            
            ax.set_xlabel('Step')
            ax.set_ylabel('Loss')
            ax.set_title('Anti-grokking Collapse and Recovery')
            ax.legend(loc='upper left')
            ax2.legend(loc='upper right')
            
            plt.savefig(f'results/grok_{self.config.experiment_type}_collapse.png', dpi=150)
            plt.close()

def experiment_runner(cfg: Dict):
    """Run a complete experiment based on configuration"""
    config = GrokConfig(**cfg)
    
    # Create model and dataset
    model = GrokNet(config.modulus, config.hidden_dim, config.num_layers)
    dataset = ModularArithmeticDataset(config.modulus)
    
    # Run training
    trainer = AlphaAwareTrainer(model, dataset, config)
    trainer.run()
    
    return trainer

def main():
    """Run all grokking experiments"""
    # Define experiment configurations
    experiments = [
        {"experiment_type": "baseline", "lr": 1e-3, "max_steps": 30000},
        {"experiment_type": "microstep", "lr": 1e-3, "max_steps": 50000},
        {"experiment_type": "widen", "lr": 1e-3, "max_steps": 40000}
    ]
    
    results_summary = {}
    
    for exp_config in experiments:
        logger.info(f"\n{'='*50}")
        logger.info(f"Starting {exp_config['experiment_type']} experiment")
        logger.info(f"{'='*50}\n")
        
        trainer = experiment_runner(exp_config)
        
        # Collect summary statistics
        results_summary[exp_config['experiment_type']] = {
            'final_train_loss': trainer.train_losses[-1] if trainer.train_losses else None,
            'final_val_loss': trainer.val_losses[-1] if trainer.val_losses else None,
            'final_alpha': trainer.alphas[-1] if trainer.alphas else None,
            'collapse_detected': trainer.collapse_detected,
            'steps_to_collapse': trainer.step if trainer.collapse_detected else None,
            'widened': trainer.widened
        }
    
    # Save summary
    with open('results/grok_experiments_summary.json', 'w') as f:
        json.dump(results_summary, f, indent=2)
    
    # Generate comparison plots
    generate_comparison_plots(experiments)
    
    logger.info("\nAll experiments completed! Results saved to 'results/' directory.")

def generate_comparison_plots(experiments):
    """Generate plots comparing all experiments"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Load and plot alpha curves
    ax = axes[0]
    for exp in experiments:
        exp_type = exp['experiment_type']
        try:
            df = pd.read_csv(f'results/grok_{exp_type}_alphas.csv')
            ax.plot(df['step'], df['mean_alpha'], label=exp_type, marker='o', markersize=3)
        except:
            pass
    
    ax.axhline(y=2.0, color='k', linestyle='--', alpha=0.5)
    ax.set_xlabel('Step')
    ax.set_ylabel('Mean α')
    ax.set_title('Alpha Evolution Comparison')
    ax.legend()
    
    # Load and plot validation losses
    ax = axes[1]
    for exp in experiments:
        exp_type = exp['experiment_type']
        try:
            df = pd.read_csv(f'results/grok_{exp_type}_alphas.csv')
            ax.plot(df['step'], df['val_loss'], label=exp_type, marker='o', markersize=3)
        except:
            pass
    
    ax.set_xlabel('Step')
    ax.set_ylabel('Validation Loss')
    ax.set_title('Validation Loss Comparison')
    ax.legend()
    ax.set_yscale('log')
    
    plt.tight_layout()
    plt.savefig('results/grok_experiments_comparison.png', dpi=150)
    plt.close()

if __name__ == "__main__":
    main()