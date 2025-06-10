import argparse
import torch
import numpy as np
from grok_experiments3 import GrokConfig, GrokNet, ModularArithmeticDataset, heavy_tail_alpha

def evaluate_checkpoint(args):
    device = torch.device(args.device)
    
    # 1. Reconstruct model
    model = GrokNet(
        modulus=args.modulus,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers
    ).to(device)
    
    # 2. Load checkpoint
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt['model_state'])
    step  = ckpt.get('step', None)
    alpha = ckpt.get('alpha', None)
    print(f">>> Loaded checkpoint '{args.checkpoint}' at step {step}, α ≈ {alpha:.3f}\n")
    
    # 3. Prepare dataset
    dataset = ModularArithmeticDataset(args.modulus)
    criterion = torch.nn.CrossEntropyLoss()
    
    # 4. Validation evaluation
    model.eval()
    total_loss, total_correct, total_samples = 0.0, 0, 0
    for batch_idx in range(args.num_val_batches):
        x, y = dataset.get_batch(args.batch_size, split="val")
        x, y = x.to(device), y.to(device)
        with torch.no_grad():
            logits = model(x)
            loss   = criterion(logits, y)
        preds = logits.argmax(dim=-1)
        
        total_loss   += loss.item()
        total_correct+= (preds == y).sum().item()
        total_samples+= y.size(0)
    
    avg_loss  = total_loss / args.num_val_batches
    accuracy  = total_correct / total_samples
    print(f"Validation   batches: {args.num_val_batches}")
    print(f"Average Loss : {avg_loss:.6f}")
    print(f"Accuracy     : {accuracy:.4%} ({total_correct}/{total_samples})\n")
    
    # 5. Heavy‐tail α analysis
    layer_alphas, mean_alpha = heavy_tail_alpha(model)
    print("Heavy-tail α analysis:")
    print(f"  Layer α’s : {layer_alphas}")
    print(f"  Mean   α  : {mean_alpha:.3f}\n")
    
    # 6. Sample predictions
    print("Sample predictions (first 10 of a fresh validation batch):")
    x_s, y_s = dataset.get_batch(args.batch_size, split="val")
    with torch.no_grad():
        logits = model(x_s.to(device))
    preds = logits.argmax(dim=-1).cpu()
    for i in range(min(10, len(y_s))):
        a, b = x_s[i].tolist()
        print(f"  {a} + {b} ≡ {preds[i].item()}  (true: {y_s[i].item()})")

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Evaluate a GrokNet checkpoint")
    p.add_argument("--checkpoint",     type=str,   required=True,
                   help="Path to .pt checkpoint file")
    p.add_argument("--modulus",        type=int,   default=97,
                   help="Modular base (default: 97)")
    p.add_argument("--hidden-dim",     type=int,   default=256,
                   help="Hidden dimension (default: 256)")
    p.add_argument("--num-layers",     type=int,   default=4,
                   help="Number of MLP layers (default: 4)")
    p.add_argument("--batch-size",     type=int,   default=512,
                   help="Batch size for eval (default: 512)")
    p.add_argument("--num-val-batches",type=int,   default=100,
                   help="How many validation batches to average over")
    p.add_argument("--device",         type=str,   default="cuda" if torch.cuda.is_available() else "cpu",
                   help="Device to run on (default: cuda if available)")
    args = p.parse_args()
    evaluate_checkpoint(args)
