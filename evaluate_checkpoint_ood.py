import argparse
import torch
import numpy as np
from grok_experiments3 import GrokConfig, GrokNet, ModularArithmeticDataset, heavy_tail_alpha

def eval_standard(model, dataset, device, batch_size, num_batches):
    model.eval()
    criterion = torch.nn.CrossEntropyLoss()
    total_loss = total_correct = total_samples = 0
    for _ in range(num_batches):
        x, y = dataset.get_batch(batch_size, split="val")
        x, y = x.to(device), y.to(device)
        with torch.no_grad():
            logits = model(x)
            loss = criterion(logits, y)
        preds = logits.argmax(dim=-1)
        total_loss   += loss.item()
        total_correct+= (preds == y).sum().item()
        total_samples+= y.size(0)
    return total_loss/num_batches, total_correct/total_samples, total_correct, total_samples

def eval_wraparound(model, modulus, device, batch_size, num_batches):
    model.eval()
    criterion = torch.nn.CrossEntropyLoss()
    total_loss = total_correct = total_samples = 0
    for _ in range(num_batches):
        xs, ys = [], []
        while len(xs) < batch_size:
            a = np.random.randint(0, modulus)
            b = np.random.randint(0, modulus)
            if a + b >= modulus:               # only wrap‐around
                xs.append([a, b])
                ys.append((a + b) % modulus)
        x = torch.tensor(xs, dtype=torch.long, device=device)
        y = torch.tensor(ys, dtype=torch.long, device=device)
        with torch.no_grad():
            logits = model(x)
            loss = criterion(logits, y)
        preds = logits.argmax(dim=-1)
        total_loss   += loss.item()
        total_correct+= (preds == y).sum().item()
        total_samples+= y.size(0)
    return total_loss/num_batches, total_correct/total_samples

def eval_edge(model, modulus, device, batch_size, num_batches):
    model.eval()
    criterion = torch.nn.CrossEntropyLoss()
    total_loss = total_correct = total_samples = 0
    half = batch_size // 2
    for _ in range(num_batches):
        # half with a = modulus-1, half with b = modulus-1
        xs, ys = [], []
        bs_rand = np.random.randint(0, modulus, size=half)
        for b in bs_rand:
            xs.append([modulus-1, int(b)])
            ys.append((modulus-1 + b) % modulus)
        as_rand = np.random.randint(0, modulus, size=batch_size - half)
        for a in as_rand:
            xs.append([int(a), modulus-1])
            ys.append((a + modulus-1) % modulus)
        x = torch.tensor(xs, dtype=torch.long, device=device)
        y = torch.tensor(ys, dtype=torch.long, device=device)
        with torch.no_grad():
            logits = model(x)
            loss = criterion(logits, y)
        preds = logits.argmax(dim=-1)
        total_loss   += loss.item()
        total_correct+= (preds == y).sum().item()
        total_samples+= y.size(0)
    return total_loss/num_batches, total_correct/total_samples

def main():
    p = argparse.ArgumentParser(description="Evaluate GrokNet checkpoint (standard + OOD)")
    p.add_argument("--checkpoint",     type=str,   required=True,
                   help="Path to .pt checkpoint file")
    p.add_argument("--modulus",        type=int,   default=97,
                   help="Modular base (default: 97)")
    p.add_argument("--hidden-dim",     type=int,   default=256,
                   help="Hidden dimension (default: 256)")
    p.add_argument("--num-layers",     type=int,   default=4,
                   help="Number of MLP layers (default: 4)")
    p.add_argument("--batch-size",     type=int,   default=512,
                   help="Batch size for each eval segment")
    p.add_argument("--num-val-batches",type=int,   default=100,
                   help="Number of in‐distribution validation batches")
    p.add_argument("--num-ood-batches",type=int,   default=50,
                   help="Number of OOD batches per test")
    p.add_argument("--device",         type=str,   default="cuda" if torch.cuda.is_available() else "cpu",
                   help="Device to run on")
    args = p.parse_args()

    device = torch.device(args.device)
    # rebuild & load
    model = GrokNet(modulus=args.modulus,
                    hidden_dim=args.hidden_dim,
                    num_layers=args.num_layers).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state'])
    print(f">>> Loaded '{args.checkpoint}' (step {ckpt.get('step')}, α≈{ckpt.get('alpha'):.3f})\n")

    # in‐distribution
    dataset = ModularArithmeticDataset(args.modulus)
    val_loss, val_acc, val_corr, val_tot = eval_standard(
        model, dataset, device, args.batch_size, args.num_val_batches
    )
    print("=== In-Distribution Validation ===")
    print(f" avg loss : {val_loss:.6f}")
    print(f" accuracy : {val_acc:.4%} ({val_corr}/{val_tot})\n")

    # heavy-tail
    layer_alphas, mean_alpha = heavy_tail_alpha(model)
    print("=== Heavy-tail α Analysis ===")
    print(f" layer α’s : {layer_alphas}")
    print(f" mean  α  : {mean_alpha:.3f}\n")

    # OOD: wrap-around
    wa_loss, wa_acc = eval_wraparound(
        model, args.modulus, device, args.batch_size, args.num_ood_batches
    )
    print("=== OOD Test: Wrap-Around (a+b ≥ modulus) ===")
    print(f" avg loss : {wa_loss:.6f}")
    print(f" accuracy : {wa_acc:.4%}\n")

    # OOD: edge
    ed_loss, ed_acc = eval_edge(
        model, args.modulus, device, args.batch_size, args.num_ood_batches
    )
    print("=== OOD Test: Edge Cases (a or b = modulus-1) ===")
    print(f" avg loss : {ed_loss:.6f}")
    print(f" accuracy : {ed_acc:.4%}\n")

    # sample a few wrap-around predictions
    print("Sample Wrap-Around Predictions:")
    xs, ys = [], []
    while len(xs) < 10:
        a = np.random.randint(0, args.modulus)
        b = np.random.randint(0, args.modulus)
        if a + b >= args.modulus:
            xs.append([a, b])
            ys.append((a + b) % args.modulus)
    x_ood = torch.tensor(xs, dtype=torch.long, device=device)
    with torch.no_grad():
        preds = model(x_ood).argmax(dim=-1).cpu().tolist()
    for (a,b), true, pred in zip(xs, ys, preds):
        print(f"  {a} + {b} mod {args.modulus} = {pred}  (true: {true})")

if __name__ == "__main__":
    main()
