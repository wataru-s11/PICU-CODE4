# train_white_line_classifier.py
import os, json, argparse, torch, torch.nn as nn, torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, WeightedRandomSampler
import numpy as np

def build_loaders(data_dir, img_h=128, img_w=512, bs=64):
    weights = models.MobileNet_V3_Small_Weights.DEFAULT
    norm = weights.transforms()
    # val/test: 純粋にリサイズ＋正規化
    val_tf = transforms.Compose([transforms.Resize((img_h, img_w)), transforms.ToTensor(),
                                 transforms.Normalize(norm.mean, norm.std)])
    # train: 軽めの増強
    train_tf = transforms.Compose([
        transforms.Resize((img_h, img_w)),
        transforms.ColorJitter(brightness=0.25, contrast=0.25),
        transforms.RandomApply([transforms.GaussianBlur(3)], p=0.3),
        transforms.RandomAdjustSharpness(1.5, p=0.3),
        transforms.RandomAffine(degrees=0, translate=(0.05, 0.02)),  # 水平±5%, 垂直±2%
        transforms.ToTensor(),
        transforms.Normalize(norm.mean, norm.std),
    ])

    train_ds = datasets.ImageFolder(os.path.join(data_dir, "train"), transform=train_tf)
    val_ds   = datasets.ImageFolder(os.path.join(data_dir, "val"),   transform=val_tf)

    # クラス不均衡対策（yes/no の枚数差を吸収）
    counts = np.bincount([y for _, y in train_ds.samples])
    weights_cls = 1.0 / np.clip(counts, 1, None)
    sample_weights = [weights_cls[y] for _, y in train_ds.samples]
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

    train_loader = DataLoader(train_ds, batch_size=bs, sampler=sampler, num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_ds, batch_size=bs*2, shuffle=False, num_workers=2, pin_memory=True)
    return train_loader, val_loader, train_ds.classes  # 例: ['no','yes']

def select_threshold(probs, labels, target_recall=0.98):
    """valでターゲット再現率(Recall)を満たす最小閾値を返す"""
    # しきい候補（確率の降順ユニーク）
    order = np.argsort(-probs)
    best_th = 0.5
    y = labels.astype(int)
    P = y.sum()
    tp = 0; fp = 0
    # 1つずつ陽性にしていくスイープ
    for i, idx in enumerate(order):
        tp += y[idx]
        fp += (1 - y[idx])
        rec = tp / (P + 1e-6)
        th  = probs[idx]  # この確率を境にすると、ここまでが陽性
        if rec >= target_recall:
            best_th = th
            break
    return float(best_th)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=r"C:\Users\sakai\OneDrive\Desktop\BOT\spon\data")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--img_h", type=int, default=128)
    ap.add_argument("--img_w", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--target_recall", type=float, default=0.98)
    ap.add_argument("--outdir", default="models")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    train_loader, val_loader, classes = build_loaders(args.data, args.img_h, args.img_w, args.bs)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    weights = models.MobileNet_V3_Small_Weights.DEFAULT
    model = models.mobilenet_v3_small(weights=weights)
    model.classifier[3] = nn.Linear(model.classifier[3].in_features, 1)  # 2値
    model.to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    def run_epoch(loader, train=True):
        model.train(train)
        tot_loss, n = 0.0, 0
        all_probs, all_labels = [], []
        for x, y in loader:
            x = x.to(device); y = y.float().to(device)
            if train: optimizer.zero_grad()
            logit = model(x).squeeze(1)
            loss = criterion(logit, y)
            if train:
                loss.backward(); optimizer.step()
            tot_loss += loss.item() * x.size(0); n += x.size(0)
            with torch.no_grad():
                all_probs.append(torch.sigmoid(logit).detach().cpu().numpy())
                all_labels.append(y.detach().cpu().numpy())
        probs = np.concatenate(all_probs); labels = np.concatenate(all_labels)
        # 仮の0.5で指標（参考）
        pred = (probs >= 0.5).astype(int)
        tp = ((pred==1)&(labels==1)).sum(); fp = ((pred==1)&(labels==0)).sum(); fn = ((pred==0)&(labels==1)).sum()
        prec = tp/(tp+fp+1e-6); rec = tp/(tp+fn+1e-6)
        return tot_loss/n, prec, rec, probs, labels

    best_f1 = -1
    best_state = None
    best_thr = 0.5

    for ep in range(args.epochs):
        tr_loss, tr_p, tr_r, *_ = run_epoch(train_loader, True)
        vl_loss, vl_p, vl_r, vl_probs, vl_labels = run_epoch(val_loader, False)
        scheduler.step()
        f1 = 2*vl_p*vl_r/(vl_p+vl_r+1e-6)
        print(f"[{ep:02d}]  train_loss {tr_loss:.4f} | val_loss {vl_loss:.4f}  P {vl_p:.3f} R {vl_r:.3f} F1 {f1:.3f}")

        if f1 > best_f1:
            best_f1 = f1
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
            best_thr = select_threshold(vl_probs, vl_labels, args.target_recall)

    # 保存
    pt_path = os.path.join(args.outdir, "white_line_cls.pt")
    json_path = os.path.join(args.outdir, "white_line_cls.meta.json")
    torch.save(best_state, pt_path)
    meta = {"classes": classes, "img_h": args.img_h, "img_w": args.img_w,
            "threshold": best_thr, "target_recall": args.target_recall}
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"saved: {pt_path}\nmeta: {json_path}\nthreshold@recall≈{args.target_recall}: {best_thr:.3f}")

if __name__ == "__main__":
    main()
