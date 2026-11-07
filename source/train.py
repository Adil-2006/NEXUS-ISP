"""Train / fine-tune a learned or hybrid ISP on synthetic RAW generated on the fly from any folder of RGB images.
   python train.py --mode hybrid  --data DIV2K_train --epochs 30 --out hybrid.pt
   python train.py --mode learned --data DIV2K_train --epochs 30 --out learned.pt
   # knowledge distillation: big teacher -> small student
   python train.py --mode hybrid --base 8 --teacher hybrid_big.pt --out hybrid_small.pt
"""
import argparse, glob, os, numpy as np, cv2, torch, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import isp_lib as L
from models import build, load_ckpt, count_params


class SynthISP(Dataset):
    def __init__(self, files, crop=128, repeat=4, val=False):
        self.files, self.crop, self.repeat, self.val = files, crop, (1 if val else repeat), val
    def __len__(self): return len(self.files) * self.repeat
    def __getitem__(self, i):
        rng = np.random.default_rng(i if self.val else None)
        img = cv2.cvtColor(cv2.imread(self.files[i % len(self.files)]), cv2.COLOR_BGR2RGB).astype(np.float32) / 255
        c = self.crop
        if min(img.shape[:2]) < c:
            s = c / min(img.shape[:2]); img = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_CUBIC)
        h, w = img.shape[:2]
        y = (h - c) // 2 if self.val else int(rng.integers(0, h - c + 1)); x = (w - c) // 2 if self.val else int(rng.integers(0, w - c + 1))
        img = np.ascontiguousarray(img[y:y + c, x:x + c])
        if not self.val:
            if rng.random() < .5: img = img[:, ::-1]
            if rng.random() < .5: img = img[::-1]
            img = np.ascontiguousarray(img)
        raw, meta = L.simulate_raw(img, rng)
        t = lambda a: torch.from_numpy(np.ascontiguousarray(a.transpose(2, 0, 1) if a.ndim == 3 else a)).float()
        return dict(packed=t(L.pack_raw(raw).transpose(1, 2, 0)), front=t(L.isp_front(raw)),
                    sigma=torch.tensor(L.estimate_raw_sigma(raw)), target=t(img))


def forward(model, mode, b, dev):
    sig = b["sigma"].to(dev)
    return model(b["packed" if mode == "learned" else "front"].to(dev), sig)

def grad_l1(a, b):
    return F.l1_loss(a[..., :, 1:] - a[..., :, :-1], b[..., :, 1:] - b[..., :, :-1]) + \
           F.l1_loss(a[..., 1:, :] - a[..., :-1, :], b[..., 1:, :] - b[..., :-1, :])

def psnr_t(a, b): return (-10 * torch.log10(F.mse_loss(a.clamp(0, 1), b) + 1e-10)).item()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["learned", "hybrid"], required=True)
    ap.add_argument("--data", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=20); ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--crop", type=int, default=128); ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--base", type=int, default=16); ap.add_argument("--depth", type=int, default=3)
    ap.add_argument("--repeat", type=int, default=4); ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--teacher", default=None, help="checkpoint of a larger model => knowledge distillation")
    ap.add_argument("--alpha", type=float, default=0.5, help="weight of distillation loss")
    a = ap.parse_args()
    files = sorted(sum([glob.glob(os.path.join(a.data, "**", e), recursive=True) for e in ("*.png", "*.jpg", "*.jpeg")], []))
    assert files, "no images found"
    nval = max(1, len(files) // 20); tr, va = files[nval:], files[:nval]
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = build(a.mode, a.base, a.depth).to(dev)
    print(f"{a.mode}: {count_params(model)/1e3:.1f}K params, {len(tr)} train imgs, device {dev}")
    teacher = None
    if a.teacher:
        teacher, tck = load_ckpt(a.teacher, dev); assert tck["mode"] == a.mode
        print(f"distilling from teacher with {count_params(teacher)/1e3:.1f}K params")
    dl = DataLoader(SynthISP(tr, a.crop, a.repeat), a.bs, shuffle=True, num_workers=a.workers, drop_last=True)
    vl = DataLoader(SynthISP(va, a.crop, val=True), a.bs, num_workers=0)
    opt = torch.optim.Adam(model.parameters(), a.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.epochs * len(dl))
    for ep in range(a.epochs):
        model.train(); tot = 0
        for b in dl:
            pred, tgt = forward(model, a.mode, b, dev), b["target"].to(dev)
            loss = F.l1_loss(pred, tgt) + 0.2 * grad_l1(pred, tgt)
            if teacher is not None:
                with torch.no_grad(): tp = forward(teacher, a.mode, b, dev)
                loss = (1 - a.alpha) * loss + a.alpha * F.l1_loss(pred, tp)
            opt.zero_grad(); loss.backward(); opt.step(); sched.step(); tot += loss.item()
        model.eval()
        with torch.no_grad(): vp = np.mean([psnr_t(forward(model, a.mode, b, dev), b["target"].to(dev)) for b in vl])
        print(f"epoch {ep+1}/{a.epochs} loss {tot/len(dl):.4f} val PSNR {vp:.2f}", flush=True)
        torch.save(dict(mode=a.mode, base=a.base, depth=a.depth, sep=True, state=model.state_dict()), a.out)
