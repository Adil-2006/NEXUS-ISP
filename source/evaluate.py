"""Compare Naive / Traditional / Learned / Hybrid ISP under varied noise + illumination, and report efficiency.
   python evaluate.py --data test_imgs --learned learned.pt --hybrid hybrid.pt [--hybrid_small hybrid_small.pt] [--fp16]
"""
import argparse, glob, os, time, csv, numpy as np, cv2, torch
from skimage.metrics import peak_signal_noise_ratio as psnr_fn, structural_similarity as ssim_fn
from skimage.color import rgb2lab, deltaE_ciede2000
import isp_lib as L
from models import load_ckpt, count_params

try:
    import lpips; _lp = lpips.LPIPS(net="alex", verbose=False)
except Exception:
    _lp = None

CONDITIONS = {   # name: (noise a, WB gains r,g,b)
    "low_noise":      (2e-4, None), "mid_noise":  (2e-3, None), "high_noise (low light)": (1e-2, None),
    "warm_light+mid": (2e-3, (2.7, 1, 1.25)), "cool_light+mid": (2e-3, (1.4, 1, 2.4)),
    "warm+high_noise": (1e-2, (2.7, 1, 1.25)),
}

def metrics(pred, gt):
    pred = np.clip(pred, 0, 1)
    m = dict(PSNR=psnr_fn(gt, pred, data_range=1), SSIM=ssim_fn(gt, pred, channel_axis=2, data_range=1),
             dE2000=float(deltaE_ciede2000(rgb2lab(gt), rgb2lab(pred)).mean()))
    if _lp is not None:
        tt = lambda a: torch.from_numpy(a.transpose(2, 0, 1)[None]).float() * 2 - 1
        with torch.no_grad(): m["LPIPS"] = float(_lp(tt(pred), tt(gt)))
    return m

def run_net(model, mode, raw, device, half):
    s = torch.tensor([L.estimate_raw_sigma(raw)]).float().to(device)
    x = L.pack_raw(raw)[None] if mode == "learned" else L.isp_front(raw).transpose(2, 0, 1)[None]
    x = torch.from_numpy(np.ascontiguousarray(x)).float().to(device)
    if half: x, s = x.half(), s.half()
    with torch.no_grad(): y = model(x, s)
    return y.float().clamp(0, 1)[0].permute(1, 2, 0).cpu().numpy()

def efficiency(model, mode, device, half, size=256, runs=20):
    raw = np.random.rand(size, size).astype(np.float32) * .5
    for _ in range(3): run_net(model, mode, raw, device, half)             # warm-up
    if device == "cuda": torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize()
    ts = []
    for _ in range(runs):
        t = time.perf_counter(); run_net(model, mode, raw, device, half)
        if device == "cuda": torch.cuda.synchronize()
        ts.append((time.perf_counter() - t) * 1e3)
    n = count_params(model); bytes_per = 2 if half else 4
    mem = torch.cuda.max_memory_allocated() / 2**20 if device == "cuda" else float("nan")
    t0 = time.perf_counter(); [L.isp_front(raw) for _ in range(5)]; front_ms = (time.perf_counter() - t0) / 5 * 1e3
    return dict(params_K=n / 1e3, size_MB=n * bytes_per / 2**20, latency_ms=float(np.median(ts)),
                peak_gpu_MB=mem, classical_frontend_ms=front_ms if mode == "hybrid" else 0.0)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True); ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--learned"); ap.add_argument("--hybrid"); ap.add_argument("--hybrid_small")
    ap.add_argument("--fp16", action="store_true", help="evaluate nets in half precision (CUDA only)")
    ap.add_argument("--n", type=int, default=20); ap.add_argument("--out", default="results")
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"; half = a.fp16 and dev == "cuda"
    nets = {}
    for name, p in [("learned", a.learned), ("hybrid", a.hybrid), ("hybrid_small", a.hybrid_small)]:
        if p: m, ck = load_ckpt(p, dev); nets[name] = (m.half() if half else m, ck["mode"])
    files = sorted(sum([glob.glob(os.path.join(a.data, e)) for e in ("*.png", "*.jpg", "*.jpeg")], []))[:a.n]
    imgs = []
    for f in files:
        im = cv2.cvtColor(cv2.imread(f), cv2.COLOR_BGR2RGB).astype(np.float32) / 255
        s = a.size / min(im.shape[:2]); im = cv2.resize(im, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s < 1 else im
        h, w = (im.shape[0] // 16) * 16, (im.shape[1] // 16) * 16
        imgs.append(np.ascontiguousarray(im[:h, :w]))
    rows, shown = [], {}
    for cname, (na, wb) in CONDITIONS.items():
        acc = {}
        for i, gt in enumerate(imgs):
            rng = np.random.default_rng(1000 + i)
            raw, meta = L.simulate_raw(gt, rng, noise_a=na, wb=wb)
            outs = {"naive (no denoise)": L.isp_front(raw), "traditional": L.isp_traditional(raw)}
            for n, (m, mode) in nets.items(): outs[n] = run_net(m, mode, raw, dev, half)
            for k, v in outs.items():
                for mk, mv in metrics(v, gt).items(): acc.setdefault((k, mk), []).append(mv)
            if i == 0: shown[cname] = [gt] + [np.clip(v, 0, 1) for v in outs.values()]; names = ["ground truth"] + list(outs)
        for (k, mk), v in acc.items(): rows.append(dict(condition=cname, method=k, metric=mk, value=float(np.mean(v))))
    # ---- print tables
    methods = list(dict.fromkeys(r["method"] for r in rows)); mets = list(dict.fromkeys(r["metric"] for r in rows))
    for mk in mets:
        print(f"\n== {mk} =="); print(f"{'condition':26s}" + "".join(f"{m[:18]:>20s}" for m in methods))
        for c in CONDITIONS:
            print(f"{c:26s}" + "".join(f"{next(r['value'] for r in rows if r['condition']==c and r['method']==m and r['metric']==mk):20.3f}" for m in methods))
    with open(f"{a.out}/quality.csv", "w", newline="") as f:
        w = csv.DictWriter(f, rows[0].keys()); w.writeheader(); w.writerows(rows)
    # ---- efficiency
    print("\n== Efficiency (256x256 RAW, device=%s, fp16=%s) ==" % (dev, half))
    eff = {n: efficiency(m, mode, dev, half) for n, (m, mode) in nets.items()}
    for n, e in eff.items(): print(n, {k: round(v, 2) for k, v in e.items()})
    t0 = time.perf_counter(); r0, _ = L.simulate_raw(imgs[0][:256, :256], np.random.default_rng(0)); [L.isp_traditional(r0) for _ in range(3)]
    print("traditional ISP latency ms (CPU, incl. NLM denoise):", round((time.perf_counter() - t0) / 3 * 1e3, 1))
    with open(f"{a.out}/efficiency.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["model"] + list(next(iter(eff.values())).keys()) if eff else ["model"])
        for n, e in eff.items(): w.writerow([n] + list(e.values()))
    # ---- visual grid: one row per condition
    tile = lambda im, t: cv2.putText(cv2.cvtColor((im * 255).astype(np.uint8), cv2.COLOR_RGB2BGR).copy(), t, (4, 14), 0, .45, (255, 255, 255), 1)
    grid = [np.hstack([tile(cv2.resize(im[:a.size, :a.size], (192, 192)).astype(np.float32), f"{n[:16]}") for im, n in zip(ims, names)]) for ims in shown.values()]
    cv2.imwrite(f"{a.out}/comparison.png", np.vstack(grid)); print(f"\nsaved {a.out}/quality.csv, efficiency.csv, comparison.png")
