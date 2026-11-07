# Hybrid Learned Smartphone ISP (Problem P2)

## Idea
RAW (RGGB) --> [classical front-end: gray-world WB -> demosaic -> CCM -> gamma -> tone]
           --> [small residual U-Net, conditioned on blind noise estimate]  --> RGB

Why hybrid: classical stages are cheap and colour-stable; the network only learns the *correction*
(denoise, detail recovery, fix for WB/CCM errors), so it can be tiny (~50-100K params).

Three systems are compared (+ the degraded "naive" output as the reference for degradation/recovery):
1. Traditional: WB -> demosaic -> CCM -> gamma -> adaptive NLM denoise -> tone -> adaptive unsharp
2. Learned: packed RAW (+ noise estimate, global channel means) -> U-Net -> RGB
3. Hybrid: classical front-end + residual U-Net

Adaptivity: per-image noise estimate (blind, MAD of G1-G2) drives NLM strength, sharpening amount and the
network's noise-map input. Illumination variation is covered by randomized WB gains + CCM jitter in training.

Optimisation: depth-wise separable convs, knowledge distillation (--teacher), FP16 (--fp16).

## Data
No RAW dataset needed to start: `simulate_raw` inverts a camera pipeline (Brooks et al. "unprocessing") on ANY RGB
images, adding Poisson-Gaussian noise, random illuminant and CCM. Use DIV2K / Flickr2K / any 500+ photos.
Optional fine-tuning on real RAW: Zurich RAW-to-RGB (ZRR, PyNET) or SIDD - mention as future work if no time.

## Run
    pip install -r requirements.txt
    python train.py --mode hybrid  --data TRAIN_IMGS --epochs 30 --base 16 --out hybrid.pt
    python train.py --mode learned --data TRAIN_IMGS --epochs 30 --base 16 --out learned.pt
    python train.py --mode hybrid  --data TRAIN_IMGS --epochs 30 --base 8 --teacher hybrid.pt --out hybrid_small.pt
    python evaluate.py --data TEST_IMGS --learned learned.pt --hybrid hybrid.pt --hybrid_small hybrid_small.pt
    python evaluate.py ... --fp16        # on a GPU, for the reduced-precision row

Outputs in results/: quality.csv (PSNR/SSIM/DeltaE2000/LPIPS per condition), efficiency.csv
(params, size, latency, peak memory), comparison.png (visual grid).

## Report checklist
Pipeline diagram - dataset/simulation description - table of 4 methods x 6 conditions x 4 metrics -
degradation (naive) vs recovery (hybrid) - efficiency table (base16 / distilled base8 / fp16) -
failure cases + honest limitations (synthetic RAW, gray-world WB, single-sensor).
