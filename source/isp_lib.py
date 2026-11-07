"""Classical ISP stages + RAW degradation simulator (numpy / OpenCV only, no torch)."""
import numpy as np, cv2

# Nominal camera->sRGB colour correction matrix (rows sum to 1 => neutral stays neutral)
NOMINAL_CCM = np.array([[1.80, -0.60, -0.20],
                        [-0.25, 1.55, -0.30],
                        [0.02, -0.55, 1.53]], np.float32)
BAYER_CODE = cv2.COLOR_BAYER_BG2RGB_EA   # OpenCV's naming is "swapped": this decodes an RGGB mosaic


# ---------------- colour / tone helpers ----------------
def srgb_to_linear(x):
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4).astype(np.float32)

def linear_to_srgb(x):
    x = np.clip(x, 0, 1)
    return np.where(x <= 0.0031308, 12.92 * x, 1.055 * x ** (1 / 2.4) - 0.055).astype(np.float32)

def tone_curve(x):                       # smoothstep S-curve (mild contrast)
    x = np.clip(x, 0, 1); return (3 * x * x - 2 * x ** 3).astype(np.float32)

def inv_tone_curve(y):
    y = np.clip(y, 0, 1); return (0.5 - np.sin(np.arcsin(1 - 2 * y) / 3)).astype(np.float32)


# ---------------- RAW simulator ("unprocessing", Brooks et al. 2019 style) ----------------
def mosaic_rggb(rgb):
    h, w, _ = rgb.shape
    raw = np.zeros((h, w), np.float32)
    raw[0::2, 0::2] = rgb[0::2, 0::2, 0]; raw[0::2, 1::2] = rgb[0::2, 1::2, 1]
    raw[1::2, 0::2] = rgb[1::2, 0::2, 1]; raw[1::2, 1::2] = rgb[1::2, 1::2, 2]
    return raw

def simulate_raw(srgb, rng, noise_a=None, wb=None, ccm_jitter=0.08):
    """srgb: HxWx3 float [0,1], H and W even. Returns noisy RGGB raw (HxW) + metadata.
    Poisson-Gaussian noise: var = a*x + b, b = 2a^2. Illuminant = per-channel WB gains."""
    lin = srgb_to_linear(inv_tone_curve(srgb))
    ccm = NOMINAL_CCM + rng.uniform(-ccm_jitter, ccm_jitter, (3, 3)).astype(np.float32)
    ccm /= ccm.sum(1, keepdims=True)
    cam = np.clip(lin @ np.linalg.inv(ccm).T.astype(np.float32), 0, 1)
    if wb is None:
        wb = (rng.uniform(1.4, 2.8), 1.0, rng.uniform(1.2, 2.4))
    gains = np.array(wb, np.float32)
    raw = mosaic_rggb(np.clip(cam / gains, 0, 1))
    if noise_a is None:
        noise_a = float(np.exp(rng.uniform(np.log(1e-4), np.log(1.2e-2))))
    b = 2 * noise_a ** 2 + 1e-6
    raw = raw + rng.standard_normal(raw.shape).astype(np.float32) * np.sqrt(noise_a * raw + b)
    return np.clip(raw, 0, 1).astype(np.float32), dict(gains=gains, a=noise_a, ccm=ccm)


# ---------------- classical ISP stages ----------------
def gray_world_gains(raw):
    r = raw[0::2, 0::2].mean(); g = 0.5 * (raw[0::2, 1::2].mean() + raw[1::2, 0::2].mean()); b = raw[1::2, 1::2].mean()
    return np.array([g / max(r, 1e-6), 1.0, g / max(b, 1e-6)], np.float32)

def apply_wb(raw, gains):
    out = raw.copy()
    out[0::2, 0::2] *= gains[0]; out[1::2, 1::2] *= gains[2]
    return np.clip(out, 0, 1)

def demosaic(raw):
    return cv2.cvtColor((np.clip(raw, 0, 1) * 65535 + .5).astype(np.uint16), BAYER_CODE).astype(np.float32) / 65535

def estimate_raw_sigma(raw):
    """Blind noise estimate: G1-G2 half-res difference, robust (MAD) so edges barely matter."""
    d = raw[0::2, 1::2] - raw[1::2, 0::2]
    return float(np.median(np.abs(d - np.median(d))) / 0.6745 / np.sqrt(2))

def denoise_rgb(img, sigma_raw, strength=1.0):
    h = float(np.clip((2 + 450 * sigma_raw) * strength, 2, 35))
    bgr = cv2.cvtColor((np.clip(img, 0, 1) * 255 + .5).astype(np.uint8), cv2.COLOR_RGB2BGR)
    out = cv2.fastNlMeansDenoisingColored(bgr, None, h, h, 5, 15)
    return cv2.cvtColor(out, cv2.COLOR_BGR2RGB).astype(np.float32) / 255

def unsharp(img, sigma_raw, radius=1.2):
    amt = float(np.clip(0.9 - 8 * sigma_raw, 0.15, 0.9))      # noisy input -> sharpen less
    blur = cv2.GaussianBlur(img, (0, 0), radius)
    return np.clip(img + amt * (img - blur), 0, 1)

def isp_front(raw, gains=None, ccm=NOMINAL_CCM):
    """Classical front-end WITHOUT denoise/sharpen: WB -> demosaic -> CCM -> gamma -> tone."""
    gains = gray_world_gains(raw) if gains is None else gains
    x = demosaic(apply_wb(raw, gains))
    x = np.clip(x @ ccm.T, 0, 1)
    return tone_curve(linear_to_srgb(x))

def isp_traditional(raw, gains=None, ccm=NOMINAL_CCM, denoise=True, sharpen=True):
    """Full classical ISP: WB -> demosaic -> CCM -> gamma -> adaptive NLM denoise -> tone -> unsharp."""
    gains = gray_world_gains(raw) if gains is None else gains
    s = estimate_raw_sigma(raw)
    x = demosaic(apply_wb(raw, gains))
    x = linear_to_srgb(np.clip(x @ ccm.T, 0, 1))
    if denoise: x = denoise_rgb(x, s)
    x = tone_curve(x)
    if sharpen: x = unsharp(x, s)
    return x

def pack_raw(raw):
    return np.stack([raw[0::2, 0::2], raw[0::2, 1::2], raw[1::2, 0::2], raw[1::2, 1::2]], 0)
