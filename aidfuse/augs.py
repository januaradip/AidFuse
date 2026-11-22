"""
Skeleton augmentations for MAA (COCO-17, box-normalised: x right, y down, box centre = 0, box height = 1).
Input/output of every transform: float32 array (T, 17, 3) = (x, y, confidence); confidence 0 = missing joint.

Generic (comparison):
  A1  strong affine: rotation +-20 deg, scale 0.8-1.2, shear +-0.2, translation +-0.1
  A2  joint dropout (10 % of joints) and body-part dropout (one of arms / legs / head, p = 0.3)
  A3  temporal: speed perturbation 0.7-1.3 with resampling to the same length
  A4  A1 + A2 + A3

Proposed mobility-aid kinematic augmentations (applied to one sample with probability P_AID):
  U1  seated (wheelchair): static seated lower body, upper body lowered by the thigh length
  U2  occupied arm (single crutch / cane, as in CARR): one arm fixed beside the body
  U3  asymmetric gait: reduced motion of one leg, slower tempo, lateral trunk sway
  U4  aid occlusion: lower legs (footrest) or aid-side forearm missing, or noisy / swapped legs, over a time segment
  U   one of U1-U4 drawn at random (U1 may be combined with U4-legs)
  U5  upper-body-centred normalisation (a representation, used at training and test time)

Proposed mobility-aid observation (sensing) augmentations, S3c. An aid changes not only how the body moves but also
what the detection and pose pipeline delivers (S2): the annotated box contains the aid, whole frames lose the pose,
and the legs are estimated with low confidence or on the aid. Ranges are physical priors, not fitted to test users.
  O1  box inflation: the box also covers the aid -> skeleton smaller and off-centre in box coordinates, box jitter
  O2  burst frame loss: 1-3 bursts of frames without any pose, 10-60 % of the window
  O3  leg corruption: leg confidence collapses, leg joints noisy or drawn towards the aid beside / below the body
  O   O1-O3 (each with its own probability, at least one), applied to a sample with probability P_AID
  K   kinematic family U1-U3 (one of them, probability P_AID)

S4, proposed ARC (Aid-Robust Canonicalisation), flags in the variant name (train and test):
  C1  confidence-aware body anchor + view-robust body scale (replaces the box frame; successor of U5)
  C2  temporal gap filling of missing joints / frames, on the whole sequence before windowing
  C3  late fusion of a box-frame stream and a body-frame stream (post hoc, see maa.compare with exp 'FUSE')
Comparators (re-implemented, same backbones and protocol): OCC body-part occlusion, ASM adaptive spatial masking
(needs the training-set missing rate, set in maa.SkelDS), MIX JMDA-style temporal / spatial mixing with soft labels.
After U1-U3 the skeleton is re-fitted to the original keypoint extent, as the annotated box would be.
"""
import math, random
import numpy as np

NOSE, LSHO, RSHO, LELB, RELB, LWRI, RWRI, LHIP, RHIP, LKNE, RKNE, LANK, RANK = 0, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16
UPPER = list(range(0, 13))              # head, arms, hips
LEGS = [LKNE, RKNE, LANK, RANK]
FLIP = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]
P_AID = 0.5


# ----------------------------------------------------------------------------- helpers
def _valid(x):
    return x[..., 2] > 0


def _mean_pt(x, j):
    """Mean position of joint j over the valid frames (or None)."""
    v = _valid(x[:, j])
    return x[v, j, :2].mean(0) if v.any() else None


def _seg_len(x, a, b):
    v = _valid(x[:, a]) & _valid(x[:, b])
    return float(np.linalg.norm(x[v, a, :2] - x[v, b, :2], axis=1).mean()) if v.any() else None


def _extent(x):
    v = _valid(x)
    if not v.any():
        return None
    p = x[v][:, :2]
    lo, hi = p.min(0), p.max(0)
    return (lo + hi) / 2, max(hi[1] - lo[1], 1e-3)


def refit(x_new, x_old, full=None):
    """Re-centre on the original keypoint extent and re-scale towards it. The box of a seated person or of a
    person with an aid is drawn around the person *and* the aid, so the skeleton fills the box only partly:
    the scale is drawn between 'unchanged' (u = 0) and 'fills the original extent' (u = 1)."""
    e0, e1 = _extent(x_old), _extent(x_new)
    if e0 is None or e1 is None:
        return x_new
    (c0, h0), (c1, h1) = e0, e1
    u = random.uniform(0, 1) if full is None else float(full)
    k = (h0 / h1) ** u
    v = _valid(x_new)
    x_new[..., :2] = np.where(v[..., None], (x_new[..., :2] - c1) * k + c0, 0)
    return x_new


def _resample(x, factor):
    """Play the sequence `factor` times faster (factor < 1: slower) and bring it back to length T."""
    T = len(x)
    src = np.clip(np.arange(T) * factor, 0, T - 1)
    if factor < 1:                                  # slower: random start so that all phases occur
        src = src + random.uniform(0, max(0.0, (T - 1) - src[-1]))
    i0 = np.floor(src).astype(int)
    i1 = np.minimum(i0 + 1, T - 1)
    w = (src - i0)[:, None, None]
    out = x[i0] * (1 - w) + x[i1] * w
    both = _valid(x[i0]) & _valid(x[i1])
    out[..., 2] = np.where(both, out[..., 2], 0)
    out[..., :2] = np.where(both[..., None], out[..., :2], 0)
    return out.astype(np.float32)


def _drop(x, joints, t0=0, t1=None):
    x[t0:t1, joints] = 0
    return x


# ----------------------------------------------------------------------------- generic
def a1_affine(x):
    a = math.radians(random.uniform(-20, 20))
    s = random.uniform(0.8, 1.2)
    sh = random.uniform(-0.2, 0.2)
    M = np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]], np.float32) @ \
        np.array([[1, sh], [0, 1]], np.float32) * s
    t = np.random.uniform(-0.1, 0.1, 2).astype(np.float32)
    v = _valid(x)
    x[..., :2] = np.where(v[..., None], x[..., :2] @ M.T + t, 0)
    return x


def a2_dropout(x):
    m = np.random.rand(17) < 0.1
    x[:, m] = 0
    if random.random() < 0.3:
        part = random.choice([[LELB, LWRI, RELB, RWRI], LEGS, [0, 1, 2, 3, 4]])
        x[:, part] = 0
    return x


def a3_speed(x):
    return _resample(x, random.uniform(0.7, 1.3))


def a4_generic(x):
    return a2_dropout(a1_affine(a3_speed(x)))


# ----------------------------------------------------------------------------- proposed
def u1_seated(x):
    """Seated in a wheelchair, seen from an elevated camera: torso lowered and slightly leaning, thighs pointing
    towards the camera (short in the image) with the knees apart, shanks foreshortened with the feet on the
    footrest, legs static. The person fills only part of the box (the box also contains the chair)."""
    old = x.copy()
    lth = _seg_len(x, LHIP, LKNE) or _seg_len(x, RHIP, RKNE) or 0.22
    lsh = _seg_len(x, LKNE, LANK) or _seg_len(x, RKNE, RANK) or 0.22
    sho_c = (x[:, LSHO, :2] + x[:, RSHO, :2]) / 2
    hip_c = (x[:, LHIP, :2] + x[:, RHIP, :2]) / 2
    # 1. lean: shoulders and head come closer to the hips
    lean = random.uniform(0.2, 0.45)                # seated torso looks shorter (lean, slumped, seen from above)
    top = [0, 1, 2, 3, 4, LSHO, RSHO, LELB, RELB, LWRI, RWRI]
    x[:, top, 1] -= np.where(_valid(x[:, top]), lean * (sho_c[:, None, 1] - hip_c[:, None, 1]), 0)
    # 2. the whole upper body comes down by about one thigh length
    drop = lth * random.uniform(0.6, 1.0)
    x[:, UPPER, 1] += np.where(_valid(x[:, UPPER]), drop, 0)
    # 3. seated legs (identical shape over time, they move with the chair)
    f = random.uniform(0.15, 0.55)                  # thigh seen almost end-on from the elevated camera
    spread = random.uniform(0.02, 0.07)             # knees apart
    shank = lsh * random.uniform(0.5, 0.8)          # feet forward on the footrest
    inward = random.uniform(-0.01, 0.03)
    hw = np.abs(x[:, LHIP, 0] - x[:, RHIP, 0]).mean() / 2 + 1e-6
    for hip, kne, ank in ((LHIP, LKNE, LANK), (RHIP, RKNE, RANK)):
        hv = _valid(x[:, hip])
        if not hv.any():
            x[:, [kne, ank]] = 0
            continue
        out = np.sign((x[hv, hip, 0] - hip_c[hv, 0]).mean()) or 1.0
        kn = x[:, hip, :2] + np.array([out * spread, lth * f], np.float32)
        an = kn + np.array([-out * inward, shank], np.float32)
        jit = lambda: np.random.normal(0, 0.003, (len(x), 2)).astype(np.float32)
        x[:, kne, :2], x[:, ank, :2] = kn + jit(), an + jit()
        x[:, kne, 2] = np.where(hv, x[:, hip, 2], 0)
        x[:, ank, 2] = np.where(hv, x[:, hip, 2] * 0.8, 0)
        x[~hv, kne, :2] = 0
        x[~hv, ank, :2] = 0
    del hw
    return refit(x, old, full=random.uniform(0.0, 0.2))


def u2_arm(x):
    old = x.copy()
    sides = [(LSHO, LELB, LWRI, LHIP), (RSHO, RELB, RWRI, RHIP)]
    random.shuffle(sides)
    sides = sides[:1]                              # CARR: crutch and cane users hold a single aid
    hip_c = (x[:, LHIP, :2] + x[:, RHIP, :2]) / 2
    out_w = random.uniform(0.05, 0.15)
    down_w = random.uniform(0.0, 0.12)
    for k, (sho, elb, wri, hip) in enumerate(sides):
        ok = _valid(x[:, sho]) & _valid(x[:, hip])
        if not ok.any():
            continue
        # outward = away from the body centre on the image (depends on whether the person faces the camera)
        outward = 1.0 if (x[ok, hip, 0] - hip_c[ok, 0]).mean() >= 0 else -1.0
        w = x[:, hip, :2] + np.array([outward * out_w, down_w], np.float32)
        e = (x[:, sho, :2] + w) / 2 + np.array([outward * 0.03, 0], np.float32)
        noise = lambda: np.random.normal(0, 0.004, (len(x), 2)).astype(np.float32)
        x[ok, wri, :2], x[ok, elb, :2] = (w + noise())[ok], (e + noise())[ok]
        x[ok, wri, 2] = x[ok, elb, 2] = np.minimum(x[ok, sho, 2], x[ok, hip, 2])
    return refit(x, old)


def u3_gait(x):
    old = x.copy()
    x = _resample(x, random.uniform(0.6, 0.85))           # slower tempo
    hip_c = (x[:, LHIP, :2] + x[:, RHIP, :2]) / 2
    side = random.choice([(LKNE, LANK, LHIP), (RKNE, RANK, RHIP)])
    a = random.uniform(0.3, 0.8)                           # remaining motion of the affected leg
    for j in side[:2]:
        v = _valid(x[:, j]) & _valid(x[:, side[2]])
        if v.any():
            rel = x[:, j, :2] - x[:, side[2], :2]
            m = rel[v].mean(0)
            x[v, j, :2] = x[v, side[2], :2] + m + (rel[v] - m) * a
    amp = random.uniform(0.015, 0.045)                     # lateral trunk sway with each step
    per = random.uniform(15, 30)
    ph = random.uniform(0, 2 * math.pi)
    sway = amp * np.sin(2 * math.pi * np.arange(len(x)) / per + ph)
    up = [0, 1, 2, 3, 4, LSHO, RSHO, LELB, RELB, LWRI, RWRI]
    x[:, up, 0] += np.where(_valid(x[:, up]), sway[:, None], 0)
    return refit(x, old)


def u4_occlusion(x, legs=None):
    T = len(x)
    if legs is None:
        legs = random.random() < 0.5
    if random.random() < 0.5:
        t0, t1 = 0, T
    else:
        L = random.randint(T // 4, T)
        t0 = random.randint(0, T - L)
        t1 = t0 + L
    if legs:
        if random.random() < 0.4:                  # aid confuses the pose estimator: noisy, crossed legs
            v = _valid(x[t0:t1, LEGS])
            x[t0:t1, LEGS, :2] += np.where(v[..., None], np.random.normal(0, 0.03, (t1 - t0, 4, 2)), 0).astype(np.float32)
            if random.random() < 0.5:
                x[t0:t1, [LKNE, RKNE, LANK, RANK]] = x[t0:t1, [RKNE, LKNE, RANK, LANK]]
            return x
        return _drop(x, random.choice([[LANK, RANK], LEGS]), t0, t1)
    return _drop(x, random.choice([[LWRI], [RWRI], [LELB, LWRI], [RELB, RWRI]]), t0, t1)


def u_any(x, which=('U1', 'U2', 'U3', 'U4')):
    if random.random() >= P_AID:
        return x
    k = random.choice(which)
    if k == 'U1':
        x = u1_seated(x)
        if random.random() < 0.3:
            x = u4_occlusion(x, legs=True)
        return x
    return {'U2': u2_arm, 'U3': u3_gait, 'U4': u4_occlusion}[k](x)


def u5_upper_norm(x):
    """Upper-body-centred normalisation of a whole window: origin = mid-shoulder of the first valid frame,
    unit = median shoulder width x 3 (about the box height of a standing adult). Falls back to the input."""
    v = _valid(x[:, LSHO]) & _valid(x[:, RSHO])
    if v.sum() < 2:
        return x
    mid = (x[v, LSHO, :2] + x[v, RSHO, :2]) / 2
    sw = np.median(np.linalg.norm(x[v, LSHO, :2] - x[v, RSHO, :2], axis=1))
    if sw < 0.03:
        return x
    o, s = mid[0], 1.0 / (3 * sw)
    vv = _valid(x)
    x[..., :2] = np.where(vv[..., None], (x[..., :2] - o) * s, 0)
    return x


# ----------------------------------------------------------------------------- proposed: observation family
def o1_box(x):
    """Box inflated by the aid: x_box' = (x_box - d) / h with h = new/old box height and d = centre shift (in old
    box heights), plus a small smooth per-frame box jitter (box-normalised coordinates, so no pixel geometry needed)."""
    T = len(x)
    h = random.uniform(1.0, 1.45)
    dx = random.choice([-1, 1]) * random.uniform(0.0, 0.25)          # aid beside the body (crutch, cane, wheel)
    dy = (h - 1) * random.uniform(-0.5, 0.5)                          # box grows up / down
    jit = np.cumsum(np.random.normal(0, 0.004, (T, 2)), 0).astype(np.float32)
    jit -= jit.mean(0)
    hj = 1 + np.cumsum(np.random.normal(0, 0.003, T)).astype(np.float32)
    d = np.array([dx, dy], np.float32) + jit
    v = _valid(x)
    x[..., :2] = np.where(v[..., None], (x[..., :2] - d[:, None]) / (h * hj[:, None, None]), 0)
    return x


def o2_frameloss(x):
    """Whole frames without a pose, in 1-3 bursts covering 10-60 % of the window (at least 25 % of frames kept)."""
    T = len(x)
    total = int(T * random.uniform(0.1, 0.6))
    n = random.randint(1, 3)
    lens = np.random.multinomial(total, [1 / n] * n)
    for L in lens:
        if L < 1:
            continue
        t0 = random.randint(0, max(0, T - L))
        x[t0:t0 + L] = 0
    return x                                                        # o_any falls back if too little is left


def o3_legs(x):
    """Legs estimated badly: confidence collapse and noise, or (30 %) leg joints pulled towards the aid."""
    v = _valid(x[:, LEGS])
    x[:, LEGS, 2] = np.where(v, x[:, LEGS, 2] * random.uniform(0.1, 0.5), 0)
    sd = random.uniform(0.01, 0.04)
    x[:, LEGS, :2] += np.where(v[..., None], np.random.normal(0, sd, (len(x), 4, 2)), 0).astype(np.float32)
    if random.random() < 0.3:
        hv = _valid(x[:, LHIP]) & _valid(x[:, RHIP])
        if hv.any():
            hip_c = (x[:, LHIP, :2] + x[:, RHIP, :2]) / 2
            tgt = hip_c + np.array([random.choice([-1, 1]) * random.uniform(0.05, 0.2), random.uniform(0.2, 0.45)],
                                   np.float32)
            w = random.uniform(0.3, 0.8)
            ok = v & hv[:, None]
            x[:, LEGS, :2] = np.where(ok[..., None], (1 - w) * x[:, LEGS, :2] + w * tgt[:, None], x[:, LEGS, :2])
    return x


def o_any(x, which=('O1', 'O2', 'O3'), p=None):
    if random.random() >= (P_AID if p is None else p):
        return x
    fn = {'O1': (o1_box, 0.7), 'O2': (o2_frameloss, 0.5), 'O3': (o3_legs, 0.5)}
    pick = [k for k in which if random.random() < fn[k][1]] or [random.choice(which)]
    x0 = x.copy()
    for k in ('O3', 'O1', 'O2'):                                     # legs, then box, then frame loss
        if k in pick:
            x = fn[k][0](x)
    return x if _valid(x).any(1).sum() >= len(x) // 4 else x0


def k_any(x):
    return u_any(x, ('U1', 'U2', 'U3'))


def single(name):
    return {'U1': u1_seated, 'U2': u2_arm, 'U3': u3_gait, 'U4': u4_occlusion}[name]


AUGS = {
    'A0': lambda x: x,
    'A1': a1_affine, 'A2': a2_dropout, 'A3': a3_speed, 'A4': a4_generic,
    'U1': lambda x: u_any(x, ('U1',)), 'U2': lambda x: u_any(x, ('U2',)),
    'U3': lambda x: u_any(x, ('U3',)), 'U4': lambda x: u_any(x, ('U4',)),
    'U': u_any,
    'A4+U': lambda x: u_any(a4_generic(x)),
    'A4+U1': lambda x: u_any(a4_generic(x), ('U1',)),
    'U1+U4': lambda x: u_any(x, ('U1', 'U4')),
    # S3c: observation (sensing) family and its combination with the kinematic family
    'O': o_any, 'K': k_any,
    'K+O': lambda x: o_any(k_any(x)),
    'A4+O': lambda x: o_any(a4_generic(x)),
    'A4+K': lambda x: k_any(a4_generic(x)),
    'A4+K+O': lambda x: o_any(k_any(a4_generic(x))),
    'A4+O1': lambda x: o_any(a4_generic(x), ('O1',)),
    'A4+O2': lambda x: o_any(a4_generic(x), ('O2',)),
    'A4+O3': lambda x: o_any(a4_generic(x), ('O3',)),
}


# =============================================================================== S4: ARC (Aid-Robust Canonicalisation)
TORSO = [LSHO, RSHO, LHIP, RHIP]


def c1_body_norm(x, k_sw=3.0, k_tl=3.3, n0=8, min_conf=0.3):
    """C1: confidence-aware body anchor that does not depend on the box.
    Origin: confidence-weighted mid-shoulder (fallback mid-hip, then nose) averaged over the first n0 frames in which
    it is reliable, so the translation of the person inside the window is kept.
    Scale: the larger of 3 x shoulder width and 3.3 x torso length (median over reliable frames) - shoulder width
    collapses in side view, torso length collapses when seated / leaning, the maximum is robust to both."""
    c = x[..., 2]
    ok = lambda *j: np.all([c[:, i] >= min_conf for i in j], axis=0)
    sho, hip = ok(LSHO, RSHO), ok(LHIP, RHIP)
    msho = (x[:, LSHO, :2] + x[:, RSHO, :2]) / 2
    mhip = (x[:, LHIP, :2] + x[:, RHIP, :2]) / 2
    if sho.any():
        idx = np.where(sho)[0][:n0]
        w = np.minimum(c[idx, LSHO], c[idx, RSHO])[:, None]
        o = (msho[idx] * w).sum(0) / w.sum()
    elif hip.any():
        idx = np.where(hip)[0][:n0]
        o = mhip[idx].mean(0)
    elif (c[:, NOSE] >= min_conf).any():
        o = x[c[:, NOSE] >= min_conf, NOSE, :2][:n0].mean(0)
    else:
        return x
    cand = []
    if sho.sum() >= 2:
        cand.append(k_sw * np.median(np.linalg.norm(x[sho, LSHO, :2] - x[sho, RSHO, :2], axis=1)))
    both = sho & hip
    if both.sum() >= 2:
        cand.append(k_tl * np.median(np.linalg.norm(msho[both] - mhip[both], axis=1)))
    s = max(cand) if cand else 0.0
    if s < 0.1:                                      # unreliable body size: keep the box scale, only re-centre
        s = 1.0
    v = c > 0
    x[..., :2] = np.where(v[..., None], (x[..., :2] - o) / s, 0)
    return x


def c2_fill(seq, max_gap=64, max_hold=8, fill_conf=0.05):
    """C2: temporal gap filling of a whole sequence (T,17,3), before windowing / augmentation (train and test).
    Interior gaps of a joint up to max_gap frames are linearly interpolated, gaps at the start / end are held for
    up to max_hold frames. Filled joints get a low confidence so that the network can tell them apart."""
    x = seq.copy()
    T = len(x)
    t = np.arange(T)
    for j in range(x.shape[1]):
        v = x[:, j, 2] > 0
        if v.all() or v.sum() < 2:
            continue
        tv = t[v]
        nxt = np.searchsorted(tv, t)                         # index of next valid frame
        prv = nxt - 1
        miss = ~v
        inner = miss & (nxt < len(tv)) & (prv >= 0)
        gap = np.where(inner, tv[np.clip(nxt, 0, len(tv) - 1)] - tv[np.clip(prv, 0, len(tv) - 1)] - 1, 10 ** 6)
        fill_in = inner & (gap <= max_gap)
        head = miss & (prv < 0) & (tv[0] - t <= max_hold)
        tail = miss & (nxt >= len(tv)) & (t - tv[-1] <= max_hold)
        for d in (0, 1):
            interp = np.interp(t, tv, x[v, j, d])
            x[fill_in | head | tail, j, d] = interp[fill_in | head | tail]
        x[fill_in | head | tail, j, 2] = fill_conf
    return x


# ----------------------------------------------------------------------------- comparators (re-implemented)
PARTS = {'head': [0, 1, 2, 3, 4], 'larm': [LSHO, LELB, LWRI], 'rarm': [RSHO, RELB, RWRI],
         'lleg': [LHIP, LKNE, LANK], 'rleg': [RHIP, RKNE, RANK], 'torso': [LSHO, RSHO, LHIP, RHIP]}


def occ_parts(x, p=0.5):
    """OCC: body-part occlusion augmentation (Mathe et al., Sensors 2025): one or two body parts removed from the
    whole window."""
    if random.random() < p:
        for part in random.sample(list(PARTS), random.choice([1, 2])):
            x[:, PARTS[part]] = 0
    return x


def asm_mask(x, rate, p=0.5, ratio=0.15):
    """ASM: adaptive spatial masking (Wang et al., 2023): joints are masked with a probability proportional to how
    often they are missing in the training data (rate: per-joint missing rate of the training set)."""
    if random.random() < p:
        q = np.clip(ratio * rate / max(rate.mean(), 1e-6), 0, 0.6)
        x[:, np.random.rand(17) < q] = 0
    return x


def jmda_mix(xa, xb):
    """MIX: JMDA-style mixing (Xiang & Wang, ACM TOMM 2024), adapted to 2D poses. TemporalMix: both sequences are
    resized in time and concatenated; SpatialMix: upper body of A with lower body of B (B aligned at the hip centre).
    Returns the mixed sequence and the weight lam of sample A for the label."""
    T = len(xa)
    if random.random() < 0.5:
        lam = random.uniform(0.3, 0.7)
        ta = max(1, int(round(lam * T)))
        ia = np.linspace(0, T - 1, ta).round().astype(int)
        ib = np.linspace(0, T - 1, T - ta).round().astype(int)
        return np.concatenate([xa[ia], xb[ib]], 0).astype(np.float32), ta / T
    lower = [LHIP, RHIP, LKNE, RKNE, LANK, RANK]
    out = xa.copy()
    va = _valid(xa[:, LHIP]) & _valid(xa[:, RHIP])
    vb = _valid(xb[:, LHIP]) & _valid(xb[:, RHIP])
    if va.any() and vb.any():
        ha = ((xa[va, LHIP, :2] + xa[va, RHIP, :2]) / 2).mean(0)
        hb = ((xb[vb, LHIP, :2] + xb[vb, RHIP, :2]) / 2).mean(0)
        sh = xb[:, lower].copy()
        sh[..., :2] = np.where(_valid(sh)[..., None], sh[..., :2] - hb + ha, 0)
        out[:, lower] = sh
    return out, 11 / 17


AUGS.update({
    'OCC': occ_parts, 'A4+OCC': lambda x: occ_parts(a4_generic(x)),
})
