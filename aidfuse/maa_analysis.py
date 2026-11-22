"""
MAA analyses without training:
  S2  pose-quality diagnosis per participant (keypoint confidence, missing joints, leg crossings, proportions)
  S8  realism of the augmentations: body-proportion and motion features of augmented skeletons (from participants
      without aids) versus real skeletons of aid users (P07-P13 are only used here, never for training or tuning)
  S10 cost: parameters, GFLOPs per 64-frame clip, latency per clip (GPU and 1 CPU thread)
Output: <DATA>/maa/results/S2_*.csv, S8_*.csv, S10_cost.csv and figures in <DATA>/maa/figures/.
"""
import os, math, random, time
import numpy as np
import augs as A

AID = {'P07': 'wheelchair', 'P11': 'wheelchair', 'P08': 'crutches', 'P10': 'crutches', 'P13': 'crutches',
       'P09': 'cane', 'P12': 'cane'}
LS, RS, LH, RH, LK, RK, LA, RA = 5, 6, 11, 12, 13, 14, 15, 16
ARMS, LEGS = [7, 8, 9, 10], [13, 14, 15, 16]


def group_of(owner):
    return AID.get(owner, 'no aid')


# ----------------------------------------------------------------------------- per-window features
def _mid(x, a, b):
    v = (x[:, a, 2] > 0) & (x[:, b, 2] > 0)
    m = (x[:, a, :2] + x[:, b, :2]) / 2
    return m, v


def features(x):
    """Body-proportion and motion features of one window (T,17,3); NaN where not computable.
    Lengths are divided by the shoulder width, which does not change between standing and sitting."""
    f = {}
    sho, vs = _mid(x, LS, RS)
    hip, vh = _mid(x, LH, RH)
    ank, va = _mid(x, LA, RA)
    kne, vk = _mid(x, LK, RK)
    sw = np.linalg.norm(x[:, LS, :2] - x[:, RS, :2], axis=1)
    ok = vs & (x[:, LS, 2] > 0) & (x[:, RS, 2] > 0) & (sw > 0.02)
    swm = np.median(sw[ok]) if ok.any() else np.nan

    def med(a, m):
        return float(np.median(a[m])) if m.any() else np.nan
    f['torso / shoulder'] = med(hip[:, 1] - sho[:, 1], vs & vh) / swm
    f['hip-knee / shoulder'] = med(kne[:, 1] - hip[:, 1], vh & vk) / swm
    f['knee-ankle / shoulder'] = med(ank[:, 1] - kne[:, 1], vk & va) / swm
    f['shoulder-ankle / shoulder'] = med(ank[:, 1] - sho[:, 1], vs & va) / swm
    f['torso / shoulder-ankle'] = f['torso / shoulder'] / f['shoulder-ankle / shoulder'] \
        if f['shoulder-ankle / shoulder'] and not np.isnan(f['shoulder-ankle / shoulder']) else np.nan
    # motion: leg swing relative to the hips, hand reach relative to the shoulders (per shoulder width)
    leg = []
    for j, h in ((LA, LH), (RA, RH), (LK, LH), (RK, RH)):
        v = (x[:, j, 2] > 0) & (x[:, h, 2] > 0)
        if v.sum() > 4:
            leg.append(np.std(x[v, j, 1] - x[v, h, 1]))
    f['leg swing / shoulder'] = (np.mean(leg) / swm) if leg else np.nan
    # arm use: motion of each wrist relative to its shoulder; an arm holding an aid moves little
    arm = []
    for w, sh in ((9, LS), (10, RS)):
        v = (x[:, w, 2] > 0) & (x[:, sh, 2] > 0)
        arm.append(np.linalg.norm(np.std(x[v, w, :2] - x[v, sh, :2], axis=0)) / swm if v.sum() > 4 else np.nan)
    f['quieter arm activity'] = float(np.nanmin(arm)) if not np.all(np.isnan(arm)) else np.nan
    f['arm activity ratio'] = float(np.nanmin(arm) / max(np.nanmax(arm), 1e-6)) if not np.any(np.isnan(arm)) else np.nan
    ls = (x[:, LA, 2] > 0) & (x[:, RA, 2] > 0) & (x[:, LK, 2] > 0) & (x[:, RK, 2] > 0)
    f['leg asymmetry'] = abs(np.std(x[ls, LA, 1] - x[ls, LK, 1]) - np.std(x[ls, RA, 1] - x[ls, RK, 1])) / swm \
        if ls.sum() > 4 else np.nan
    return f


def quality(x):
    """Pose-quality indicators of one window."""
    c = x[..., 2]
    q = dict(conf_all=float(c[c > 0].mean()) if (c > 0).any() else 0.0,
             missing_all=float((c <= 0).mean()),
             missing_arms=float((c[:, ARMS] <= 0).mean()), missing_legs=float((c[:, LEGS] <= 0).mean()),
             lowconf_legs=float((c[:, LEGS] < 0.5).mean()))
    # left/right ankles swapped relative to the hips (crossed legs) in frames with all four joints
    v = (c[:, LA] > 0) & (c[:, RA] > 0) & (c[:, LH] > 0) & (c[:, RH] > 0)
    if v.any():
        hs = np.sign(x[v, LH, 0] - x[v, RH, 0])
        as_ = np.sign(x[v, LA, 0] - x[v, RA, 0])
        q['crossed_legs'] = float((hs != as_).mean())
    else:
        q['crossed_legs'] = np.nan
    return q


# ----------------------------------------------------------------------------- S2
def s2_quality(seq, F, out):
    import pandas as pd
    rows = []
    for v, x in seq.items():
        m = F['videos'][v]
        if len(x) < 8:
            continue
        r = quality(x)
        r.update(video=v, participant=m['owner'], aid=group_of(m['owner']), cls=m['cls'])
        rows.append(r)
    df = pd.DataFrame(rows)
    cols = ['conf_all', 'missing_all', 'missing_arms', 'missing_legs', 'lowconf_legs', 'crossed_legs']
    by_p = df.groupby(['aid', 'participant'])[cols].mean().round(3)
    by_g = df.groupby('aid')[cols].mean().round(3)
    by_p.to_csv(f'{out}/S2_pose_quality_participant.csv')
    by_g.to_csv(f'{out}/S2_pose_quality_group.csv')
    return by_g, by_p


# ----------------------------------------------------------------------------- S8
def _windows(seq, vids, T=64, per_video=4, rng=None):
    rng = rng or np.random.default_rng(0)
    out = []
    for v in vids:
        x = seq[v]
        if len(x) < T:
            out.append(np.concatenate([x, np.zeros((T - len(x), 17, 3), np.float32)]))
            continue
        for s in rng.choice(len(x) - T + 1, min(per_video, len(x) - T + 1), replace=False):
            out.append(x[s:s + T].copy())
    return out


def _wasserstein(a, b):
    """1-D Wasserstein distance on the 5-95 % quantile range (ratios explode when a denominator is ~0, e.g.
    side views with a tiny shoulder width; the tails would otherwise dominate)."""
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    if len(a) < 5 or len(b) < 5:
        return np.nan
    q = np.linspace(0.05, 0.95, 91)
    return float(np.mean(np.abs(np.quantile(a, q) - np.quantile(b, q))))


def s8_realism(seq, F, out, figdir, n_per_video=4, seed=0):
    """Compare features of: real windows without aids, the same windows after U1 / U2 / U3 / U4 / U,
    and real windows of aid users per aid type. Distance = 1-D Wasserstein per feature, scaled by the
    spread of the real no-aid feature; lower = closer to the real aid users."""
    import pandas as pd
    random.seed(seed); np.random.seed(seed)
    rng = np.random.default_rng(seed)
    vids = {g: [v for v in seq if group_of(F['videos'][v]['owner']) == g] for g in
            ('no aid', 'wheelchair', 'crutches', 'cane')}
    base = _windows(seq, vids['no aid'], per_video=n_per_video, rng=rng)
    sets = {'real: no aid': base}
    for g in ('wheelchair', 'crutches', 'cane'):
        sets[f'real: {g}'] = _windows(seq, vids[g], per_video=n_per_video, rng=rng)
    aug = {'U1 seated': A.u1_seated, 'U2 occupied arm': A.u2_arm, 'U3 gait': A.u3_gait,
           'U4 occlusion': A.u4_occlusion, 'A4 generic': A.a4_generic}
    for k, fn in aug.items():
        sets[f'aug: {k}'] = [fn(w.copy()) for w in base]
    feats = {k: pd.DataFrame([features(w) for w in ws]) for k, ws in sets.items()}
    names = list(feats['real: no aid'].columns)
    rows = []
    for k, df in feats.items():
        r = {'set': k, 'windows': len(df)}
        r.update({n: round(float(np.nanmedian(df[n])), 3) for n in names})
        rows.append(r)
    med = pd.DataFrame(rows)
    med.to_csv(f'{out}/S8_feature_medians.csv', index=False)
    # distances to each real aid group
    pairs = {'wheelchair': ['real: no aid', 'aug: U1 seated', 'aug: A4 generic'],
             'crutches': ['real: no aid', 'aug: U2 occupied arm', 'aug: U3 gait', 'aug: U4 occlusion', 'aug: A4 generic'],
             'cane': ['real: no aid', 'aug: U3 gait', 'aug: U2 occupied arm', 'aug: A4 generic']}
    drows = []
    for g, srcs in pairs.items():
        tgt = feats[f'real: {g}']
        for s in srcs:
            r = {'target': g, 'source': s}
            ds = []
            for n in names:
                ref = feats['real: no aid'][n].values
                ref = ref[np.isfinite(ref)]
                sd = (np.percentile(ref, 75) - np.percentile(ref, 25)) or 1.0      # robust spread (IQR)
                d = _wasserstein(feats[s][n].values, tgt[n].values) / sd
                r[n] = round(d, 3)
                if not np.isnan(d):
                    ds.append(d)
            r['mean'] = round(float(np.mean(ds)), 3) if ds else np.nan
            drows.append(r)
    dist = pd.DataFrame(drows)
    dist.to_csv(f'{out}/S8_distance_to_real_aid_users.csv', index=False)
    _plot_s8(feats, figdir)
    return med, dist


def _plot_s8(feats, figdir):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    show = ['torso / shoulder', 'hip-knee / shoulder', 'shoulder-ankle / shoulder', 'leg swing / shoulder']
    groups = [('real: no aid', '#888888', '-'), ('real: wheelchair', '#c0392b', '-'), ('aug: U1 seated', '#c0392b', '--'),
              ('real: crutches', '#2471a3', '-'), ('aug: U3 gait', '#2471a3', '--')]
    fig, axs = plt.subplots(1, len(show), figsize=(4.2 * len(show), 3.4))
    for ax, n in zip(axs, show):
        allv = np.concatenate([feats[g][n].dropna().values for g, _, _ in groups])
        lo, hi = np.nanpercentile(allv, [1, 99])
        bins = np.linspace(lo, hi, 40)
        for g, col, ls in groups:
            v = feats[g][n].dropna().values
            if len(v) > 5:
                h, e = np.histogram(np.clip(v, lo, hi), bins=bins, density=True)
                ax.plot((e[1:] + e[:-1]) / 2, h, color=col, ls=ls, lw=1.6, label=g)
        ax.set_title(n, fontsize=10)
        ax.set_yticks([])
    axs[0].legend(fontsize=7, frameon=False)
    plt.tight_layout()
    os.makedirs(figdir, exist_ok=True)
    plt.savefig(f'{figdir}/S8_feature_distributions.png', dpi=150)
    plt.close(fig)


# ----------------------------------------------------------------------------- S10
def s10_cost(build, models, out, reps=30):
    import torch
    import pandas as pd
    rows = []
    x1 = torch.randn(1, 3, 64, 17) * 0.3
    x1[:, 2] = torch.rand(1, 64, 17)
    for m in models:
        net = build(m).eval()
        r = dict(model=m, params_M=round(sum(p.numel() for p in net.parameters()) / 1e6, 2))
        try:
            from thop import profile
            macs, _ = profile(net, inputs=(x1,), verbose=False)
            r['GFLOPs_per_clip'] = round(2 * macs / 1e9, 3)
        except Exception as e:                       # thop missing or op not supported
            r['GFLOPs_per_clip'] = np.nan
        with torch.no_grad():
            torch.set_num_threads(1)
            for _ in range(3):
                net(x1)
            t = time.perf_counter()
            for _ in range(reps):
                net(x1)
            r['cpu1_ms_per_clip'] = round((time.perf_counter() - t) / reps * 1000, 2)
            torch.set_num_threads(os.cpu_count() or 2)
            if torch.cuda.is_available():
                g = net.cuda()
                xg = x1.cuda()
                for _ in range(10):
                    g(xg)
                torch.cuda.synchronize()
                t = time.perf_counter()
                for _ in range(reps * 3):
                    g(xg)
                torch.cuda.synchronize()
                r['gpu_ms_per_clip'] = round((time.perf_counter() - t) / (reps * 3) * 1000, 2)
        r['ms_per_frame_cpu1'] = round(r['cpu1_ms_per_clip'] / 64, 3)
        rows.append(r)
    df = pd.DataFrame(rows)
    df.to_csv(f'{out}/S10_cost.csv', index=False)
    return df
