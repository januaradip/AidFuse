"""
S9: external validation on SAFER-Activities (CC BY-NC-SA 4.0, Lamsal et al., 2026).

Conversion (once, by the first session; the others wait):
  * 2D poses (YOLOv8x + ViTPose-H, COCO-17) and boxes from pose_bboxes/3d_keypoints_pickle_ntu_format.zip
  * action segments from the CSVs (Action, Start, End) at 25 fps; every (camera view, segment) becomes one sample
  * classes = actions present in both the non-wheelchair and the wheelchair subset with at least MIN_TEST_SEG
    segments in the wheelchair subject-test split
  * keypoints normalised exactly as in CARR: box centre = 0, box height = 1, confidence as third channel
Protocols:
  SZ  zero-shot to wheelchair: train on non-wheelchair subjects that are NOT in the wheelchair test split;
      validation = half of the non-wheelchair test subjects; test = wheelchair subject-test split ('dis')
      + the other half of the non-wheelchair test subjects ('nondis', check that nothing is lost for them)
  SO  oracle: train and validate on the wheelchair subject-train split, test on the wheelchair subject-test split
"""
import os, re, csv, glob, json, time, gc, random, collections, zipfile
import numpy as np

import maa
from carr import common as C

FPS = 25.0
MIN_TEST_SEG = 10            # minimum segments (all camera views) of a class in the wheelchair test split
MAX_SEG_SEC = 20.0           # longer segments: the middle 20 s are kept
MIN_SEG_FRAMES = 8
TRAIN_WIN_PER_SEG = 2        # windows per segment for training (evenly spaced), evaluation: up to 8
EVAL_WIN_PER_SEG = 8


def _secs(t):
    h, m, s = t.strip().split(':')
    return int(h) * 3600 + int(m) * 60 + float(s)


def _labels(safer):
    lab = {}
    for f in glob.glob(f'{safer}/raw/Safer-Activities-Full-Dataset/*/CSVs/*.csv'):
        rows = list(csv.reader(open(f, encoding='utf-8-sig')))[1:]
        segs = []
        for r in rows:
            if len(r) >= 4 and r[2].strip() and r[3].strip():
                try:
                    segs.append((r[1].strip().lower(), _secs(r[2]), _secs(r[3])))
                except ValueError:
                    pass
        lab[os.path.basename(f)[:-4]] = segs
    return lab


def _recording(frame_dir):
    return re.sub(r'_d\d+$', '', frame_dir.replace('.mp4', ''))


def _subject(frame_dir):
    m = re.search(r'_(p\d+)_d\d+', frame_dir)
    return m.group(1) if m else frame_dir


def _normalise(kp, sc, bb, fmt):
    """kp (T,17,2) pixels, sc (T,17), bb (T,4) -> (T,17,3) box-normalised like CARR."""
    bb = bb.astype(np.float32)
    if fmt == 'xywh':
        x1, y1, w, h = bb.T
    else:
        x1, y1, x2, y2 = bb.T
        w, h = x2 - x1, y2 - y1
    cx, cy = x1 + w / 2, y1 + h / 2
    ok = h > 2
    hh = np.where(ok, h, 1.0)
    valid = (sc > 0.05) & ok[:, None]
    xs = (kp[..., 0] - cx[:, None]) / hh[:, None]
    ys = (kp[..., 1] - cy[:, None]) / hh[:, None]
    c = np.clip(sc, 0, 1)
    return np.stack([np.where(valid, xs, 0), np.where(valid, ys, 0), np.where(valid, c, 0)], -1).astype(np.float16)


def convert(safer, out_file, work='/content/safer_pkl'):
    import pickle
    lab = _labels(safer)
    os.makedirs(work, exist_ok=True)
    zf = f'{safer}/pose_bboxes/3d_keypoints_pickle_ntu_format.zip'
    with zipfile.ZipFile(zf) as z:
        for n in z.namelist():
            if n.endswith(('aic_normal_dataset_with_3d.pkl', 'aic_wheelchair_dataset_with_3d.pkl')) and \
                    not os.path.exists(f'{work}/{os.path.basename(n)}'):
                C.log(f'extract {n}')
                with z.open(n) as src, open(f'{work}/{os.path.basename(n)}', 'wb') as dst:
                    while True:
                        b = src.read(1 << 24)
                        if not b:
                            break
                        dst.write(b)
    # pass 1: class counts in the wheelchair test split, to choose the classes
    seg_meta = []
    for subset, fn in (('normal', 'aic_normal_dataset_with_3d.pkl'), ('wheelchair', 'aic_wheelchair_dataset_with_3d.pkl')):
        C.log(f'read {fn}')
        d = pickle.load(open(f'{work}/{fn}', 'rb'))
        split_of = {}
        for k in ('sub_train', 'sub_test'):
            for v in d['split'][k]:
                split_of[v] = k
        for a in d['annotations']:
            fd = a['frame_dir']
            if fd not in split_of:
                continue
            rec = _recording(fd)
            if rec not in lab:
                continue
            T = int(a['total_frames'])
            kp = np.asarray(a['keypoint'])[0]
            sc = np.asarray(a['keypoint_score'])[0]
            bb = np.asarray(a['bboxes'])
            for i, (act, t0, t1) in enumerate(lab[rec]):
                f0, f1 = int(round(t0 * FPS)), min(T, int(round(t1 * FPS)))
                if f1 - f0 < MIN_SEG_FRAMES:
                    continue
                if (f1 - f0) > MAX_SEG_SEC * FPS:
                    mid = (f0 + f1) // 2
                    f0, f1 = int(mid - MAX_SEG_SEC * FPS / 2), int(mid + MAX_SEG_SEC * FPS / 2)
                x = _normalise(kp[f0:f1], sc[f0:f1], bb[f0:f1], a.get('bbox_format', 'xyxy'))
                if (x[..., 2] > 0).mean() < 0.2:       # person mostly not detected (blacked-out gap)
                    continue
                seg_meta.append(dict(key=f'{fd}#{i:04d}', subset=subset, split=split_of[fd], video=fd,
                                     subject=_subject(fd), action=act, x=x))
        del d
        gc.collect()
    cnt_wc_test = collections.Counter(s['action'] for s in seg_meta if s['subset'] == 'wheelchair' and s['split'] == 'sub_test')
    acts_n = {s['action'] for s in seg_meta if s['subset'] == 'normal'}
    acts_w = {s['action'] for s in seg_meta if s['subset'] == 'wheelchair'}
    shared = sorted(acts_n & acts_w)
    classes = [a for a in shared if cnt_wc_test[a] >= MIN_TEST_SEG]
    C.log(f'shared classes {shared}; kept (>= {MIN_TEST_SEG} test segments): {classes}')
    keep = [s for s in seg_meta if s['action'] in classes]
    obj = dict(classes=classes, shared=shared, wc_test_counts=dict(cnt_wc_test), fps=FPS,
               seq={s['key']: s['x'] for s in keep},
               meta={s['key']: {k: s[k] for k in ('subset', 'split', 'video', 'subject', 'action')} for s in keep})
    C.save_pickle(obj, out_file)
    C.log(f'{len(keep)} segments saved to {out_file}')
    return obj


def load_or_convert(safer, out_file):
    """The first session converts (lock), the others wait until the file exists."""
    if not os.path.exists(out_file):
        ldir = os.path.dirname(out_file) + '/convert_lock'
        if C.claim_run(ldir):
            try:
                if not os.path.exists(out_file):
                    convert(safer, out_file)
            finally:
                C.release_run(ldir)
        while not os.path.exists(out_file):
            C.log('another session is converting SAFER, waiting 60 s...')
            time.sleep(60)
    return C.load_pickle(out_file)


# ----------------------------------------------------------------------------- data for the trainer
def prepare(obj):
    classes = obj['classes']
    K = len(classes)
    seq = {k: v.astype(np.float32) for k, v in obj['seq'].items()}
    F = {'videos': {k: dict(cls=classes.index(m['action']), group='dis' if m['subset'] == 'wheelchair' else 'nondis',
                            owner=m['subject'], split=m['split'], subset=m['subset'])
                    for k, m in obj['meta'].items()}}
    meta = obj['meta']
    wc_test_subj = {m['subject'] for m in meta.values() if m['subset'] == 'wheelchair' and m['split'] == 'sub_test'}
    n_test_subj = sorted({m['subject'] for m in meta.values() if m['subset'] == 'normal' and m['split'] == 'sub_test'}
                         - wc_test_subj)
    val_subj = set(n_test_subj[::2])
    keys = list(meta)
    sz = dict(train=[k for k in keys if meta[k]['subset'] == 'normal' and meta[k]['split'] == 'sub_train'
                     and meta[k]['subject'] not in wc_test_subj],
              val=[k for k in keys if meta[k]['subset'] == 'normal' and meta[k]['split'] == 'sub_test'
                   and meta[k]['subject'] in val_subj],
              test=[k for k in keys if (meta[k]['subset'] == 'wheelchair' and meta[k]['split'] == 'sub_test') or
                    (meta[k]['subset'] == 'normal' and meta[k]['split'] == 'sub_test'
                     and meta[k]['subject'] not in val_subj and meta[k]['subject'] not in wc_test_subj)])
    wtr = sorted({meta[k]['subject'] for k in keys if meta[k]['subset'] == 'wheelchair' and meta[k]['split'] == 'sub_train'})
    wval = {wtr[0]} if wtr else set()
    so = dict(train=[k for k in keys if meta[k]['subset'] == 'wheelchair' and meta[k]['split'] == 'sub_train'
                     and meta[k]['subject'] not in wval],
              val=[k for k in keys if meta[k]['subset'] == 'wheelchair' and meta[k]['split'] == 'sub_train'
                   and meta[k]['subject'] in wval],
              test=[k for k in keys if meta[k]['subset'] == 'wheelchair' and meta[k]['split'] == 'sub_test'])
    info = dict(classes=classes, wc_test_subjects=sorted(wc_test_subj), normal_val_subjects=sorted(val_subj),
                wheelchair_val_subject=sorted(wval))
    F['n_classes'] = K
    return seq, F, {'SZ': sz, 'SO': so}, K, info


def windows(keys, seq, per_seg):
    W, S = maa.SK.CFG['WIN'], maa.SK.CFG['STRIDE']
    items = []
    for k in keys:
        L = len(seq[k])
        if L >= W:
            starts = list(range(0, L - W + 1, S))
            if len(starts) > per_seg:
                starts = [starts[int(round(i))] for i in np.linspace(0, len(starts) - 1, per_seg)]
            for s in starts:
                items.append((k, np.arange(s, s + W)))
        else:
            items.append((k, np.arange(W) % L))
    return items


def _patch_metrics(K):
    from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

    def cm(y_true, y_pred):
        return dict(acc=float(accuracy_score(y_true, y_pred)),
                    macro_f1=float(f1_score(y_true, y_pred, average='macro', labels=list(range(K)), zero_division=0)),
                    cm=confusion_matrix(y_true, y_pred, labels=list(range(K))).tolist())
    C.classification_metrics = cm


def build_k(name, K):
    import torch.nn as nn
    net = maa.build(name)
    last = None
    for mod_name, mod in net.named_modules():
        if isinstance(mod, nn.Linear):
            last = mod_name
    parent = net
    parts = last.split('.')
    for p in parts[:-1]:
        parent = getattr(parent, p)
    old = getattr(parent, parts[-1])
    setattr(parent, parts[-1], nn.Linear(old.in_features, K))
    return net


def unit_dir(P, u):
    return f"{P.work}/maa/runs/{u['exp']}/{u['aug']}/{u['model']}/{u['split']}/s{u['seed']}"


def run_unit(P, seq, F, prots, K, u):
    import torch
    rdir = unit_dir(P, u)
    C.seed_everything(u['seed'])
    torch.backends.cudnn.benchmark = True
    parts = prots[u['split']]
    tr = windows(parts['train'], seq, TRAIN_WIN_PER_SEG)
    va = windows(parts['val'], seq, EVAL_WIN_PER_SEG)
    te = windows(parts['test'], seq, EVAL_WIN_PER_SEG)
    m = u['model']
    CF = maa.CFG
    cfg = dict(epochs=40, lr=CF['LR'][m], wd=CF['WD'], batch=CF['BATCH'][m], accum=CF['ACCUM'][m],
               patience=8, workers=CF['WORKERS'], warmup=CF['WARMUP'], amp=True)
    C.log(f'windows train {len(tr)} / val {len(va)} / test {len(te)}')
    t0 = time.time()
    C.train_classifier(rdir, build_k(m, K), maa.SkelDS(tr, seq, F, True, u['aug']),
                       maa.SkelDS(va, seq, F, False, u['aug']), cfg)
    res = maa.evaluate(rdir, build_k(m, K), maa.SkelDS(te, seq, F, False, u['aug']), F, cfg)
    res.update(u, minutes=round((time.time() - t0) / 60, 1), gpu=torch.cuda.get_device_name(0))
    C.save_json(res, f'{rdir}/DONE.json')
    C.log(f"{u} -> wheelchair {res.get('vid_f1_dis', float('nan')):.4f}, non-wheelchair {res.get('vid_f1_nondis', float('nan')):.4f}")


def execute(P, seq, F, prots, K, todo):
    rnd = random.Random(C.SESSION)
    todo = list(todo)
    rnd.shuffle(todo)
    for u in todo:
        rdir = unit_dir(P, u)
        if os.path.exists(f'{rdir}/DONE.json') or not C.claim_run(rdir):
            continue
        if maa._done_elsewhere(rdir):
            C.release_run(rdir)
            continue
        C.log(f'===== {u} =====')
        try:
            run_unit(P, seq, F, prots, K, u)
        finally:
            C.release_run(rdir)
    left = [u for u in todo if not os.path.exists(f'{unit_dir(P, u)}/DONE.json')]
    C.log(f'Session finished. Open units: {len(left)}')


def summary(P, exp='S9', K=None, n_boot=2000, base='A0'):
    """Segment-level macro-F1 (%) mean +- sd over seeds, and paired bootstrap difference vs A0 of the same model."""
    import pandas as pd
    rows = [json.load(open(f)) for f in glob.glob(f'{P.work}/maa/runs/{exp}/*/*/*/*/DONE.json')]
    if not rows:
        return None, None
    df = pd.DataFrame(rows)
    out = []
    for (aug, model, prot), g in df.groupby(['aug', 'model', 'split']):
        r = dict(protocol=prot, model=model, aug=aug, seeds=len(g))
        for col, name in (('vid_f1_dis', 'wheelchair'), ('vid_f1_nondis', 'non-wheelchair')):
            if col in g and g[col].notna().any():
                v = g[col].dropna() * 100
                r[name] = f'{v.mean():.1f} ± {v.std(ddof=1) if len(v) > 1 else 0:.1f}'
        out.append(r)
    s = pd.DataFrame(out).sort_values(['protocol', 'model', 'aug'])
    # paired bootstrap on the wheelchair test segments, vs A0
    rng = np.random.default_rng(0)
    comp = []
    for (model, prot), g in df.groupby(['model', 'split']):
        for aug in sorted(set(g['aug']) - {base}):
            pairs = []
            for sd in sorted(set(g['seed'])):
                fb = f"{P.work}/maa/runs/{exp}/{base}/{model}/{prot}/s{sd}/video_probs.npz"
                fv = f"{P.work}/maa/runs/{exp}/{aug}/{model}/{prot}/s{sd}/video_probs.npz"
                if os.path.exists(fb) and os.path.exists(fv):
                    pairs.append((np.load(fb, allow_pickle=True), np.load(fv, allow_pickle=True)))
            if not pairs:
                continue
            for grp in ('dis', 'nondis'):
                idx = np.where(pairs[0][0]['group'] == grp)[0]
                if len(idx) == 0:
                    continue
                k = int(K or pairs[0][0]['probs'].shape[1])
                Y = np.stack([b['y'][idx] for b, _ in pairs])
                PB = np.stack([b['probs'][idx].argmax(1) for b, _ in pairs])
                PV = np.stack([v['probs'][idx].argmax(1) for _, v in pairs])
                f = lambda y, p: maa._f1(y, p, k)
                d0 = np.mean([f(Y[i], PV[i]) - f(Y[i], PB[i]) for i in range(len(pairs))])
                bs = []
                for _ in range(n_boot):
                    r_ = rng.integers(0, len(idx), len(idx))
                    bs.append(np.mean([f(Y[i, r_], PV[i, r_]) - f(Y[i, r_], PB[i, r_]) for i in range(len(pairs))]))
                lo, hi = np.percentile(bs, [2.5, 97.5]) * 100
                comp.append(dict(protocol=prot, model=model, aug=aug, group='wheelchair' if grp == 'dis' else 'non-wheelchair',
                                 seeds=len(pairs), segments=len(idx), diff=round(d0 * 100, 1), ci_lo=round(lo, 1),
                                 ci_hi=round(hi, 1), significant=bool(lo > 0 or hi < 0)))
    c = pd.DataFrame(comp)
    os.makedirs(f'{P.work}/maa/results', exist_ok=True)
    s.to_csv(f'{P.work}/maa/results/S9_summary.csv', index=False)
    c.to_csv(f'{P.work}/maa/results/S9_compare_vs_{base}.csv', index=False)
    return s, c


# ----------------------------------------------------------------------------- S9d: fusion (QDRF) on SAFER
def pose_quality(seq, F):
    """Fraction of frames without any joint, mean per group (input only)."""
    q = maa.video_quality(seq)
    out = {}
    for g in ('dis', 'nondis'):
        ks = [k for k in seq if F['videos'][k]['group'] == g]
        out['wheelchair' if g == 'dis' else 'non-wheelchair'] = float(np.mean([q[str(k)] for k in ks])) if ks else None
    return out


def fusion_compare(P, seq, F, K, variants, bases, models=('ST-GCN', 'ST-GCN++', 'CTR-GCN'), exp='S9', prot='SZ',
                   seeds=(0, 1, 2), n_boot=2000, name='S9_fusion_compare'):
    """variants / bases: aug name, 'FUSE:augA|augB' or 'QFUSE:augA|augB' (box stream | body stream, same exp).
    Paired bootstrap over test segments, per group. Writes S9_fusion_compare.csv (or name)."""
    import pandas as pd
    maa.video_quality(seq)
    rng = np.random.default_rng(0)

    def probs(spec, m, sd):
        def one(a):
            s_ = sd
            if '#' in a:                                # 'A4#1': same variant, another seed (ensemble control)
                a, k = a.split('#')
                s_ = seeds[(seeds.index(sd) + int(k)) % len(seeds)]
            f = f"{P.work}/maa/runs/{exp}/{a}/{m}/{prot}/s{s_}/video_probs.npz"
            if not os.path.exists(f):
                return None
            z = np.load(f, allow_pickle=True)
            return {str(v): (int(y), np.asarray(p, np.float64), str(g)) for v, y, p, g in
                    zip(z['videos'], z['y'], z['probs'], z['group'])}
        if ':' not in spec:
            return one(spec)
        kind, pair = spec.split(':', 1)
        a, b = pair.split('|')
        pa, pb = one(a), one(b)
        if pa is None or pb is None:
            return None
        out = {}
        for v, (y, p1, g) in pa.items():
            w = 0.5 * (1 + maa.QUAL.get(v, 0.0)) if kind == 'QFUSE' else 0.5
            out[v] = (y, (1 - w) * p1 + w * pb[v][1], g)
        return out

    rows = []
    for m in models:
        for base in bases:
            for var in variants:
                if var == base:
                    continue
                pairs = []
                for sd in seeds:
                    b, v = probs(base, m, sd), probs(var, m, sd)
                    if b and v:
                        pairs.append((b, v))
                if not pairs:
                    continue
                keys = sorted(pairs[0][0])
                grp = np.array([pairs[0][0][k][2] for k in keys])
                for g in ('dis', 'nondis'):
                    idx = np.where(grp == g)[0]
                    if len(idx) == 0:
                        continue
                    Y = np.array([[pr[0][keys[i]][0] for i in idx] for pr in pairs])
                    PB = np.array([[int(np.argmax(pr[0][keys[i]][1])) for i in idx] for pr in pairs])
                    PV = np.array([[int(np.argmax(pr[1][keys[i]][1])) for i in idx] for pr in pairs])
                    f = lambda y, p: maa._f1(y, p, K)
                    fb = np.mean([f(Y[k], PB[k]) for k in range(len(pairs))])
                    fv = np.mean([f(Y[k], PV[k]) for k in range(len(pairs))])
                    bs = []
                    for _ in range(n_boot):
                        r = rng.integers(0, len(idx), len(idx))
                        bs.append(np.mean([f(Y[k, r], PV[k, r]) - f(Y[k, r], PB[k, r]) for k in range(len(pairs))]))
                    lo, hi = np.percentile(bs, [2.5, 97.5]) * 100
                    rows.append(dict(model=m, variant=var, base=base, group='wheelchair' if g == 'dis' else 'non-wheelchair',
                                     seeds=len(pairs), base_f1=round(fb * 100, 1), variant_f1=round(fv * 100, 1),
                                     diff=round((fv - fb) * 100, 1), ci_lo=round(lo, 1), ci_hi=round(hi, 1),
                                     significant=bool(lo > 0 or hi < 0)))
    df = pd.DataFrame(rows)
    os.makedirs(f'{P.work}/maa/results', exist_ok=True)
    if len(df):
        df.to_csv(f'{P.work}/maa/results/{name}.csv', index=False)
    return df
