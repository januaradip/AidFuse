"""
MAA: Mobility-Aid-Aware skeleton augmentation for activity recognition on CARR.

Experiment S1 (this version): baselines without the proposed augmentation (aug 'A0') for four backbones
(ST-GCN, ST-GCN++, CTR-GCN, PoseC3D) on the nine CARR protocols (CS1-4, CL1-4, CG) and several seeds.

Builds on the public CARR benchmark code (github.com/januaradip/CARR):
  * frame index, protocol splits, run locks, trainer with checkpoint/resume  -> carr.common
  * 17 COCO keypoints (YOLOv8m-Pose), box-normalised sequences, ST-GCN / CTR-GCN -> carr.skeleton

Layout of the output (WORK):
  WORK/cache/frames.pkl, WORK/cache/seq_box.pkl         shared caches
  WORK/maa/runs/<exp>/<aug>/<model>/<split>/s<seed>/    checkpoints, DONE.json, video_probs.npz
  WORK/maa/results/<exp>_runs.csv, <exp>_summary.csv    collected results

Every unit (exp, aug, model, split, seed) is claimed with a lock file, so several Colab sessions (also from
different Google accounts sharing the same Drive folder) can work on the same experiment in parallel.
"""
import os, sys, glob, json, math, time, random, shutil, collections
import numpy as np

CARR_REPO = os.environ.get('CARR_REPO', '/content/CARR')
if CARR_REPO not in sys.path:
    sys.path.insert(0, CARR_REPO)
from carr import common as C                   # noqa: E402
from carr import skeleton as SK                # noqa: E402

MODELS = ['ST-GCN', 'ST-GCN++', 'CTR-GCN', 'PoseC3D']
SPLITS = ['CS1', 'CS2', 'CS3', 'CS4', 'CL1', 'CL2', 'CL3', 'CL4', 'CG']

CFG = dict(
    WIN=64, STRIDE=32,                         # windows as in the CARR benchmark
    EPOCHS=60, PATIENCE=12, WARMUP=3, WD=1e-4, WORKERS=2,
    BATCH={'ST-GCN': 64, 'ST-GCN++': 64, 'CTR-GCN': 64, 'PoseC3D': 32},
    ACCUM={'ST-GCN': 1, 'ST-GCN++': 1, 'CTR-GCN': 1, 'PoseC3D': 2},     # PoseC3D: effective batch 64
    LR={'ST-GCN': 1e-3, 'ST-GCN++': 1e-3, 'CTR-GCN': 1e-3, 'PoseC3D': 1e-3},
    HM_SIZE=56, HM_SIGMA=0.6, HM_RANGE=0.75, HM_FRAMES=32,              # PoseC3D heatmap volumes
)


# =============================================================================== data
def paths(data, work):
    return C.Paths(data, work)


def load_seq(P, F, pose_dir=None):
    """Box-normalised keypoint sequences {video: (T,17,3)}, cached once in WORK/cache/seq_box.pkl.
    pose_dir: folder with <video>.npz from the CARR benchmark (default WORK/cache/pose)."""
    cache = f'{P.cache}/seq_box.pkl'
    if os.path.exists(cache):
        obj = C.load_pickle(cache)
        if obj.get('sig') == P.signature() and len(obj['seq']) == len(F['videos']):
            return obj['seq']
    pose_dir = pose_dir or f'{P.cache}/pose'
    missing = [v for v in F['videos'] if not os.path.exists(f'{pose_dir}/{v}.npz')]
    if missing:
        raise FileNotFoundError(
            f'{len(missing)} of {len(F["videos"])} pose files missing in {pose_dir} (e.g. {missing[:3]}). '
            f'Set POSE_DIR to the pose cache of the CARR benchmark (CARR_work/cache/pose) or run the '
            f'pose-extraction cell.')
    C.log('Building box-normalised sequences (once)...')
    # carr.skeleton.load_sequences reads <cache>/pose/<video>.npz
    if os.path.basename(pose_dir.rstrip('/')) != 'pose':
        raise ValueError('POSE_DIR must be a folder named "pose" (e.g. .../CARR_work/cache/pose)')
    q = type('Q', (), {})()
    q.cache = os.path.dirname(pose_dir.rstrip('/'))
    seq = SK.load_sequences(q, F)
    C.save_pickle(dict(sig=P.signature(), seq=seq), cache)
    C.log(f'{len(seq)} sequences cached in {cache}')
    return seq


def make_windows(videos, seq):
    return SK.make_windows(videos, seq, None)


# ------------------------------------------------------------------ augmentations
def base_aug(x):
    """Augmentation of the CARR benchmark (kept for every variant): flip, rotation +-10 deg, scale 0.9-1.1."""
    if random.random() < 0.5:
        x = x[:, SK.FLIP]
        x[:, :, 0] *= -1
    a = math.radians(random.uniform(-10, 10))
    sc = random.uniform(0.9, 1.1)
    R = np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]], np.float32) * sc
    x[:, :, :2] = x[:, :, :2] @ R.T
    return x


try:
    import augs as _AG                         # generic (A1-A4) and proposed (U, U1-U4, U5) augmentations
    AUGS = dict(_AG.AUGS)
except ImportError:
    _AG, AUGS = None, {'A0': lambda x: x}


FLAGS = ('U5', 'C1', 'C2', 'ASM', 'MIX')       # representation / dataset-level parts of a variant name
_FILLED = {}                                    # C2-filled sequences, computed once per process and sequence dict


def parse_aug(name):
    """'A4+C1+C2' -> ('A4', {'C1', 'C2'}); 'U+U5' -> ('U', {'U5'}); 'U5' alone -> ('A0', {'U5'})."""
    parts = name.split('+')
    flags = {p for p in parts if p in FLAGS}
    rest = '+'.join(p for p in parts if p not in FLAGS) or 'A0'
    return rest, flags


def _filled(seq, keys):
    d = _FILLED.setdefault(id(seq), {})
    for v in keys:
        if v not in d:
            d[v] = _AG.c2_fill(np.asarray(seq[v]))
    return d


class SkelDS:
    def __init__(self, items, seq, F, train, aug='A0'):
        name, self.flags = parse_aug(aug)
        self.items, self.F, self.train, self.aug = items, F, train, AUGS[name]
        self.seq = _filled(seq, sorted({v for v, _ in items})) if 'C2' in self.flags else seq
        self.K = F.get('n_classes') or len(C.CLASSES)
        self.rate = None
        if 'ASM' in self.flags and train:            # per-joint missing rate of the training windows
            miss = np.zeros(17); n = 0
            for v, idx in items[::max(1, len(items) // 2000)]:
                w = np.asarray(self.seq[v][idx])
                miss += (w[..., 2] <= 0).sum(0); n += len(w)
            self.rate = miss / max(1, n)

    def __len__(self):
        return len(self.items)

    def _get(self, i):
        v, idx = self.items[i]
        return self.seq[v][idx].copy(), self.F['videos'][v]['cls']   # (T,17,3): x, y (box-normalised), confidence

    def _view(self, x):
        if self.train:
            x = self.aug(base_aug(x))
            if self.rate is not None:
                x = _AG.asm_mask(x, self.rate)
        if 'C1' in self.flags:
            x = _AG.c1_body_norm(x)
        elif 'U5' in self.flags:
            x = _AG.u5_upper_norm(x)
        return x

    def __getitem__(self, i):
        import torch
        x, y = self._get(i)
        if self.train and 'MIX' in self.flags:        # soft label (CrossEntropyLoss accepts class probabilities)
            t = np.zeros(self.K, np.float32); t[y] = 1
            if random.random() < 0.5:
                xb, yb = self._get(random.randrange(len(self.items)))
                x, lam = _AG.jmda_mix(self._view(x), self._view(xb))
                t = lam * t; t[yb] += 1 - lam
            else:
                x = self._view(x)
            y = torch.from_numpy(t)
        else:
            x = self._view(x)
        x = torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32)).permute(2, 0, 1).contiguous()
        return x, y, i


# =============================================================================== models
def build(name):
    import torch
    import torch.nn as nn
    if name in ('ST-GCN', 'CTR-GCN'):
        return SK.build_model(name)

    A0 = torch.tensor(SK.graph_A())                   # (3,17,17)

    class DataBN(nn.Module):
        def __init__(s, C=3, V=17):
            super().__init__()
            s.bn = nn.BatchNorm1d(C * V)

        def forward(s, x):
            N, Cc, T, V = x.shape
            x = x.permute(0, 3, 1, 2).reshape(N, V * Cc, T)
            return s.bn(x).reshape(N, V, Cc, T).permute(0, 2, 3, 1).contiguous()

    # ------------------------------------------------ ST-GCN++ (PYSKL: adaptive 'init' graph + multi-scale TCN)
    class AGCN(nn.Module):
        def __init__(s, cin, cout):
            super().__init__()
            s.K = A0.shape[0]
            s.A = nn.Parameter(A0.clone())             # initialised with the skeleton graph, fully learnable
            s.conv = nn.Conv2d(cin, cout * s.K, 1)
            s.bn = nn.BatchNorm2d(cout)
            s.res = nn.Identity() if cin == cout else nn.Sequential(nn.Conv2d(cin, cout, 1), nn.BatchNorm2d(cout))
            s.relu = nn.ReLU(inplace=True)

        def forward(s, x):
            N, _, T, V = x.shape
            y = s.conv(x).view(N, s.K, -1, T, V)
            y = torch.einsum('nkctv,kwv->nctw', y, s.A)
            return s.relu(s.bn(y) + s.res(x))

    class MSTCN(nn.Module):
        def __init__(s, cin, cout, stride=1):
            super().__init__()
            nb = 6
            mids = [cout // nb] * (nb - 1) + [cout - (cout // nb) * (nb - 1)]
            br = []
            for d, m in zip((1, 2, 3, 4), mids[:4]):
                br.append(nn.Sequential(nn.Conv2d(cin, m, 1), nn.BatchNorm2d(m), nn.ReLU(inplace=True),
                                        nn.Conv2d(m, m, (3, 1), (stride, 1), (d, 0), (d, 1))))
            br.append(nn.Sequential(nn.Conv2d(cin, mids[4], 1), nn.BatchNorm2d(mids[4]), nn.ReLU(inplace=True),
                                    nn.MaxPool2d((3, 1), (stride, 1), (1, 0))))
            br.append(nn.Conv2d(cin, mids[5], 1, (stride, 1)))
            s.br = nn.ModuleList(br)
            s.out = nn.Sequential(nn.BatchNorm2d(cout), nn.ReLU(inplace=True), nn.Conv2d(cout, cout, 1),
                                  nn.BatchNorm2d(cout))

        def forward(s, x):
            return s.out(torch.cat([b(x) for b in s.br], 1))

    class PPBlock(nn.Module):
        def __init__(s, cin, cout, stride=1, residual=True):
            super().__init__()
            s.gcn, s.tcn = AGCN(cin, cout), MSTCN(cout, cout, stride)
            if not residual:
                s.res = lambda x: 0
            elif cin == cout and stride == 1:
                s.res = nn.Identity()
            else:
                s.res = nn.Sequential(nn.Conv2d(cin, cout, 1, (stride, 1)), nn.BatchNorm2d(cout))
            s.relu = nn.ReLU(inplace=True)

        def forward(s, x):
            return s.relu(s.tcn(s.gcn(x)) + s.res(x))

    class STGCNpp(nn.Module):
        def __init__(s):
            super().__init__()
            s.bn = DataBN()
            cfg = [(3, 64, 1, False), (64, 64, 1, True), (64, 64, 1, True), (64, 64, 1, True),
                   (64, 128, 2, True), (128, 128, 1, True), (128, 128, 1, True),
                   (128, 256, 2, True), (256, 256, 1, True), (256, 256, 1, True)]
            s.blocks = nn.ModuleList(PPBlock(*c) for c in cfg)
            s.drop = nn.Dropout(0.1)
            s.fc = nn.Linear(256, 6)

        def forward(s, x):
            x = s.bn(x)
            for b in s.blocks:
                x = b(x)
            return s.fc(s.drop(x.mean((2, 3))))

    # ------------------------------------------------ PoseC3D (PYSKL: SlowOnly-R50 pose variant on joint heatmaps)
    class Bottleneck3d(nn.Module):
        def __init__(s, cin, planes, stride_s=1, inflate=False):
            super().__init__()
            kt = 3 if inflate else 1
            s.c1 = nn.Sequential(nn.Conv3d(cin, planes, (kt, 1, 1), padding=(kt // 2, 0, 0), bias=False),
                                 nn.BatchNorm3d(planes), nn.ReLU(inplace=True))
            s.c2 = nn.Sequential(nn.Conv3d(planes, planes, (1, 3, 3), (1, stride_s, stride_s), (0, 1, 1), bias=False),
                                 nn.BatchNorm3d(planes), nn.ReLU(inplace=True))
            s.c3 = nn.Sequential(nn.Conv3d(planes, planes * 4, 1, bias=False), nn.BatchNorm3d(planes * 4))
            s.down = None
            if stride_s != 1 or cin != planes * 4:
                s.down = nn.Sequential(nn.Conv3d(cin, planes * 4, 1, (1, stride_s, stride_s), bias=False),
                                       nn.BatchNorm3d(planes * 4))
            s.relu = nn.ReLU(inplace=True)

        def forward(s, x):
            r = x if s.down is None else s.down(x)
            return s.relu(s.c3(s.c2(s.c1(x))) + r)

    class PoseC3D(nn.Module):
        """Input (N,3,T,17) box-normalised keypoints -> joint heatmap volume (N,17,T/2,56,56) built on the GPU."""

        def __init__(s):
            super().__init__()
            S = CFG['HM_SIZE']
            g = torch.linspace(-CFG['HM_RANGE'], CFG['HM_RANGE'], S)
            s.register_buffer('gx', g.view(1, 1, 1, 1, S))
            s.register_buffer('gy', g.view(1, 1, 1, S, 1))
            s.sig = CFG['HM_SIGMA'] * (2 * CFG['HM_RANGE'] / (S - 1))     # sigma in grid cells -> coordinates
            s.stem = nn.Sequential(nn.Conv3d(17, 32, (1, 7, 7), padding=(0, 3, 3), bias=False),
                                   nn.BatchNorm3d(32), nn.ReLU(inplace=True))
            layers, cin = [], 32
            for planes, n, inflate in ((32, 4, False), (64, 6, True), (128, 3, True)):
                for i in range(n):
                    layers.append(Bottleneck3d(cin, planes, 2 if i == 0 else 1, inflate))
                    cin = planes * 4
            s.body = nn.Sequential(*layers)
            s.drop = nn.Dropout(0.5)
            s.fc = nn.Linear(cin, 6)

        def heatmaps(s, x):
            T = x.shape[2]
            step = max(1, T // CFG['HM_FRAMES'])
            x = x[:, :, ::step][:, :, :CFG['HM_FRAMES']].float()       # (N,3,T',17)
            u = x[:, 0].permute(0, 2, 1)[..., None, None]               # (N,17,T',1,1)
            v = x[:, 1].permute(0, 2, 1)[..., None, None]
            c = x[:, 2].permute(0, 2, 1)[..., None, None]
            return c * torch.exp(-((s.gx - u) ** 2 + (s.gy - v) ** 2) / (2 * s.sig ** 2))

        def forward(s, x):
            h = s.heatmaps(x)
            return s.fc(s.drop(s.body(s.stem(h)).mean((2, 3, 4))))

    return {'ST-GCN++': STGCNpp, 'PoseC3D': PoseC3D}[name]()


def n_params(m):
    return sum(p.numel() for p in m.parameters())


# =============================================================================== units
def units(exp, augs, models, splits, seeds):
    return [dict(exp=exp, aug=a, model=m, split=s, seed=int(sd))
            for a in augs for m in models for s in splits for sd in seeds]


def unit_dir(P, u):
    return f"{P.work}/maa/runs/{u['exp']}/{u['aug']}/{u['model']}/{u['split']}/s{u['seed']}"


def unit_tag(u):
    return f"{u['exp']} {u['aug']} {u['model']} {u['split']} s{u['seed']}"


def evaluate(rdir, model, test_ds, F, cfg):
    """Video-level macro-F1 (mean of window probabilities per video), overall and per group; saves the
    per-video probabilities for the paired bootstrap."""
    import torch
    from torch.utils.data import DataLoader
    model.load_state_dict(torch.load(f'{rdir}/best.pt', map_location='cpu', weights_only=False)['model'])
    model.to('cuda')
    dl = DataLoader(test_ds, batch_size=cfg['batch'], shuffle=False, num_workers=cfg['workers'], pin_memory=True)
    yt, yp, probs, idx = C.predict(model, dl, 'cuda', True)
    wv = np.array([test_ds.items[i][0] for i in idx])
    vids = sorted(set(wv))
    vp = np.stack([probs[wv == v].mean(0) for v in vids])
    vy = np.array([F['videos'][v]['cls'] for v in vids])
    grp = np.array([F['videos'][v]['group'] for v in vids])
    own = np.array([F['videos'][v]['owner'] for v in vids])
    np.savez_compressed(f'{rdir}/video_probs.npz', videos=np.array(vids), y=vy, probs=vp, group=grp, owner=own)
    res = {}
    for tag, m in (('all', np.ones(len(vids), bool)), ('dis', grp == 'dis'), ('nondis', grp == 'nondis')):
        if m.sum():
            r = C.classification_metrics(vy[m], vp[m].argmax(1))
            res[f'vid_f1_{tag}'], res[f'vid_acc_{tag}'], res[f'n_{tag}'] = r['macro_f1'], r['acc'], int(m.sum())
            if tag == 'all':
                C.save_json(r['cm'], f'{rdir}/confusion_video.json')
    r = C.classification_metrics(yt, yp)
    res['win_f1_all'], res['win_acc_all'] = r['macro_f1'], r['acc']
    return res


def run_unit(P, F, seq, u):
    import torch
    rdir = unit_dir(P, u)
    C.seed_everything(u['seed'])
    torch.backends.cudnn.benchmark = True
    parts = C.load_split(P, u['split'])
    tr, va, te = (make_windows(parts[k], seq) for k in ('train', 'val', 'test'))
    m = u['model']
    cfg = dict(epochs=CFG['EPOCHS'], lr=CFG['LR'][m], wd=CFG['WD'], batch=CFG['BATCH'][m], accum=CFG['ACCUM'][m],
               patience=CFG['PATIENCE'], workers=CFG['WORKERS'], warmup=CFG['WARMUP'], amp=True)
    C.log(f'windows train {len(tr)} / val {len(va)} / test {len(te)}')
    t0 = time.time()
    C.train_classifier(rdir, build(m), SkelDS(tr, seq, F, True, u['aug']), SkelDS(va, seq, F, False, u['aug']), cfg)
    res = evaluate(rdir, build(m), SkelDS(te, seq, F, False, u['aug']), F, cfg)
    hist = json.load(open(f'{rdir}/history.json')) if os.path.exists(f'{rdir}/history.json') else []
    res.update(u, params=n_params(build(m)), epochs=len(hist),
               best_val_f1=max([h['val_f1'] for h in hist], default=None),
               gpu=torch.cuda.get_device_name(0), minutes=round((time.time() - t0) / 60, 1))
    C.save_json(res, f'{rdir}/DONE.json')
    C.log(f"{unit_tag(u)} finished: video macro-F1 {res['vid_f1_all']:.4f}"
          + (f" (dis {res['vid_f1_dis']:.4f})" if 'vid_f1_dis' in res else ''))


def _done_elsewhere(rdir):
    """True if DONE.json exists (fresh directory listing; Google Drive caches os.path.exists across sessions)."""
    try:
        return 'DONE.json' in os.listdir(rdir) or os.path.exists(f'{rdir}/DONE.json')
    except OSError:
        return False


def execute(P, F, seq, todo, order='forward', session_seed=None):
    """todo: list of units, or list of stages (lists of units); stages are done in order, shuffled within."""
    stages = todo if todo and isinstance(todo[0], list) else [todo]
    rnd = random.Random(session_seed if session_seed is not None else C.SESSION)
    todo = []
    for st in stages:
        st = list(st)
        if order == 'reverse':
            st = st[::-1]
        elif order == 'shuffle':
            rnd.shuffle(st)
        todo += st
    busy = []
    for u in todo:
        rdir = unit_dir(P, u)
        if os.path.exists(f'{rdir}/DONE.json'):
            continue
        if not C.claim_run(rdir):
            busy.append(unit_tag(u))
            continue
        if _done_elsewhere(rdir):                     # Drive of another session may sync late: re-check after claim
            C.release_run(rdir)
            continue
        C.log(f'===== {unit_tag(u)} =====')
        try:
            run_unit(P, F, seq, u)
        finally:
            C.release_run(rdir)
    left = [u for u in todo if not os.path.exists(f'{unit_dir(P, u)}/DONE.json')]
    C.log(f'Session finished. Open units: {len(left)} (in progress elsewhere: {len(busy)}).')
    return left


def status(P, todo):
    import pandas as pd
    rows = []
    for u in todo:
        d = unit_dir(P, u)
        st = 'done' if os.path.exists(f'{d}/DONE.json') else ('running' if os.path.exists(f'{d}/LOCK') else
                                                              ('started' if os.path.exists(f'{d}/history.json') else 'open'))
        rows.append(dict(u, status=st))
    df = pd.DataFrame(rows)
    return df.pivot_table(index=['aug', 'model'], columns='status', values='seed', aggfunc='count', fill_value=0)


# =============================================================================== results
def collect(P, exp):
    """exp: name or list of names (e.g. ['S1', 'S3'] so that the A0 baselines of S1 are included)."""
    import pandas as pd
    exps = [exp] if isinstance(exp, str) else list(exp)
    rows = []
    for e in exps:
        for f in glob.glob(f'{P.work}/maa/runs/{e}/*/*/*/*/DONE.json'):
            with open(f) as fh:
                rows.append(json.load(fh))
    if not rows:
        C.log('No finished units yet.')
        return None
    df = pd.DataFrame(rows)
    os.makedirs(f'{P.work}/maa/results', exist_ok=True)
    df.to_csv(f'{P.work}/maa/results/{"_".join(exps)}_runs.csv', index=False)
    return df


def summary(P, exp):
    """Video macro-F1 (%) per model and protocol: CS (mean of 4 folds), CS-dis, CL, CG; mean +- sd over seeds."""
    import pandas as pd
    df = collect(P, exp)
    if df is None:
        return None
    df['protocol'] = df['split'].str.replace(r'\d+$', '', regex=True)
    out = []
    for (aug, model), g in df.groupby(['aug', 'model']):
        row = dict(aug=aug, model=model)
        for prot, col, name in (('CS', 'vid_f1_all', 'CS'), ('CS', 'vid_f1_dis', 'CS-dis'),
                                ('CL', 'vid_f1_all', 'CL'), ('CG', 'vid_f1_all', 'CG')):
            h = g[g.protocol == prot]
            if h.empty or col not in h:
                continue
            per_seed = h.groupby('seed')[col].mean() * 100          # fold mean per seed
            nf = h.groupby('seed')['split'].nunique()
            full = nf.max() == (1 if prot == 'CG' else 4)
            row[name] = f"{per_seed.mean():.1f} ± {per_seed.std(ddof=1) if len(per_seed) > 1 else 0:.1f}" \
                        f" (n={len(per_seed)}{'' if full else ', incomplete'})"
        row['params_M'] = round(g['params'].iloc[0] / 1e6, 2)
        row['min_per_run'] = round(g['minutes'].median(), 1)
        out.append(row)
    s = pd.DataFrame(out)
    tag = exp if isinstance(exp, str) else '_'.join(exp)
    s.to_csv(f'{P.work}/maa/results/{tag}_summary.csv', index=False)
    return s


# ------------------------------------------------------------------ paired comparison with bootstrap CI
def _f1(y, p, k=6):
    f = []
    for c in range(k):
        if (y == c).any():
            tp = np.sum((y == c) & (p == c)); fp = np.sum((y != c) & (p == c)); fn = np.sum((y == c) & (p != c))
            f.append(2 * tp / max(1, 2 * tp + fp + fn))
    return float(np.mean(f)) if f else 0.0


QUAL = {}                                       # video -> fraction of frames without any pose (set by video_quality)


def video_quality(seq):
    """Fraction of frames without any detected joint, per video (input-only, no labels). Fills maa.QUAL."""
    QUAL.clear()
    for v, x in seq.items():
        x = np.asarray(x)
        QUAL[str(v)] = float((~(x[..., 2] > 0).any(1)).mean()) if len(x) else 0.0
    return QUAL


def _pooled(P, exp, aug, model, splits, seed):
    """Video predictions of one seed pooled over the folds of a protocol (each video is tested once).
    exp 'FUSE': aug = 'expA@augA|expB@augB|...' -> late fusion (mean of the video probabilities) of those runs (C3)."""
    if exp == 'QFUSE':                         # quality-gated fusion: box stream | body stream
        (ea, aa), (eb, ab) = [p.split('@') for p in aug.split('|')]
        pa, pb = _pooled_probs(P, ea, aa, model, splits, seed), _pooled_probs(P, eb, ab, model, splits, seed)
        if pa is None or pb is None:
            return None
        out = {}
        for v, (y, p1, g) in pa.items():
            w = 0.5 * (1 + QUAL.get(v, 0.0))       # weight of the body stream grows with the missing-pose fraction
            out[v] = (y, int(np.argmax((1 - w) * p1 + w * pb[v][1])), g)
        return out
    if exp == 'FUSE':
        parts = [p.split('@') for p in aug.split('|')]
        probs = [_pooled_probs(P, e, a, model, splits, seed) for e, a in parts]
        if any(p is None for p in probs):
            return None
        return {v: (y, int(np.argmax(np.mean([p[v][1] for p in probs], 0))), g) for v, (y, _, g) in probs[0].items()}
    pr = _pooled_probs(P, exp, aug, model, splits, seed)
    return None if pr is None else {v: (y, int(np.argmax(p)), g) for v, (y, p, g) in pr.items()}


def _pooled_probs(P, exp, aug, model, splits, seed):
    if '#' in aug:                                  # 'A4#1': same variant, another seed (seed + 1 mod 5) - ensemble control
        aug, k = aug.split('#')
        seed = (seed + int(k)) % 5
    out = {}
    for sp in splits:
        f = f"{P.work}/maa/runs/{exp}/{aug}/{model}/{sp}/s{seed}/video_probs.npz"
        if not os.path.exists(f):
            return None
        z = np.load(f, allow_pickle=True)
        for v, y, p, g in zip(z['videos'], z['y'], z['probs'], z['group']):
            out[str(v)] = (int(y), np.asarray(p, np.float64), str(g))
    return out


def compare(P, variants, models, seeds=(0, 1, 2, 3, 4), base=('S1', 'A0'), n_boot=2000, name=None):
    """Macro-F1 difference variant - base (points), mean over the seeds both have, with a paired bootstrap over
    videos (95 % CI). Rows: CG, CS (pooled over 4 folds) all / dis / nondis, CL all / dis (if available)."""
    import pandas as pd
    rng = np.random.default_rng(0)
    prots = {'CG': ['CG'], 'CS': ['CS1', 'CS2', 'CS3', 'CS4'], 'CL': ['CL1', 'CL2', 'CL3', 'CL4']}
    rows = []
    for (vexp, vaug) in variants:
        for m in models:
            for prot, sps in prots.items():
                pairs = []
                for sd in seeds:
                    b, v = _pooled(P, base[0], base[1], m, sps, sd), _pooled(P, vexp, vaug, m, sps, sd)
                    if b and v:
                        pairs.append((b, v))
                if not pairs:
                    continue
                vids = sorted(pairs[0][0])
                grp = np.array([pairs[0][0][k][2] for k in vids])
                for sub in (['all', 'dis', 'nondis'] if prot != 'CG' else ['all']):
                    if prot == 'CL' and sub == 'nondis':
                        continue
                    idx = np.arange(len(vids)) if sub == 'all' else np.where(grp == sub)[0]
                    if len(idx) == 0:
                        continue
                    Y = np.array([[pr[0][vids[i]][0] for i in idx] for pr in pairs])
                    PB = np.array([[pr[0][vids[i]][1] for i in idx] for pr in pairs])
                    PV = np.array([[pr[1][vids[i]][1] for i in idx] for pr in pairs])
                    fb = np.mean([_f1(Y[k], PB[k]) for k in range(len(pairs))])
                    fv = np.mean([_f1(Y[k], PV[k]) for k in range(len(pairs))])
                    d = []
                    for _ in range(n_boot):
                        r = rng.integers(0, len(idx), len(idx))
                        d.append(np.mean([_f1(Y[k, r], PV[k, r]) - _f1(Y[k, r], PB[k, r]) for k in range(len(pairs))]))
                    lo, hi = np.percentile(d, [2.5, 97.5]) * 100
                    rows.append(dict(variant=f'{vexp}:{vaug}', model=m, protocol=prot if sub == 'all' else f'{prot}-{sub}',
                                     seeds=len(pairs), videos=len(idx), base=round(fb * 100, 1), variant_f1=round(fv * 100, 1),
                                     diff=round((fv - fb) * 100, 1), ci_lo=round(lo, 1), ci_hi=round(hi, 1),
                                     significant=bool(lo > 0 or hi < 0)))
    df = pd.DataFrame(rows)
    if len(df):
        os.makedirs(f'{P.work}/maa/results', exist_ok=True)
        df.to_csv(f"{P.work}/maa/results/{name or f'compare_vs_{base[0]}_{base[1]}'}.csv", index=False)
    return df


# ----------------------------------------------------------------------------- S5: per mobility-aid type
AID_TYPE = {'P07': 'wheelchair', 'P11': 'wheelchair', 'P08': 'crutches', 'P10': 'crutches', 'P13': 'crutches',
            'P09': 'cane', 'P12': 'cane'}


def _owners(P, exp, aug, model, splits, seed):
    out = {}
    if '|' in aug:
        exp, aug = aug.split('|')[0].split('@')
    aug = aug.split('#')[0]
    for sp in splits:
        f = f"{P.work}/maa/runs/{exp}/{aug}/{model}/{sp}/s{seed}/video_probs.npz"
        if os.path.exists(f):
            z = np.load(f, allow_pickle=True)
            out.update({str(v): str(o) for v, o in zip(z['videos'], z['owner'])})
    return out


def aid_breakdown(P, variants, models, seeds=(0, 1, 2, 3, 4), base=('S3', 'A4'), n_boot=2000, name='S5_per_aid'):
    """Macro-F1 and accuracy per mobility-aid type (wheelchair / crutches / cane) on the aid-user videos of CG,
    CS and CL (each pooled over its folds), with a paired bootstrap over videos against base."""
    import pandas as pd
    rng = np.random.default_rng(0)
    prots = {'CG': ['CG'], 'CS': ['CS1', 'CS2', 'CS3', 'CS4'], 'CL': ['CL1', 'CL2', 'CL3', 'CL4']}
    rows = []
    for (vexp, vaug) in variants:
        for m in models:
            for prot, sps in prots.items():
                pairs = []
                for sd in seeds:
                    b, v = _pooled(P, base[0], base[1], m, sps, sd), _pooled(P, vexp, vaug, m, sps, sd)
                    if b and v:
                        pairs.append((b, v))
                if not pairs:
                    continue
                own = _owners(P, base[0], base[1], m, sps, seeds[0])
                vids = sorted(pairs[0][0])
                for t in ('wheelchair', 'crutches', 'cane'):
                    idx = np.array([i for i, k in enumerate(vids) if AID_TYPE.get(own.get(k, ''), '') == t])
                    if len(idx) == 0:
                        continue
                    Y = np.array([[pr[0][vids[i]][0] for i in idx] for pr in pairs])
                    PB = np.array([[pr[0][vids[i]][1] for i in idx] for pr in pairs])
                    PV = np.array([[pr[1][vids[i]][1] for i in idx] for pr in pairs])
                    fb = np.mean([_f1(Y[k], PB[k]) for k in range(len(pairs))])
                    fv = np.mean([_f1(Y[k], PV[k]) for k in range(len(pairs))])
                    ab, av = (PB == Y).mean(), (PV == Y).mean()
                    bs = []
                    for _ in range(n_boot):
                        r = rng.integers(0, len(idx), len(idx))
                        bs.append(np.mean([_f1(Y[k, r], PV[k, r]) - _f1(Y[k, r], PB[k, r]) for k in range(len(pairs))]))
                    lo, hi = np.percentile(bs, [2.5, 97.5]) * 100
                    rows.append(dict(variant=f'{vexp}:{vaug}', model=m, protocol=prot, aid=t, videos=len(idx),
                                     seeds=len(pairs), base_f1=round(fb * 100, 1), variant_f1=round(fv * 100, 1),
                                     diff=round((fv - fb) * 100, 1), ci_lo=round(lo, 1), ci_hi=round(hi, 1),
                                     significant=bool(lo > 0 or hi < 0), base_acc=round(ab * 100, 1),
                                     variant_acc=round(av * 100, 1)))
    df = pd.DataFrame(rows)
    os.makedirs(f'{P.work}/maa/results', exist_ok=True)
    if len(df):
        df.to_csv(f'{P.work}/maa/results/{name}.csv', index=False)
    return df
