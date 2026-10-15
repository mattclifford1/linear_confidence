'''
GPU half of the extra-experiments export.

    uv run python export_torch.py features
    uv run python export_torch.py train --seeds 10

features  extracts frozen ImageNet embeddings of every image dataset once into
          features/; export_projections.py then reads "<dataset> [resnet18]"
          from there and fits its sklearn models on the embeddings.
train     trains the projection_models presets and writes the same .npz format
          as export_projections.py, so run_wide.py reads both. Each fit is
          exported twice, as "<net>-prob" (sigmoid - 1/2, bounded) and
          "<net>-logit", with Balanced Weights, Focal and LDAM refits as
          baselines.
'''
import argparse
import json
import os
import time
import warnings

import numpy as np

import fingerprint as fp

warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split       # noqa: E402
from projection_models.datasets import trainer_kwargs      # noqa: E402
from projection_models.torch import presets                # noqa: E402

import export_projections as ep                            # noqa: E402
import extra_grid                                          # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
FEATURES = os.path.join(HERE, 'features')

#: dataset family -> {net name: (preset, epochs)}
NETS = {
    'MNIST': {'ConvNet': (presets.MnistConvTrainer, 10)},
    'Fashion-MNIST': {'ConvNet': (presets.MnistConvTrainer, 10)},
    'MedMNIST': {'ConvNet': (presets.MedMnistConvTrainer, 15)},
    'CIFAR': {'ConvNet': (presets.CifarConvTrainer, 20),
              'ResNet': (presets.CifarResNetTrainer, 20)},
    'EuroSAT': {'ConvNet': (presets.EuroSatConvTrainer, 20),
                'ResNet': (presets.EuroSatResNetTrainer, 20)},
}
#: refits exported as test-set predictions, like SMOTE/BW in export_projections
BASELINES = {'bw': {'class_weight': 'balanced'},
             'focal': {'loss': 'focal'},
             'ldam': {'loss': 'ldam'}}


def base_name(spec):
    '''the toy_datasets name behind a task or variant name'''
    spec = ep.source_specs(spec)[0]
    spec = ep.VARIANTS.get(spec, (spec,))[0]
    return ep.resolve_dataset(spec)[0]


def family(base):
    if base.endswith('MNIST') and base not in ('MNIST', 'Fashion-MNIST'):
        return 'MedMNIST'
    if base.startswith('CIFAR') or base == 'SVHN':
        return 'CIFAR'
    return base


# --------------------------------------------------------------- features ---
def extract(spec, backbone, force=False):
    from projection_models.torch.features import FrozenFeatureProbe
    path = ep.features_path(spec, backbone)
    if os.path.exists(path) and not force:
        return path, 'cached'
    X, _, _ = ep._load_xy(spec)
    probe = FrozenFeatureProbe(backbone=backbone,
                               **trainer_kwargs(base_name(spec)))
    t0 = time.time()
    Z = probe.transform(X.astype(np.float32))
    os.makedirs(FEATURES, exist_ok=True)
    np.savez(path, Z=Z.astype(np.float32), x_hash=ep.x_hash(X),
             seconds=time.time() - t0)
    return path, f'{Z.shape} in {time.time() - t0:.0f}s'


# ------------------------------------------------------------------ train ---
def _fit(preset, epochs, X, y, seed, base, **kw):
    return preset(seed=seed, **trainer_kwargs(base), **kw).fit(X, y, epochs=epochs)


def one_case(spec, net, seed):
    '''both calibration modes of one (task, net, seed), keyed like export_projections'''
    base = base_name(spec)
    preset, epochs = NETS[family(base)][net]
    t0 = time.time()
    train, test, meta = ep.load_split(spec, seed)
    Xtr, ytr = train['X'].astype(np.float32), train['y']
    Xte, yte = test['X'].astype(np.float32), test['y']

    data = np.array([data_hash(spec, seed, (train, test))])
    out = {space: {f's{seed}_data': data} for space in ('prob', 'logit')}
    for mode in ('naive', 'split'):
        if mode == 'naive':
            X_fit, y_fit, X_cert, y_cert = Xtr, ytr, Xtr, ytr
        else:
            n_min = int(np.bincount(ytr, minlength=2)[1])
            k = int(np.floor(n_min * ep.CAL_FRAC))
            if k < 2 or n_min - k < 2:
                continue
            X_fit, X_cert, y_fit, y_cert = train_test_split(
                Xtr, ytr, test_size=ep.CAL_FRAC, stratify=ytr, random_state=seed)
        model = _fit(preset, epochs, X_fit, y_fit, seed, base)
        preds = {label: np.asarray(_fit(preset, epochs, X_fit, y_fit, seed, base,
                                        **kw).predict(Xte)).reshape(-1).astype(np.int8)
                 for label, kw in BASELINES.items()}
        p = f's{seed}_{mode}_'
        for space in ('prob', 'logit'):
            model.net.projection_space = space
            o = out[space]
            o[p + 'z_cert'] = ep._proj(model, X_cert)
            o[p + 'y_cert'] = y_cert.astype(np.int8)
            o[p + 'z_test'] = ep._proj(model, Xte)
            o[p + 'y_test'] = yte.astype(np.int8)
            o[p + 'z_fit'] = ep._proj(model, X_fit)
            o[p + 'y_fit'] = y_fit.astype(np.int8)
            # the logit's boundary is 0 too: sigmoid(0) - 1/2 = 0
            o[p + 'threshold'] = np.array([ep._thresh(model)])
            for label, pred in preds.items():
                o[p + f'pred_{label}'] = pred
    for space, o in out.items():
        if not o:
            raise ValueError('no usable calibration mode')
        o['meta'] = np.array([json.dumps(
            {**meta, 'dataset': spec, 'model': f'{net}-{space}', 'epochs': epochs,
             'n_train': int(len(ytr)), 'n_test': int(len(yte)),
             'train_counts': np.bincount(ytr, minlength=2).tolist(),
             'test_counts': np.bincount(yte, minlength=2).tolist(),
             'n_features': int(Xtr.shape[1]),
             'seconds': round(time.time() - t0, 2)})])
    return out


def data_hash(spec, seed, split=None):
    train, test = split or ep.load_split(spec, seed)[:2]
    return fp.arrays_hash(train['X'], train['y'], test['X'], test['y'])


def export_settings(spec, net):
    '''what a (task, net) export depends on besides the seed'''
    base = base_name(spec)
    out = ep.export_settings(spec, net, code_of=(
        one_case, _fit, data_hash, ep.load_split, ep.split_options, ep.resolve_dataset,
        ep.shift_specs, ep.source_specs, ep._load_xy, ep._as_minority_class_1,
        ep._proj, ep._thresh))
    return {**out, 'net': fp.stable(NETS[family(base)][net]),
            'trainer': fp.stable(trainer_kwargs(base)), 'baselines': BASELINES}


def run_pair(spec, net, seeds, force, out_dir):
    paths = {space: os.path.join(out_dir, f'{ep._slug(spec)}__{net}-{space}.npz')
             for space in ('prob', 'logit')}
    merged = {space: {} if force else ep._read_npz(p) for space, p in paths.items()}
    settings = export_settings(spec, net)
    present = sorted(int(k[1:].split('_')[0]) for k in merged['prob']
                     if k.endswith('_naive_z_test'))
    todo = [s for s in seeds
            if not all(f's{s}_naive_z_test' in m for m in merged.values())]
    for m in merged.values():
        why = ep.check_existing(m, settings, present if todo else [],
                                lambda s: data_hash(spec, s))
        if why:
            return 'stale', why
    if not todo:
        for space, m in merged.items():
            if m and 'settings' not in m:            # adopt, on trust
                m['settings'] = np.array([json.dumps(settings)])
                ep._write_npz(paths[space], m)
        return 'cached', None
    for m in merged.values():
        m['settings'] = np.array([json.dumps(settings)])
    errors = []
    for seed in todo:
        try:
            out = one_case(spec, net, seed)
        except Exception as exc:
            errors.append(f'seed {seed}: {type(exc).__name__}: {exc}')
            continue
        for space, o in out.items():
            merged[space].update(o)
            ep._write_npz(paths[space], merged[space])
    if not merged['prob']:
        return 'failed', '; '.join(errors[:2])
    return 'ok', ('; '.join(errors[:2]) if errors else None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('job', choices=['features', 'train'])
    ap.add_argument('--datasets', nargs='*', default=extra_grid.IMAGE)
    ap.add_argument('--backbones', nargs='*', default=extra_grid.BACKBONES)
    ap.add_argument('--nets', nargs='*', default=['ConvNet', 'ResNet'])
    ap.add_argument('--seeds', type=int, default=10)
    ap.add_argument('--force', action='store_true')
    ap.add_argument('--out', default=extra_grid.OUT)
    args = ap.parse_args()

    t0 = time.time()
    if args.job == 'features':
        for spec in args.datasets:
            for source in ep.source_specs(spec):
                for backbone in args.backbones:
                    path, status = extract(source, backbone, args.force)
                    print(f'{source} [{backbone}]: {status}', flush=True)
    else:
        seeds = list(range(args.seeds))
        for spec in args.datasets:
            for net in args.nets:
                if net not in NETS[family(base_name(spec))]:
                    continue
                status, err = run_pair(spec, net, seeds, args.force, args.out)
                print(f'{spec} / {net}: {status} ({time.time() - t0:.0f}s)'
                      + (f'  {err}' if err else ''), flush=True)
    print(f'done ({time.time() - t0:.0f}s)')


if __name__ == '__main__':
    main()
