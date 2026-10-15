'''
The extra-experiments grid (../papers/ecai-extra-experiments): which datasets,
tasks, models and methods each stage runs, and a driver for export_projections.

    uv run python extra_grid.py tabular --seeds 10 --jobs 24
    uv run python extra_grid.py image-fast --seeds 10 --jobs 8
    uv run python extra_grid.py --list                    # print every stage

Projections go to projections-extra/ (override with --out), so the original
34-dataset grid in projections/ is untouched. Then run_wide.py
--proj projections-extra --methods <EXTRA_METHODS> evaluates them.
'''
import argparse
import os
import time

from export_projections import MODELS, VARIANTS, run_pair

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'projections-extra')

SYNTHETIC = ['Gaussian', 'Moons', 'Circles', 'Blobs', 'XOR', 'Sklearn Normal',
             'Truncated Normal', 'Madelon']
MEDICAL_TABULAR = [
    'Arrhythmia', 'Breast Cancer', 'Breast Cancer Coimbra',
    'Breast Cancer Prognostic', 'Cervical Cancer', 'Chronic Kidney Disease',
    'Diabetes Pima Indian', 'Framingham CHD', 'Habermans Breast Cancer',
    'HCC Survival', 'Heart Disease', 'Heart Failure', 'Hepatitis',
    'Indian Liver Patient', 'Mammographic Mass', 'Parkinsons', 'SPECTF Heart',
    'Stroke Prediction', 'Thoracic Surgery', 'Thyroid Sick',
    'Z-Alizadeh Sani CAD']          # Breast Cancer Wisconsin is the same data
DISEASE_POSITIVE = [v for v in VARIANTS if v.endswith('(disease-positive)')]
CLINICAL = ['MIMIC-III Mortality', 'MIMIC-III Sepsis',
            'MIMIC-IV Ready for Discharge']
OTHER_TABULAR = ['Abalone Gender', 'Banknote Authentication', 'Ionosphere',
                 'Sonar Rocks vs Mines', 'Wheat Seeds', 'Iris', 'Wine']
COST = ['German Credit', 'Costcla Credit Scoring Kaggle 2011',
        'Costcla Credit Scoring PAKDD 2009', 'Costcla Direct Marketing']
OPENML = ['Spambase', 'Phoneme', 'MAGIC Gamma Telescope', 'EEG Eye State',
          'Default of Credit Card Clients', 'QSAR Biodegradation', 'Bioresponse',
          'Hill-Valley', 'Steel Plates Fault', 'KC1 Software Defects',
          'Blood Transfusion', 'Ozone Level 8hr']

MEDICAL_IMAGE = [
    'PneumoniaMNIST: pneumonia vs normal',
    'BreastMNIST: malignant vs normal, benign',
    'DermaMNIST: melanoma vs rest',
    'DermaMNIST: melanoma vs melanocytic nevi',
    'DermaMNIST: dermatofibroma vs rest',
    'BloodMNIST: basophil vs rest',
    'BloodMNIST: erythroblast vs rest',
    'PathMNIST: colorectal adenocarcinoma epithelium vs normal colon mucosa',
    'PathMNIST: colorectal adenocarcinoma epithelium vs rest',
    'OCTMNIST: drusen vs normal',
    'OCTMNIST: diabetic macular edema vs normal',
    'OCTMNIST: choroidal neovascularization vs normal',
]
NATURAL_IMAGE = [
    'MNIST: 0 vs rest', 'MNIST: 4 vs 9', 'MNIST: 3 vs 8', 'MNIST: 1 vs 7',
    'MNIST: 0 vs 1', 'MNIST: 3+5 vs rest',
    'Fashion-MNIST: Shirt vs T-shirt/top', 'Fashion-MNIST: Pullover vs Coat',
    'Fashion-MNIST: Trouser vs Bag',
    'Fashion-MNIST: Sandal+Sneaker+Ankle boot vs rest',
    'SVHN: 0 vs rest', 'SVHN: 3 vs 8', 'SVHN: 1 vs 7',
    'EuroSAT: Pasture vs rest', 'EuroSAT: AnnualCrop vs PermanentCrop',
    'EuroSAT: Highway vs River', 'EuroSAT: Forest vs SeaLake',
    'CIFAR-10: cat vs dog', 'CIFAR-10: automobile vs truck',
    'CIFAR-10: airplane vs frog', 'CIFAR-10: cat vs rest',
    'CIFAR-100: apple vs rest', 'CIFAR-100: fish vs rest',
    'CIFAR-100: vehicles 1 vs rest',
    'CIFAR-100: large carnivores vs large omnivores and herbivores',
    'CIFAR-100: leopard vs tiger',
]
LABEL_NOISE = [v for v in VARIANTS if v.startswith('CIFAR-10N')]
#: E8: the test minority is a class never seen in training, beside the same
#: training task tested as usual
SHIFT = ['CIFAR-100: leopard vs rest except tiger @test tiger',
         'CIFAR-10: automobile vs rest except truck @test truck',
         'OCTMNIST: drusen vs normal @test diabetic macular edema']
SHIFT += [t.split(' @')[0] for t in SHIFT[:2]]
IMAGE = MEDICAL_IMAGE + NATURAL_IMAGE + LABEL_NOISE + SHIFT
GAUSSIAN_GRID = [v for v in VARIANTS if v.startswith('Gaussian sep')]

#: models cheap enough for tens of thousands of rows of hundreds of features
FAST_MODELS = ['Linear', 'LDA', 'NearestClassMean', 'HistGradientBoosting',
               'LinearSVM', 'DecisionTree', 'MLP-small']
SLOW_MODELS = [m for m in MODELS if m not in FAST_MODELS]

#: E7: frozen ImageNet embeddings (export_torch.py features) and the heads on them
BACKBONES = ['resnet18', 'convnext_tiny']
FEATURE_HEADS = ['Linear', 'NearestClassMean', 'HistGradientBoosting', 'MLP-small']

#: E5's tasks and minority sizes; each task stops at its own maximum
N_SWEEP_TASKS = ['MNIST: 0 vs rest',
                 'PathMNIST: colorectal adenocarcinoma epithelium vs rest',
                 'OCTMNIST: drusen vs normal', 'DermaMNIST: melanoma vs rest',
                 'Framingham CHD', 'MIMIC-III Mortality',
                 'MIMIC-IV Ready for Discharge']
N_SWEEP = [5, 10, 25, 50, 100, 250, 1000, 2500]

#: run_wide.py --methods for the extension
EXTRA_METHODS = ['Slacks Deltas', 'Min Deltas', 'F Deltas', 'CP Sum',
                 'CP Minimax', 'DKW Sum', 'DKW Minimax', 'Gaussian Predictive',
                 'Gaussian Band', 'Logistic Band']

TABULAR = (MEDICAL_TABULAR + DISEASE_POSITIVE + CLINICAL + OTHER_TABULAR + COST
           + OPENML)

def with_backbone(task, backbone):
    '''the task on frozen embeddings: the [backbone] tag goes before any @ option'''
    name, at, opts = task.partition(' @')
    return f'{name} [{backbone}]{at}{opts}'


#: stage -> (datasets, models)
STAGES = {
    'tabular': (TABULAR + SYNTHETIC, list(MODELS)),
    'image-fast': (IMAGE, FAST_MODELS),
    'image-slow': (IMAGE, SLOW_MODELS),
    'features': ([with_backbone(t, b) for t in IMAGE for b in BACKBONES],
                 FEATURE_HEADS),
    'gaussian': ([f'{g} @n{n}' for g in GAUSSIAN_GRID
                  for n in (5, 10, 25, 50, 100, 250, 1000)],
                 ['Linear', 'NearestClassMean', 'SVM-rbf']),
    'n-sweep': ([f'{t} @n{n}' for t in N_SWEEP_TASKS for n in N_SWEEP],
                ['Linear', 'HistGradientBoosting', 'NearestClassMean']),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('stage', nargs='?', choices=list(STAGES))
    ap.add_argument('--list', action='store_true')
    ap.add_argument('--datasets', nargs='*',
                    help='subset of the stage; a list name here (MEDICAL_IMAGE) expands')
    ap.add_argument('--models', nargs='*', help='subset of the stage')
    ap.add_argument('--skip-models', nargs='*', default=[])
    ap.add_argument('--seeds', type=int, default=10)
    ap.add_argument('--jobs', type=int, default=1)
    ap.add_argument('--force', action='store_true')
    ap.add_argument('--out', default=OUT)
    args = ap.parse_args()

    if args.list or args.stage is None:
        for name, (datasets, models) in STAGES.items():
            print(f'{name}: {len(datasets)} datasets x {len(models)} models')
        return

    datasets, models = STAGES[args.stage]
    if args.datasets:
        datasets = [d for name in args.datasets
                    for d in (globals()[name] if name.isupper() and name in globals()
                              else [name])]
    models = [m for m in (args.models or models) if m not in args.skip_models]
    seeds = list(range(args.seeds))
    pairs = [(d, m) for d in datasets for m in models]
    print(f'{args.stage}: {len(pairs)} pairs x {len(seeds)} seeds -> {args.out}',
          flush=True)

    t0 = time.time()
    if args.jobs > 1:
        from joblib import Parallel, delayed
        results = Parallel(n_jobs=args.jobs, verbose=5)(
            delayed(run_pair)(d, m, seeds, args.force, args.out)
            for d, m in pairs)
    else:
        results = [run_pair(d, m, seeds, args.force, args.out)
                   for d, m in pairs]
    failed = [(d, m, err) for (d, m), (_, status, err) in zip(pairs, results)
              if status == 'failed' or err]
    for d, m, err in failed:
        print(f'  {d} / {m}: {err}', flush=True)
    print(f'{len(pairs) - len(failed)} ok, {len(failed)} with errors '
          f'({time.time() - t0:.0f}s)', flush=True)


if __name__ == '__main__':
    main()
