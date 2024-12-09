from datetime import datetime
import os

from deltas.pipeline import data, classifier, evaluation
from deltas.model import non_sep, downsample
from tqdm import tqdm

import numpy as np
import pandas as pd


DIR_NAME = 'results-non-sep'
TABLE_NAME = 'combined_table-non-sep.txt'


def generator():
  num = 0
  while True:
      yield num
      num += 1

def run_single(dataset, model, len_required=10, exp_num=0):
    dfs = []
    print(dataset, model)
    for i in tqdm(generator()):
        data_clf = data.get_real_dataset(
            dataset, _print=False, seed=i, scale=True)

        classifiers_dict = classifier.get_classifier(
            data_clf=data_clf,
            model=model,
            _plot=False,
            _print=False)
        data_clf['clf'] = classifiers_dict['Baseline']
        X = data_clf['data']['X']
        y = data_clf['data']['y']
        clf = data_clf['clf']
        deltas_min = non_sep.deltas(
            clf).fit(X, y, loss_type='min')
        deltas_avg = non_sep.deltas(
            clf).fit(X, y, loss_type='mean')
        deltas_f = non_sep.deltas(
            clf).fit(X, y, only_furtherest_k=True)
        deltas_downsample = downsample.downsample_deltas(clf).fit(X, y,
                                                             max_trials=10000,
                                                             parallel=True)

        if (deltas_min.is_fit == True and
            deltas_avg.is_fit == True and
            deltas_f.is_fit == True and
            deltas_downsample.is_fit == True):

            classifiers_dict['Slacks Deltas'] = deltas_downsample
            classifiers_dict['Min Deltas'] = deltas_min
            classifiers_dict['Avg Deltas'] = deltas_avg
            classifiers_dict['F Deltas'] = deltas_f
            scores_df = evaluation.eval_test(classifiers_dict,
                                             data_clf['data_test'], _print=False, _plot=False)
            dfs.append(scores_df)
            if len(dfs) > 1:
                write_results(dfs, dataset, model, data_clf, exp_num)

        if len(dfs) == len_required:
            break


def write_results(dfs, dataset, model, data_clf, exp_num=0):
    df = pd.concat(dfs, axis=0)
    mean = {}
    std = {}
    index = df.index.unique().to_list()
    cols = df.columns.to_list()
    for method in index:
        mean[method] = df.loc[method].mean().to_list()
        std[method] = df.loc[method].std().to_list()

    mean_df = pd.DataFrame.from_dict(mean, orient='index', columns=cols)
    std_df = pd.DataFrame.from_dict(std, orient='index', columns=cols)

    m = mean_df.to_dict('list')
    s = std_df.to_dict('list')
    metrics = mean_df.columns.to_list()
    methods = mean_df.index.to_list()
    sf = 5
    for metric in metrics:
        means = m[metric]
        sts = s[metric]
        mx = np.argmax(means)
        for i in range(len(means)):
            m_str = str(means[i])[1:sf]
            if i == mx:
                m_str = f"\\textbf{{{m_str}}}"
            s_str = str(sts[i])[1:sf-1]
            m[metric][i] = f'${m_str} \\pm {s_str}$'

    method_map = {
        'Baseline': 'Baseline',
        'SMOTE': "SMOTE \cite{Chawla_2002_JAIR}",
        'Balanced Weights': 'BW',
        'BMR': 'BMR \cite{Bahnsen_2014_SIAM}',
        'Threshold': 'Thresh \cite{Sheng_2006_AAAI}',
        'Slacks Deltas': 'Old Deltas \cite{Clifford_2024_ECAI}',
        'Min Deltas': 'Min Deltas',
        'Avg Deltas': 'Avg Deltas',
        'F Deltas': 'F Deltas',
    }
    meths_new = []
    for me in methods:
        meths_new.append(method_map[me])
    m['Methods'] = meths_new
    df = pd.DataFrame(m)  # .set_index('Methods')
    meths = df.pop('Methods')
    df.insert(0, 'Methods', meths)
    latex_str = df.to_latex(index=False)

    # write to file
    now = datetime.now()
    dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
    # make folder if we dont have already
    save_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), DIR_NAME)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_file = os.path.join(save_dir, f'Results-{exp_num}-{dataset}.txt')
    with open(save_file, "w") as text_file:
        text_file.write(f'% {dt_string}\n')
        text_file.write(f'% {dataset} - {len(dfs)} runs {model} model\n')
        train = np.unique(data_clf['data']['y'], return_counts=True)[1]
        test = np.unique(data_clf['data_test']['y'], return_counts=True)[1]
        text_file.write(f'% train:{train}, test:{test}\n')
        format = '\\begin{tabular}{@{}l' + 'c'*len(mean_df.columns) + '@{}}'
        text_file.write(format+latex_str[18+len(mean_df.columns):])


def combine_tables(tables):
    dir = os.path.dirname(os.path.abspath(__file__))
    short_names = {'Pima Indian Diabetes': 'Pima Diabetes',
                'Breast Cancer': 'Breast Cancer',
                'Hepatitis': 'Hepatitis',
                'Heart Disease': 'Heart Disease',
                'MIMIC-III-mortality': 'MIMIC ICU'}

    # gopen write file
    with open(os.path.join(dir, TABLE_NAME), 'w') as w_file:
        # get the first table to get table starter
        with open(os.path.join(dir, DIR_NAME, tables[0]), 'r') as r_table:
            table = r_table.read()
            lines = table.split('\n')
            lines[3] = lines[3][:20] + 'l' + lines[3][20:]
            lines[5] = '& ' + lines[5][:]
            for l in [3, 4, 5]:
                w_file.write(lines[l] + '\n')
        # loop through all tables
        for table in tables:
            with open(os.path.join(dir, DIR_NAME, table), 'r') as t:
                lines = t.read().split('\n')
                # meta info
                for l in range(0, 3):
                    w_file.write(lines[l] + '\n')
                w_file.write('\midrule\n')
                # dataset name
                dataset_name = lines[1].split(' - ')[0][2:]
                w_file.write(
                    '\multirow{6}{*}{\\rotatebox{90}{'+short_names[dataset_name]+'}}\n')
                # table info
                for l in range(7, 13):
                    w_file.write('& ' + lines[l] + '\n')
        w_file.write('\\bottomrule\n\end{tabular}')


def main():
    experiments = {
        0: {'dataset': 'Gaussian', 'model': 'Linear'},
        1: {'dataset': 'Pima Indian Diabetes', 'model': 'SVM-rbf'},
        2: {'dataset': 'Breast Cancer', 'model': 'SVM-rbf'},
        3: {'dataset': 'Hepatitis', 'model': 'SVM-rbf'},
        4: {'dataset': 'Heart Disease', 'model': 'SVM-rbf'},
        5: {'dataset': 'MIMIC-III-mortality', 'model': 'MIMIC'},
    }

    for exp_num, exp in experiments.items():
        dataset = exp['dataset']
        model = exp['model']
        run_single(dataset, model, exp_num=exp_num)

    tables = [
        'Results-1-Pima Indian Diabetes.txt',
        'Results-2-Breast Cancer.txt',
        'Results-3-Hepatitis.txt',
        'Results-4-Heart Disease.txt',
        'Results-5-MIMIC-III-mortality.txt'
                ]
    combine_tables(tables)


if __name__ == '__main__':
    main()