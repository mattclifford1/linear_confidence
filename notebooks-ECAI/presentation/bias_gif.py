import numpy as np
from deltas.model import base
from deltas.pipeline import data, classifier, evaluation
import deltas.plotting.plots as plots
import matplotlib.pyplot as plt
import os
import shutil

START = -2.9
END = 3.1
TOTAL = 51
FPS = 1

def plot_bias_stills(clfs, test_data, dim_reducer=None, save_dir=None):
    str_len = 3
    if save_dir != None:
        if os.path.exists(save_dir):
            # delete dir and all files
            shutil.rmtree(save_dir)
        os.makedirs(save_dir, exist_ok=True)

    # plot in original space
    i = 1
    for clf_info in clfs:
        clf, name = clf_info

        print(name)
        # get subplots with first subplot smaller than the second

        _, axs = plt.subplots(
            2, 1, gridspec_kw={'height_ratios': [1, 4]},
            # figsize=(6, 12)),
            ) 
        # get the first axis
        ax1 = axs[1]
        ax2 = axs[0]
        # ax2 = axs[2].axes[0]
        # ax, _ = plots._get_axes(None)
        # data = {'X': test_data['X'], 'y': preds[name]}
        data = test_data
        # data = test_data
        plots.plot_classes(data, ax=ax1, dim_reducer=dim_reducer)
        plots.plot_decision_boundary(
            clf, test_data, ax=ax1, probs=False, dim_reducer=dim_reducer)
        
        # ax1.set_title(name)
        
        # plot the bias term
        ax2.scatter([name], [0], c='k', s=1000, marker='|')
        # remove ylabel
        ax2.set_yticks([])
        # set fixed scale
        ax2.set_xlim([START, END])
        # set title
        ax2.set_title('Bias')
        # tight layout
        plt.tight_layout()

        if save_dir == None:
            plots.plt.show()
        else:
            save_str = str(i)
            while len(save_str) < str_len:
                save_str = '0' + save_str
            plots.plt.savefig(os.path.join(save_dir, f'{save_str}.png'))
        i += 1


class adjust_bias(base.base_deltas):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.is_fit = True
        self.class_nums = [0, 1]

    def set_bias(self, bias):
        self.boundary = bias
        return self
    

if __name__ == '__main__':
    dataset = 'Breast Cancer'
    # dataset = 'Pima Indian Diabetes'
    # dataset = 'Hepatitis'
    # dataset = 'Heart Disease'

    model = 'SVM-rbf'

    data_clf = data.get_real_dataset(dataset, scale=True, seed=0)

    classifiers_dict = classifier.get_classifier(
        data_clf=data_clf,
        model=model,
        _plot=False)

    data_clf['clf'] = classifiers_dict['Baseline']

    runs = []
    for i in np.linspace(START, END, TOTAL):
        runs.append(round(i, 1))
    for i in np.linspace(END, START, TOTAL):
        runs.append(round(i, 1))

    classifiers_list = []
    for i in runs:
        _clf = adjust_bias(clf=data_clf['clf']).set_bias(i)
        _name = i
        classifiers_list.append((_clf, _name))
    
    curr_dir = os.path.dirname(__file__)
    path = os.path.abspath(os.path.join(curr_dir, 'bias_stills'))

    plot_bias_stills(classifiers_list, data_clf['data_test'],
                     dim_reducer=data_clf['dim_reducer'], save_dir=path)
    
    # remove file if it exists
    if os.path.exists(f'{curr_dir}/bias.gif'):
        os.remove(f'{curr_dir}/bias.gif')
    # convert to gif
    os.system(
        f'ffmpeg -i {path}/%0{3}d.png {curr_dir}/bias.gif -framerate {FPS}')
    # delete all stills
    shutil.rmtree(path)
