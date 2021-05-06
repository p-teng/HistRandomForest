import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import openml
from sklearn.metrics import zero_one_loss, cohen_kappa_score

sns.set_context("paper", font_scale=1.5)

clfs = ['BagDT', 'RF', 'HistRF']
color_dict = {
    'BagDT': '#e41a1c',
    'RF': '#377eb8',
    'HistRF': '#4daf4a',
}
tag = 'results_cv'

if not os.path.exists(f'./figures/{tag}/'):
    os.makedirs(f'./figures/{tag}/')

results_dir = f'./cc18_results/{tag}/'


def bin_data(y, n_bins):
    """
    Partitions the data into ordered bins based on
    the probabilities. Returns the binned indices.
    """
    edges = np.linspace(0, 1, n_bins)
    bin_idx = np.digitize(y, edges, right=True)
    binned_idx = [np.where(bin_idx == i)[0] for i in range(n_bins)]

    return binned_idx


def bin_stats(y_true, y_proba, bin_idx):
    # mean accuracy within each bin
    bin_acc = [
        np.equal(np.argmax(y_proba[idx], axis=1),
                 y_true[idx]).mean() if len(idx) > 0 else 0
        for idx in bin_idx
    ]
    # mean confidence of prediction within each bin
    bin_conf = [
        np.mean(np.max(y_proba[idx], axis=1)) if len(idx) > 0 else 0
        for idx in bin_idx
    ]

    return np.asarray(bin_acc), np.asarray(bin_conf)


def ece(y_true, y_proba, n_bins=10):
    bin_idx = bin_data(y_proba.max(axis=1), n_bins)
    n = len(y_true)

    bin_acc, bin_conf = bin_stats(y_true, y_proba, bin_idx)
    bin_sizes = [len(idx) for idx in bin_idx]

    ece = np.sum(np.abs(bin_acc - bin_conf) * np.asarray(bin_sizes)) / n

    return ece


def mce(y_true, y_proba, n_bins=10):
    bin_idx = bin_data(y_proba.max(axis=1), n_bins)

    bin_acc, bin_conf = bin_stats(y_true, y_proba, bin_idx)
    mce = np.max(np.abs(bin_acc - bin_conf))

    return mce

def ece_score(y_true, y_proba, num_bins=20):
    y_hat = y_proba.argmax(axis=1)
    y_hat_proba = y_proba.max(axis=1)
    
    score = 0
    for bin in range(num_bins):
        indx = np.where(
            (y_hat_proba > bin / num_bins) &
            (y_hat_proba <= (bin + 1) / num_bins)
        )[0]
        
        acc = np.nan_to_num(
            np.mean(
            y_hat[indx] == y_true[indx]
        )
        ) if indx.size!=0 else 0
        conf = np.nan_to_num(
            np.mean(
            y_hat_proba[indx]
        )
        ) if indx.size!=0 else 0
        score += len(indx)*np.abs(
            acc - conf
        )
    
    score /= len(y_true)
    return score

def mce_score(y_true, y_proba, num_bins=20):
    y_hat = y_proba.argmax(axis=1)
    y_hat_proba = y_proba.max(axis=1)
    poba_hist = []
    
    score = 0
    for bin in range(num_bins):
        indx = np.where(
            (y_hat_proba > bin / num_bins) &
            (y_hat_proba <= (bin + 1) / num_bins)
        )[0]
        
        acc = np.nan_to_num(
            np.mean(
            y_hat[indx] == y_true[indx]
        )
        ) if indx.size!=0 else 0
        conf = np.nan_to_num(
            np.mean(
            y_hat_proba[indx]
        )
        ) if indx.size!=0 else 0
        score = max((score, np.abs(acc - conf)))
    
    return score


def brier_score_mvloss(y_true, y_proba):
    if y_true.ndim == 1:
        y_true = np.squeeze(np.eye(len(np.unique(y_true)))[y_true.reshape(-1)])
    return np.mean(
        np.mean((y_proba - y_true)**2, axis=1)
    )


def get_task_scores(score_fn, results_dir):
    tasks = []
    fold_scores = []
    # task_score_means = []
    # task_score_stds = []
    for file in os.listdir(results_dir)[::-1]:
        with open(results_dir + file, 'rb') as f:
            results_dict = pickle.load(f)

        scores = np.asarray([
            [
                score_fn(
                    results_dict['y'][idx], y_proba[-1]
                ) for y_proba, idx in zip(results_dict[name], results_dict['test_indices'])
            ] for name in clfs
        ])

        tasks.append(
            {k: results_dict[k] for k in ['task_id', 'task', 'n_samples', 'n_classes', 'n_features']
        })
        # task_score_means.append(score_means)
        # task_score_stds.append(score_stds)
        fold_scores.append(scores)

    # return tasks, task_score_means, task_score_stds
    return tasks, np.asarray(fold_scores)


def continuous_pairplot(df, vars, hue, cmap, diag_kind, scale='log'):
    import matplotlib as mpl
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    vmin = min(np.min([df[hue]]), -1e-6)
    vmax = max(np.max([df[hue]]), 1e-6)
    g = sns.pairplot(
        df,
        vars=vars,
        diag_kind=diag_kind,
        plot_kws=dict(
            # The effort I put into figuring this out.....
            c=df[hue], cmap=cmap, norm=mpl.colors.TwoSlopeNorm(
                vcenter=0, vmin=vmin, vmax=vmax)
        ),
    )
    if scale:
        for r in range(len(g.axes)):
            for c in range(len(g.axes)):
                g.axes[r, c].set_xscale(scale)
                if r != c:
                    g.axes[c, r].set_yscale(scale)

    sm = mpl.cm.ScalarMappable(
        mpl.colors.TwoSlopeNorm(vcenter=0, vmin=vmin, vmax=vmax),
        cmap=cmap
    )
    plt.colorbar(sm, ax=g.axes, label=hue, aspect=40)
    return g


def _bin_scores_from_fold(y_true, y_proba, n_bins):
    bin_idx = bin_data(y_proba.max(axis=1), n_bins)
    n = len(y_true)

    bin_acc, bin_conf = bin_stats(y_true, y_proba, bin_idx)
    bin_sizes = np.asarray([len(idx) for idx in bin_idx])

    ece = np.sum(np.abs(bin_acc - bin_conf) * bin_sizes) / n

    return ece, bin_acc, bin_sizes


def get_bin_scores(results_dir, n_bins=10):
    tasks = []
    task_bin_accs = []
    task_bin_eces = []
    task_bin_sizes = []
    for file in os.listdir(results_dir):
        with open(results_dir + file, 'rb') as f:
            results_dict = pickle.load(f)

        clf_accs = []
        clf_eces = []
        clf_bin_sizes = []
        for name in clfs:
            stats = [_bin_scores_from_fold(
                results_dict['y'][idx], y_proba[-1], n_bins
            ) for y_proba, idx in zip(results_dict[name], results_dict['test_indices'])]
            eces, bin_accs, bin_sizes = list(zip(*stats))
            clf_accs.append(np.mean(bin_accs, axis=0))
            clf_eces.append(np.mean(eces))
            clf_bin_sizes.append(np.mean(bin_sizes, axis=0))

        tasks.append(
            {k: results_dict[k] for k in ['task_id', 'task', 'n_samples', 'n_classes', 'n_features']
        })
        task_bin_accs.append(clf_accs)
        task_bin_eces.append(clf_eces)
        task_bin_sizes.append(clf_bin_sizes)

    return tasks, task_bin_accs, task_bin_eces, task_bin_sizes


def score_plots(loss_fn, loss_name):
    print("Score plots fn: " + loss_name)
    tasks, task_scores = get_task_scores(
        loss_fn, results_dir)
    print(task_scores)
    df_rows = []
    for task, scores in zip(tasks, task_scores):
        print("Task: " + str(task))
        row = np.round(np.mean(scores, axis=1), 5)
        row = [task['task'], task['n_classes'],
            task['n_samples'], task['n_features']] + list(row)
        df_rows.append(row)
    header = ['Dataset', 'n_classes', 'n_samples', 'n_features'] + clfs

    score_df = pd.DataFrame(
        df_rows, columns=header, index=[task['task_id'] for task in tasks]
    ).sort_index()
        
    for i, clf in enumerate(clfs[:-1]): # clfs[:-1] = ['RF', 'IRF', 'SigRF']
        print(clf)
        score_df[f'{clf}-HistRF'] = np.mean(task_scores[:, i] - task_scores[:, -1], axis=-1)

    score_df['HistRF_diff_max'] = score_df.apply(
        lambda row: max([row[f'{col}-HistRF'] for col in ['BagDT', 'RF']]), axis=1)
    print(score_df)
    # Wilcoxon csv
    # m = len(clfs)
    # stat_mat = []
    # for r in range(m):
    #     stat_mat.append([])
    #     for c in range(m):
    #         if r == c:
    #             stat_mat[r].append('')
    #             continue
    #         # Wilcoxon(x, y) significant if x < y
    #         # stat, pval = wilcoxon(
    #         #     score_df[clfs[r]], score_df[clfs[c]], zero_method='zsplit', alternative='less')
    #         stat, pval = wilcoxon(
    #             np.mean(task_scores[:, r] - task_scores[:, c], axis=-1), zero_method='zsplit', alternative='less')
    #         stat_mat[r].append(f'{stat:.3f} ({pval:.3f})')

    # stat_mat = pd.DataFrame(stat_mat, columns=clfs)
    # stat_mat.index = clfs
    # stat_mat.to_csv(f'./figures/{tag}/{loss_name}_wilcoxon_cv10.csv')
    # print(stat_mat)

    # Pairplots
    for clf in ['BagDT', 'RF']:
        g = continuous_pairplot(
            score_df,
            vars=['n_classes', 'n_samples', 'n_features'],
            hue=f'{clf}-HistRF',
            cmap='coolwarm',
            diag_kind='hist',
        )
        g.fig.suptitle(f'{loss_name} loss: HistRF vs {clf}', y=1.03)
        plt.savefig(f'./figures/{tag}/{loss_name}_pairplot_HistRF-vs-{clf}.pdf')
        plt.show()

    # stripplot
    f, ax = plt.subplots(figsize=(12, 16))
    sns.despine(bottom=True, left=True)

    df = score_df[['Dataset'] + clfs]
    df = df.melt(
        id_vars=['Dataset'], value_vars=clfs, var_name='Classifier', value_name=f'{loss_name} loss'
    ).sort_values(f'{loss_name} loss')

    # Show each observation with a scatterplot
    sns.stripplot(x=f"{loss_name} loss", y="Dataset", hue="Classifier",
        data=df, dodge=False, alpha=1, zorder=1, palette=color_dict)

    plt.savefig(f'./figures/scatter/{loss_name}_scatterplot.pdf')
    plt.show()


losses = [
    (mce_score, 'MCE'),
    (ece_score, 'ECE'),
    (brier_score_mvloss, 'Brier'),
    # (lambda y_true, y_proba: zero_one_loss(y_true, y_proba.argmax(1)), '0-1')
    (lambda y_true, y_proba: -cohen_kappa_score(y_true, y_proba.argmax(1)), 'Cohen\'s Kappa')
]

for loss_fn, loss_name in losses:
    print(loss_name + ' loss')
    score_plots(loss_fn, loss_name)

tasks, task_bin_accs, task_eces, task_bin_sizes = get_bin_scores(results_dir)

n_cols = 10
n_rows = len(tasks) // n_cols + 1
plt.rcParams['figure.facecolor'] = 'white'
fig = plt.figure(figsize=(8*n_cols, 5*n_rows))

samples_sizes = sorted(
    [(i, task['n_samples']) for i, task in enumerate(tasks)],
    key=lambda x: x[1])

n_bins = 10
bin_centers = np.linspace(0.05, 0.95, n_bins)

for rc, (idx, _) in enumerate(samples_sizes):
    row = rc // n_cols
    col = rc % n_cols

    ax1 = plt.subplot2grid((3*n_rows, 1*n_cols), (row*3, col), rowspan=2)
    ax2 = plt.subplot2grid((3*n_rows, 1*n_cols), (2+row*3, col))

    # fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 5))

    ax1.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")

    bin_accs = task_bin_accs[idx]
    eces = task_eces[idx]
    bin_sizes = task_bin_sizes[idx]
    task = tasks[idx]
    task_name = task['task']
    task_id = task['task_id']
    n_samples = task['n_samples']
    n_classes = task['n_classes']
    n_features = task['n_features']

    # Initialize network
    for i, clf in enumerate(clfs):
        accs = bin_accs[i]
        ax1.plot(bin_centers[accs > 0], accs[accs > 0], "o-",
                 label=f'{clf} (ECE={eces[i]:.3f})', c=color_dict[clf])

#         ax1.bar(
#             bin_centers, accs, width=0.1, label=f'{clf} (ECE={eces[i]:.3f})', ec=color_dict[clf], fill=False
#         )

        sizes = bin_sizes[i]
        ax2.bar(
            bin_centers[sizes > 0],
            sizes[sizes > 0] / np.sum(sizes),
            width=0.1,
            alpha=1,
            label=clf,
            fill=False,
            ec=color_dict[clf])

    ax1.set_ylabel("Mean Accuracy")
    ax1.set_ylim([0, 1])
    ax1.set_xlim([0, 1])
    ax2.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1])
    ax1.legend(loc="upper left")
    ax1.set_title(
        f'{task_name} ({task_id}) [dim={n_samples}x{n_features}, classes={n_classes}]')

    ax2.set_xlabel("Prediction confidence (probability)")
    ax2.set_ylim([-0.05, 1])
    ax2.set_xlim([0, 1])
    ax2.set_yticks([0, 0.5, 1])
    ax2.set_ylabel("Bin density")

plt.tight_layout()
plt.savefig(f'./figures/{tag}/ECE_density_plots.pdf', dpi=300)
plt.show()
