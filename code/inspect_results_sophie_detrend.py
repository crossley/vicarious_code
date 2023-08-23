from imports import *
from util_funcs import *

mpl.use('TkAgg')

raw_data_dir = '/Users/mq20185996/projects/crazly/raw_data/'
detrend_epoch_dir = '/Users/mq20185996/projects/crazly/detrended_epoched_data/'

dir_figs = "../figures/sophie_tmrdt/"
dir_results = "../results/sophie_tmrdt/"

sub_dirs = np.sort([x[0] for x in os.walk(raw_data_dir)])
sub_nums = np.arange(1, len(sub_dirs), 1)

exclude_subs = [1, 10, 13, 16, 18, 32, 33]
sub_nums = [x for x in sub_nums if x not in exclude_subs]

subs_non_mts = [1, 2, 3, 4, 5, 6, 7, 10, 15, 17, 19, 22, 25, 26, 27, 34, 37]

subs_mts = [
    8, 9, 12, 13, 14, 16, 18, 20, 21, 23, 24, 28, 29, 30, 31, 32, 33, 35, 36,
    38, 39, 40
]

# include_subs = subs_mts
# include_subs = subs_non_mts
# include_subs = [40]
# sub_nums = [x for x in sub_nums if x in include_subs]

sub_dirs = sub_dirs[sub_nums]

print(sub_dirs)

timegen_list = []
for s in sub_dirs:

    print()
    print(s[-3:])
    print(s[-2:])
    print()

    trial_mat = load_trial_mat(raw_data_dir, s)
    epochs, X, y = load_epochs(s, trial_mat, detrend_epoch_dir)

    # NOTE: Combine across trials but keep sensors separate
    fig, ax = plt.subplots(2, 2, squeeze=False, figsize=(8, 5))
    for i, k in enumerate(epochs.event_id.keys()):
        epochs[k].average().plot(spatial_colors=True, axes=ax.flatten()[i], show=False)
        ax.flatten()[i].set_title(k)
    plt.tight_layout()
    plt.savefig(dir_figs + "fig_epochs_sub_" + s[-2:] + ".pdf")

    epochs.resample(sfreq=150)
    scores_timegen = run_time_gens(epochs)
    timegen_list.append(scores_timegen)

    del raw, epochs

epochs = mne.concatenate_epochs(epochs_list)
fig, ax = plt.subplots(2, 2, squeeze=False, figsize=(8, 5))
for i, k in enumerate(epochs.event_id.keys()):
    epochs[k].average().plot(spatial_colors=True, axes=ax.flatten()[i], show=False)
    ax.flatten()[i].set_title(k)
plt.tight_layout()
plt.savefig(dir_figs + "fig_epochs_sub_" + "ave" + ".pdf")

timegen_touchtouch = [x[0] for x in timegen_list]
timegen_touchtouch = np.dstack(timegen_touchtouch)
np.save(dir_results + 'timegen_touchtouch.npy', timegen_touchtouch)

timegen_visvis = [x[1] for x in timegen_list]
timegen_visvis = np.dstack(timegen_visvis)
np.save(dir_results + 'timegen_visvis.npy', timegen_visvis)

timegen_vistouch = [x[2] for x in timegen_list]
timegen_vistouch = np.dstack(timegen_vistouch)
np.save(dir_results + 'timegen_vistouch.npy', timegen_vistouch)

timegen_touchvis = [x[3] for x in timegen_list]
timegen_touchvis = np.dstack(timegen_touchvis)
np.save(dir_results + 'timegen_touchvis.npy', timegen_touchvis)

fig, ax = plt.subplots(2, 2, squeeze=False)
sns.heatmap(timegen_touchtouch.mean(axis=2).T, ax=ax[0, 0])
sns.heatmap(timegen_visvis.mean(axis=2).T, ax=ax[0, 1])
sns.heatmap(timegen_vistouch.mean(axis=2).T, ax=ax[1, 0])
sns.heatmap(timegen_touchvis.mean(axis=2).T, ax=ax[1, 1])
[x.invert_yaxis() for x in ax.flatten()]
plt.tight_layout()
plt.savefig(dir_figs + '/timegen.pdf')
