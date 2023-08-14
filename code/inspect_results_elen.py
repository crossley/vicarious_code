from imports import *
from util_funcs_elen import *

mpl.use("TkAgg")

data_dir_behaviour = "../Experiment_Elen/data_behaviour"
data_dir_eeg = "../Experiment_Elen/data_eeg"

sub_dirs_eeg = np.sort(os.listdir(data_dir_eeg))
sub_dirs_eeg = sub_dirs_eeg[sub_dirs_eeg != ".DS_Store"]
sub_nums_eeg = [x[1:3] for x in sub_dirs_eeg]

sub_dirs_behaviour = np.sort(os.listdir(data_dir_behaviour))
sub_dirs_behaviour = sub_dirs_behaviour[sub_dirs_behaviour != ".DS_Store"]
sub_nums_behaviour = [x[1:3] for x in sub_dirs_behaviour]

sub_nums = np.intersect1d(sub_nums_eeg, sub_nums_behaviour)

for s in sub_nums:
    print(s)

    trial_mat = load_trial_mat(data_dir_behaviour, s)
    raw, events = load_raw(data_dir_eeg, s)
    epochs = compute_epochs(raw, events, trial_mat)

    # fig, ax = plt.subplots(2, 2, squeeze=False, figsize=(8, 5))
    # for i, k in enumerate(epochs.event_id.keys()):
    #     epochs[k].average().plot(
    #         spatial_colors=True, axes=ax.flatten()[i], show=False
    #     )
    #     ax.flatten()[i].set_title(k)
    # plt.tight_layout()
    # plt.savefig('figures/fig_epochs_sub_' + s + '.pdf')

    fig, ax = plt.subplots(2, 2, squeeze=False, figsize=(8, 5))
    for i, k in enumerate(epochs.event_id.keys()):
        epochs[k].plot(axes=ax.flatten()[i], show=False)
        ax.flatten()[i].set_title(k)
    plt.tight_layout()
    plt.savefig('figures/fig_epochs_sub_' + s + '2.pdf')
