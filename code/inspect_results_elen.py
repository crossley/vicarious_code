from imports import *
from util_funcs_elen import *

mpl.use("TkAgg")

data_dir_behaviour = "../data/elen/data_behaviour"
data_dir_eeg = "../data/elen/data_eeg"

sub_dirs_eeg = np.sort(os.listdir(data_dir_eeg))
sub_dirs_eeg = sub_dirs_eeg[sub_dirs_eeg != ".DS_Store"]
sub_nums_eeg = [x[1:3] for x in sub_dirs_eeg]

sub_dirs_behaviour = np.sort(os.listdir(data_dir_behaviour))
sub_dirs_behaviour = sub_dirs_behaviour[sub_dirs_behaviour != ".DS_Store"]
sub_nums_behaviour = [x[1:3] for x in sub_dirs_behaviour]

sub_nums = np.intersect1d(sub_nums_eeg, sub_nums_behaviour)

sub_nums = [sub_nums[0]]

for s in sub_nums:
    print(s)

    trial_mat = load_trial_mat(data_dir_behaviour, s)
    raw, events = load_raw(data_dir_eeg, s)
    epochs = compute_epochs(raw, events, trial_mat)

    # TODO: finish development of a preprocessing pipeline
    # TODO: trial-masked robust detrending

    # NOTE: Combine across trials but keep sensors separate
    fig, ax = plt.subplots(2, 2, squeeze=False, figsize=(8, 5))
    for i, k in enumerate(epochs.event_id.keys()):
        epochs[k].average().plot(
            spatial_colors=True, axes=ax.flatten()[i], show=False
        )
        ax.flatten()[i].set_title(k)
    plt.tight_layout()
    plt.show()
    # plt.savefig('figures/fig_epochs_sub_' + s + '.pdf')

    # TODO: Is information present at all (CSP classifier)?
    # TODO: When is information present (time gen)?
    # TODO: Do both in the time domain as wel in the frequency domain
    # TODO: what is the mutual information between visual and tactile trials?

    # # TODO: why are scores chance at 25%?
    # scores = run_time_gens(epochs)
    # scores_touchtouch = scores[0]
    # scores_visvis = scores[1]
    # scores_touchvis = scores[2]
    # scores_vistouch = scores[3]

    # # TODO plot all on a grid with color bar
    # plt.imshow(scores_touchtouch)
    # plt.show()
