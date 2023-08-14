from imports import *
from util_funcs import *

mpl.use('TkAgg')

raw_data_dir = '../raw_data/'
detrend_epoch_dir = '../detrended_epoched_data/'

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
sub_dirs = [sub_dirs[12]]

print(sub_dirs)

reactivation_time_rec = []
scores_rec = []
for s in sub_dirs:

    print()
    print(s[-3:])
    print(s[-2:])
    print()

    trial_mat = load_trial_mat(s)
    epochs, X, y = load_epochs(s, trial_mat, detrend_epoch_dir)

    fig, ax = plt.subplots(1, 1, squeeze=False, figsize=(4, 4))
    for trial in range(0, X.shape[0], 100):
        for channel in range(0, X.shape[1], 10):
            ax[0, 0].plot(X[trial, channel, :])
    plt.savefig('/Users/mq20185996/Dropbox/crazly/epochs.pdf', dpi=10)

    # reject_criteria = dict(eeg=100e-6, eog=200e-6)  # 100 µV, 200 µV
    # epochs.drop_bad(reject=reject_criteria)

    # epochs.plot_drop_log()
