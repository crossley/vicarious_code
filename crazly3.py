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

    # NOTE: Not sure why they do this step
    # epochs_train = epochs.copy().crop(tmin=0.5, tmax=1.0)
    # epochs_train = epochs.copy()
    # XX_train = epochs_train.get_data()
    # XX_train = XX_train[(y == 3) | (y == 4)]

    # tag = 'touchtouch'
    XX = X[(y == 3) | (y == 4)]
    yy = y[(y == 3) | (y == 4)]

    XX_train = np.array(XX)

    # NOTE: I don't think the label encoder is needed here.
    le = LabelEncoder()
    yy = le.fit_transform(yy)

    # tag = 'visvis'
    # XX = X[(y == 1) | (y == 2)]
    # yy = y[(y == 1) | (y == 2)]

    # tag = 'touchvis'

    # scl = Scaler(epochs.info)
    # X = scl.fit_transform(X)

    # X_train = X[(y == 3) | (y == 4)]
    # y_train = y[(y == 3) | (y == 4)]
    # X_test = X[(y == 1) | (y == 2)]
    # y_test = y[(y == 1) | (y == 2)]

    # X_train = X_train[:, :, 280:]
    # X_test = X_test[:, :, 25:]

    # print(X_train.shape, X_test.shape)

    cv = ShuffleSplit(5, test_size=0.2, random_state=42)
    cv_split = cv.split(XX_train)

    # Assemble a classifier
    lda = LinearDiscriminantAnalysis()
    csp = CSP(n_components=4, reg=None, log=True, norm_trace=False)

    # Use scikit-learn Pipeline with cross_val_score function
    clf = Pipeline([('CSP', csp), ('LDA', lda)])
    scores = cross_val_score(clf, XX, yy, cv=cv, n_jobs=None)

    # Printing the results
    class_balance = np.mean(yy == yy[0])
    class_balance = max(class_balance, 1. - class_balance)
    print("Classification accuracy: %f / Chance level: %f" %
          (np.mean(scores), class_balance))

    # plot CSP patterns estimated on full data for visualization
    csp.fit_transform(XX, yy)

    # csp.plot_patterns(epochs.info,
    #                   ch_type='eeg',
    #                   units='Patterns (AU)',
    #                   size=1.5)

    # look at performance over time
    sfreq = epochs.info['sfreq']
    w_length = int(sfreq * 0.5)  # running classifier: window length
    w_step = int(sfreq * 0.1)  # running classifier: window step size
    w_start = np.arange(0, XX.shape[2] - w_length, w_step)

    scores_windows = []

    for train_idx, test_idx in cv_split:
        y_train, y_test = yy[train_idx], yy[test_idx]

        X_train = csp.fit_transform(XX_train[train_idx], y_train)
        X_test = csp.transform(XX_train[test_idx])

        # fit classifier
        lda.fit(X_train, y_train)

        # running classifier: test classifier on sliding window
        score_this_window = []
        for n in w_start:
            X_test = csp.transform(XX[test_idx][:, :, n:(n + w_length)])
            score_this_window.append(lda.score(X_test, y_test))
        scores_windows.append(score_this_window)

    # Plot scores over time
    w_times = (w_start + w_length / 2.) / sfreq + epochs.tmin

    plt.figure()
    plt.plot(w_times, np.mean(scores_windows, 0), label='Score')
    plt.axvline(0, linestyle='--', color='k', label='Onset')
    plt.axhline(0.5, linestyle='-', color='k', label='Chance')
    plt.xlabel('time (s)')
    plt.ylabel('classification accuracy')
    plt.title('Classification score over time')
    plt.legend(loc='lower right')
    plt.show()
