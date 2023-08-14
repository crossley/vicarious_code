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

    n_splits = 12
    metric = 'accuracy'
    # metric = 'roc_auc'

    # tag = 'touchtouch'
    # XX = X[(y == 3) | (y == 4)]
    # yy = y[(y == 3) | (y == 4)]

    tag = 'visvis'
    XX = X[(y == 1) | (y == 2)]
    yy = y[(y == 1) | (y == 2)]

    scl = Scaler(epochs.info)
    XX = scl.fit_transform(XX)

    ## 0. split data in half. One half to be used to estimate "activation" and
    ## the other half to be used to estimate "reactivation".
    X_train, X_test, y_train, y_test = train_test_split(XX,
                                                        yy,
                                                        test_size=0.5,
                                                        random_state=0)

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

    n_train_trial = X_train.shape[0]
    n_train_point = X_train.shape[2]
    n_test_trial = X_test.shape[0]
    n_test_point = X_test.shape[2]

    # 1. Estimate "activation" classifiers: train an LDA classifier for each
    # training time point
    train_clf = []
    for i in range(n_train_point):
        clf = LinearDiscriminantAnalysis(solver='svd')
        clf.fit(X_train[:, :, i], y_train)
        train_clf.append(clf)

    # 2. get distance to bound for each test trial, train_point, and test_point
    #    Each element in d is the distance for a given trial and given test
    #    point from a given train point classifier boundary.
    d = np.zeros((n_test_trial, n_train_point, n_test_point))
    for i in range(n_test_trial):
        for j in range(n_train_point):
            clf = train_clf[j]
            # TODO: verify that this is correct
            y = clf.decision_function(X_test[i, :, :].T)
            w_norm = np.linalg.norm(clf.coef_)
            dist = y / w_norm
            if y_test[i] == np.sort(np.unique(y_test))[0]:
                d[i, j, :] = -dist
            else:
                d[i, j, :] = dist

    # 3. apply LPF to resulting distance measures
    d_filt = np.zeros((n_test_trial, n_train_point, n_test_point))
    sos = signal.butter(2, 0.03, output='sos')
    for i in range(n_test_trial):
        for j in range(n_train_point):
            d_filt[i, j, :] = signal.sosfiltfilt(sos, d[i, j, :])

    # fig, ax = plt.subplots(1, 1, squeeze=False)
    # plt.plot(d[0, 0, :])
    # plt.plot(d_filt[0, 0, :])
    # plt.savefig('/Users/mq20185996/Dropbox/crazly/fig_reactivation_filt.pdf')
    # plt.close()

    # 4. find time of peak distance to bound
    reactivation_time = np.argmax(d_filt, axis=-1)

    # fig, ax = plt.subplots(1, 1, squeeze=False)
    # for i in range(reactivation_time.shape[0]):
    #     x = reactivation_time[i, :]
    #     y = np.arange(0, x.shape[0])
    #     plt.scatter(x, y, c='C0', alpha=0.01)
    # plt.savefig('/Users/mq20185996/Dropbox/crazly/fig_reactivation_' + s[-2:] +
    #             '.pdf')
    # plt.close()

    reactivation_time = pd.DataFrame(reactivation_time).melt(
        var_name='activation_time', value_name='reactivation_time')

    # TODO: Fix this hack
    reactivation_time = reactivation_time.loc[
        reactivation_time['reactivation_time'] != 0]
    reactivation_time = reactivation_time.loc[
        reactivation_time['reactivation_time'] != 261]

    reactivation_time = reactivation_time.groupby(['activation_time'
                                                   ]).mean().reset_index()

    reactivation_time['ppt'] = s[-2:]

    reactivation_time_rec.append(reactivation_time)

    # 5. train and plot perception model time on the x and reactivation time on
    #    the y. If processing happens in a similar order for train and test
    #    points, then positive slope
    # fig, ax = plt.subplots(1, 1, squeeze=False)
    # sns.regplot(data=reactivation_time,
    #             x='activation_time',
    #             y='reactivation_time',
    #             scatter_kws={'alpha':0.5},
    #             ax=ax[0, 0])
    # plt.savefig('/Users/mq20185996/Dropbox/crazly/fig_reactivation_' + s[-2:] +
    #             '.pdf')
    # plt.close()

# for i in range(len(reactivation_time_rec)):
#     reactivation_time_rec[i]['ppt'] = i

d = pd.concat(reactivation_time_rec)

d['ppt'] = d['ppt'].astype(int)
d['mts'] = np.isin(d['ppt'], subs_mts)

dd = d.groupby(['activation_time',
                'mts'])['reactivation_time'].mean().reset_index()

fig, ax = plt.subplots(1, 2, figsize=(8, 4), squeeze=False)
sns.regplot(data=dd.loc[dd['mts'] == True],
            x='activation_time',
            y='reactivation_time',
            scatter_kws={'alpha': 0.5},
            ax=ax[0, 0])
sns.regplot(data=dd.loc[dd['mts'] == False],
            x='activation_time',
            y='reactivation_time',
            scatter_kws={'alpha': 0.5},
            ax=ax[0, 1])
ax[0, 0].set_title('MTS')
ax[0, 1].set_title('Non-MTS')
plt.tight_layout()
plt.savefig('/Users/mq20185996/Dropbox/crazly/fig_reactivation_mean.pdf')
plt.close()

ddd = dd.loc[dd['mts'] == True]
x = ddd.activation_time
y = ddd.reactivation_time
lm = pg.linear_regression(x, y)


ddd = dd.loc[dd['mts'] == False]
x = ddd.activation_time
y = ddd.reactivation_time
lm = pg.linear_regression(x, y)
