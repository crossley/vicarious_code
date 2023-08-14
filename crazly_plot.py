import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

scores_rec = []
for f in os.listdir('/Volumes/LaCie/Sophie/mjc/code'):
    if f.endswith('.txt'):
        d = np.loadtxt(f)
        scores_rec.append(d)

scores = np.vstack(scores_rec)
scores_mean = np.mean(scores, 0)
err = np.std(scores, axis=0, ddof=1) / len(scores_rec)

fig, ax = plt.subplots()
x = np.linspace(-1.1, 1.8, scores_mean.shape[0])
ax.plot(x, scores_mean, label='score')
ax.fill_between(x, scores_mean-err, scores_mean+err, alpha=0.2)
ax.axhline(.5, color='k', linestyle='--', label='chance')
ax.set_xlabel('Times')
ax.set_ylabel('AUC')
ax.legend()
ax.axvline(.0, color='k', linestyle='-')
ax.set_title('Sensor space decoding')
ax.set_ylim(0.47, 0.56)
plt.show()
