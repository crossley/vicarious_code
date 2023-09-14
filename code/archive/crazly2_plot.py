import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
 
scores_rec = []
for f in os.listdir('/Volumes/LaCie/Sophie/mjc/code'):
    if f.startswith('crazly2_scores'):
        d = np.loadtxt(f)
        scores_rec.append(d)

scores = np.stack(scores_rec)
scores_mean = np.mean(scores, 0)
err = np.std(scores, axis=0, ddof=1) / len(scores_rec)

print(scores_mean.max())

fig, ax = plt.subplots()
ax.imshow(scores_mean, origin='lower')
# ax.set_xlabel('Times')
# ax.set_ylabel('AUC')
# ax.legend()
# ax.set_title('Sensor space decoding')
plt.show()
