import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

X_train = np.load('path/pca_train_images.npy')
y_train = np.load('path/pca_train_labels.npy')

plot = sns.scatterplot(x=X_train[..., 0], y=X_train[..., 1],
                       hue=y_train.squeeze(), legend=False)
plt.savefig('pca.png')
plt.show()
