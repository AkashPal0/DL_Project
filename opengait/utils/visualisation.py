from tsnecuda import TSNE
from einops import rearrange
import matplotlib.pyplot as plt
import numpy as np


def visual_summary(x, labs):
        x = rearrange(x, 'b t n c -> b (t n c)')
        X_embedded = TSNE(n_components=2, perplexity=15, learning_rate=10).fit_transform(x)

        def scale_to_01_range(x):
            value_range = (np.max(x) - np.min(x))
        
            starts_from_zero = x - np.min(x)
        
            return starts_from_zero / value_range
 
        tx = X_embedded[:, 0]
        ty = X_embedded[:, 1]
        
        tx = scale_to_01_range(tx)
        ty = scale_to_01_range(ty)

        fig = plt.figure()
        ax = fig.add_subplot(111)
        
        for label in labs:
            indices = [i for i, l in enumerate(labs) if l == label]
        
            current_tx = np.take(tx, indices)
            current_ty = np.take(ty, indices)
        
            color = np.array(colors_per_class[label], dtype=np.float) / 255
        
            ax.scatter(current_tx, current_ty, c=color, label=label)
        
        ax.legend(loc='best')
        plt.show()
