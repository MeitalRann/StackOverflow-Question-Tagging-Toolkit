import numpy as np

def get_tags(pred, n_tags, max_tags=5, th=0.9):
    n = len(pred)
    tag = np.zeros((n,n_tags))
    for i in range(n):
        ind = np.argpartition(pred[i,:], -max_tags)[-max_tags:]
        max_tag = max(pred[i,:])
        for j in range(max_tags):
            val_j = pred[i,ind[j]]
            if val_j >= th*max_tag:
                tag[i, ind[j]] = 1
    return tag
