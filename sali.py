import numpy as np
from sklearn.preprocessing import StandardScaler

def calc_sali(X, y, k=3):
    
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    res = []
    for n, (xi, yi) in enumerate(zip(X, y)):
        dists = []
        for i in range(len(X)):

            dist = np.linalg.norm(X[i] - xi)
            if dist == 0:
                continue
            else:
                dists.append((i, dist))

        dists = sorted(dists, key=lambda x: x[1], reverse=False)[:k]
        sali = np.median([abs(abs(yi) - abs(y[i])) / x for i, x in dists])

        res.append((n, np.log10(sali)))
        
    return res
