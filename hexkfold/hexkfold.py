import h3
import numpy as np


class HexKFold:

    def __init__(self, n_splits, lat_col='lat', lon_col='lon', h3_size=6):
        self.n_splits = n_splits
        self.h3_size = h3_size
        self.lat_col = lat_col
        self.lon_col = lon_col

    def split(self, X):

        hex_ids = X.reset_index()[[self.lat_col, self.lon_col]] \
            .apply(lambda p: h3.geo_to_h3(p[self.lat_col],
                                          p[self.lon_col],
                                          self.h3_size), axis=1)
        hex_unique = hex_ids.unique()
        hex_nunique = len(hex_unique)
        indices = np.arange(hex_nunique)
        np.random.shuffle(hex_unique)
        fold_hex_sizes = np.full(self.n_splits, hex_nunique // self.n_splits, dtype=int)
        fold_hex_sizes[:hex_nunique % self.n_splits] += 1

        current = 0
        for fold_size in fold_hex_sizes:
            start, stop = current, current + fold_size
            hex_test = hex_unique[indices[start:stop]]
            hex_train = [h for h in hex_unique if h not in hex_test]

            indices_train = list(X.reset_index()[hex_ids.isin(hex_train)].index)
            indices_test = list(X.reset_index()[hex_ids.isin(hex_test)].index)

            yield [indices_train, indices_test]
            current = stop

    def get_n_splits(self, X, y, groups=None):
        return self.n_splits
