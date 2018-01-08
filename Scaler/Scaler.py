class Scaler:
    def __init__(self):
        self.means = None
        self.stdevs = None

    def fit(self, X):
        """Store the mean and standard deviation of each column"""
        self.means = np.apply_along_axis(self._findMean, 0, X)
        self.stdevs = np.apply_along_axis(self._findStdev, 0, X)

    def scale(self, X):
        """Scale the columns of X based on the fit"""
        if(self.means == None or self.stdevs == None):
            raise Exception("Scaler has not been fitted yet! Use Scaler.fit(ndarray: X)")

        if(X.shape[1] != len(self.means)):
            raise ValueError("Passed ndarray does not have the same number of columns as the fitting dataset!")
            
        itr = (self._standardize(X, col) for col in range(X.shape[1]))
        return np.fromiter(itr, X.dtype, count=X.shape[1])

    def _findMean(self, x):
        return x.mean()

    def _findStdev(self, x):
        return x.std()

    def _standardize(self, X, col):
        return (X[:,col] - self.means(col))/self.stdevs(col)
