import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA
#from _config import config
       
class PCAHandler(BaseEstimator, TransformerMixin):
    def __init__(self, apply_pca=False, variance=0.95):
        self.apply_pca = apply_pca
        self.variance = variance
        self.pca = None
        self.num_cols = None  # To store the indices of numerical columns

    def fit(self, X, y=None):
        if self.apply_pca:
            # Select only numerical columns and exclude 'groupid'
            self.num_cols = X.select_dtypes(include='number').columns
            self.num_cols = self.num_cols.difference(['groupid'])  # Exclude 'groupid'
            # Apply PCA only to the remaining numerical columns
            self.pca = PCA(n_components=self.variance)
            self.pca.fit(X[self.num_cols])
        return self

    def transform(self, X):
        if self.apply_pca and self.pca:
            # Transform only the numerical columns excluding 'groupid'
            X_transformed = self.pca.transform(X[self.num_cols])
            # Create a DataFrame for the PCA-transformed columns
            X_pca = pd.DataFrame(X_transformed, index=X.index, columns=[f'PCA_{i+1}' for i in range(X_transformed.shape[1])])
            # Combine non-PCA columns with PCA-transformed ones
            X = X.drop(columns=self.num_cols).join(X_pca)
        return X
