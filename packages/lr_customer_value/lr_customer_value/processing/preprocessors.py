import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from lr_customer_value.processing.errors import InvalidModelInputError


class CategoricalImputer(BaseEstimator, TransformerMixin):
    """categorical missing value imputer"""

    def __init__(self, variables=None):
        if not isinstance(variables, list):
            self.variables = [variables]
        else:
            self.variables = variables

    def fit(self, x, y=None):
        # just to keep on with scikit-learn api
        return self

    def transform(self, x):
        x = x.copy()
        for var in self.variables:
            x[var] = x[var].fillna('missing')
        return x


class ChangeDataType(BaseEstimator, TransformerMixin):
    """Change type of input data"""

    def __init__(self, target_type, variables=None):

        self.target_type = target_type

        if not isinstance(variables, list):
            self.variables = [variables]
        else:
            self.variables = variables

    def fit(self, x, y=None):
        return self

    def transform(self, x):

        x = x.copy()
        for var in self.variables:
            x[var] = x[var].astype(self.target_type)

        return x


class RareLabelCategoricalEncoder(BaseEstimator, TransformerMixin):
    """Handles the rare levels in categorical variables"""

    def __init__(self, tol=0.01, variables=None):

        self.tol = tol
        # to be used in fit method
        self.encoder_dict_ = {}

        if not isinstance(variables, list):
            self.variables = [variables]
        else:
            self.variables = variables

    def fit(self, x, y=None):

        # Encoder learns the most frequent levels for each variable
        for var in self.variables:
            tmp = pd.Series(x[var].value_counts() / np.float(len(x)))
            self.encoder_dict_[var] = list(tmp[tmp >= self.tol].index)

        return self

    def transform(self, x):
        x = x.copy()
        for var in self.variables:
            x[var] = np.where(x[var].isin(self.encoder_dict_[var]), x[var], 'rare')
        return x


class CategoricalEncoder(BaseEstimator, TransformerMixin):
    """Encodes string levels to ordinal numbers"""

    def __init__(self, variables=None):
        self.encoder_dict_ = {}

        if not isinstance(variables, list):
            self.variables = [variables]
        else:
            self.variables = variables

    def fit(self, x, y):
        # add target to dataframe temporarily
        tmp = pd.concat([x, y], axis=1)
        tmp.columns = list(x.columns) + ['outcome']

        for var in self.variables:
            z = tmp.groupby([var])['outcome'].agg(['sum', 'count'])
            t = (z['sum'] / z['count']).sort_values(ascending=True).index
            # persist transforming dictionary
            self.encoder_dict_[var] = {k: i for i, k in enumerate(t, 0)}

        return self

    def transform(self, x):
        x = x.copy()
        for var in self.variables:
            x[var] = x[var].map(self.encoder_dict_[var])

        # check if transformer returns NaN
        if x[self.variables].isnull().any().any():
            null_counts = x[self.variables].isnull().any()
            vars_ = {
                key: value for (key, value) in null_counts.items() if value is True
            }
            raise InvalidModelInputError(
                f"Categorical encoder has introduced NaN when "
                f"transforming categorical variables: {vars_.keys()}"
            )
        return x


class FeatureScaler(BaseEstimator, TransformerMixin):
    """This is essentially the same as MinMaxScaler"""

    def __init__(self, variables=None):
        self.min_ = {}
        self.max_ = {}
        if not isinstance(variables, list):
            self.variables = [variables]
        else:
            self.variables = variables

    def fit(self, x, y=None):
        for var in self.variables:
            self.min_[var] = np.nanmin(x[var])
            self.max_[var] = np.nanmax(x[var])
        return self

    def transform(self, x):
        x = x.copy()
        for var in self.variables:
            x[var] = (x[var] - self.min_[var]) / np.float(self.max_[var] - self.min_[var])
        return x


class DropUnwantedFeatures(BaseEstimator, TransformerMixin):

    def __init__(self, variables_to_drop=None):

        if not isinstance(variables_to_drop, list):
            self.variables = [variables_to_drop]
        else:
            self.variables = variables_to_drop

    def fit(self, x, y=None):
        return self

    def transform(self, x):
        x = x.copy()
        x = x.drop(self.variables, axis=1)
        return x
