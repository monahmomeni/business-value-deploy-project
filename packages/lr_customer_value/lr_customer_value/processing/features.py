import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin

from lr_customer_value.processing.errors import InvalidModelInputError


class TemporalVariableEstimator(BaseEstimator, TransformerMixin):
    """Extracts temporal information of a timestamp variable"""

    def __init__(self, variables=None):
        if not isinstance(variables, list):
            self.variables = [variables]
        else:
            self.variables = variables

    def fit(self, x, y=None):
        # to keep on with scikit-learn api
        return self

    def transform(self, x):
        x = x.copy()
        for var in self.variables:
            x[var] = x[var].astype('datetime64')
            x[var] = x[var].dt.year
        return x


class MergeHighDimCategories(BaseEstimator, TransformerMixin):
    """Merge the levels in high dimensional categorical variables"""

    def __init__(self, variables=None):
        self.hdim_dict_ = {}
        assert (isinstance(variables, dict))
        self.variables = variables

    def fit(self, x, y):

        # add target to dataframe temporarily
        tmp = pd.concat([x, y], axis=1)
        tmp.columns = list(x.columns) + ['outcome']

        for var, interval in self.variables.items():
            z = tmp.groupby(var)['outcome'].agg(['sum', 'count'])
            t = (z['sum'] / z['count'])

            # persist transforming dictionary
            self.hdim_dict_[var] = {}
            for i in range(len(interval[:-1])):
                # find indices that correspond to mapping interval
                ix = z[(interval[i] <= t) & (t <= interval[i + 1])]['count'].sort_values().index
                # assign it to i_th merging category
                self.hdim_dict_[var]['merge_' + str(i)] = list(ix.values)
        return self

    def transform(self, x):

        x = x.copy()
        for var in list(self.variables.keys()):
            for key, _ in self.hdim_dict_[var].items():
                x[var] = np.where(x[var].isin(self.hdim_dict_[var][key]), key, x[var])

        # check if transformer returns NaN
        if x[self.variables].isnull().any().any():
            null_counts = x[self.variables].isnull().any()
            vars_ = {
                key: value for (key, value) in null_counts.items() if value is True
            }
            raise InvalidModelInputError(
                f"Merging high-dim categories has introduced NaN when "
                f"transforming categorical variables: {vars_.keys()}"
            )

        return x
