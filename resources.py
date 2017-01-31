from collections import defaultdict

import numpy as np
import pandas as pd
import scipy.sparse as sp

class IncrementerDict(defaultdict):

    def __init__(self, base_value=0):
        self.__incrementer__ = base_value
        super(IncrementerDict, self).__init__(self.__incrementer_factory__)

    def __incrementer_factory__(self):
        result = self.__incrementer__
        self.__incrementer__ += 1
        return result


def build_dict(input_list):
    output_dict = IncrementerDict()

    for element in input_list:
        output_dict[element]

    return output_dict


def build_users_dataframe(data, features_dict):
    df = pd.DataFrame(columns=['user', 'feature', 'value'])

    for i in range(len(data)):
        row = data.iloc[i]
        gender_key = 'female'
        if row['gender'] == 'M':
            gender_key = 'male'

        df.loc[i * 3] = [row['user'], features_dict['age'], row['age']]
        df.loc[i * 3 + 1] = [row['user'], features_dict[gender_key], 1]
        df.loc[i * 3 + 2] = [row['user'], features_dict[row['occupation']], 1]

    return df


def build_items_dataframe(data, features_dict):
    df = data.copy()
    df['release'] = data['release'].str[-4:].astype(int)
    df = df.ix[:, features_dict.keys()]
    return df


# Function from: https://github.com/lyst/lightfm/blob/
# 4c658e6be477fc4be39aada2e2001642d1c80489/lightfm/datasets/movielens/__init__.py#L57
def build_interaction_matrix(num_rows, num_cols=None, data=None, function=None, function_args=None):
    if num_cols is None:
        return sp.coo_matrix(num_rows.values, dtype=np.float32).tocoo()

    mat = sp.lil_matrix((num_rows, num_cols), dtype=np.float32)

    for _, row in data.iterrows():
        x, y, value = function(row, **function_args)
        if x is not None:
            mat[x, y] = value

    return mat.tocoo()


def collaborative_filter(row, min_rating=3):
    if row['rating'] >= min_rating:
        return row['user'], row['item'], row['rating']
    else:
        return None, None, None


def content_filter(row, kind):
    return row[kind], row['feature'], row['value']


def normalize_output(output, max_rating, min_rating, max_output, min_output):
    range_ouput = max_output - min_output
    range_rating = max_rating - min_rating
    return max_rating - ((range_rating * (max_output - output)) / range_ouput)
