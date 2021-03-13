import pandas as pd
import numpy as np

import seaborn as sns
from matplotlib import pyplot as plt

class CarPrice:

    def __init__(self):
        self.df = pd.read_csv('data/data.csv')
        print(f'${len(self.df)} lines loaded')

    def trim(self):
        self.df.columns = self.df.columns.str.lower().str.replace(' ', '_')
        string_columns = list(self.df.dtypes[self.df.dtypes == 'object'].index)
        for col in string_columns:
            self.df[col] = self.df[col].str.lower().str.replace(' ', '_')

    def validate(self):
        pass

    def linear_regression(self, X, y):
        ones = np.ones(X.shape[0])
        X = np.column_stack([ones, X])
        XTX = X.T.dot(X)
        XTX_inv = np.linalg.inv(XTX)
        w = XTX_inv.dot(X.T).dot(y)
        return w[0], w[1:]

    def plot_price_distribution(self, entity=None, max_threshold=None, min_threshold=None):
        if entity is None:
            entity=self.df.msrp
        plt.figure(figsize=(6, 4))
        if max_threshold:
            sns.distplot(entity[entity < max_threshold], kde=False, hist_kws=dict(color='black', alpha=1))
        elif min_threshold:
            sns.distplot(entity[entity > min_threshold], kde=False, hist_kws=dict(color='black', alpha=1))
        else:
            sns.distplot(entity, kde=False, hist_kws=dict(color='black', alpha=1))
        plt.ylabel('Frequency')
        plt.xlabel('Price')
        plt.title('Distribution of prices')
        plt.show()

def prepare_X(df):
        base = ['engine_hp', 'engine_cylinders',
                'highway_mpg', 'city_mpg', 'popularity']
        df_num = df[base]
        df_num = df_num.fillna(0)
        X = df_num.values
        return X

def rmse(y, y_pred):
    error = y_pred - y
    mse = (error ** 2).mean()
    return np.sqrt(mse)

def main():
    obj = CarPrice()
    obj.trim()
    #obj.plot_price_distribution()
    #obj.plot_price_distribution(max_threshold=100000)
    #log_price = np.log1p(obj.df.msrp)
    #print(obj.df.engine_hp.isnull())
    #obj.plot_price_distribution(log_price)
    #spliting the dataset for test, train and validate
    np.random.seed(2)

    n = len(obj.df)

    n_val = int(0.2 * n)
    n_test = int(0.2 * n)
    n_train = n - (n_val + n_test)

    idx = np.arange(n)
    np.random.shuffle(idx)

    df_shuffled = obj.df.iloc[idx]

    obj.df_train = df_shuffled.iloc[:n_train].copy()
    obj.df_val = df_shuffled.iloc[n_train:n_train+n_val].copy()
    obj.df_test = df_shuffled.iloc[n_train+n_val:].copy()

    obj.y_train_orig = obj.df_train.msrp.values
    obj.y_val_orig = obj.df_val.msrp.values
    obj.y_test_orig = obj.df_test.msrp.values

    obj.y_train = np.log1p(obj.df_train.msrp.values)
    obj.y_val = np.log1p(obj.df_val.msrp.values)
    obj.y_test = np.log1p(obj.df_test.msrp.values)

    del obj.df_train['msrp']
    del obj.df_val['msrp']
    del obj.df_test['msrp']

    X_train = prepare_X(obj.df_train)
    w_0, w = obj.linear_regression(X_train, obj.y_train)

    obj.y_pred = w_0 + X_train.dot(w)
    print('train:', rmse(obj.y_train, obj.y_pred))

    X_val = prepare_X(obj.df_val)
    y_pred = w_0 + X_val.dot(w)
    print('validation:', rmse(obj.y_val, y_pred))

    #Test The model
    X_test = prepare_X(obj.df_test)
    y_pred_test = w_0 + X_val.dot(w)
    y_pred_test_val = np.expm1(y_pred_test)

    #print the result
    obj.df_test['msrp'] = obj.y_test_orig
    obj.df_test['msrp_predicted'] = y_pred_test_val

    print(obj.df_test[['make', 'model', 'engine_cylinders', 'transmission_type', 'driven_wheels', 'number_of_doors',
             'market_category', 'vehicle_size', 'vehicle_style', 'highway_mpg', 'city_mpg', 'popularity', 'msrp', 'msrp_predicted']].head().to_markdown())


main()