import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error


def save_fig(fig_id, tight_layout=True):
  path = "foo" + ".png"
  print("Saving figure", fig_id)
  if tight_layout:
    plt.tight_layout()
  plt.savefig(path, format='png', dpi=300)


def random_digit(X):
  some_digit = X[36000]
  some_digit_image = some_digit.reshape(28, 28)
  plt.imshow(some_digit_image, cmap=mpl.cm.binary,
             interpolation="nearest")
  plt.axis("off")

  save_fig(some_digit)
  plt.show()


def load_and_sort():
  try:
    from sklearn.datasets import fetch_openml
    mnist = fetch_openml(name='mnist_784', version=1, cache=True)
    mnist.target = mnist.target.astype(np.int8)
  except ImportError:
    from sklearn.datasets import fetch_mldata
    mnist = fetch_mldata('MNIST original')
  return mnist


def sort_by_target(mnist):
  reorder_train = np.array(sorted([(target, i) for i, target in enumerate(mnist.target[:60000])]))[:, 1]
  reorder_test = np.array(sorted([(target, i) for i, target in enumerate(mnist.target[60000:])]))[:, 1]
  X_train = mnist.data.loc[reorder_train]
  y_train = mnist.target.loc[reorder_train]
  X_test = mnist.data.loc[reorder_test + 60000]
  y_test = mnist.target.loc[reorder_test + 60000]
  return X_train, y_train, X_test, y_test


def train_predict(some_digit, X_train, y_train):
  s_classifier = train(X_train, y_train)
  return s_classifier.predict(pd.DataFrame(some_digit).transpose())


class_to_learn = 5


def train(X_train, y_train):
  shuffle_index = np.random.permutation(60000)
  X_train, y_train = X_train.iloc[shuffle_index], y_train.iloc[shuffle_index]
  y_train_classified = (y_train == class_to_learn)
  from sklearn.linear_model import SGDClassifier
  s_classifier = SGDClassifier()
  s_classifier.fit(X_train, y_train_classified)
  return s_classifier


def calculate_cross_val_score(k, X):
  rmse_sum = 0
  for i in range(k):
    X_train, y_train, X_test, y_test = sort_by_target(X)
    s_classifier = train(X_train, y_train)
    y_test_pred = s_classifier.predict(X_test)
    y_test_actual = (y_test == class_to_learn)
    rmse_i = rmse(pd.DataFrame(y_test_actual.to_numpy()), pd.DataFrame(y_test_pred))
    print("rmse of fold %s: %s", ((i+1),rmse_i))
    rmse_sum += rmse_i
  return rmse_sum / k


def rmse(y_actual, y_predict):
  return np.sqrt(mean_squared_error(y_actual, y_predict))


def main():
    np.random.seed(42)
    mpl.rc('axes', labelsize=14)
    mpl.rc('xtick', labelsize=12)
    mpl.rc('ytick', labelsize=12)
    PROJECT_ROOT_DIR = "."
    IMAGE_DIR = "FIXME"

    mnist = load_and_sort()
    print("cross validation score: %s", calculate_cross_val_score(5, mnist))

main()
