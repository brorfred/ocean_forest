
import os 

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.model_selection import RepeatedKFold
from lineartree import LinearForestRegressor
from joblib import dump as joblib_dump
from joblib import load as joblib_load

from .config import settings

def clean_data(df, env="pp-mattei", depths=False, dropna=True):
    settings.setenv(env=env)
    df = df.copy(deep=True)
    feature_keys = settings.features
    if depths or (not settings.above_zeu): 
        feature_keys.append("depth")
        env = env + "_depths"
    df = df[feature_keys + [settings["y_feature"]]]
    for key in settings.log_features:
        if key in df:
            df[key] = np.log(df[key])
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    if dropna:
        df.dropna(inplace=True)
    if not settings.above_zeu:
        df = df[df.Zeu<df.depth]
    return df[feature_keys], df[settings["y_feature"]]


def regress(df=None, env="pp-mattei", random_state=None, depths=True, 
            rerf=False, **kw):
    # evaluate random forest ensemble for regression
    # https://machinelearningmastery.com/random-forest-ensemble-in-python/
    settings.setenv(env=env)
    if df is None:
        print("load dataframe")
        df = load(env=env)
    else:
        df = df.copy(deep=True)
    X,y = clean_data(df, env=env, depths=depths)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33, random_state=random_state)

    # define the model
    if len(kw) == 0:
        kw = dict(n_estimators=3000, max_depth=200, min_samples_leaf=7)
    if rerf:
        model = LinearForestRegressor(base_estimator=Lasso(), **kw)
    else:
        print("Use normal RF regressor")
        model = RandomForestRegressor(**kw)
    model.env = env
    model.fit(X_train, y_train)
    model.X_test = X_test
    model.y_test = y_test
    model.X_train = X_train
    model.y_train = y_train
    print(r'R2 train: %.3f' % (model.score(model.X_train, model.y_train)))
    print(r'R2 test:  %.3f' % (model.score(model.X_test,  model.y_test)))
    return model


def dump_model(model, filename=None):
    name = f"model_{getattr(model, 'env', '')}"
    if "depth" in model.X_train:
        name += "_with_depths"
    r2str = model.score(model.X_test, model.y_test).astype(str)[2:7]
    if filename is None:
        filename = f"rf_models/{name}_{r2str}.joblib"
    joblib_dump(model, filename, compress=9)

def load_model(datadir="rf_models", filename=None):
    return joblib_load(os.path.join(datadir, filename))

