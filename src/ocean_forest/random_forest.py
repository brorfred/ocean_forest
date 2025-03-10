
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.model_selection import RepeatedKFold
from lineartree import LinearForestRegressor
from joblib import dump as joblib_dump
from joblib import load as joblib_load

from .config import settings

def load(env, datadir=None, filename=None):
    settings.setenv(env=env)
    datadir = settings.input_datadir if datadir is None else datadir
    filename = settings.input_file if filename is None else filename
    return pd.read_hdf(os.path.join(datadir, filename))

def clean_data(df, env="pp-mattei", depths=False, dropna=True):
    settings.setenv(env=env)
    df = df.copy(deep=True)
    feature_keys = settings.features
    if (not settings.above_zeu) and ("depth" in df):
        df = df[df.Zeu<df.depth]
    if depths:
        feature_keys.append("depth")
        env = env + "_depths"
        #print("Use depth as feature")
    df = df[feature_keys + [settings["y_feature"]]]
    for key in settings.log_features:
        if key in df:
            df[key] = np.log(df[key])
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    if dropna:
        df.dropna(inplace=True)
    return df[feature_keys], df[settings["y_feature"]]

def regress(df=None, env="pp-mattei", random_state=None, depths=False, 
            rerf=False, test_size=0.25, xydict=None, **kw):
    # evaluate random forest ensemble for regression
    # https://machinelearningmastery.com/random-forest-ensemble-in-python/
    settings.setenv(env=env)
    if df is None:
        print(f"load dataframe from '{settings['INPUT_FILE']}'")
        df = load(env=env)
    else:
        df = df.copy(deep=True)
    if xydict is None:
        X,y = clean_data(df, env=env, depths=depths)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state)
    else:
        X_train = xydict["X_train"]
        X_test  = xydict["X_test"]
        y_train = xydict["y_train"]
        y_test  = xydict["y_test"]
    #Set hyper parameters
    rfkw = settings.get("rf_params", {})
    for key in kw:
        rfkw[key] = kw[key]
    if rerf:
        print("Use extended RF regressor")
        model = LinearForestRegressor(base_estimator=Lasso(), **rfkw)
    else:
        print("Use normal RF regressor")
        model = RandomForestRegressor(**rfkw)
    model.env = env
    model.fit(X_train, y_train)
    model.X_test = X_test
    model.y_test = y_test
    model.X_train = X_train
    model.y_train = y_train
    print(r'R2 train: %.3f' % (model.score(model.X_train, model.y_train)))
    print(r'R2 test:  %.3f' % (model.score(model.X_test,  model.y_test)))
    return model


def dump_model(model, datadir="rf_models", filename=None):
    Path(datadir).mkdir(parents=True, exist_ok=True)
    name = f"model_{getattr(model, 'env', '')}"
    if "depth" in model.X_train:
        name += "_with_depths"
    r2str = model.score(model.X_test, model.y_test).astype(str)[2:7]
    if filename is None:
        filename = f"rf_models/{name}_{r2str}.joblib"
    joblib_dump(model, filename, compress=9)

def load_model(datadir="rf_models", filename=None):
    return joblib_load(os.path.join(datadir, filename))

