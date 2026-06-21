import sys
sys.path.insert(0, "playground-series-s6e6/src")
import train_xgboost
train_xgboost.XGB_PARAMS["device"] = "cpu"
train_xgboost.XGB_PARAMS["n_jobs"] = 4
train_xgboost.main()
