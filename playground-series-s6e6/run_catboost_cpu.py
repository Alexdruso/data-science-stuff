import sys
sys.path.insert(0, "playground-series-s6e6/src")
import train_catboost
# Override to CPU — GPU is busy with Optuna LGBM tuning
train_catboost.CB_PARAMS["task_type"] = "CPU"
train_catboost.CB_PARAMS["thread_count"] = 4
train_catboost.main()
