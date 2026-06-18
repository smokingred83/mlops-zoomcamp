import pickle

import click
import mlflow
import numpy as np
from hyperopt import Trials, fmin, tpe, hp, STATUS_OK
from hyperopt.pyll import scope
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error

mlflow.set_tracking_uri("sqlite:///homework.db")
mlflow.set_experiment("random-forest-hyperopt")

def load_pickle(filename: str):
    with open(filename, "rb") as f_in:
        return pickle.load(f_in)

@click.command()
@click.option(
    "--data_path",
    is_flag=False,
    default="./output",
    help="Path to load data from."
)
@click.option(
    "--num_trials",
    is_flag=False,
    default=15,
    help="Number of evaluations to run."
)
def run_optimization(data_path: str, num_trials: int):
    X_train, y_train = load_pickle(f"{data_path}/train.pkl")
    X_val, y_val = load_pickle(f"{data_path}/val.pkl")

    def objective(params):
        with mlflow.start_run():
            mlflow.log_params(params)
            regressor = RandomForestRegressor(**params)
            regressor.fit(X_train, y_train)
            y_pred = regressor.predict(X_val)
            rmse = root_mean_squared_error(y_val, y_pred)
            mlflow.log_param("rmse", rmse)
            return {'loss': rmse, 'status': STATUS_OK}

    search_space = {
        'max_depth': scope.int(hp.quniform('max_depth', 1, 20, 1)),
        'n_estimators': scope.int(hp.quniform('n_estimators', 10, 50, 1)),
        'min_samples_split': scope.int(hp.quniform('min_samples_split', 2, 10, 1)),
        'min_samples_leaf': scope.int(hp.quniform('min_samples_leaf', 1, 4, 1)),
        'random_state': 42
    }
    rstate = np.random.default_rng(42)
    fmin(
        fn=objective,
        space=search_space,
        algo=tpe.suggest,
        max_evals=num_trials,
        trials=Trials(),
        rstate=rstate
    )

if __name__ == "__main__":
    run_optimization()