import pickle

import click
from mlflow import MlflowClient
import mlflow
from mlflow.entities import ViewType
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error

HPO_EXPERIMENT_NAME = "random-forest-hyperopt"
EXPERIMENT_NAME = "random-forest-best-models"
RF_PARAMS = ['max_depth', 'n_estimators', 'min_samples_split', 'min_samples_leaf', 'random_state']
client = MlflowClient(tracking_uri="sqlite:///homework.db")
mlflow.set_tracking_uri("sqlite:///homework.db")
mlflow.set_experiment(EXPERIMENT_NAME)

def load_pickle(filename):
    with open(filename, "rb") as f_in:
        return pickle.load(f_in)

def train_and_log_model(data_path, params):
    X_train, y_train = load_pickle(f"{data_path}/train.pkl")
    X_val, y_val = load_pickle(f"{data_path}/val.pkl")
    X_test, y_test = load_pickle(f"{data_path}/test.pkl")

    with mlflow.start_run():
        new_params = {}
        for param in RF_PARAMS:
            new_params[param] = int(params[param])

        r = RandomForestRegressor(**new_params)
        r.fit(X_train, y_train)
        rmse_val = root_mean_squared_error(y_val, r.predict(X_val))
        rmse_test = root_mean_squared_error(y_test, r.predict(X_test))
        mlflow.sklearn.log_model(sk_model=r, artifact_path="model")
        mlflow.log_param("rmse_val", rmse_val)
        mlflow.log_param("rmse_test", rmse_test)

@click.command()
@click.option(
    "--data_path",
    is_flag=False,
    default="./output",
    help="Path to load data from."
)
@click.option(
    "--top_n",
    is_flag=False,
    default=5,
    help="Number of top models that need to be evaluated"
)
def register_best(data_path: str, top_n: int):
    exp_id = client.get_experiment_by_name(HPO_EXPERIMENT_NAME).experiment_id
    runs = client.search_runs(
        experiment_ids=exp_id,
        run_view_type=ViewType.ACTIVE_ONLY,
        max_results=top_n,
        order_by=["metrics.rmse ASC"]
    )
    for run in runs:
        train_and_log_model(data_path=data_path, params=run.data.params)

    exp_id = client.get_experiment_by_name(EXPERIMENT_NAME).experiment_id
    best_run = client.search_runs(
        experiment_ids=exp_id,
        run_view_type=ViewType.ACTIVE_ONLY,
        max_results=1,
        order_by=["parameter.rmse_test ASC"]
    )[0]
    mlflow.register_model(model_uri=f"runs:/{best_run.info.run_id}/model", name="random-forest-best-model")

if __name__ == "__main__":
    register_best()