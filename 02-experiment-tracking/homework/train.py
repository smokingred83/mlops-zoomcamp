import pickle

import click
import mlflow
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error


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
def train(data_path: str = "./output"):
    mlflow.set_tracking_uri("sqlite:///homework.db")
    mlflow.set_experiment("exp-tracking-homework")
    mlflow.autolog()
    with mlflow.start_run():
        X_train, y_train = load_pickle(f"{data_path}/train.pkl")
        X_val, y_val = load_pickle(f"{data_path}/train.pkl")
        regressor = RandomForestRegressor(max_depth=10, random_state=0)
        regressor.fit(X_train, y_train)
        y_pred = regressor.predict(X_val)
        rmse = root_mean_squared_error(y_val, y_pred)

if __name__ == "__main__":
    train()