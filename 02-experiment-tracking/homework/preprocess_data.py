import os.path
import pickle

import click
import pandas as pd
from sklearn.feature_extraction import DictVectorizer


def dump_pickle(obj, filename: str):
    with open(filename, "wb") as f_out:
        pickle.dump(obj, f_out)

def read_data(path: str):
    df = pd.read_parquet(path)
    df['duration'] = df.lpep_dropoff_datetime - df.lpep_pickup_datetime
    df.duration = df.duration.apply(lambda t: t.seconds / 60)
    df = df[(df.duration >= 1) & (df.duration <= 60)]
    categorical = ['PULocationID', 'DOLocationID']
    df[categorical] = df[categorical].astype(str)
    return df

def preprocess(df: pd.DataFrame, dv: DictVectorizer, fit_dv: bool = False):
    df['PU_DO'] = df['PULocationID'] + "_" + ['PULocationID']
    categorical = ['PU_DO']
    numerical = ['trip_distance']
    dicts = df[categorical + numerical].to_dict(orient="records")
    X = dv.fit_transform(dicts) if fit_dv else dv.transform(dicts)
    return X, dv

@click.command(
    help="""
Testing preprocessing data.
    """
)
@click.option(
    "--raw_data_path",
    is_flag=False,
    default="",
    help="Path to data."
)
@click.option(
    "--dest_path",
    is_flag=False,
    default="",
    help="Path to save data to."
)
def main(
    raw_data_path: str = "",
    dest_path: str = ""
):
    df_train = read_data(f"{raw_data_path}/green_tripdata_2023-01.parquet")
    df_val = read_data(f"{raw_data_path}/green_tripdata_2023-02.parquet")
    df_test = read_data(f"{raw_data_path}/green_tripdata_2023-03.parquet")

    X_train, dv = preprocess(df_train, DictVectorizer(), True)
    X_val, _ = preprocess(df_val, dv)
    X_test, _ = preprocess(df_test, dv)

    target = "duration"
    y_train = df_train[target].values
    y_val = df_val[target].values
    y_test = df_test[target].values

    os.makedirs(dest_path, exist_ok=True)
    dump_pickle(dv, f"{dest_path}/dv.pkl")
    dump_pickle((X_train, y_train), f"{dest_path}/train.pkl")
    dump_pickle((X_val, y_val), f"{dest_path}/val.pkl")
    dump_pickle((X_test, y_test), f"{dest_path}/test.pkl")


if __name__ == "__main__":
    main()