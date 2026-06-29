#!/usr/bin/env python
# coding: utf-8
import os
import sys
import pickle
from typing import Any
import pandas as pd
from pandas import DataFrame, Series
import pyarrow.fs as fs


def read_data(filename, categorical):
    url = os.getenv("S3_ENDPOINT_URL")
    if url is not None:
        options = {
            'client_kwargs': {
                'endpoint_url': url
            }
        }

        df = pd.read_parquet(
            filename,
            engine='pyarrow',
            storage_options=options)
        return df

    df = pd.read_parquet(filename)
    return prepare_data(categorical, df)

def save_data(df, path):
    url = os.getenv("S3_ENDPOINT_URL")
    if url is not None:
        options = {
            'client_kwargs': {
                'endpoint_url': url
            }
        }
        df.to_parquet(
            path,
            engine='pyarrow',
            compression=None,
            index=False,
            storage_options=options
        )
    else:
        df.to_parquet(path, engine='pyarrow', index=False)

def prepare_data(categorical, df: DataFrame) -> Series | DataFrame | Any:
    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')

    return df

def get_input_path(year, month):
    default_input_pattern = 's3://nyc-duration/in/{year:04d}-{month:02d}.parquet'
    input_pattern = os.getenv('INPUT_FILE_PATTERN', default_input_pattern)
    return input_pattern.format(year=year, month=month)


def get_output_path(year, month):
    default_output_pattern = 's3://nyc-duration/out/{year:04d}-{month:02d}.parquet'
    output_pattern = os.getenv('OUTPUT_FILE_PATTERN', default_output_pattern)
    return output_pattern.format(year=year, month=month)


def main(year, month):
    input_file = get_input_path(year, month)
    output_file = get_output_path(year, month)

    with open('model.bin', 'rb') as f_in:
        dv, lr = pickle.load(f_in)

    categorical = ['PULocationID', 'DOLocationID']
    df = read_data(input_file, categorical)
    df['ride_id'] = f'{year:04d}/{month:02d}_' + df.index.astype('str')

    dicts = df[categorical].to_dict(orient='records')
    X_val = dv.transform(dicts)
    y_pred = lr.predict(X_val)

    print('predicted mean duration:', y_pred.mean())

    df_result = pd.DataFrame()
    df_result['ride_id'] = df['ride_id']
    df_result['predicted_duration'] = y_pred
    save_data(df_result, output_file)

if __name__ == "__main__":
    year = int(sys.argv[1])
    month = int(sys.argv[2])
    main(year, month)