import os

import pandas as pd

from batch import prepare_data, get_input_path, read_data, get_output_path
from tests.test_batch import dt


def store_input_data():
    data = [
        (None, None, dt(1, 1), dt(1, 10)),
        (1, 1, dt(1, 2), dt(1, 10)),
        (1, None, dt(1, 2, 0), dt(1, 2, 59)),
        (3, 4, dt(1, 2, 0), dt(2, 2, 1)),
    ]

    columns = ['PULocationID', 'DOLocationID', 'tpep_pickup_datetime', 'tpep_dropoff_datetime']
    df = pd.DataFrame(data, columns=columns)
    df_input = prepare_data(columns, df)
    url = os.getenv("S3_ENDPOINT_URL")
    options = {
        'client_kwargs': {
            'endpoint_url': url
        }
    }
    path = get_input_path(2023, 1)
    df_input.to_parquet(
        path,
        engine='pyarrow',
        compression=None,
        index=False,
        storage_options=options
    )

def test_end_to_end():
    os.system("./../batch.py 2023 1")
    filename = get_output_path(2023, 1)
    df = read_data(filename, [])

    actual = round(df['predicted_duration'].sum(), 2)
    expected = 36.28

    assert expected == actual


if __name__ == "__main__":
    store_input_data()