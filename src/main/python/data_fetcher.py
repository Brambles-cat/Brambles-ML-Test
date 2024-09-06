import pandas as pd
import io
import json
from PIL import Image

splits = {'train': 'mnist/train-00000-of-00001.parquet', 'test': 'mnist/test-00000-of-00001.parquet'}

for key, val in splits.items():

    try:
        df = pd.read_pickle(f"outputs/{key}.pkl")

    except FileNotFoundError:
        df = pd.read_parquet(f"hf://datasets/ylecun/mnist/{val}")
        df.to_pickle(f"{key}.pkl")

    dataset = []

    for pd_tuple in df.itertuples():
        p = Image.open(io.BytesIO(pd_tuple.image["bytes"]))

        dataset.append({ "label": pd_tuple.label, "image": [float(val / 255.0) for val in p.getdata()] })

    with open(f"outputs/{key}.json", "w") as output_file:
        json.dump(dataset, output_file)

print("done")