import pandas as pd
import io
import json
from PIL import Image

def pull_data() -> dict[str, list[dict]]:
    splits = {'train': 'mnist/train-00000-of-00001.parquet', 'test': 'mnist/test-00000-of-00001.parquet'}

    for key, val in splits.items():

        try:
            df = pd.read_pickle(f"outputs/{key}.pkl")

        except FileNotFoundError:
            df = pd.read_parquet(f"hf://datasets/ylecun/mnist/{val}")
            df.to_pickle(f"outputs/{key}.pkl")

        dataset = []

        for pd_tuple in df.itertuples():
            p = Image.open(io.BytesIO(pd_tuple.image["bytes"]))

            dataset.append({ "label": pd_tuple.label, "image": [float(val / 255.0) for val in p.getdata()] })
        
        splits[key] = dataset

    return splits

if __name__ == "__main__":
    datasets = pull_data()

    for dataset_type, dataset in datasets.items():
        with open(f"outputs/{dataset_type}.json", "w") as output_file:
            json.dump(dataset, output_file)

    print("done")