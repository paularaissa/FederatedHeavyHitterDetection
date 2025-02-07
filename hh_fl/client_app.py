import pandas as pd
import csv
from datetime import datetime
from river import sketch
from river import stream
import pickle
from flwr.client import NumPyClient, Client, ClientApp
from flwr.common.config import unflatten_dict
from flwr.common.context import Context

from hh_fl.task import (
    load_subfolders
)
import os

delimiter = ","

class FlowerRiverClient(NumPyClient):

    def __init__(self, cid: str, only_folder: str, col_name: str,
                 fading_factor: float, epsilon: float, support: float,
                 main_src: str, data_set_partitioned_src: str, top_n: int,
                 nnodes: str):
        self.cid = cid
        self.onlyfolder = only_folder
        self.colname = col_name
        self.previous_sketch_path = "heavy_hitters_sketch_" + str(self.cid)+".pkl"
        self.fading_factor = fading_factor
        self.epsilon = epsilon
        self.support = support
        self.main_src = main_src
        self.data_set_partitioned_src = data_set_partitioned_src
        self.top_n = top_n
        self.nnodes = nnodes

    def get_parameters(self, config):
        """
        Return the initial model parameters.
        In this example, we return a dummy list. You could instead serialize your heavy hitters sketch
        or any other model state you want to share with the server.
        """
        # Attempt to load an existing sketch or create a new one if it doesn't exist.
        heavy_hitters_sketch = self.load_previous_sketch()
        if heavy_hitters_sketch is None:
            heavy_hitters_sketch = sketch.HeavyHitters(
                fading_factor=self.fading_factor,
                epsilon=self.epsilon,
                support=self.support,
            )
            # Optionally, save the newly created sketch for future rounds
            self.save_sketch(heavy_hitters_sketch)

        # Return a representation of the initial parameters.
        # Depending on your application, this could be a list of numpy arrays or any serializable structure.
        # Here, we return an empty list as a placeholder.
        return []

    def load_previous_sketch(self):
        """Load the heavy hitters sketch if the file exists; otherwise, return None."""
        if os.path.exists(self.previous_sketch_path):
            with open(self.previous_sketch_path, 'rb') as f:
                heavy_hitters_sketch = pickle.load(f)
            if heavy_hitters_sketch is None:
                print("DEBUG: Loaded sketch is None.")
            return heavy_hitters_sketch
        else:
            print("DEBUG: No previous sketch file found.")
            return None


    def load_previous_sketch(self):
        """Carrega o heavy_hitters_sketch se o arquivo existir e round > 1."""
        if os.path.exists(self.previous_sketch_path):
            with open(self.previous_sketch_path, 'rb') as f:
                heavy_hitters_sketch = pickle.load(f)
        else:
            heavy_hitters_sketch = None
        return heavy_hitters_sketch

    def save_sketch(self, heavy_hitters_sketch):
        """Save the heavy_hitters_sketch into file"""
        # Validade directory
        directory = os.path.dirname(self.previous_sketch_path)
        if directory:  # Evita erro se o diretório não for especificado
            os.makedirs(directory, exist_ok=True)
        # Save object sketch on file
        with open(self.previous_sketch_path, 'wb') as f:
            pickle.dump(heavy_hitters_sketch, f)

    # def save_previous_data(self):
    #     with open('previous_data.pkl', 'wb') as f:
    #         pickle.dump(self.previous_round_data, f) FUNÇÃO SEM UTILIZAÇÃO


    def get_next_batch(self, config):
        data = []
        labels = []
        new_file = self.onlyfolder + "/" + "output_" + str(config["current_round"]) + ".csv"
        return new_file

    def train(self, config):
        # Create a new sketch by default.
        heavy_hitters_sketch = sketch.HeavyHitters(
            fading_factor=self.fading_factor,
            epsilon=self.epsilon,
            support=self.support,
        )

        # For rounds beyond the first, try to load a previous sketch.
        if int(config["current_round"]) > 1:
            heavy_hitters_sketch = self.load_previous_sketch()
            if heavy_hitters_sketch is None:
                heavy_hitters_sketch = sketch.HeavyHitters(
                    fading_factor=self.fading_factor,
                    epsilon=self.epsilon,
                    support=self.support,
                )

        cols = ['ANumber', 'BNumber', 'DateTime', 'Action', 'Result', 'CountryCode']
        df_bucket_a = pd.DataFrame(columns=['ANumber', 'Count', 'DateTime'])
        new_batch = self.get_next_batch(config)
        X_y = stream.iter_csv(new_batch, delimiter=delimiter, fieldnames=cols)

        for x, y in X_y:
            if len(x) > 0:
                # Update the sketch with the value from the specified column.
                heavy_hitters_sketch.update(x[self.colname])

                # Retrieve the two most common elements.
                most_common_iter_a = heavy_hitters_sketch.most_common(self.top_n)

                # Check if the bucket is complete.
                if heavy_hitters_sketch._n % heavy_hitters_sketch._bucket_width == 0:
                    bucket_data = x['DateTime']
                    df_new_row = pd.DataFrame(most_common_iter_a, columns=['ANumber', 'Count'])
                    df_new_row['DateTime'] = bucket_data
                    df_new_row = df_new_row.dropna()
                    df_bucket_a = pd.concat([df_bucket_a, df_new_row], axis=0)

        # Create the results directory if it doesn't exist.
        path_results = (
                self.main_src
                + "results_federated/"
                + self.data_set_partitioned_src
                + self.nnodes
                + "/node_"
                + str(self.cid)
        )
        if not os.path.exists(path_results):
            os.makedirs(path_results)

        # Save the results.
        df_bucket_a.to_csv(
            path_results + '/' + "df_bucket_anumber_" + str(config["current_round"]) + ".csv",
            index=False
        )

        # Save the updated sketch for the next round.
        self.save_sketch(heavy_hitters_sketch)

        return heavy_hitters_sketch, df_bucket_a


    def read_csv_streaming(self, file_path):
        with open(file_path, 'r') as csv_file:
            csv_reader = csv.reader(csv_file)
            for row in csv_reader:
                yield row

    def fit(self, parameters, config):
        hh, df_bucket_hh = self.train(config)

        tuple_bucket_hh = list(df_bucket_hh.values)

        # Convertendo a lista de tuplas
        tuple_list = [(x[0], x[1], x[2]) for x in tuple_bucket_hh]

        return tuple_list, 0, {}

    # def evaluate(self, parameters, config):
    #     print(f"[Client {self.cid}] evaluate, config: {config}")
    #     set_parameters(self.net, parameters)
    #     loss, accuracy = test(self.net, self.valloader)
    #     return float(loss), len(self.valloader), {"accuracy": float(accuracy)}

def client_fn(context: Context) -> Client:
    cid = context.node_config["partition-id"]
    col_name = "ANumber"
    fading_factor = context.run_config["fading-factor"]
    epsilon = context.run_config["epsilon"]
    support = context.run_config["support"]
    top_n = context.run_config["top-n"]
    main_src = context.run_config["main-src"]
    data_set_partitioned_src = context.run_config["data-set-partitioned-src"]
    nnodes = str(context.node_config["num-partitions"]) + "nodes"

    onlyfolders = load_subfolders(main_src + data_set_partitioned_src + nnodes)
    onlyfolder = onlyfolders[cid]

    #onlyfolders = load_subfolders(main_src + data_set_partitioned_src + nnodes)
    #print(onlyfolders)
    #print(cid)

    return FlowerRiverClient(
        cid=cid,
        only_folder=onlyfolder,
        col_name=col_name,
        fading_factor=fading_factor,
        epsilon=epsilon,
        support=support,
        main_src=main_src,
        data_set_partitioned_src=data_set_partitioned_src,
        top_n=top_n,
        nnodes=nnodes
    ).to_client()

# Create ClientApp
app = ClientApp(client_fn=client_fn)