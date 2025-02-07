from typing import Callable, Dict, List, Optional, Tuple, Union
import os
import numpy as np
import pandas as pd
import flwr as fl
from flwr.common import (
    EvaluateIns,
    EvaluateRes,
    FitRes,
    Scalar,
    parameters_to_ndarrays,
    ndarrays_to_parameters,
    Parameters
)
from flwr.common.context import Context
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg
from flwr.server import ServerApp, ServerAppComponents, ServerConfig

class FedAnalytics(FedAvg):
    def __init__(
            self, min_fit_clients,min_evaluate_clients, min_available_clients, on_fit_config_fn, context,
            num_partitions
    ) -> None:
        super().__init__()
        self.min_fit_clients = min_fit_clients
        self.min_evaluate_clients = min_evaluate_clients
        self.min_available_clients = min_available_clients
        self.on_fit_config_fn = on_fit_config_fn
        self.context = context
        self.num_partitions = num_partitions

    def __repr__(self) -> str:
        return "FedAnalytics"

    def initialize_parameters(
            self, client_manager: Optional[ClientManager] = None
    ) -> Optional[Parameters]:
        return None

    # Função personalizada para selecionar a data mais recente
    def most_actual_datetime(series):
        return series.loc[series.idxmax()]

    # Aggregation Strategy - aggregate training results
    def aggregate_fit(
            self,
            server_round: int,
            results: List[Tuple[ClientProxy, FitRes]],
            failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        # Get results from fit

        # Convert results
        values_aggregated = [
            (parameters_to_ndarrays(fit_res.parameters)) for _, fit_res in results if len(fit_res.parameters.tensors) > 0
        ]

        lista_filtrada = []
        for lista_de_arrays in values_aggregated:
            nova_lista = [arr for arr in lista_de_arrays if len(arr) > 0]
            lista_filtrada.append(nova_lista)

        lista_filtrada = [arr for arr in lista_filtrada if len(arr) > 0]

        values_hh_numpy = np.empty((0, 3))
        if (len(lista_filtrada) > 0):
            values_hh_numpy = np.concatenate(lista_filtrada)

        # Create DataFrame
        if len(values_hh_numpy) > 0:
            df_hh = pd.DataFrame(values_hh_numpy, columns=['Code', 'Value', 'DateTime'])

            # # # Converter a coluna 'Valor' para tipo numérico
            df_hh['Value'] = pd.to_numeric(df_hh['Value'])
            df_hh['DateTime'] = pd.to_datetime(df_hh['DateTime'])

            # # Agrupar e agregar os dados
            df_agrupado = df_hh.groupby('Code').agg({'Value': 'sum', 'DateTime': 'max'}).reset_index()
            df_sorted = df_agrupado.sort_values(by='Value', ascending=False)

            # Cortando as n primeiras linhas
            #df_cortado = df_sorted.iloc[:top_n]
            df_cortado = df_sorted.iloc[:5].copy()

            df_cortado['Window'] = server_round

            main_src = self.context.run_config["main-src"]
            data_set_partitioned_src = self.context.run_config["data-set-partitioned-src"]
            nnodes = str(self.num_partitions) + "nodes"
            #nnodes = self.context.run_config["num-partitions"]

            path_results =  main_src + "results_federated/" + data_set_partitioned_src + nnodes + "/server/"

            if not os.path.exists(path_results):
                os.makedirs(path_results)

            df_cortado.to_csv(path_results + "df_bucket_anumber_" + str(server_round) + ".csv", index=False)

        #print(df_cortado)

        return ndarrays_to_parameters([]), {}

    def evaluate(self, server_round: int, parameters: Parameters
                 ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        return 0, {}

    def configure_evaluate(
            self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, EvaluateIns]]:
        pass

    def aggregate_evaluate(
            self,
            server_round: int,
            results: List[Tuple[ClientProxy, EvaluateRes]],
            failures: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        pass

def config_func(rnd: int) :
    """ Return configuration with global epochs """
    config = {
        "current_round": rnd,
    }
    return config

def server_fn(context: Context) -> ServerAppComponents:
    # Read from config
    num_rounds = context.run_config["num-server-rounds"]
    #fraction_fit = context.run_config["fraction-fit"]
    #fraction_evaluate = context.run_config["fraction-evaluate"]

    # Init an empty Parameter
    parameters = Parameters(tensor_type="", tensors=[])

    num_partitions = context.run_config["num-supernodes"]

    strategy = FedAnalytics(
        min_fit_clients=1,
        min_evaluate_clients=1,
        min_available_clients=1,
        on_fit_config_fn=config_func,
        context = context,
        num_partitions = num_partitions
        #initial_parameters=fl.common.ndarrays_to_parameters(params)
    )

    # Defines strategy
    # strategy = FedAnalytics(
    #     fraction_fit = fraction_fit,
    #     fraction_evaluate = fraction_evaluate,
    #     evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation,
    #     on_evaluate_config_fn=config_func,
    #     on_fit_config_fn=config_func,
    #     initial_parameters=parameters,
    # )
    config = ServerConfig(num_rounds=num_rounds)

    return ServerAppComponents(strategy=strategy, config=config)

# Create ServerApp
app = ServerApp(
    server_fn = server_fn,
)