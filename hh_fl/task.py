import os
from pathlib import Path
from flwr.common.context import Context
import numpy as np

#nnodes = str(Context.run_config["num-nodes"]) + "nodes"
main_src = "trees/data/"
data_set_partitioned_src = "_june_1000/"
delimiter= ","
top_n = "30"

def load_datasets(dataset_path: str):
    """Carrega os arquivos CSV para os clientes."""
    files = os.listdir(dataset_path)
    csv_files = [os.path.join(dataset_path, file) for file in files if file.endswith('.csv')]
    return csv_files

def load_subfolders(folder: str):
    """Carrega subpastas para identificar diferentes clientes."""
    subfolders = []
    for root, dirs, _ in os.walk(folder):
        dirs[:] = [d for d in dirs if d != '.ipynb_checkpoints']
        for dir in dirs:
            subfolders.append(os.path.join(root, dir))
    return subfolders

def count_file_by_folder(directory: str) :
    dir_list = Path(directory)
    files = [item for item in dir_list.iterdir() if item.is_file()]
    return len(files)

# cwd = "/home/paula/ecml2024/data_june_1000/" + str(nnodes)
# onlyfolders = load_subfolders(main_src + data_set_partitioned_src + nnodes)
#
# print(onlyfolders)
#
# num_files = 0
# for folder in onlyfolders:
#     num_files = count_file_by_folder(folder)


