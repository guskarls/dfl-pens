# Credits
# 1. CIFAR-10 training/validation/testing code: https://medium.com/@sergioalves94/deep-learning-in-pytorch-with-cifar-10-dataset-858b504a6b54
# 2. Model fusion: https://towardsdatascience.com/preserving-data-privacy-in-deep-learning-part-1-a04894f78029

# General imports
import argparse

# Torch
import torch

# Our code
import train_functions
from train_utils import *
from train_functions import *
from client_manager import *
from data_manager import *
from models import *


def parseargs():
    parser = argparse.ArgumentParser()

    # General
    parser.add_argument(
        "--gpu", help="Which gpu.", type=str, default=None,
    )
    parser.add_argument(
        "--pid", help="Process number, 1, 2, 3 etc.", type=int, default=None,
    )
    parser.add_argument(
        "--start_date", help="Start date, used as folder name.", type=str,
    )
    parser.add_argument(
        "-rep", "--n_repetitions", type=int, default=1,
    )
    parser.add_argument(
        "--independent_model_initialization", type=bool, default=True,
    )

    # Training
    parser.add_argument("--n_comrounds", help="Number of epochs", type=int, default=333)
    parser.add_argument(
        "--lr", help="Learning rate.", type=float, default=1e-3,
    )
    parser.add_argument(
        "--batch_size", help="Batch size for train loader", type=int, default=8,
    )
    parser.add_argument(
        "--early_stopping", type=bool, default=True,
    )
    parser.add_argument(
        "--patience", help="Early stopping patience.", type=int, default=50,
    )
    parser.add_argument(
        "--moving_average_window_size", help="No.", type=int, default=5,
    )

    # Plotting
    parser.add_argument(
        "--plot", type=bool, default=True,
    )
    parser.add_argument(
        "--plot_title", type=str, default=None,
    )

    # Federation
    parser.add_argument(
        "--n_clients", help="Number of clients.", type=int, default=200,
    )
    parser.add_argument(
        "--n_neighbours",
        help="Number of neighbours to communicate with.",
        type=int,
        default=2,
    )
    parser.add_argument(
        "-fed",
        "--federation",
        help="Type of cooperation among clients.",
        type=str,
        default="no_cooperation",
    )
    parser.add_argument(
        "--local_epochs",
        help="Number of local epochs between averaging.",
        type=int,
        default=3,
    )

    # Settings for specific federation types
    parser.add_argument(
        "--C", help="Client fraction for FedAvg.", type=float, default=1,
    )

    # Neighbour selection
    parser.add_argument(
        "--neighbour_selection",
        help="Method for deciding which client communicates with which.",
        type=str,
        default="random",
    )
    parser.add_argument("--neighbour_exploration", type=str)
    parser.add_argument("--topk", type=int)
    parser.add_argument(
        "-metric", "--neighbour_selection_metric", default="acc", type=str
    )
    parser.add_argument(
        "--neighbour_epsilon", type=float, default=99,
    )
    parser.add_argument(
        "--epsilon_decay_rate", type=float, default=1,
    )
    parser.add_argument(
        "--min_epsilon", type=float, default=0,
    )
    parser.add_argument(
        "--n_clients_to_consider", type=int, default=20,
    )

    # Data
    parser.add_argument(
        "--dataset", type=str, default="CIFAR-10",
    )
    parser.add_argument(
        "--n_train_and_val",
        help="Number of train+val datapoints.",
        type=int,
        default=400,
    )
    parser.add_argument(
        "--n_test", help="Number of test datapoints.", type=int, default=1,
    )
    parser.add_argument(
        "--val_proportion", type=float, default=0.75,
    )
    parser.add_argument(
        "--data_sampling",
        "-data",
        help="Data sampling method.",
        type=str,
        default="dirichlet",
    )
    parser.add_argument(
        "--clustering", help="Clustering method.", type=str, default="rotation180",
    )
    parser.add_argument(
        "--n", type=int, default=2,
    )
    parser.add_argument(
        "-mf",
        "--majority_fraction",
        help="Majority class fraction for non-heterogenous data. 0.2 corresponds to iid data.",
        type=float,
        default=0.2,
    )
    parser.add_argument(
        "--n_majority_classes", help="Number of majority classes.", type=int, default=2,
    )
    parser.add_argument(
        "--alpha", help="Dirichlet parameter.", type=float, default=1,
    )

    parser.add_argument(
        "--use_clients_in_cluster", "-icc", type=bool, default=False,
    )

    # Edu
    parser.add_argument(
        "--edu", type=bool, default=False,
    )
    parser.add_argument(
        "--edu_fed_step1", type=str, default="random_subset",
    )
    parser.add_argument(
        "--edu_fed_step2", type=str, default="random_subset",
    )
    parser.add_argument(
        "--client_clustering_method", type=str, default="above_average",
    )
    parser.add_argument(
        "--n_clusters", type=int, default=2,
    )
    parser.add_argument(
        "--n_final_epochs", type=int, default=1000,
    )
    parser.add_argument(
        "--initial_local_epochs", type=int, default=3,
    )
    parser.add_argument(
        "--n_comrounds_with_client_cluster", type=int, default=1,
    )
    parser.add_argument(
        "--n_initial_communication_rounds", type=int, default=200,
    )
    parser.add_argument(
        "--n_final_communication_rounds", type=int, default=333,
    )
    parser.add_argument(
        "--reset_models_before_final_training", type=bool, default=False,
    )
    parser.add_argument(
        "--start_from_step2", type=bool, default=False,
    )
    parser.add_argument("--clients_path", type=str)
    parser.add_argument("--ratio", type=float, default=1.0)
    parser.add_argument(
        "--n_neighbours_in_cluster", type=int, default=99,
    )
    parser.add_argument(
        "--evaluate_ideal", type=bool, default=False,
    )
    parser.add_argument(
        "--n_samplings", type=int, default=1,
    )
    return parser.parse_args()


if __name__ == "__main__":
    # Argument parsing
    args = parseargs()

    # Set device
    args.gpu, device = set_gpu_and_device(args.gpu)

    # Initialize clients
    clients = create_clients(args, TensorFlowCIFAR10Net, device)

    # Train models
    train(args, clients)
