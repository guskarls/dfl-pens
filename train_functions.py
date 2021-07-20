import torch
import numpy as np
from tqdm import tqdm
from tqdm.auto import trange
import copy
from collections import Counter
import shutil
import pickle
from pathlib import Path
import os
import sys

from train_utils import *
from data_manager import to_device, DeviceDataLoader
from models import *


def train(args, clients):
    """
    Generic train
    """
    if args.federation == "no_cooperation":
        args.local_epochs = 1

    histories = {}
    avg_history = {
        "train_losses": np.zeros(0),
        "train_accs": np.zeros(0),
        "val_losses": np.zeros(0),
        "val_accs": np.zeros(0),
    }
    num_selected_clients = round(args.C * len(clients))  # for fed_avg

    # Evaluate and save models before training
    (histories, avg_history,) = evaluate_and_save_models_before_training(
        args, clients, histories, avg_history, args.plot_title, args.start_date,
    )

    t_comround = trange(args.n_comrounds)
    for comround in t_comround:
        t_comround.set_description("epochs")

        # Select clients to train and participate in averaging
        selected_clients = np.random.choice(
            clients, num_selected_clients, replace=False
        )

        # Local update
        if args.federation == "fed_avg":
            for epoch_ in range(args.local_epochs):
                train_clients_one_epoch(selected_clients)
            perform_averaging(args, clients, selected_clients, comround)
        else:
            # Average models
            for client in selected_clients:
                perform_averaging(args, clients, [client], comround)
                # Local update
                for epoch_ in range(args.local_epochs):
                    train_clients_one_epoch([client])

        # Evaluate models
        (histories, avg_history,) = evaluate_clients(clients, histories, avg_history)

        # Save checkpoints. Do so after each communication in fed_avg, otherwise every comround.
        if args.federation == "fed_avg":
            checkpoints = save_checkpoints(
                args,
                [clients[0]],
                [avg_history],
                comround,
                args.plot_title,
                args.moving_average_window_size,
                args.start_date,
            )
        else:
            checkpoints = save_checkpoints(
                args,
                clients,
                histories,
                comround,
                args.plot_title,
                args.moving_average_window_size,
                args.start_date,
            )
        t_comround.set_postfix(
            average_val_loss=avg_history["val_losses"][-1],
            average_val_acc=avg_history["val_accs"][-1],
        )
        # # Plot
        # if comround % 20 == 0 and comround != 0 and args.plot:
        #     save_plots_for_clients_and_average(
        #         args,
        #         clients,
        #         checkpoints,
        #         histories,
        #         avg_history,
        #     )

        # Finish training if all clients have stopped
        if args.federation == "fed_avg":
            clients_stopped_early = update_checkpoints_with_stop_early_values(
                args,
                [clients[0]],
                [avg_history],
                args.plot_title,
                args.patience,
                args.start_date,
                args.moving_average_window_size,
            )
        else:
            clients_stopped_early = update_checkpoints_with_stop_early_values(
                args,
                clients,
                histories,
                args.plot_title,
                args.patience,
                args.start_date,
                args.moving_average_window_size,
            )
        if all(clients_stopped_early):
            if args.plot:
                stopped_at_communication_round = comround - args.patience
                save_plots_for_clients_and_average(
                    args,
                    clients,
                    checkpoints,
                    histories,
                    avg_history,
                    stopped_at_communication_round,
                )
            return clients
    return clients


def train_with_client_clusters(args, clients):
    # Initial local epochs
    args.plot = False
    args.n_comrounds = (
        args.n_initial_local_epochs
    )  # TODO: skapa args.n_initial_local_epochs
    args.federation = "no_cooperation"
    clients = train(args, clients)

    args.federation = "random_subset"
    args.neighbour_selection = "random"

    folders_to_remove = [
        f"data/output/{args.start_date}/checkpoints/checkpoint{args.pid}/",
        f"data/output/{args.start_date}/plots/{args.pid}/",
        f"data/output/{args.start_date}/json/",
    ]
    for f in folders_to_remove:
        shutil.rmtree(f)

    # Train random N times
    for n in range(args.N):  # skapa args.N
        args.n_comrounds = (
            args.n_comrounds_with_client_cluster * args.local_epochs
        )  # TODO: definiera args.n_comrounds_with_client_cluster

        create_clusters(args, clients, args.n_clusters)  # TODO: definiera clustering
        clients = train(args, clients)

        files_to_remove = [
            f"data/output/{args.start_date}/checkpoints/checkpoint{args.pid}/",
            f"data/output/{args.start_date}/plots/{args.pid}/",
        ]
        for file in files_to_remove:
            os.remove(file)

    # Train clients with their final clusters
    args.plot = True
    args.n_comrounds = args.n_final_epochs  # TODO: definiera args.n_final_epochs
    clients = train(args, clients)


def train_edu(args, clients):
    if not args.start_from_step2:
        # Initial local epochs
        args.plot = False
        # args.n_comrounds = args.n_initial * args.initial_local_epochs
        # args.local_epochs = args.initial_local_epochs
        args.federation = "random_subset"
        args.neighbour_selection = "performance_based"
        args.neighbour_exploration = "greedy"
        args.use_clients_in_cluster = False
        args.local_epochs = int(
            1250 // (args.n_train_and_val * (1 - args.val_proportion)) + 1
        )
        args.n_comrounds = 20
        # # Continue step 1: Read clients from file
        # infile = open(args.clients_path, "rb")
        # clients = pickle.load(infile)
        # infile.close()

        clients = train(args, clients)
        # Define clients_in_same_cluster
        for c, client in enumerate(clients):
            counts = Counter(client.neighbours_history.flatten())
            if args.client_clustering_method == "topk":
                n_other_clients_in_same_cluster = int(args.n_clients / 2 - 1)
                if len(counts.values()) >= n_other_clients_in_same_cluster:
                    most_common = [
                        item[0]
                        for item in counts.most_common(n_other_clients_in_same_cluster)
                    ]
                else:
                    most_common = counts.keys()
                client.clients_in_same_cluster = [
                    client_ for client_ in clients if client_.id in most_common
                ]
            elif args.client_clustering_method == "above_average":
                cut_off_value = (
                    args.n_neighbours
                    * args.n_comrounds
                    * args.n_samplings
                    / (args.n_clients - 1)
                ) - 1
                chosen_clients = []
                counts = dict(counts)
                for i in counts.keys():
                    if counts[i] > cut_off_value:
                        chosen_clients.append(i)
                client.clients_in_same_cluster = [
                    client_ for client_ in clients if client_.id in chosen_clients
                ]
        # if args.n_clients != 1000:
        #     # save clients to file
        #     Path(f"data/output/{args.start_date}/clients").mkdir(
        #         parents=True, exist_ok=True
        #     )
        #     outfile = open(f"data/output/{args.start_date}/clients/client-{args.pid}", "wb")
        #     pickle.dump(clients, outfile)
        #     outfile.close()

    else:
        # Read clients from file
        infile = open(args.clients_path, "rb")
        clients = pickle.load(infile)
        infile.close()
        _, device = set_gpu_and_device(args.gpu)
        for client in clients:
            client.train_loader = DeviceDataLoader(client.train_loader, device)
            client.val_loader = DeviceDataLoader(client.val_loader, device)
            client.test_loader = DeviceDataLoader(client.test_loader, device)

    # Reset models
    if args.reset_models_before_final_training:
        _, device = set_gpu_and_device(args.gpu)
        net = TensorFlowCIFAR10Net()  # in the case of dependent model initialization
        for client in clients:
            if args.independent_model_initialization:
                client.model = to_device(TensorFlowCIFAR10Net(), device)
            else:
                client.model = copy.deepcopy(to_device(net, device))
            client.optimizer = torch.optim.Adam(client.model.parameters(), args.lr)

    # Remove checkpoints and plots
    f = f"data/output/{args.start_date}/checkpoints/checkpoint{args.pid}/"
    if Path(f).exists():
        shutil.rmtree(f)
    Path(f).mkdir(parents=True, exist_ok=True)
    # Reset training related client attributes
    for client in clients:
        client.stopped_early = False
        client.history = {
            "train_losses": np.zeros(0),
            "train_accs": np.zeros(0),
            "val_losses": np.zeros(0),
            "val_accs": np.zeros(0),
        }
    # Remaining training after clustering
    args.plot = True
    args.local_epochs = 3
    args.n_comrounds = args.n_final_communication_rounds
    args.federation = args.edu_fed_step2
    args.neighbour_selection = "random"
    args.use_clients_in_cluster = True
    args.n_neighbours = 20
    for client in clients:
        n_neighbours = (
            args.n_neighbours
            if args.n_neighbours <= len(client.clients_in_same_cluster)
            else len(client.clients_in_same_cluster)
        )
        client.neighbours_history2 = np.empty((0, n_neighbours), int)
    clients = train(args, clients)
    return clients, args


def evaluate_ideal(args, clients):
    n_correct = int(args.ratio * args.n_neighbours_in_cluster)
    n_incorrect = args.n_neighbours_in_cluster - n_correct
    n_clients_in_first_cluster = int(args.n_clients / 2)
    for c, client in enumerate(clients):
        if c < n_clients_in_first_cluster:
            correct_clients = np.random.choice(
                clients[:n_clients_in_first_cluster], n_correct, replace=False
            )
            incorrect_clients = np.random.choice(
                clients[n_clients_in_first_cluster:], n_incorrect, replace=False
            )
            client.clients_in_same_cluster = list(correct_clients) + list(
                incorrect_clients
            )
        else:
            correct_clients = np.random.choice(
                clients[n_clients_in_first_cluster:], n_correct, replace=False
            )
            incorrect_clients = np.random.choice(
                clients[:n_clients_in_first_cluster], n_incorrect, replace=False
            )
            client.clients_in_same_cluster = list(correct_clients) + list(
                incorrect_clients
            )

    args.plot = False
    args.local_epochs = 1
    args.n_comrounds = args.n_final * args.local_epochs
    args.federation = "random_subset"
    args.neighbour_selection = "random"
    args.use_clients_in_cluster = True
    clients = train(args, clients)

    return args