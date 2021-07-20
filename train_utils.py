import copy
import pathlib
import matplotlib.pyplot as plt
import numpy as np
from math import *
from collections import OrderedDict

# from jenkspy import JenksNaturalBreaks

import torch


# Evaluation
def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))


def evaluate(model, data_loader):
    outputs = [model.validation_step(batch) for batch in data_loader]
    return model.validation_epoch_end(outputs)


def evaluate_and_update_history(client):
    train_loss, train_acc = evaluate(client.model, client.train_loader)
    val_loss, val_acc = evaluate(client.model, client.val_loader)
    for key, var in {
        "train_losses": train_loss,
        "train_accs": train_acc,
        "val_losses": val_loss,
        "val_accs": val_acc,
    }.items():
        client.history[key] = np.append(client.history[key], var)


# Average over first dimension
def tensor_average(x, weights):
    if len(weights) != x.shape[0]:
        print("Weights has wrong shape")
        exit()
    axis = 0
    weighted_average = torch.zeros(x.shape[1:], dtype=x.dtype, device=x.device)
    for i in range(x.shape[axis]):
        weighted_average += weights[i] * x[i]
    return weighted_average / sum(weights)


# Averaging
def model_average(models, weights=None):  # Credit 2
    models_copy = copy.deepcopy(models)
    for m in range(len(models)):
        model_dict = models_copy[m].state_dict()
        if weights is not None:
            for k in model_dict.keys():
                model_dict[k] = tensor_average(
                    torch.stack(
                        [models[i].state_dict()[k].float() for i in range(len(models))],
                        0,
                    ),
                    weights=weights,
                )
        else:
            for k in model_dict.keys():
                model_dict[k] = torch.stack(
                    [models[i].state_dict()[k].float() for i in range(len(models))], 0,
                ).mean(0)
        models_copy[m].load_state_dict(model_dict)
    return models_copy[0]


# def model_average(models, weights=None):  # Credit 2
#    models_copy = copy.deepcopy(models)
#    for m in range(len(models)):
#        model_dict = models_copy[m].state_dict()
#        for k in model_dict.keys():
#            model_dict[k] = torch.tensor(np.average(torch.stack(
#                [models[i].state_dict()[k].float() for i in range(len(models))], 0,
#            ).cpu(), axis=0, weights=weights)).to_device(models[0].device)
#        models_copy[m].load_state_dict(model_dict)
#    return models_copy[0]


def average_history(histories, avg_history):
    for key in ["train_losses", "train_accs", "val_losses", "val_accs"]:
        average_loss = np.mean([histories[c][key][-1] for c in range(len(histories))])
        avg_history[key] = np.append(avg_history[key], average_loss)
    return avg_history


# Plotting
def plot_losses(train_losses, val_losses, imagefolder, title):
    plt.plot(train_losses, label="Train loss")
    plt.plot(val_losses, label="Val loss")
    plt.xlabel("communication round")
    plt.ylabel("loss")
    plt.legend()
    plt.title(title)
    plt.savefig(imagefolder + title + " loss.png")
    plt.clf()


def plot_accuracies(
    train_accs, val_accs, imagefolder, title, stopped_at_communication_round=False
):
    plt.plot(train_accs, label="Train accuracy")
    plt.plot(val_accs, label="Validation accuracy")
    if stopped_at_communication_round:
        plt.scatter(
            stopped_at_communication_round + 1,
            val_accs[stopped_at_communication_round + 1],
            color="red",
            marker="v",
            label="Early stopping",
        )
    plt.xlabel("communication round")
    plt.ylabel("accuracy")
    plt.legend()
    plt.title(title)
    plt.savefig(imagefolder + title + " accuracy.png")
    plt.clf()


def save_plots(args, history, plot_title, stopped_at_communication_round=False):
    pathlib.Path(f"data/output/{args.start_date}/plots/plots{args.pid}").mkdir(
        parents=True, exist_ok=True
    )
    imagefolder = f"{str(pathlib.Path(__file__).resolve().parents[1])}/data/output/{args.start_date}/plots/plots{args.pid}/"
    plot_losses(
        history["train_losses"], history["val_losses"], imagefolder, str(plot_title)
    )
    plot_accuracies(
        history["train_accs"],
        history["val_accs"],
        imagefolder,
        str(plot_title),
        stopped_at_communication_round,
    )


def save_plots_for_clients_and_average(
    args,
    clients,
    checkpoints,
    histories_communication_rounds,
    avg_history_communication_rounds,
    stopped_at_communication_round=False,
):
    for c, client in enumerate(clients):
        client_stopped_at_communication_round = False

        if args.federation != "fed_avg":
            if checkpoints[c]["stopped_early"]:
                client_stopped_at_communication_round = checkpoints[c][
                    "communication_round"
                ]
        save_plots(
            args,
            histories_communication_rounds[c],
            f"{args.plot_title}, client_{c}",
            client_stopped_at_communication_round,
        )
    save_plots(
        args,
        avg_history_communication_rounds,
        f"{args.plot_title}, client average",
        stopped_at_communication_round,
    )


# Other
def train_one_model_one_epoch(client):
    # n_batches = int(100 / client.batch_size) + 1
    # data_fraction = 0.2
    # batch_indices = np.random.randint(0, n_batches, round(data_fraction * n_batches))
    for b, batch in enumerate(client.train_loader):
        # if b in batch_indices:
        loss = client.model.training_step(batch)
        loss.backward()
        client.optimizer.step()
        client.optimizer.zero_grad()


def moving_average(x, moving_average_window_size):
    return (
        np.convolve(x, np.ones(moving_average_window_size), "same")
        / moving_average_window_size
    )


def stop_early(history, patience, moving_average_window_size):
    acc = history["val_accs"]
    if moving_average_window_size:
        acc = moving_average(acc, moving_average_window_size)
    if np.argmax(acc) + 1 < len(acc) - patience:
        return True
    else:
        return False


def clients_to_communicate_with(args, client, clients, communication_round):
    weights = None
    if args.neighbour_selection == "random":
        n_neighbours = (
            args.n_neighbours if len(clients) >= args.n_neighbours else len(clients)
        )
        neighbours = np.random.choice(
            [client_ for client_ in clients if client_ != client],
            n_neighbours,
            replace=False,
        )
        if args.edu:
            client.neighbours_history2 = np.vstack(
                (client.neighbours_history2, [neighbour.id for neighbour in neighbours])
            )
        else:
            client.neighbours_history = np.vstack(
                (client.neighbours_history, [neighbour.id for neighbour in neighbours])
            )

    elif args.neighbour_selection == "ideal":
        clients_to_choose_from = [
            client_
            for client_ in clients
            if client_ != client and client_.cluster == client.cluster
        ]

        neighbours = np.random.choice(
            clients_to_choose_from, args.n_neighbours, replace=False
        )

    elif args.neighbour_selection == "performance_based":
        # Sample 10 times
        for _ in range(args.n_samplings):
            clients_to_consider = [client_ for client_ in clients if client_ != client]
            clients_to_consider = np.random.choice(
                clients_to_consider, args.n_clients_to_consider, replace=False
            )
            other_clients_metric = OrderedDict()
            for other_client in clients_to_consider:
                train_loss, train_acc = evaluate(
                    client.model, other_client.train_loader
                )
                other_clients_metric[other_client] = (
                    1 / (train_loss + 1e-5)
                    if args.neighbour_selection_metric == "loss"
                    else train_acc
                )
            other_clients_sorted_by_metric = sorted(
                other_clients_metric, key=other_clients_metric.get, reverse=True
            )
            pathlib.Path(
                f"data/output/{args.start_date}/local epochs={args.local_epochs}"
            ).mkdir(parents=True, exist_ok=True)
            # cluster0 = [client_ for client_ in other_clients_metric if client_.cluster == 0]
            # cluster1 = [client_ for client_ in other_clients_sorted_by_metric if client_.cluster == 1]
            # cluster0_accs = [other_clients_metric[client_] for client_ in cluster0]
            # cluster1_accs = [other_clients_metric[client_] for client_ in cluster1]
            # col0 = "green" if client.cluster == 0 else "red"
            # col1 = "green" if client.cluster == 1 else "red"
            # plt.bar(x=range(len(cluster0)), height=cluster0_accs, color=col0)
            # plt.bar(x=range(len(cluster0), 20), height=cluster1_accs, color=col1)
            # plt.title(f"client {client.id}")
            # plt.savefig(f"data/output/{args.start_date}/local epochs={args.local_epochs}/client_id={client.id}, comround={communication_round}.png")
            # plt.clf()
            if args.neighbour_exploration == "greedy":
                neighbours = other_clients_sorted_by_metric[: args.n_neighbours]
            elif args.neighbour_exploration == "sampling":
                probs = np.array(list(other_clients_metric.values()))
                probs = probs / sum(probs)
                neighbours = np.random.choice(
                    list(other_clients_metric.keys()),
                    size=args.n_neighbours,
                    replace=False,
                    p=probs,
                )
            elif args.neighbour_exploration == "weights":
                val_loss, val_acc = (
                    client.history["val_losses"][-1],
                    client.history["val_accs"][-1],
                )
                own_client_metric = (
                    1 / (val_loss + 1e-5)
                    if args.neighbour_selection_metric == "loss"
                    else val_acc
                )
                weights = np.array(
                    list(other_clients_metric.values()) + [own_client_metric]
                )
                neighbours = clients_to_consider
            elif args.neighbour_exploration == "epsilon_greedy":
                # Epsilon decay
                epsilon = (
                    args.neighbour_epsilon
                    * np.power(args.epsilon_decay_rate, communication_round)
                    if args.neighbour_epsilon
                    * np.power(args.epsilon_decay_rate, communication_round)
                    > args.min_epsilon
                    else args.min_epsilon
                )

                neighbours = other_clients_sorted_by_metric[: args.n_neighbours]
                n_neighbours_to_swap = np.random.binomial(args.n_neighbours, epsilon)
                neighbours_to_swap = np.random.choice(
                    neighbours, n_neighbours_to_swap, replace=False
                )
                remaining_neighbours = [
                    neighbour
                    for neighbour in neighbours
                    if neighbour not in neighbours_to_swap
                ]
                clients_to_choose_from = [
                    client_
                    for client_ in clients_to_consider
                    if client_ not in remaining_neighbours
                ]
                new_neighbours = np.random.choice(
                    clients_to_choose_from, n_neighbours_to_swap, replace=False
                )
                neighbours = list(remaining_neighbours) + list(new_neighbours)

            elif args.neighbour_exploration == "topk":
                candidates = other_clients_sorted_by_metric[: args.topk]
                neighbours = np.random.choice(candidates, size=args.n_neighbours)
            if not (args.edu and args.neighbour_selection == "random"):
                client.neighbours_history = np.vstack(
                    (
                        client.neighbours_history,
                        [neighbour.id for neighbour in neighbours],
                    )
                )

            # if not client.id in list(range(5))+list(range(100, 105)):
            #     return neighbours, weights

    return neighbours, weights


def save_checkpoints(
    args,
    clients,
    histories,
    comround,
    plot_title,
    moving_average_window_size,
    start_date,
):
    """
    Saves checkpoints corresponding to the models that reach a higher-than-previous accuracy.
    """
    checkpoints = [
        torch.load(
            f"data/output/{start_date}/checkpoints/checkpoint{args.pid}/{plot_title}/client_{c}.tar"
        )
        for c in range(len(clients))
    ]
    for c, client in enumerate(clients):
        checkpoint = checkpoints[c]
        # problem här vid ifsatsen, sista MA värdet här ej samma som i stop_early funktionen
        current_moving_average = moving_average(
            histories[c]["val_accs"], moving_average_window_size
        )[-1]
        current_accuracy = np.average(
            histories[c]["val_accs"][-moving_average_window_size:]
        )
        if (
            current_accuracy > checkpoint["val_acc"]
            and checkpoint["stopped_early"] == False
        ):
            torch.save(
                {
                    "communication_round": comround,
                    "stopped_early": False,
                    "model_state_dict": client.model.state_dict(),
                    "val_acc": current_accuracy,
                },
                f"data/output/{start_date}/checkpoints/checkpoint{args.pid}/{plot_title}/client_{c}.tar",
            )
    return checkpoints


def update_checkpoints_with_stop_early_values(
    args,
    clients,
    histories_communication_rounds,
    plot_title,
    patience,
    start_date,
    moving_average_window_size,
):
    """
    Updates checkpoints with stop_early values and returns a bool list with clients saying which clients have stopped early.
    """
    checkpoints = [
        torch.load(
            f"data/output/{start_date}/checkpoints/checkpoint{args.pid}/{plot_title}/client_{c}.tar"
        )
        for c in range(len(clients))
    ]
    for c, client in enumerate(clients):
        if not checkpoints[c]["stopped_early"]:
            if args.federation == "fed_avg":
                histories_communication_rounds_c = histories_communication_rounds[
                    0
                ]  # avg_history_communication_rounds
            else:
                histories_communication_rounds_c = histories_communication_rounds[c]
            if stop_early(
                histories_communication_rounds_c,
                patience=patience,
                moving_average_window_size=moving_average_window_size,
            ):
                # Use best model
                client.model.load_state_dict(checkpoints[c]["model_state_dict"])

                client.stopped_early = True
                checkpoints[c]["stopped_early"] = True
                torch.save(
                    checkpoints[c],
                    f"data/output/{start_date}/checkpoints/checkpoint{args.pid}/{plot_title}/client_{c}.tar",
                )
    clients_stopped_early = [checkpoint["stopped_early"] for checkpoint in checkpoints]
    return clients_stopped_early


def evaluate_and_save_models_before_training(
    args, clients, histories, avg_history, plot_title, start_date
):
    """
    Evaluate and save models before training.
    """
    for c, client in enumerate(clients):
        client.model.eval()
        evaluate_and_update_history(client)
        histories[c] = client.history

        # Save model
        pathlib.Path(
            f"data/output/{start_date}/checkpoints/checkpoint{args.pid}/{plot_title}/"
        ).mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "communication_round": None,
                "stopped_early": False,
                "model_state_dict": client.model.state_dict(),
                "val_acc": histories[c]["val_accs"][-1],
            },
            f"data/output/{start_date}/checkpoints/checkpoint{args.pid}/{plot_title}/client_{c}.tar",
        )
    avg_history = average_history(histories, avg_history)

    return histories, avg_history


def train_clients_one_epoch(clients):
    """
    Train clients one epoch.
    """
    for c, client in enumerate(clients):
        client.model.train()
        if not client.stopped_early:
            train_one_model_one_epoch(client)


def evaluate_clients(clients, histories, avg_history):
    """
    Evaluate clients and return histories and avg_history.
    """
    for c, client in enumerate(clients):
        client.model.eval()
        evaluate_and_update_history(client)
        histories[c] = client.history
    avg_history = average_history(histories, avg_history)
    return histories, avg_history


def euclidean_distance(x, y):
    return sqrt(sum(pow(a - b, 2) for a, b in zip(x, y)))


def initialize_neighbours_dict(args, clients):
    neighbours_dict = dict()
    if args.neighbour_selection == "ideal":
        for client in clients:
            distances = dict()
            own_label_dist = client.label_dist
            for client_ in clients:
                if client_ != client:
                    distances[client_] = euclidean_distance(
                        own_label_dist, client_.label_dist
                    )
            best_neighbours = sorted(distances, key=distances.get, reverse=False)
            neighbours_dict[client.id] = best_neighbours[: args.n_neighbours]
    else:  # random assignment
        for client in clients:
            neighbours_dict[client.id] = np.random.choice(
                [client_ for client_ in clients if client_ != client],
                args.n_neighbours,
                replace=False,
            )
    return neighbours_dict


def update_history_communication_rounds(
    args,
    comround,
    clients,
    histories,
    histories_communication_rounds,
    avg_history,
    avg_history_communication_rounds,
):
    for key in ["train_losses", "train_accs", "val_losses", "val_accs"]:
        for c, client in enumerate(clients):
            histories_communication_rounds[c][key] = np.append(
                histories_communication_rounds[c][key], histories[c][key][comround + 1],
            )
        avg_history_communication_rounds[key] = np.append(
            avg_history_communication_rounds[key], avg_history[key][comround + 1],
        )
    return histories_communication_rounds, avg_history_communication_rounds


def perform_averaging(args, clients, selected_clients, communication_round):
    if args.federation == "no_cooperation":
        # # Sample 10 times
        # for _ in range(args.n_samplings):
        #     clients_to_consider = [client_ for client_ in clients if client_ != client]
        #     clients_to_consider = np.random.choice(
        #         clients_to_consider, args.n_clients_to_consider, replace=False
        #     )
        #     other_clients_metric = OrderedDict()
        #     for other_client in clients_to_consider:
        #         train_loss, train_acc = evaluate(client.model, other_client.train_loader)
        #         other_clients_metric[other_client] = (
        #             1 / (train_loss + 1e-5)
        #             if args.neighbour_selection_metric == "loss"
        #             else train_acc
        #         )
        #     other_clients_sorted_by_metric = sorted(
        #         other_clients_metric, key=other_clients_metric.get, reverse=True
        #     )
        #     neighbours = other_clients_sorted_by_metric[: args.n_neighbours]
        #     client.neighbours_history = np.vstack(
        #         (client.neighbours_history, [neighbour.id for neighbour in neighbours])
        #     )

        #     if not client.id in list(range(5))+list(range(100, 105)):
        #         break
        pass
    if args.federation == "fed_avg":
        models = [client.model for client in selected_clients]
        averaged_model = model_average(models)
        model_dict = averaged_model.state_dict()
        for c, client in enumerate(clients):
            client.model.load_state_dict(model_dict)
            client.optimizer = torch.optim.Adam(client.model.parameters(), args.lr)

    if args.federation == "random_subset":
        for c, client in enumerate(selected_clients):
            if not client.stopped_early:
                if args.use_clients_in_cluster:
                    other_clients, weights = clients_to_communicate_with(
                        args,
                        client,
                        client.clients_in_same_cluster,
                        communication_round,
                    )

                else:
                    other_clients, weights = clients_to_communicate_with(
                        args, client, clients, communication_round
                    )
                models = [client_.model for client_ in np.append(other_clients, client)]
                averaged_model = model_average(models, weights)
                model_dict = averaged_model.state_dict()
                client.model.load_state_dict(model_dict)
                client.optimizer = torch.optim.Adam(client.model.parameters(), args.lr)

    if args.federation == "gossip":
        for c, client in enumerate(selected_clients):
            for other_client in other_clients:
                if not other_client.stopped_early:
                    if args.use_clients_in_cluster:
                        other_clients, weights = clients_to_communicate_with(
                            args,
                            client,
                            client.clients_in_same_cluster,
                            communication_round,
                        )

                    else:
                        other_clients, weights = clients_to_communicate_with(
                            args, client, clients, communication_round
                        )
                    models = [client.model, other_client.model]
                    averaged_model = model_average(models, weights)
                    model_dict = averaged_model.state_dict()
                    other_client.model.load_state_dict(model_dict)
                    other_client.optimizer = torch.optim.Adam(
                        other_client.model.parameters(), args.lr
                    )
    return clients


def create_clusters(args, clients, n_clusters):
    # Loopa igenom våra klienter
    for client_main in clients:
        accuracies = (
            []
        )  # np.zeros(args.n_clients-1) #np.ones()*0.1 elr använda val_loader för den egna
        for other_client in clients:
            if client_main != other_client:
                _, acc = evaluate(client_main.model, other_client.train_loader)
                accuracies.append(acc)
                labels = cluster(accuracies, n_clusters)
                labels.insert(client_main.id, 0)
                clients_in_same_cluster = [
                    client for client in clients if labels[client.id] == n_clusters - 1
                ]
                if len(clients_in_same_cluster) < args.n_neighbours:
                    remaining_clients = list(
                        set(clients) - set(clients_in_same_cluster)
                    )
                    extrclient_s = np.random.choice(
                        remaining_clients,
                        args.n_neighbours - len(clients_in_same_cluster),
                        replace=False,
                    )
                clients_in_same_cluster += extrclient_s
                client_main.clients_in_same_cluster


# va ska hända om antalet punkter i rätt cluster är mindre än n_nieghbours?
def cluster(data, n_clusters):
    jnb = JenksNaturalBreaks(nb_class=n_clusters)
    jnb.fit(data)
    # best_group = jnb.groups_[-1]
    # print(jnb.labels_)
    return jnb.labels_


def set_gpu_and_device(gpu):
    repo_name = str(pathlib.Path(__file__).resolve().parents[1]).split("/")[-1]
    if torch.cuda.is_available():
        if gpu != None:
            device = "cuda:" + str(gpu)
        else:
            if repo_name == "noa":
                device = "cuda:0"
                gpu = 0
            elif repo_name == "gustav":
                device = "cuda:1"
                gpu = 1
    else:
        device = "cpu"

    return gpu, device