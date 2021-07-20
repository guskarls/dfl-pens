import copy
import numpy as np

import torch
from torch.utils.data import random_split, DataLoader
from sklearn.model_selection import train_test_split
from PIL import ImageOps

from data_manager import *
from train_utils import *
from models import *


class Client:
    def __init__(
        self,
        args,
        id,
        device,
        model,
        train_set,
        test_set,
        train_idxs,
        val_idxs,
        test_idxs,
        label_dist,
        cluster=None,
        opt_fun=torch.optim.SGD,
    ):
        self.args = args
        self.id = id
        self.label_dist = label_dist
        self.device = device
        self.model = model
        self.history = {
            "train_losses": np.zeros(0),
            "train_accs": np.zeros(0),
            "val_losses": np.zeros(0),
            "val_accs": np.zeros(0),
        }
        self.optimizer = opt_fun(self.model.parameters(), self.args.lr)
        self.stopped_early = False
        self.cluster = cluster
        self.neighbours_history = np.empty((0, args.n_neighbours), int)
        self.neighbours_history2 = None
        self.clients_in_same_cluster = []

        # Train/val split
        """
        train_set, val_set = random_split(
            train_set,
            [
                len(train_idxs) - int(args.val_proportion * len(train_idxs)),
                int(args.val_proportion * len(train_idxs)),
            ],
        )
        """

        # train_targets = np.array(get_targets(train_set))
        # test_targets = np.array(get_targets(test_set))

        # print(np.unique(np.array(train_targets), return_counts=True))
        # print(np.unique(np.array(test_targets), return_counts=True))
        # print(np.unique(np.array(train_targets)[train_idxs], return_counts=True))
        # print(np.unique(np.array(test_targets)[test_idxs], return_counts=True))
        # exit()

        # Select datapoints from indices for train and val
        val_set = DatasetSplit(train_set, val_idxs)
        # test_set = DatasetSplit(test_set, test_idxs)   # Laddar in hela testsetet (tillhörande klientens kluster) istället för bara en andel
        train_set = DatasetSplit(train_set, train_idxs)

        # print(len(train_set), len(val_set), len(test_set))

        # Define data loaders
        self.train_loader = DataLoader(train_set, self.args.batch_size, shuffle=True)
        self.val_loader = DataLoader(val_set, self.args.batch_size * 2)
        self.test_loader = DataLoader(test_set, self.args.batch_size * 2)

        self.train_loader = DeviceDataLoader(self.train_loader, device)
        self.val_loader = DeviceDataLoader(self.val_loader, device)
        self.test_loader = DeviceDataLoader(self.test_loader, device)

    @property
    def neighbours_history(self):
        return self._neighbours_history

    @neighbours_history.setter
    def neighbours_history(self, neighbours_history):
        self._neighbours_history = neighbours_history


def get_targets(data):
    return [data.__getitem__(i)[1] for i in range(data.__len__())]


def create_clients(args, net_class, device, opt_fun=torch.optim.Adam):
    if args.dataset == "CIFAR-10":
        normalize = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    elif args.dataset == "Fashion-MNIST":
        normalize = transforms.Normalize((0.5), (0.5))
    else:
        print(f"Unknown dataset {args.dataset}")

    nets = np.empty(args.n_clients, dtype=object)
    clients = np.empty(args.n_clients, dtype=object)
    dict_label_dist = {i: np.array([], dtype="int64") for i in range(args.n_clients)}
    net = net_class()
    for c in range(args.n_clients):
        if args.independent_model_initialization:
            nets[c] = to_device(net_class(), device)
        else:
            nets[c] = copy.deepcopy(to_device(net, device))

    # Load data

    transform_standard = transforms.Compose([transforms.ToTensor(), normalize,])
    train_set, test_set, classes = load_dataset(args, transform_standard)
    transforms_dict = {
        "rotation180": transforms.Compose(
            [transforms.RandomVerticalFlip(p=1.0), transforms.ToTensor(), normalize,]
        ),
        "rotation90": transforms.Compose(
            [transforms.RandomRotation([90, 90]), transforms.ToTensor(), normalize,]
        ),
        "inversion": transforms.Compose(
            [ImageOps.invert, transforms.ToTensor(), normalize,]
        ),
        "rotation180_inversion": transforms.Compose(
            [
                transforms.RandomVerticalFlip(p=1.0),
                ImageOps.invert,
                transforms.ToTensor(),
                normalize,
            ]
        ),
        "grayscale": transforms.Compose(
            [transforms.RandomGrayscale(p=1), transforms.ToTensor(), normalize,]
        ),
    }
    if args.clustering == "label":
        train_sets, test_sets = split_data_label(train_set, test_set, 2)
        client_id = 0
        for cluster_idx in range(len(train_sets.keys())):
            client_indxs_train, client_indxs_val, client_indxs_test = generate_iid_data(
                args,
                train_sets[cluster_idx],
                test_sets[cluster_idx],
                args.n_clients // args.n,
            )
            for c in range(args.n_clients // 2):
                clients[client_id] = Client(
                    args,
                    client_id,
                    device,
                    nets[client_id],
                    train_sets[cluster_idx],
                    test_sets[cluster_idx],
                    client_indxs_train[c],
                    client_indxs_val[c],
                    client_indxs_test[c],
                    dict_label_dist[c],
                    cluster=cluster_idx,
                    opt_fun=opt_fun,
                )
                client_id += 1
    elif args.clustering not in ["False", "label"]:
        if args.n_clients == 1:
            train_set, test_set, _ = load_dataset(
                args,
                transforms_dict[args.clustering],
                two_clusters_central_training=True,
            )
            client_indxs_train, client_indxs_val, client_indxs_test = generate_iid_data(
                args, train_set, test_set, args.n_clients
            )
            clients[0] = Client(
                args,
                0,
                device,
                nets[0],
                train_set,
                test_set,
                client_indxs_train[0],
                client_indxs_val[0],
                client_indxs_test[0],
                dict_label_dist[0],
                cluster=0,
                opt_fun=opt_fun,
            )
        else:
            train_set_transformed, test_set_transformed, _ = load_dataset(
                args, transforms_dict[args.clustering]
            )

            train_sets = {0: train_set, 1: train_set_transformed}
            test_sets = {0: test_set, 1: test_set_transformed}
            client_indxs_train, client_indxs_val, client_indxs_test = generate_iid_data(
                args, train_set, test_set, args.n_clients
            )
            client_id = 0
            for cluster_idx in range(len(train_sets.keys())):
                for c in range(args.n_clients // 2):
                    clients[client_id] = Client(
                        args,
                        client_id,
                        device,
                        nets[client_id],
                        train_sets[cluster_idx],
                        test_sets[cluster_idx],
                        client_indxs_train[client_id],
                        client_indxs_val[client_id],
                        client_indxs_test[client_id],
                        dict_label_dist[client_id],
                        cluster=cluster_idx,
                        opt_fun=opt_fun,
                    )
                    client_id += 1
    elif args.clustering == "False":
        # OBS ej stratified men dirichlet med hög alfa är som iid och är stratified
        if args.data_sampling == "iid":
            client_indxs_train, client_indxs_val, client_indxs_test = generate_iid_data(
                args, train_set, test_set, args.n_clients
            )

        elif args.data_sampling == "mcf":
            client_indxs_train = 99
            while client_indxs_train == 99:
                (
                    client_indxs_train,
                    client_indxs_val,
                    client_indxs_test,
                    dict_label_dist,
                ) = generate_noniid_mcf_data(train_set, test_set,)

        elif args.data_sampling == "dirichlet":
            client_indxs_train = 99
            while client_indxs_train == 99:
                (
                    client_indxs_train,
                    client_indxs_val,
                    client_indxs_test,
                    dict_label_dist,
                ) = generate_dirichlet_data(args, train_set, test_set)
        else:
            print("No such data sampling exists")
            exit()

        for c in range(args.n_clients):
            clients[c] = Client(
                args,
                c,
                device,
                nets[c],
                train_set,
                test_set,
                client_indxs_train[c],
                client_indxs_val[c],
                client_indxs_test[c],
                dict_label_dist[c],
                opt_fun=opt_fun,
            )

    return clients