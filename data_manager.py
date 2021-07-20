from torch.utils.data import Dataset, ConcatDataset
import torchvision
from torchvision import datasets, transforms
from torchvision.transforms import ToTensor
from collections import Counter
from pathlib import Path

# from client_manager import get_targets # Varför funkar ej detta!?
import math
import numpy as np


def get_targets(data):
    return [data.__getitem__(i)[1] for i in range(data.__len__())]


def load_dataset(
    args, transform, n_train_and_val=None, two_clusters_central_training=False,
):
    # torch.manual_seed(43)
    if args.dataset == "CIFAR-10":
        download = not Path("data/cifar-10-batches-py").is_dir()
        train_set = torchvision.datasets.CIFAR10(
            root="./data", train=True, download=download, transform=transform
        )
        test_set = torchvision.datasets.CIFAR10(
            root="./data", train=False, download=download, transform=transform
        )
    elif args.dataset == "Fashion-MNIST":
        download = not Path("data/FashionMNIST").is_dir()
        train_set = torchvision.datasets.FashionMNIST(
            root="./data", train=True, download=download, transform=transform
        )
        test_set = torchvision.datasets.FashionMNIST(
            root="./data", train=False, download=download, transform=transform
        )
    if n_train_and_val == None:
        n_train_and_val = len(train_set)
    if two_clusters_central_training:
        transform_standard = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )
        train_set_non_rotated = torchvision.datasets.CIFAR10(
            root="./data", train=True, download=download, transform=transform_standard
        )
        train_set = DatasetSplit(train_set, range(n_train_and_val // 2))
        train_set_non_rotated = DatasetSplit(
            train_set_non_rotated, range(n_train_and_val // 2, n_train_and_val)
        )
        train_set = ConcatDataset([train_set, train_set_non_rotated])

        test_set_non_rotated = torchvision.datasets.CIFAR10(
            root="./data", train=False, download=download, transform=transform_standard
        )
        test_set = DatasetSplit(test_set, range(len(test_set) // 2))
        test_set_non_rotated = DatasetSplit(
            test_set_non_rotated, range(len(test_set) // 2, len(test_set))
        )
        test_set = ConcatDataset([test_set, test_set_non_rotated])
    else:
        train_set = DatasetSplit(train_set, range(n_train_and_val))

    classes = (
        "plane",
        "car",
        "bird",
        "cat",
        "deer",
        "dog",
        "frog",
        "horse",
        "ship",
        "truck",
    )
    return train_set, test_set, classes


class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label


class DeviceDataLoader:
    """Wrap a dataloader to move data to a device"""

    def __init__(self, dl, device):
        self.dl = dl
        self.device = device

    def __iter__(self):
        """Yield a batch of data after moving it to device"""
        for b in self.dl:
            yield to_device(b, self.device)

    def __len__(self):
        """Number of batches"""
        return len(self.dl)


def to_device(data, device):
    """Move tensor(s) to chosen device"""
    if isinstance(data, (list, tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)


def split_data_label(dataset, dataset_test, n):
    idxs = np.arange(len(dataset), dtype=int)
    labels = np.array(get_targets(dataset))
    unique_labels = np.unique(labels)
    num_classes = len(unique_labels)

    idxs_test = np.arange(len(dataset_test), dtype=int)
    labels_test = np.array(get_targets(dataset_test))

    labels_cluster = {}
    for i in range(n):
        labels_cluster[i] = np.random.choice(
            unique_labels, int(num_classes / n), replace=False
        )
        unique_labels = list(set(unique_labels) - set(labels_cluster[i]))

    train_sets = {}
    test_sets = {}

    for i in range(n):
        train_idxs = np.array([], dtype="int64")
        test_idxs = np.array([], dtype="int64")
        for label in labels_cluster[i]:
            train_idxs_ = idxs[label == labels[idxs]]
            train_idxs = np.concatenate((train_idxs, train_idxs_))
            test_idxs_ = idxs_test[label == labels_test[idxs_test]]
            test_idxs = np.concatenate((test_idxs, test_idxs_))

        train_sets[i] = DatasetSplit(dataset, train_idxs)
        test_sets[i] = DatasetSplit(dataset_test, test_idxs)

        # print(np.unique(labels[train_idxs], return_counts=True))
        # print(np.unique(labels_test[test_idxs], return_counts=True))

    return train_sets, test_sets


def generate_iid_data(args, dataset, dataset_test, n_clients):
    """
    Sample I.I.D. client data from CIFAR10 dataset
    returns dict of image index where the key is the client id
    """
    dict_clients = {}
    dict_clients_val = {}
    all_idxs = [i for i in range(len(dataset))]
    if n_clients * args.n_train_and_val > len(dataset):
        print(f"{n_clients} * {args.n_train_and_val} > {len(dataset)}")
        print("Not enough data")
        exit()
    for i in range(n_clients):
        dict_clients[i] = np.random.choice(
            all_idxs,
            round(args.n_train_and_val * (1 - args.val_proportion)),
            replace=False,
        )
        all_idxs = list(set(all_idxs) - set(dict_clients[i]))
        dict_clients_val[i] = np.random.choice(
            all_idxs, round(args.n_train_and_val * args.val_proportion), replace=False
        )
        all_idxs = list(set(all_idxs) - set(dict_clients_val[i]))

    dict_clients_test = {}
    all_idxs_test = [i for i in range(len(dataset_test))]
    if n_clients * args.n_test > len(dataset_test):
        print(f"{n_clients} * {args.n_test} > {len(dataset_test)}")
        print("Not enough data")
        exit()
    for i in range(n_clients):
        dict_clients_test[i] = np.random.choice(
            all_idxs_test, int(args.n_test), replace=False
        )
        all_idxs_test = list(set(all_idxs_test) - set(dict_clients_test[i]))

    return dict_clients, dict_clients_val, dict_clients_test


def generate_noniid_mcf_data(dataset, dataset_test):
    """
    Sample non-I.I.D. client data with n majority classes from CIFAR10 dataset
    args.majority_fraction is the majority class fraction
    returns dict of image index where the key is the client id
    """
    idxs = np.arange(len(dataset), dtype=int)
    labels = np.array(get_targets(dataset))
    # labels = np.array(dataset.targets)
    unique_labels = np.unique(labels)
    num_classes = len(unique_labels)

    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    # print(idxs_labels)
    idxs = idxs_labels[0, :]
    idxs = idxs.astype(int)

    dict_users = {i: np.array([], dtype="int64") for i in range(args.n_clients)}
    dict_users_val = {i: np.array([], dtype="int64") for i in range(args.n_clients)}
    user_majority_labels = []

    # Test data
    idxs_test = np.arange(len(dataset_test), dtype=int)
    labels_test = np.array(get_targets(dataset_test))
    unique_labels_test = np.unique(labels_test)

    # sort labels
    idxs_labels_test = np.vstack((idxs_test, labels_test))
    idxs_labels_test = idxs_labels_test[:, idxs_labels_test[1, :].argsort()]
    idxs_test = idxs_labels_test[0, :]
    idxs_test = idxs_test.astype(int)

    dict_users_test = {i: np.array([], dtype="int64") for i in range(args.n_clients)}

    label_list = np.tile(
        unique_labels, math.ceil(args.n_clients / num_classes) * args.n_majority_classes
    )

    for i in range(args.n_clients):
        if len(np.unique(label_list)) < args.n_majority_classes:
            return 99, 99, 99, 99
        majority_labels = np.random.choice(
            np.unique(label_list), args.n_majority_classes, replace=False
        )

        for label in majority_labels:
            label_list = np.delete(label_list, np.where(label == label_list)[0][0])
        # print('choosen labels: ', majority_labels)
        # print('labels left to choose from: ', label_list)
        # print('unique labels to pick from for next user: ', np.unique(label_list))
        # print('-------------------------------------- \n')

        # print(np.array(majority_labels))

        user_majority_labels.append(majority_labels)

        for j in range(args.n_majority_classes):
            majority_labels_idxs = idxs[majority_labels[j] == labels[idxs]]

            sub_data_idxs = np.random.choice(
                majority_labels_idxs,
                round(
                    args.majority_fraction
                    * args.n_train_and_val
                    * (1 - args.val_proportion)
                    / args.n_majority_classes
                ),
                replace=False,
            )

            dict_users[i] = np.concatenate((dict_users[i], sub_data_idxs))
            idxs = np.array(list(set(idxs) - set(sub_data_idxs)))

            # Validation data
            majority_labels_idxs = idxs[majority_labels[j] == labels[idxs]]
            sub_data_idxs_val = np.random.choice(
                majority_labels_idxs,
                round(
                    args.majority_fraction
                    * args.n_train_and_val
                    * args.val_proportion
                    / args.n_majority_classes
                ),
                replace=False,
            )
            dict_users_val[i] = np.concatenate((dict_users_val[i], sub_data_idxs_val))
            idxs = np.array(list(set(idxs) - set(sub_data_idxs_val)))

            # Test data
            majority_labels_idxs_test = idxs_test[
                majority_labels[j] == labels_test[idxs_test]
            ]
            sub_data_idxs_test = np.random.choice(
                majority_labels_idxs_test,
                int(args.majority_fraction * args.n_test / args.n_majority_classes),
                replace=False,
            )
            dict_users_test[i] = np.concatenate(
                (dict_users_test[i], sub_data_idxs_test)
            )

    if args.majority_fraction < 1.0:
        for i in range(args.n_clients):
            if len(idxs) >= args.n_train_and_val * (1 - args.majority_fraction):
                majority_labels = user_majority_labels[i]

                non_majority_labels_idxs = idxs
                for j in range(args.n_majority_classes):
                    non_majority_labels_idxs2 = idxs[
                        (majority_labels[j] != labels[idxs])
                    ]
                    non_majority_labels_idxs = np.array(
                        list(
                            set(non_majority_labels_idxs).intersection(
                                set(non_majority_labels_idxs2)
                            )
                        )
                    )
                try:
                    sub_data_idxs1 = np.random.choice(
                        non_majority_labels_idxs,
                        round(
                            (1 - args.majority_fraction)
                            * args.n_train_and_val
                            * (1 - args.val_proportion)
                        ),
                        replace=False,
                    )
                except:
                    return 99, 99, 99, 99
                dict_users[i] = np.concatenate((dict_users[i], sub_data_idxs1))
                idxs = np.array(list(set(idxs) - set(sub_data_idxs1)))

                # Validation data
                non_majority_labels_idxs = idxs
                for j in range(args.n_majority_classes):
                    non_majority_labels_idxs2 = idxs[
                        (majority_labels[j] != labels[idxs])
                    ]
                    non_majority_labels_idxs = np.array(
                        list(
                            set(non_majority_labels_idxs).intersection(
                                set(non_majority_labels_idxs2)
                            )
                        )
                    )

                sub_data_idxs1_val = np.random.choice(
                    non_majority_labels_idxs,
                    round(
                        (1 - args.majority_fraction)
                        * args.n_train_and_val
                        * args.val_proportion
                    ),
                    replace=False,
                )
                dict_users_val[i] = np.concatenate(
                    (dict_users_val[i], sub_data_idxs1_val)
                )
                idxs = np.array(list(set(idxs) - set(sub_data_idxs1_val)))

                # Test data
                non_majority_labels_idxs_test = idxs_test
                for j in range(args.n_majority_classes):
                    non_majority_labels_idxs2_test = idxs_test[
                        (majority_labels[j] != labels_test[idxs_test])
                    ]
                    non_majority_labels_idxs_test = np.array(
                        list(
                            set(non_majority_labels_idxs_test).intersection(
                                set(non_majority_labels_idxs2_test)
                            )
                        )
                    )

                sub_data_idxs1_test = np.random.choice(
                    non_majority_labels_idxs_test,
                    int((1 - args.majority_fraction) * args.n_test),
                    replace=False,
                )
                dict_users_test[i] = np.concatenate(
                    (dict_users_test[i], sub_data_idxs1_test)
                )
                # idxs_test = np.array(list(set(idxs_test) - set(sub_data_idxs1_test))) No need
            else:
                print("not enough data")
                dict_users[i] = np.concatenate((dict_users[i], idxs))
                dict_users_test[i] = np.concatenate((dict_users_test[i], idxs_test))

    # Find label distribution for each client
    dict_label_dist = {}
    for c in range(args.n_clients):
        new_counts = np.zeros(num_classes, dtype=int)
        label_count = np.unique(np.array(labels)[dict_users[c]], return_counts=True)
        for i, label in enumerate(label_count[0]):
            new_counts[label] = label_count[1][i]
        dict_label_dist[c] = new_counts
        # print(dict_label_dist[i])

    return dict_users, dict_users_val, dict_users_test, dict_label_dist


def generate_dirichlet_data(args, dataset, dataset_test):
    """
    Sample non-I.I.D. client data with dirichlet distribution from CIFAR10 dataset
    returns dict of image index where the key is the client id
    """
    idxs = np.arange(len(dataset), dtype=int)
    labels = np.array(get_targets(dataset))
    unique_labels = np.unique(labels)
    num_classes = len(unique_labels)

    dict_users = {i: np.array([], dtype="int64") for i in range(args.n_clients)}
    dict_users_val = {i: np.array([], dtype="int64") for i in range(args.n_clients)}

    # Test data
    idxs_test = np.arange(len(dataset_test), dtype=int)
    labels_test = np.array(get_targets(dataset_test))
    unique_labels_test = np.unique(labels_test)

    dict_users_test = {i: np.array([], dtype="int64") for i in range(args.n_clients)}
    dict_label_dist = {}

    for c in range(args.n_clients):
        label_distr = np.random.dirichlet(
            args.alpha * np.ones(num_classes)
        )  # number of samples of each class
        label_distr_train = np.random.multinomial(
            round(args.n_train_and_val * (1 - args.val_proportion)), label_distr
        )
        label_distr_val = np.random.multinomial(
            round(args.n_train_and_val * args.val_proportion), label_distr
        )
        label_distr_test = np.random.multinomial(args.n_test, label_distr)
        dict_label_dist[c] = label_distr_train

        """
        label_distr = np.random.multinomial(args.n_train_and_val * (1 - args.val_proportion), label_distr)
        dict_label_dist[c] = label_distr
        # print('label dist: ', dict_label_dist[c])
        label_distr_val = label_distr / sum(label_distr) * args.n_train_and_val * args.val_proportion
        label_distr_val = [int(x) for x in label_distr_val]
        label_distr_test = label_distr / sum(label_distr) * args.n_test
        label_distr_test = [int(x) for x in label_distr_test]
        """

        for i in range(num_classes):
            try:
                # if label_distr_train[i] > 0:
                sub_idx = np.random.choice(
                    idxs[labels[idxs] == i], label_distr_train[i], replace=False
                )  # sample class i
                dict_users[c] = np.concatenate((dict_users[c], sub_idx))
                idxs = np.array(list(set(idxs) - set(sub_idx)))

                sub_idx_val = np.random.choice(
                    idxs[labels[idxs] == i], label_distr_val[i], replace=False
                )
                dict_users_val[c] = np.concatenate((dict_users_val[c], sub_idx_val))
                idxs = np.array(list(set(idxs) - set(sub_idx_val)))

                sub_idx_test = np.random.choice(
                    idxs_test[labels_test[idxs_test] == i],
                    label_distr_test[i],
                    replace=False,
                )
                dict_users_test[c] = np.concatenate((dict_users_test[c], sub_idx_test))
                # idxs_test = np.array(list(set(idxs_test) - set(sub_idx_test)))
            except:
                return 99, 99, 99, 99

    return dict_users, dict_users_val, dict_users_test, dict_label_dist