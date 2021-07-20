import datetime
import pandas as pd

from main import *
from train_utils import set_gpu_and_device


class ExperimentRunner:
    def __init__(self, args):
        self.args = args
        _, self.device = set_gpu_and_device(args.gpu)
        self.experiment_folder = f"data/output/{self.args.start_date}"

        # Create json folder
        pathlib.Path(f"{self.experiment_folder}/json").mkdir(
            parents=True, exist_ok=True
        )

        if self.args.neighbour_exploration == "epsilon_greedy":
            self.neighbour_exploration_parameter_name = "neighbour_epsilon"
            self.neighbour_exploration_parameter = round(self.args.neighbour_epsilon, 2)
        elif self.args.neighbour_exploration == "topk":
            self.neighbour_exploration_parameter_name = "topk"
            self.neighbour_exploration_parameter = self.args.topk
        else:
            self.neighbour_exploration_parameter_name = (
                "neighbour_selection_parameter_value"
            )
            self.neighbour_exploration_parameter = 99
        # else:
        #     print("Handle this type of neighbour exploration")
        #     exit()

    @property
    def args(self):
        return self._args

    @args.setter
    def args(self, args):
        # Error handling
        if args.neighbour_selection == "performance_based":
            if args.n_clients_to_consider == None:
                print("n_clients_to_consider needs to be defined.")
                exit()
            elif args.n_clients_to_consider > args.n_clients - 1:
                print(
                    f"n_clients_to_consider ({args.n_clients_to_consider}) needs to be less than or equal to n_clients-1 ({args.n_clients-1})."
                )
                exit()
            elif args.neighbour_exploration == None:
                print("neighbour_exploration not defined")
                exit()
            elif args.neighbour_selection_metric == None:
                print("neighbour_selection_metric not defined")
                exit()
        self._args = args

    def run_one_combo_of_hyperparams(self):
        results = dict()
        results["test_accs_own_data"] = np.zeros(
            (self.args.n_repetitions, self.args.n_clients)
        )
        # results["train_accs_other_data"] = np.empty(
        #     (self.args.n_repetitions, self.args.n_clients), dtype=list
        # )
        results["epochs"] = np.zeros((self.args.n_repetitions, self.args.n_clients))
        clients_array_over_repetitions = np.empty(self.args.n_repetitions, dtype=object)
        for rep in range(self.args.n_repetitions):
            # Initialize clients
            if self.args.dataset == "CIFAR-10":
                net_class = TensorFlowCIFAR10Net
            elif self.args.dataset == "Fashion-MNIST":
                net_class = FashionMNISTNet
            else:
                print(f"Unknown dataset {self.args.dataset}")
            clients = create_clients(self.args, net_class, self.device)

            # Set plot title
            self.args.plot_title = f"{self.args.federation}, {self.args.neighbour_selection}, n_clients={self.args.n_clients}, n_neighbours={self.args.n_neighbours}, neighbour_exploration={self.args.neighbour_exploration}, {self.neighbour_exploration_parameter_name}={self.neighbour_exploration_parameter}, {self.args.clustering}, local_epochs={self.args.local_epochs}, reset={self.args.reset_models_before_final_training}, lr={self.args.lr}, run={rep}, pid={self.args.pid}"

            # Train models
            if self.args.edu:
                clients, self.args = train_edu(self.args, clients)
            elif self.args.evaluate_ideal:
                self.args = evaluate_ideal(self.args, clients)
            else:
                train(self.args, clients)
            clients_array_over_repetitions[rep] = clients

            # Load checkpoints
            checkpoints = [
                torch.load(
                    f"data/output/{self.args.start_date}/checkpoints/checkpoint{self.args.pid}/{self.args.plot_title}/client_{c}.tar"
                )
                for c in range(self.args.n_clients)
            ]

            # Test models
            for c, checkpoint in enumerate(checkpoints):
                model = to_device(net_class(), self.device)
                model.load_state_dict(checkpoint["model_state_dict"])
                _, results["test_accs_own_data"][rep, c] = evaluate(
                    model, clients[c].test_loader
                )

                # # Evaluate on all clients' train set
                # l = self.args.n_clients * [0]
                # for c_ in range(self.args.n_clients):
                #     _, l[c_] = evaluate(model, clients[c_].train_loader)
                # results["train_accs_other_data"][rep, c] = l
                results["epochs"][rep, c] = checkpoint["communication_round"]
        return results, clients_array_over_repetitions

    def update_results_df_and_json(self, results_new_row, path_to_json):
        try:
            results = pd.read_json(path_to_json)  # read json
        except Exception:
            results = pd.DataFrame(
                columns=[
                    "federation",
                    "n_clients",
                    "n_clients_to_consider",
                    "neighbour_selection",
                    "neighbour_exploration",
                    self.neighbour_exploration_parameter_name,
                    "edu",
                    "n_neighbours",
                    "clustering",
                    "n_train",
                    "initial_local_epochs",
                    "lr",
                    "client_id",
                    "cluster_id",
                    "reset_models_before_final_training",
                    "test_accs_own_data",
                    # "train_accs_other_data",
                    "neighbours_history",
                    "neighbours_history2",
                    "epochs",
                    "repetition",
                    "pid",
                ]
            )

        if str(results.index.max()) == "nan":
            results.loc[0] = results_new_row
        else:
            results = pd.read_json(path_to_json)  # read json
            results.loc[
                results.index.max() + 1
            ] = results_new_row  # append new row to df
        results.to_json(path_to_json)

        return results

    def save_results(self, results_new_row):
        # Save to unique process file
        process_results = self.update_results_df_and_json(
            results_new_row,
            f"{self.experiment_folder}/json/results{self.args.pid}.json",
        )