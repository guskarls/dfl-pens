import numpy as np

import experiment_runner
from train_utils import *
from main import *
from shutil import copyfile
from shutil import copytree
from time import sleep, time
import datetime

# Parse args
args = parseargs()
if args.start_date == None:
    print("Define start_date")
    exit()

er = experiment_runner.ExperimentRunner(copy.copy(args))

# Copy source code into experiment folder
sleep(5*args.pid) # sleep 5 seconds between each process to give the CPU some slack
if not pathlib.Path(
    f"{pathlib.Path(__file__).parents[1].absolute()}/data/output/{er.args.start_date}/src"
).exists():
    copytree(
        f"{pathlib.Path(__file__).parents[1].absolute()}/src",
        f"{pathlib.Path(__file__).parents[1].absolute()}/data/output/{er.args.start_date}/src",
    )

# Test hyperparameters
new_results, clients_array_over_repetitions = er.run_one_combo_of_hyperparams()
for rep in range(er.args.n_repetitions):
    clients = clients_array_over_repetitions[rep]
    for c in range(er.args.n_clients):
        results_new_row = [
            er.args.federation,
            er.args.n_clients,
            er.args.n_clients_to_consider,
            er.args.neighbour_selection,
            er.args.neighbour_exploration,
            er.neighbour_exploration_parameter,
            er.args.edu,
            er.args.n_neighbours,
            er.args.clustering,
            round(er.args.n_train_and_val*(1-er.args.val_proportion)),
            er.args.initial_local_epochs,
            er.args.lr,
            c,
            clients[c].cluster,
            er.args.reset_models_before_final_training,
            new_results["test_accs_own_data"][rep, c],
            # new_results["train_accs_other_data"][rep, c],
            clients[c].neighbours_history,
            clients[c].neighbours_history2,
            new_results["epochs"][rep, c],
            rep,
            er.args.pid,
        ]
        er.save_results(results_new_row)
print("Finished")