import copy
import os
import pickle
import ray
import sys
import yaml

import main


def run_config(config, path, n_runs=10, start_run=0):
    os.makedirs(path, exist_ok=True)

    with open(os.path.join(path, "config.yaml"), "w") as f:
        f.write(yaml.dump(config))

    for i in range(start_run, n_runs):
        print(f"Starting run {i + 1} of {n_runs} with config {path}")
        stats = main.run(copy.copy(config))
        print("Results:", stats)

        with open(os.path.join(path, f"run_{i}_stats.pkl"), "wb") as f:
            pickle.dump(stats, f)


if __name__ == "__main__":
    config_path = os.path.join("configs", "config-" + sys.argv[1] + ".yaml")
    init_config = yaml.safe_load(open(config_path, "r"))

    binary_switches = [
        "priority_replay",
        "reanalyse",
        "off_policy_correction",
        "consistency_loss",
        "value_prefix",
    ]

    start_config, start_run = 0, 0

    for config_ndx in range(start_config, 2 ** len(binary_switches)):
        config = copy.copy(init_config)
        ndx_bin = bin(config_ndx)[2:].zfill(len(binary_switches))
        bool_switches = [bool(int(i)) for i in list(ndx_bin)]
        config_update_dict = dict(zip(binary_switches, bool_switches))
        print(config_update_dict)
        config.update(config_update_dict)

        sub_names = [sys.argv[1]] + [
            binary_switches[i][:3]
            for i in range(len(binary_switches))
            if bool_switches[i]
        ]
        run_name = "_".join(sub_names)
        print(run_name)
        if len(sys.argv) > 2 and sys.argv[2] == "colab":
            path = os.path.join("/content/gdrive/My Drive/muz/comp", run_name)
        else:
            path = os.path.join("testcomp", run_name)
        begin_run = start_run if config_ndx == start_config else 0
        run_config(config=config, path=path, n_runs=10, start_run=begin_run)
