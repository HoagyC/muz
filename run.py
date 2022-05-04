import copy
import os
import pickle
import sys
import yaml

import main


def run_config(config, path, n_runs=10):
    os.makedirs(path, exist_ok=True)

    with open(os.path.join(path, "config.yaml"), "w") as f:
        f.write(yaml.dump(config))

    for i in range(n_runs):
        stats = main.run(copy.copy(init_config))

        with open(os.path.join(path, f"run{i}.txt"), "w") as f:
            pickle.dump(stats, f)


config_path = os.path.join("configs", "config-" + sys.argv[1] + ".yaml")
init_config = yaml.safe_load(open(config_path, "r"))

binary_switches = [
    "priority_replay",
    "reanalyse",
    "off_policy_correction",
    "consistency_loss",
    "value_prefix",
]

for run_ndx in range(2 ** len(binary_switches)):
    config = copy.copy(init_config)
    ndx_bin = bin(run_ndx)[2:].zfill(len(binary_switches))
    bool_switches = list(ndx_bin)
    config_update_dict = dict(zip(binary_switches, bool_switches))
    config.update(config_update_dict)

    run_name = "_".join(
        [binary_switches[i] for i in range(len(binary_switches)) if bool_switches[i]]
    )
    path = os.path.join("comps", run_name)
    run_config(config=config, path=path)
