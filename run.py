import os
import sys
import yaml

import main


def run_config(config, path, n_runs=10):
    os.makedirs(path, exist_ok=True)

    with open(os.path.join(path, f"config.yaml"), "w") as f:
        f.write(yaml.dump(init_config))

    for i in range(n_runs):
        scores = main.run(init_config)

        with open(os.path.join(path, f"run{i}.txt"), "w") as f:
            f.write(" ".join([str(x) for x in scores]))


init_config = yaml.safe_load(open("config-" + sys.argv[1] + ".yaml", "r"))

config = init_config
config["priority_replay"] = False
run_config(config, os.path.join("comps", "no_replay"))

config = init_config
config["priority_replay"] = True
run_config(config, os.path.join("comps", "with_replay"))
