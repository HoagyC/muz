import os
import time

import numpy as np
import ray
import torch

from mcts import search, MinMax
from utils import convert_to_int, convert_from_int


@ray.remote
class Reanalyser:
    def __init__(self, config, log_dir, device=torch.device("cpu")):
        self.device = device
        self.config = config
        self.log_dir = log_dir

    def reanalyse(self, mu_net, memory, buffer):
        while not ray.get(memory.is_finished.remote()):
            if "latest_model_dict.pt" in os.listdir(self.log_dir):
                mu_net = ray.get(memory.load_model.remote(self.log_dir, mu_net))
                mu_net.to(device=self.device)

            # No point reanalysing until there are multiple games in the history
            while True:
                buffer_len = ray.get(buffer.get_buffer_len.remote())
                train_stats = ray.get(memory.get_data.remote())
                current_game = train_stats["games"]
                if buffer_len >= 1 and current_game >= 2:
                    break

                time.sleep(1)

            mu_net.train()
            mu_net = mu_net.to(self.device)

            p = ray.get(buffer.get_reanalyse_probabilities.remote())

            if len(p) > 0:
                ndxs = ray.get(buffer.get_buffer_ndxs.remote())
                try:
                    ndx = np.random.choice(ndxs, p=p)
                except ValueError:
                    print(p, ndxs)
                game_rec = ray.get(buffer.get_buffer_ndx.remote(ndx))
                minmax = ray.get(memory.get_minmax.remote())

                vals = []
                search_stats = []

                for i in range(len(game_rec.observations) - 1):
                    if self.config["obs_type"] == "image":
                        obs = game_rec.get_last_n(pos=i)
                    else:
                        obs = convert_from_int(
                            game_rec.observations[i], self.config["obs_type"]
                        )

                    new_root = search(
                        config=self.config,
                        mu_net=mu_net,
                        current_frame=obs,
                        minmax=minmax,
                        log_dir=self.log_dir,
                        device=torch.device("cpu"),
                    )
                    vals.append(new_root.average_val)
                    search_stats.append(
                        [c.num_visits if c else 0 for c in new_root.children]
                    )

                buffer.update_game_info.remote(
                    ndx=ndx, vals=vals, search_stats=search_stats
                )
                buffer.add_priorities.remote(ndx=ndx, reanalysing=True)
                print(f"Reanalysed game {ndx}")
            else:
                time.sleep(5)
