import numpy as np
import torch


def convert_to_int(obs, game_type):
    if game_type == "image":
        return (obs * 256).astype(np.ubyte)
    elif game_type == "cartpole":
        return ((obs * 64) + 128).astype(np.ubyte)
    else:
        raise ValueError("Not a recognised game type")


def convert_from_int(obs, game_type):
    if game_type == "image":
        return obs.astype(np.float32) / 256
    elif game_type == "cartpole":
        return (obs.astype(np.float32) - 128) / 64
