import os

import torch


def load_model(log_dir, model, device=torch.device("cpu")):
    try:
        path = os.path.join(log_dir, "latest_model_dict.pt")
        if os.path.exists(path):
            model.load_state_dict(torch.load(path, map_location=device))
        else:
            print(f"no dict to load at {path}")

        return model
    except:
        print(log_dir, model, device, path)
        breakpoint()
