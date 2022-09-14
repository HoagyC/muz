import importlib


def make_env(config):
    env_name = config["obs_name"]

    try:
        env_mod = importlib.import_module("envs." + env_name + "_env")
    except NameError:
        print(f"No env named {env_name} available")

    return env_mod.make_env(config)
