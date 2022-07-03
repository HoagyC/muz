import os
import pickle

import numpy as np

from matplotlib import pyplot as plt

MAX_FRAMES = 10_000


def process_runs(data_dict):
    n_runs = len(data_dict.keys())
    average_scores = np.zeros(MAX_FRAMES)
    last_scores = {key: 0 for key in data_dict.keys()}
    run_frames = {}

    for key, run in data_dict.items():
        run_frames[key] = {x["total frames"]: x["score"] for x in run}

    for frame in range(MAX_FRAMES):
        for key in data_dict.keys():
            if frame in run_frames[key]:
                last_scores[key] = run_frames[key][frame]

        average_scores[frame] = np.mean(list(last_scores.values()))

    return average_scores


def plot_switch_comparisons(data_dict):
    config_switches = ["con", "val", "rea", "off"]

    switch_dict = {
        "con": "consistency loss",
        "val": "value prefix",
        "rea": "reanalyser",
        "off": "off-policy correction",
    }

    n_switches = len(config_switches)
    cols = (n_switches + 1) // 2

    fig, axs = plt.subplots(2, cols, figsize=(12, 10))

    for i, switch in enumerate(config_switches):
        with_switch_dict = {
            key: value for key, value in data_dict.items() if switch in key
        }
        average_scores = process_runs(with_switch_dict)

        without_switch_dict = {
            key: value for key, value in data_dict.items() if switch not in key
        }
        average_scores_without = process_runs(without_switch_dict)

        c, r = divmod(i, 2)
        ax = axs[r, c]

        ax.plot(average_scores, label="with switch")
        ax.plot(average_scores_without, label="without switch")
        ax.legend()
        ax.set_xlabel("Number of elapsed frames")
        ax.set_ylabel("Average score across runs")
        ax.set_title(f"Comparison with and without {switch_dict[switch]}")
    plt.show()


def plot_all(data_dict, config_names):
    config_names.sort(key=lambda x: len(x))
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))

    for i, config_name in enumerate(config_names):
        config_dict = {
            key: value for key, value in data_dict.items() if config_name in key
        }

        width = 2 if i in [0, len(config_names) - 1] else 1

        average_scores = process_runs(config_dict)
        ax.plot(average_scores, label=config_name, linewidth=width)

    ax.legend()
    ax.set_xlabel("Number of elapsed frames")
    ax.set_ylabel("Average score across runs")
    ax.set_title("Comparison between all switch combinations")
    plt.show()


def get_data_dict():
    data_folder = "colab_comps"

    data_dict = {}
    config_names = []

    for config_name in os.listdir(data_folder):
        config_names.append(config_name)
        for file_name in os.listdir(os.path.join(data_folder, config_name)):
            if "config" in file_name:
                continue
            else:
                with open(os.path.join(data_folder, config_name, file_name), "rb") as f:
                    run_name = config_name + file_name[4]
                    data_dict[run_name] = pickle.load(f)

    return data_dict, config_names


def main():
    data_dict, config_names = get_data_dict()

    plot_switch_comparisons(data_dict)
    plot_all(data_dict, config_names)


if __name__ == "__main__":
    main()
