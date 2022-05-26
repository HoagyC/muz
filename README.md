### Running the repository

The repository contains an implementatino
I use this github to push changes to a Colab Notebook so may contain temporary changes and bugs.

To test it on, for example, Atari Breakout, set your configuration in configs/config-breakout.yaml and run the following commands in terminal.

```
git clone https://github.com/hoagyc/muz.git
cd muz
python -m virtualenv venv
source venv/bin/activate
pip install -r requirements.txt
python main.py breakout
```

The algorithm when run on a CPU is heavily constrained by quantity of training, rather than the amount of game data generated. To exclusively train without generating new runs, set the `train_only` flag to `True` in the appropriate config file.

The settings are optimized for a CPU without any GPU support, but if you add `colab` to the end of the command, as in `python main.py colab`, if will switch to more appropriate settings for a computer with a GPU.