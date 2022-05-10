### Hello

The repository contains a 

To test it on breakout, 
```
pip install -r /path/to/requirements.txt
python breakjout
```

The algorithm when run on a CPU is heavily constrained by quantity of training, rather than the amount of game data generated. To exclusively train without generating new runs, set the `train_only` flag to `True` in the appropriate config file.

The settings are optimized for a CPU without any GPU support, but if you add `colab` to the end of the command, as in `python main.py colab`, if will switch to more appropriate values for a 