## Overview of Analysis tools

Use `analyze_threshold.py` to perform analysis of already trained models. This script can be used to generate and save statistics about the peformance. Then, the same script can be used to load previously generated data and produce plots. The behaviour of this script is goverened by passing flags/arguments at runtime. 

See `post-run-analysis.sh` for an example deploy script to run the analysis. This will produce data files (in csv format).
It is recommended to copy these csv files to your local machine and run `analyze_threshold.py` with the `--produce_plots` flag (and with correct parameters for the path to find results) to obtain a summary plot.

### Misc

A data class to handle metadata about training runs is defined in `training_run_class.py`.

`analysis_util.py` defines the main workload for the evaluation workflow, besides additional, smaller utility functions.

`output_activation.ipynb` aims to investigate how a network reacts to different scenarios by looking at the network output when faced with different syndromes.

In this folder, there is also the `quick_config.json` configuration file. This can be used when running `analyze_threshold.py` to define which models to load and analyze.