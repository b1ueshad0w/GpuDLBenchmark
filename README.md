# Overview

A framework to benchmark common deep-learning frameworks. Easy to use yet fully customizable.

# Usage

Create a config file to specify running details. A sample config is already provided:`example_config.csv`

Symply run the following command to start benchmarking:

```
python -config your_config.csv -log_dir DirPathToSaveLogs -test_summary_file FilePathToSaveReport
```

As the name says,

- __config__ filepath of your config csv file
- __log_dir__ directory to save intermediate logs
- __test_summary_file__ filepath to save the benchmark results (in `csv` format)



# Prerequisites

You should have the corresponding deep learning frameworks installed (e.g. TensorFlow, Coffee, etc.). 

Also all cuda-related things (GPU driver, cuDNN library, etc.) is already set up.

Finally specify the CIFAR10 and MNIST dataset path in `globalconfig.py`.

> You should create an image or use docker to simplify your setup work across different test machines.