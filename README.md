# Overview

A framework to benchmark common deep-learning frameworks. Easy to use yet fully customizable.

# Usage

## 1. Set up your config file

Create a config file to specify running details. A sample config file is already provided under the root path of the project:`example_config.csv`

It's a csv-format file. You can copy this file and change the content at your will. 

```Csv
framework,network_type,network_name,device_id,device_count,cpu_count,batch_size,number_of_epochs,epoch_size,learning_rate,synthetic,enabled
tensorflow,cnn,alexnet,0,1,0,64,40,50000,0.01,0,1
... <more rows>
```

Explanation of csv head row fields:

* __framework__ 

  Deep learning framework used to perform benchamark. Only support Tensorflow so far.

* __network_type__ 

  Deep learning network type. Support `cnn`/`rnn`/`fc` corresponding to CNN/RNN/FCN network type.

* __network_name__ 

  Deep learning network name. Support: 

  1) cnn: `alexnet`, `resnet` 

  2) rnn: `lstm` 

  3) fcn: `fcn5`

* __device_id__

  Specify which GPU(s) to use. Use semicolon as a seperator if more than one GPU is used.

* __device_count__

  number of GPU(s) used.

* __cpu_count__

  number of CPU core used.

  > Attention: __cpu_count__ with a value __0__ means using all CPU cores.

* __batch_size__

  The number of training examples in one forward/backward pass.

* __number_of_epochs__

  One epoch means one forward pass and one backward pass of all the training examples.

* __epoch_size__

  Size of the trainning set.

* __learning_rate__

  Learning rate in deep learning.

* __synthetic__

  Whether use synthetic data or not. __1__ for true, use __0__ otherwise. 

* __enabled__

  Toggle a config row. __0__ will not be executed.

## 2. Start the benchmark

Simply run the following command to start benchmarking:

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