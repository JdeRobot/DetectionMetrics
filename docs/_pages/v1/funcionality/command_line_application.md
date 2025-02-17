---
layout: home
title: Command line application
permalink: /v1/functionality/command_line_application/


sidebar:
  nav: "main_v1"
---

Detection Metrics supports both a Qt based UI and some command line based applications both requiring a config file to run.
Some users might prefer using the command line tools which can give results in a single run without the need to use the Graphical User Interface.

The current supported command line applications are:

* [AutoEvaluator](#auto-evaluator)
* [Converter](/v1/functionality/converter/)
* [Evaluator](/v1/functionality/evaluator/)
* [Splitter](#splitter)
* [Detector](/v1/functionality/detector/)
* [Viewer](/v1/functionality/viewer/)

To access te different tools, navigate to ```build/Tools/``` and then enter the desired tool. One in the tool's directory
run

```
./[Tool name] -c config.yml
```

and the selected tool will be executed. On the configuration file, the exactly configuration to run the tool is needed, because no
GUI will appear.

<a name="auto-evaluator"></a>
## Auto Evaluator

One such significant tool is **Automatic Evaluator**, which can evaluate multiple networks on a single dataset or multiple datasets in a single run.

All you need is config file containing details about the dataset(s) and network(s).

The results are then written in CSV files in the output directory specified.

To run this tool simply build this repository and navigate to ```build/Tools/AutoEvaluator```
and run

```
    ./autoEvaluator -c config.yml
```

Here ```config.yml``` is your required config file and some examples to create them are detailed below.

### Creating Config File

Below is a sample config file to run **Automatic Evaluator** on COCO dataset for 2 inferencers.

```
Datasets:

-
    inputPath: /opt/datasets/coco/annotations/instances_train2014.json
    readerImplementation: COCO
    readerNames: /opt/datasets/names/coco.names


Inferencers:

-
    inferencerWeights:         /opt/datasets/weights/ssd_mobilenet_v2_coco_2018_03_29/frozen_inference_graph.pb
    inferencerConfig:          /opt/datasets/cfg/foo.cfg
    inferencerImplementation:  tensorflow
    inferencerNames:           /opt/datasets/names/coco.names

-
    inferencerWeights:         /opt/datasets/weights/ssd_inception_v2_coco_2017_11_17/frozen_inference_graph.pb
    inferencerConfig:          /opt/datasets/cfg/foo.cfg
    inferencerImplementation:  tensorflow
    inferencerNames:           /opt/datasets/names/coco.names


outputCSVPath: /opt/datasets/output
```


As you can see there are two networks being used for inferencing: ```SSD_MobileNet``` and ```SSD_Inception```. Therefore, Inferencers contain an array of size 2.


### Using Multiple Frameworks

Below is a sample file for inferencing using multiple frameworks:

```
Datasets:

-
    inputPath: /opt/datasets/coco/annotations/instances_train2014.json
    readerImplementation: COCO
    readerNames: /opt/datasets/names/coco.names


Inferencers:

-
    inferencerWeights:         /opt/datasets/weights/ssd_mobilenet_v2_coco_2018_03_29/frozen_inference_graph.pb
    inferencerConfig:          /opt/datasets/cfg/foo.cfg              # TensorFlow doesn't need any config file, hence any
    inferencerImplementation:  tensorflow                             # empty foo.cfg file
    inferencerNames:           /opt/datasets/names/coco.names

-
    inferencerWeights:         /opt/datasets/weights/VGG_VOC0712_SSD_512x512_iter_120000.h5
    inferencerConfig:          /opt/datasets/cfg/foo.cfg              # New version Keras also doesn't need any file, all the
    inferencerImplementation:  keras                                  # data is stored in the HDF5 file including model
    inferencerNames:           /opt/datasets/names/voc.names          # weights, configuration and optimizer state, hence we
                                                                    # are using an empty foo.cfg file
-
    inferencerWeights:         /opt/datasets/weights/VGG_VOC0712_SSD_512x512_iter_240000.h5
    inferencerConfig:          /opt/datasets/cfg/foo.cfg
    inferencerImplementation:  keras
    inferencerNames:           /opt/datasets/names/voc.names



outputCSVPath: /opt/datasets/output
```



**Note:** In the above example, you can see that a VOC trained network is being used to evaluate on COCO Ground Truth. This tool supports such evaluation by mapping Pascal VOC class names to COCO class names.
This mapping is very robust, it can also map synonyms and subclasses.


<a name="splitter"></a>
## Splitter

This tool takes a dataset and split it in two different parts (test set and train set). It needs a trainRatio that set the amount of data that goes into each set.
An example of config.yml file would be:

```
    inputPath: /opt/datasets/weights/annotations/instances_val2017.json
    readerImplementation: COCO
    writerImplementation: COCO
    outputPath: /opt/output/new-output-folder
    trainRatio: 0.8
    readerNames: /opt/datasets/names/coco.names
```

Having the config, it is executed as always:

```
    ./splitter -c config.yml
```

The results are written in a folder specified as `outputPath`.