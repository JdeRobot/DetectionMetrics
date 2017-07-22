# DeepLearningSuite
DeepLearning Suite is a set of tool that simplify the evaluation of most common object detection datasets with several object detection neural networks.

The idea is to offer a generic infrastructure to evaluates object detection algorithms againts a dataset and compute most common statistics:
* Intersecion Over Union
* Precision
* Recall 



#####Supported datasets formats:
* YOLO

##### Supported object detection frameworks/algorithms
* YOLO (darknet)
* Background substraction



# Sample generation Tool
Sample Generation Tool has been developed in order to simply the process of generation samples for datasets focused on object detection. The tools provides some features to reduce the time on labeling objects as rectangles. 


# Requirements
In you want to use darknet with yolo detection we will need to use a modified implementation of darknet just with a c++ api. You can find this implementation here [2]. We will try to merge with the original repository asap.


# References.
[1] YOLO: https://pjreddie.com/darknet/yolo/ \
[2] YOLO with c++ API: https://github.com/chanfr/darknet
