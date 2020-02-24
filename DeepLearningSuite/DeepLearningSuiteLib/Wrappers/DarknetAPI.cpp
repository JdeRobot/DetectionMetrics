#include "DarknetAPI.h"
#include <iostream>
#include "DarknetAPIConversions.h"


char *voc_names[] = {"aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "dinningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"};
image voc_labels[20];


DarknetAPI::DarknetAPI(char *cfgfile, char *weightfile) {
    net = c_parse_network_cfg(cfgfile);
    if (weightfile != nullptr) {
        c_load_weights(&net, weightfile);
    }
}

DarknetAPI::~DarknetAPI() {
    c_free_network(net);
}

void addDetection(image& im, int num, float threshold, box *boxes, float **probs, char **names, image *labels, int classes, DarknetDetections& detections) {
    int i;
    for(i = 0; i< num; i++) {
        int classid = c_max_index(probs[i], classes);
	float prob = probs[i][classid];
	if (prob > threshold) {
	    int width =pow(prob, 1./2.) * 10 + 1;
	    int offset = classid * 17 % classes;
	    box b = boxes[i];

	    int left = (b.x-b.w/2.) * im.w;
            int right = (b.x + b.w / 2.) * im.w;
	    int top = (b.y - b.h / 2.) * im.h;
	    int bottom = (b.y + b.h / 2.) * im.h;
			
	    if (left < 0) left = 0;
	    if (right > im.w - 1) right = im.w - 1;
	    if (top < 0) top = 0;
	    if (bottom > im.h - 1) bottom = im.h - 1;

	    box detectedBox;
	    detectedBox.x = left;
	    detectedBox.y = top;
	    detectedBox.h = bottom - top;
	    detectedBox.w = right - left;

	    DarknetDetection detection(detectedBox, classid, prob);
	    detections.push_back(detection);
        }
    }
}


DarknetDetections processImageDetection(network& net, image& im, float threshold) {
    float hier_threshold = 0.5;
    DarknetDetections detections;
    c_set_batch_network(&net, 1);
    srand(2222222);
    clock_t time;
    char buffer[256];
    char *input = buffer;
    int j;
    float nms = .3;

    image sized = c_letterbox_image(im, net.w, net.h);
    layer l = net.layers[net.n - 1];

    box *boxes = (box*) calloc(l.w * l.h * l.n, sizeof(box));
    float **probs = (float **) calloc(l.w * l.h * l.n, sizeof(float *));
    for ( j=0; j < l.w * l.h * l.n; ++j) probs[j] = (float *) calloc(l.classes + 1, sizeof(float *));
    float **masks = 0;
    if (l.coords > 4) {
        masks = (float**) calloc(l.w * l.h * l.n, sizeof(float*));
        for (j = 0; j < l.w * l.h * l.n; ++j) masks[j] = (float*) calloc(l.coords-4, sizeof(float *));
    }

    float *X = sized.data;
    time = clock();
    c_network_predict(net, X);
    printf("%s: Prodicted in %f second.\n", input, c_sec(clock()-time));
    c_get_region_boxes(l, im.w, im.h, net.w, net.h, threshold, probs, boxes, masks, 0, 0, hier_threshold, 1);
    if (l.softmax_tree && nms)
        c_do_nms_obj(boxes, probs, l.w * l.h * l.h, l.classes, nms);
    else if (nms)
        c_do_nms_sort(boxes, probs, l.w * l.h * l.n, l.classes, nms);

    addDetection(im, l.w * l.h * l.n, threshold, boxes, probs, voc_names, 0, l.classes, detections);
    c_free_image(sized);

    return detections;
}

DarknetDetections processImageDetection(network& net, const cv::Mat & im, float threshold) {
    image imageDarknet = cv_to_image(im);
    auto detection = processImageDetection(net, imageDarknet, threshold);
    c_free_image(imageDarknet);
    return detection;
}

DarknetDetections DarknetAPI::process(image& im, float threshold) {
    return processImageDetection(this->net, im, threshold);
}

DarknetDetections DarknetAPI::process(const cv::Mat &im, float threshold) {
    image imageDarknet = cv_to_image(im);
    return processImageDetection(this->net, imageDarknet, threshold);
}

std::string DarknetAPI::processToJson(const cv::Mat &im, float threshold) {
    return process(im, threshold).serialize();
}



