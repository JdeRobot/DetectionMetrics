#include "darknet.h"
#include "DarknetAPI.h"
#include <iostream>
#include "DarknetAPIConversions.h"
#include <glog/logging.h>






//char *voc_names[] = {"aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "dinningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"};
//image voc_labels[20];


DarknetAPI::DarknetAPI(char *cfgfile, char *weightfile) {
    LOG(INFO) << "Darknet initialization 1" << '\n';
    LOG(INFO) << cfgfile << '\n';
    //net = c_parse_network_cfg(cfgfile);
    LOG(INFO) << "Darknet initialization 1.1" << '\n';
    if (weightfile != nullptr) {
	LOG(INFO) << "Darknet initialization 1.2" << '\n';
        //c_load_weights(&net, weightfile);
    }
    LOG(INFO) << "Darknet initialization 1.3" << '\n';

   // net = load_network(cfgfile, weightfile, 0);
    //set_batch_network(&net, 1);

    LOG(INFO) << "Darknet initialization 1.4" << '\n';
}

DarknetAPI::~DarknetAPI() {
    LOG(INFO) << "Darknet initialization 2" << '\n';
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
    LOG(INFO) << "process Image Detection 1-1" << '\n';
    float hier_threshold = 0.5;
    DarknetDetections detections;
    LOG(INFO) << "process Image Detection 1-2" << '\n';
    c_set_batch_network(&net, 1);
    LOG(INFO) << "process Image Detection 1-3" << '\n';
    srand(2222222);
    clock_t time;
    char buffer[256];
    char *input = buffer;
    int j;
    float nms = .3;


    // PRUEBA
    // str filename = "/home/docker/Projects/DetectionSuite/datasets/coco/oneval2014/COCO_val2014_000000397133.jpg"

   //c_test_detector("cfg/coco.data", "/home/docker/Projects/DetectionSuite/datasets/cfg/yolov3.cfg", "/home/docker/Projects/DetectionSuite/datasets/weights/yolov3.weights", "/home/docker/Projects/DetectionSuite/datasets/coco/oneval2014/COCO_val2014_000000397133.jpg", .5, .5, "", 0);

    network net2 = load_network("/home/docker/Projects/DetectionSuite/datasets/cfg/yolov3.cfg", "/home/docker/Projects/DetectionSuite/datasets/weights/yolov3.weights", 0);
    set_batch_network(&net2, 1);
    strncpy(input, "/home/docker/Projects/DetectionSuite/datasets/coco/oneval2014/COCO_val2014_000000397133.jpg", 256);
    image im2 = load_image_color(input,0,0);
    printf("INPUT -> %s \n", input);
    image sized2 = c_letterbox_image(im2, net2.w, net2.h);
    printf("IM -> %i %i %i %5.1f \n", im2.w, im2.h, im2.c, *im2.data);
        printf("SIZED -> %i %i %i %5.1f  \n", sized2.w, sized2.h, sized2.c, *sized2.data);
    printf("Width and height -> %i %i \n", net2.w, net2.h);
    layer l2 = net2.layers[net2.n - 1];
    float *X2 = sized2.data;
    printf("TODO CARGADO");
    c_network_predict(net2, X2);
    printf("TODO TERMINADO");
    printf("%s: Prodicted in 1 second.\n", input);
    // FIN PRUEBa



    LOG(INFO) << "process Image Detection 1-4" << '\n';
    image sized = c_letterbox_image(im, net.w, net.h);
    layer l = net.layers[net.n - 1];

    LOG(INFO) << "process Image Detection 1-5" << '\n';
   /* box *boxes = (box*) calloc(l.w * l.h * l.n, sizeof(box));
    float **probs = (float **) calloc(l.w * l.h * l.n, sizeof(float *));
    for ( j=0; j < l.w * l.h * l.n; ++j) probs[j] = (float *) calloc(l.classes + 1, sizeof(float *));
    float **masks = 0;
    if (l.coords > 4) {
        masks = (float**) calloc(l.w * l.h * l.n, sizeof(float*));
        for (j = 0; j < l.w * l.h * l.n; ++j) masks[j] = (float*) calloc(l.coords-4, sizeof(float *));
    }
*/

    LOG(INFO) << "process Image Detection 1-6" << '\n';
    float *X = sized.data;
    time = clock();
    LOG(INFO) << "pdkljsakldjsarocess Image Detection 1-7" << '\n';
    //LOG(INFO) <<  net << '\n';
    LOG(INFO) << "X -> " << *X << '\n';
    c_network_predict(net, X);
    LOG(INFO) << "process Image Detection 1-8" << '\n';
    printf("%s: Prodicted in %f second.\n", input, c_sec(clock()-time));
    LOG(INFO) << im.w << '\n';
    LOG(INFO) << im.h << '\n';
    //LOG(INFO) << net.w << '\n';
    //LOG(INFO) << net.h << '\n';
    LOG(INFO) << "process Image Detection 1-8.1" << '\n';
    /*c_get_region_boxes(l, im.w, im.h, net.w, net.h, threshold, probs, boxes, masks, 0, 0, hier_threshold, 1);
    LOG(INFO) << "process Image Detection 1-9" << '\n';
    if (l.softmax_tree && nms)
        c_do_nms_obj(boxes, probs, l.w * l.h * l.h, l.classes, nms);
    else if (nms)
        c_do_nms_sort(boxes, probs, l.w * l.h * l.n, l.classes, nms);
    LOG(INFO) << "process Image Detection 1-10" << '\n';
    addDetection(im, l.w * l.h * l.n, threshold, boxes, probs, voc_names, 0, l.classes, detections);
    c_free_image(sized);
LOG(INFO) << "process Image Detection 1-max" << '\n';*/
    return detections;
}

DarknetDetections processImageDetection(network& net, const cv::Mat & im, float threshold) {
    LOG(INFO) << "process Image Detection 1" << '\n';
    image imageDarknet = cv_to_image(im);
    LOG(INFO) << "process Image Detection 2" << '\n';
    auto detection = processImageDetection(net, imageDarknet, threshold);
    LOG(INFO) << "process Image Detection 3" << '\n';
    c_free_image(imageDarknet);
    LOG(INFO) << "process Image Detection 4" << '\n';
    return detection;
}


DarknetDetections DarknetAPI::process(image& im, float threshold) {
    std::cout << "OpenCV version: "
			<< CV_MAJOR_VERSION << "." 
			<< CV_MINOR_VERSION << "."
			<< CV_SUBMINOR_VERSION
			<< std::endl;
    LOG(INFO) << "HHHHH" << '\n';
    return processImageDetection(this->net, im, threshold);
}

DarknetDetections DarknetAPI::process(const cv::Mat &im, float threshold) {
    std::cout << "OpenCV version: "
                        << CV_MAJOR_VERSION << "."
                        << CV_MINOR_VERSION << "."
                        << CV_SUBMINOR_VERSION
                        << std::endl;
    LOG(INFO) << "HHHHHH" << '\n';
    LOG(INFO) << "HHHHH" << '\n';
    image imageDarknet = cv_to_image(im);
    LOG(INFO) << "HHHHHHiA" << '\n';
    return processImageDetection(this->net, imageDarknet, threshold);
}

std::string DarknetAPI::processToJson(const cv::Mat &im, float threshold) {
    return process(im, threshold).serialize();
}



