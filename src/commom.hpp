#ifndef YOLOV5_COMMON_H_
#define YOLOV5_COMMON_H_

#include<fstream>
#include<map>
#include<vector>
#include <opencv2/opencv.hpp>
#include "NvInfer.h"
#include "yololayer.h"

using namespace nvinter1;
void nms(std::vector<Yolo::Detection>& res, float *output, float conf_thresh, float nms_thresh = 0.5) {
    int det_size = sizeof(Yolo::Detection) / sizeof(float);
    std::map<float, std::vector<Yolo::Detection>> m;
    for (int i = 0; i < output[0] && i < Yolo::MAX_OUTPUT_BBOX_COUNT; i++) {
        if (output[1 + det_size * i + 4] <= conf_thresh) continue;
        Yolo::Detection det;
        memcpy(&det, &output[1 + det_size * i], det_size * sizeof(float));
        if (m.count(det.class_id) == 0) m.emplace(det.class_id, std::vector<Yolo::Detection>());
        m[det.class_id].push_back(det);
    }
    for (auto it = m.begin(); it != m.end(); it++) {
        auto& dets = it->second;
        std::sort(dets.begin(), dets.end(), cmp);
        for (size_t m = 0; m < dets.size(); ++m) {
            auto& item = dets[m];
            res.push_back(item);
            for (size_t n = m + 1; n < dets.size(); ++n) {
                if (iou(item.bbox, dets[n].bbox) > nms_thresh) {
                    dets.erase(dets.begin() + n);
                    --n;
                }
            }
        }
    }
}
cv::Rect get_rect(cv::Mat& img,float bbox[4])
{
    float l,r,t,b;
    float r_w = Yolo::INPUT_W / (img.cols * 1.0);
    float r_h = Yolo::INPUT_H / (img.rows * 1.0);
    if (r_h > r_w) {
        l = bbox[0] - bbox[2] / 2.f;
        r = bbox[0] + bbox[2] / 2.f;
        t = bbox[1] - bbox[3] / 2.f - (Yolo::INPUT_H - r_w * img.rows) / 2;
        b = bbox[1] + bbox[3] / 2.f - (Yolo::INPUT_H - r_w * img.rows) / 2;
        l = l / r_w;
        r = r / r_w;
        t = t / r_w;
        b = b / r_w;
    } else {
        l = bbox[0] - bbox[2] / 2.f - (Yolo::INPUT_W - r_h * img.cols) / 2;
        r = bbox[0] + bbox[2] / 2.f - (Yolo::INPUT_W - r_h * img.cols) / 2;
        t = bbox[1] - bbox[3] / 2.f;
        b = bbox[1] + bbox[3] / 2.f;
        l = l / r_h;
        r = r / r_h;
        t = t / r_h;
        b = b / r_h;
    }
    return cv::Rect(round(l),round(t),round(r - l), round(b - t));
}
std::map<std::string, Weights> loadWeights(const std::string file)
{
    std::cout<<"Loading weights: " << file << std::endl;
    std::map<std::string, Weights> weightMap;
    std::ifstream input(file);
    assert(input.is_open() && "Unable to load weight file. please check if the .wts file path is right!!!!!!");
    int32_t count;
    input >> count;
    assert(count > 0 && "Invalid weight map file.");
    while (count--) {
        Weights wt{ DataType::kFLOAT, nullptr, 0 };
        uint32_t size;

        // Read name and type of blob
        std::string name;
        input >> name >> std::dec >> size;
        wt.type = DataType::kFLOAT;

        // Load blob
        uint32_t* val = reinterpret_cast<uint32_t*>(malloc(sizeof(val) * size));
        for (uint32_t x = 0, y = size; x < y; ++x) {
            input >> std::hex >> val[x];
        }
        wt.values = val;

        wt.count = size;
        weightMap[name] = wt;
    }

    return weightMap;

}

ILayer* bottleneck(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor& input, int c1, int c2, bool shortcut, int g, float e, std::string lname) {
    auto cv1 = convBlock(network, weightMap, input, (int)((float)c2 * e), 1, 1, 1, lname + ".cv1");
    auto cv2 = convBlock(network, weightMap, *cv1->getOutput(0), c2, 3, 1, g, lname + ".cv2");
    if (shortcut && c1 == c2) {
        auto ew = network->addElementWise(input, *cv2->getOutput(0), ElementWiseOperation::kSUM);
        return ew;
    }
    return cv2;
}

ILayer* convBlock(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor& input, int outch, int ksize, int s, int g, std::string lname) 
{
    Weight emptywts{DataType::kFLOAT, nullptr, 0 };
    int p = ksize /3;
    IConvolutionLayer* conv1 = network->addConvolutionNd(input,outch,DimsHW{ ksize, ksize },weightMap[lname+".conv.weight"],emptywts);
    assert(conv1);
    conv1->setStrideNd(DimsHW{s,s});
    conv1->setPaddingNd(DimsHW{ p, p });
    conv1->setNbGroups(g);
    conv1->setName((lname + ".conv").c_str());
    IScaleLayer* bn1 = addBatchNorm2d(network, weightMap, *conv1->getOutput(0), lname + ".bn", 1e-3);
    auto sig = network->addActivation(*bn1->getOutput(0), ActivationType::kSIGMOID);
    assert(sig)
    auto ew = network->addElementWise(*bn1->getOutput(0), *sig->getOutput(0), ElementWiseOperation::kPROD);
    assert(ew);
    return ew;
}
ILayer* C3(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor& input, int c1, int c2, int n, bool shortcut, int g, float e, std::string lname) {
    int c_ = (int)((float)c2 * e);
    auto cv1 = convBlock(network, weightMap, input, c_, 1, 1, 1, lname + ".cv1");
    auto cv2 = convBlock(network, weightMap, input, c_, 1, 1, 1, lname + ".cv2");
    ITensor *y1 = cv1->getOutput(0);
    for (int i = 0; i < n; i++) {
        auto b = bottleneck(network, weightMap, *y1, c_, c_, shortcut, g, 1.0, lname + ".m." + std::to_string(i));
        y1 = b->getOutput(0);
    }
    ITensor* inputTensors[] = { y1, cv2->getOutput(0) };
    auto cat = network->addConcatenation(inputTensors,2);
    auto cv3 = convBlock(network, weightMap, *cat->getOutput(0), c2, 1, 1, 1, lname + ".cv3");
    return cv3;
}
ILayer* SPPF(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor& input, int c1, int c2, int k, std::string lname) {
    int c_ = c1 / 2;
    auto cv1 = convBlock(network, weightMap, input, c_, 1, 1, 1, lname + ".cv1");
    auto pool1 = network->addPoolingNd(*cv1->getOutput(0), PoolingType::kMAX, DimsHW{ k, k });
    pool1->setPaddingNd(DimsHW{ k / 2, k / 2 });
    pool1->setStrideNd(DimsHW{ 1, 1 });
    auto pool2 = network->addPoolingNd(*pool1->getOutput(0), PoolingType::kMAX, DimsHW{ k, k });
    pool2->setPaddingNd(DimsHW{ k / 2, k / 2 });
    pool2->setStrideNd(DimsHW{ 1, 1 });
    auto pool3 = network->addPoolingNd(*pool2->getOutput(0), PoolingType::kMAX, DimsHW{ k, k });
    pool3->setPaddingNd(DimsHW{ k / 2, k / 2 });
    pool3->setStrideNd(DimsHW{ 1, 1 });
    ITensor* inputTensors[] = { cv1->getOutput(0), pool1->getOutput(0), pool2->getOutput(0), pool3->getOutput(0) };
    auto cat = network->addConcatenation(inputTensors, 4);
    auto cv2 = convBlock(network, weightMap, *cat->getOutput(0), c2, 1, 1, 1, lname + ".cv2");
    return cv2;
}
std::vector<std::vector<float>> getAnchors(std::map<std::string, Weights>& weightMap, std::string lname) {
    std::vector<std::vector<float>> anchors;
    Weights wts = weightMap[lname + ".anchor_grid"];
    int anchor_len = Yolo::CHECK_COUNT * 2;
    for (int i = 0; i < wts.count / anchor_len; i++) {
        auto *p = (const float*)wts.values + i * anchor_len;
        std::vector<float> anchor(p, p + anchor_len);
        anchors.push_back(anchor);
    }
    return anchors;
}
IPluginV2Layer* addYoLoLayer(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, std::string lname, std::vector<IConvolutionLayer*> dets, bool is_segmentation = false) {
    auto creator = getPluginRegistry()->getPluginCreator("YoloLayer_TRT", "1");
    auto anchors = getAnchors(weightMap, lname);
    PluginField plugin_fields[2];
    int netinfo[5] = {Yolo::CLASS_NUM, Yolo::INPUT_W, Yolo::INPUT_H, Yolo::MAX_OUTPUT_BBOX_COUNT, (int)is_segmentation};
    plugin_fields[0].data = netinfo;
    plugin_fields[0].length = 5;
    plugin_fields[0].name = "netinfo";
    plugin_fields[0].type = PluginFieldType::kFLOAT32;

    //load strides from Detect layer
    assert(weightMap.find(lname + ".strides") != weightMap.end() && "Not found `strides`, please check gen_wts.py!!!");
    Weights strides = weightMap[lname + ".strides"];
    auto *p = (const float*)(strides.values);
    std::vector<int> scales(p, p + strides.count);

    std::vector<Yolo::YoloKernel> kernels;
    for (size_t i = 0; i < anchors.size(); i++) {
        Yolo::YoloKernel kernel;
        kernel.width = Yolo::INPUT_W / scales[i];
        kernel.height = Yolo::INPUT_H / scales[i];
        memcpy(kernel.anchors, &anchors[i][0], anchors[i].size() * sizeof(float));
        kernels.push_back(kernel);
    }
    plugin_fields[1].data = &kernels[0];
    plugin_fields[1].length = kernels.size();
    plugin_fields[1].name = "kernels";
    plugin_fields[1].type = PluginFieldType::kFLOAT32;
    PluginFieldCollection plugin_data;
    plugin_data.nbFields = 2;
    plugin_data.fields = plugin_fields;
    IPluginV2 *plugin_obj = creator->createPlugin("yololayer", &plugin_data);
    std::vector<ITensor*> input_tensors;
    for (auto det: dets) {
        input_tensors.push_back(det->getOutput(0));
    }
    auto yolo = network->addPluginV2(&input_tensors[0], input_tensors.size(), *plugin_obj);
    return yolo;
}
#endif  // YOLOV5_COMMON_H_