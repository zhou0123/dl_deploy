#ifndef _YOLO_LAYER_H
#define _YOLO_LAYER_H

#include <vector>
#include <string>
#include <NvInfer.h>
#include "macros.h"
namespace Yolo
{
    static constexpr int CHECK_COUNT=3;
    static constexpr float IGNORE_THRESH = 0.1f;
    struct YoloKernel
    {
        int width;
        int height;
    }
}