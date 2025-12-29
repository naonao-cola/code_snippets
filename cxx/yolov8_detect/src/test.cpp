#include <iostream>
#include "yolo.hpp"


int main()
{
    YOLO yolo;
    yolo.init("/home/nvidia/wangw/demo/yolo_test/model/PERSON.engine");
    yolo.detect_img("/home/nvidia/wangw/demo/yolo_test/data/2222.jpg");
    yolo.destroy();
}