#include "defines.h"

void write_rgb_img(std::string fpath, cv::Mat img)
{
    bool cvtBGR = false;
    if (cvtBGR) {
        cv::Mat rgb_img;
        cv::cvtColor(img, rgb_img, cv::COLOR_RGB2BGR);
        cv::imwrite(fpath, rgb_img);
    } else {
        cv::imwrite(fpath, img);
    }
    return;
}

void write_debug_img(std::string fpath, cv::Mat img, DebugType debug_type)
{
switch (debug_type)
{

    case DebugType::FIND_C:
#ifdef DEBUG_SHOW_C
        write_rgb_img(fpath, img);
#endif
    break;

    case DebugType::FIND_O:
#ifdef DEBUG_SHOW_O
        write_rgb_img(fpath, img);
#endif
    break;  

    case DebugType::FIND_PLUG:
#ifdef DEBUG_SHOW_PLUG
        write_rgb_img(fpath, img);
#endif
    break;

    case DebugType::FIND_C_RING:
#ifdef DEBUG_SHOW_C_RING
        write_rgb_img(fpath, img);
#endif
    break;

    case DebugType::FIND_O_RING:
#ifdef DEBUG_SHOW_O_RING
        write_rgb_img(fpath, img);
#endif
    break;  

    case DebugType::CIRCLE_EA:
#ifdef DEBUG_CIRCLE_EA
        write_rgb_img(fpath, img);
#endif
    break;

    case DebugType::CIRCLE_HD:
#ifdef DEBUG_CIRCLE_HD
        write_rgb_img(fpath, img);
#endif
    break;

    default:
    break;
}
    return;
}