#pragma once

#define DEBUG_ON
#define DEBUG_ON_SHOW_OK2
#define DEBUG_ON_SHOW_NG2
#define COLLECT_OCR_DATA2

/* PI */
#ifndef M_PI
#define M_PI   3.14159265358979323846
#endif /* !M_PI */

#define RadToDeg(x) (double)(x) * 57.295779513082
#define DegToRad(x) (double)(x) / 57.295779513082

struct LocateInfo 
{
    double x1 = 0;
    double y1 = 0;
    double x2 = 0;
    double y2 = 0;
    double x = 0;
    double y = 0;
    double angle = 0;
    bool empty() {return x==0 && y==0 && angle==0;}
};

enum PaperType
{
    PT_UNKOWN = 0, HGZ_A, HGZ_B, HBZ_A, HBZ_B_RY1, HBZ_B_RY2, HBZ_B_HD1, HBZ_B_HD2, HBZ_B_CD, COC_RY, COC_HD, COC_V4, RYZ_RY, RYZ_HD
};

enum DebugType
{
    NORMAL = 0, CHAR_DET_OK, CHAR_DET_NG, COLLECT_OCR_DATA
};
