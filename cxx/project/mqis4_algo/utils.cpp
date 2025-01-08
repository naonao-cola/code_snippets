#include <windows.h>
#include <locale.h>
#include "utils.h"
#include "logger.h"
#include <opencv2/imgproc/types_c.h>

wchar_t* AnsiToUnicode(const char* lpszStr)
{
    wchar_t* lpUnicode;
    int nLen;

    if (NULL == lpszStr)
        return NULL;

    nLen = MultiByteToWideChar(CP_ACP, 0, lpszStr, -1, NULL, 0);
    if (0 == nLen)
        return NULL;

    lpUnicode = new wchar_t[nLen + 1];
    if (NULL == lpUnicode)
        return NULL;

    memset(lpUnicode, 0, sizeof(wchar_t)* (nLen + 1));
    nLen = MultiByteToWideChar(CP_ACP, 0, lpszStr, -1, lpUnicode, nLen);
    if (0 == nLen)
    {
        delete[]lpUnicode;
        return NULL;
    }

    return lpUnicode;
}

char* UnicodeToAnsi(const wchar_t* lpszStr)
{
    char* lpAnsi;
    int nLen;

    if (NULL == lpszStr)
        return NULL;

    nLen = WideCharToMultiByte(CP_ACP, 0, lpszStr, -1, NULL, 0, NULL, NULL);
    if (0 == nLen)
        return NULL;

    lpAnsi = new char[nLen + 1];
    if (NULL == lpAnsi)
        return NULL;

    memset(lpAnsi, 0, nLen + 1);
    nLen = WideCharToMultiByte(CP_ACP, 0, lpszStr, -1, lpAnsi, nLen, NULL, NULL);
    if (0 == nLen)
    {
        delete[]lpAnsi;
        return NULL;
    }

    return lpAnsi;
}

char* AnsiToUtf8(const char* lpszStr)
{
    wchar_t* lpUnicode;
    char* lpUtf8;
    int nLen;

    if (NULL == lpszStr)
        return NULL;

    nLen = MultiByteToWideChar(CP_ACP, 0, lpszStr, -1, NULL, NULL);
    if (0 == nLen)
        return NULL;

    lpUnicode = new wchar_t[nLen + 1];
    if (NULL == lpUnicode)
        return NULL;

    memset(lpUnicode, 0, sizeof(wchar_t)* (nLen + 1));
    nLen = MultiByteToWideChar(CP_ACP, 0, lpszStr, -1, lpUnicode, nLen);
    if (0 == nLen)
    {
        delete[]lpUnicode;
        return NULL;
    }

    nLen = WideCharToMultiByte(CP_UTF8, 0, lpUnicode, -1, NULL, 0, NULL, NULL);
    if (0 == nLen)
    {
        delete[]lpUnicode;
        return NULL;
    }

    lpUtf8 = new char[nLen + 1];
    if (NULL == lpUtf8)
    {
        delete[]lpUnicode;
        return NULL;
    }

    memset(lpUtf8, 0, nLen + 1);
    nLen = WideCharToMultiByte(CP_UTF8, 0, lpUnicode, -1, lpUtf8, nLen, NULL, NULL);
    if (0 == nLen)
    {
        delete[]lpUnicode;
        delete[]lpUtf8;
        return NULL;
    }

    delete[]lpUnicode;

    return lpUtf8;
}

char* Utf8ToAnsi(const char* lpszStr)
{
    wchar_t* lpUnicode;
    char* lpAnsi;
    int nLen;

    if (NULL == lpszStr)
        return NULL;

    nLen = MultiByteToWideChar(CP_UTF8, 0, lpszStr, -1, NULL, NULL);
    if (0 == nLen)
        return NULL;

    lpUnicode = new wchar_t[nLen + 1];
    if (NULL == lpUnicode)
        return NULL;

    memset(lpUnicode, 0, sizeof(wchar_t)* (nLen + 1));
    nLen = MultiByteToWideChar(CP_UTF8, 0, lpszStr, -1, lpUnicode, nLen);
    if (0 == nLen)
    {
        delete[]lpUnicode;
        return NULL;
    }

    nLen = WideCharToMultiByte(CP_ACP, 0, lpUnicode, -1, NULL, 0, NULL, NULL);
    if (0 == nLen)
    {
        delete[]lpUnicode;
        return NULL;
    }

    lpAnsi = new char[nLen + 1];
    if (NULL == lpAnsi)
    {
        delete[]lpUnicode;
        return NULL;
    }

    memset(lpAnsi, 0, nLen + 1);
    nLen = WideCharToMultiByte(CP_ACP, 0, lpUnicode, -1, lpAnsi, nLen, NULL, NULL);
    if (0 == nLen)
    {
        delete[]lpUnicode;
        delete[]lpAnsi;
        return NULL;
    }

    delete[]lpUnicode;

    return lpAnsi;
}

char* UnicodeToUtf8(const wchar_t* lpszStr)
{
    char* lpUtf8;
    int nLen;

    if (NULL == lpszStr)
        return NULL;

    nLen = WideCharToMultiByte(CP_UTF8, 0, lpszStr, -1, NULL, 0, NULL, NULL);
    if (0 == nLen)
        return NULL;

    lpUtf8 = new char[nLen + 1];
    if (NULL == lpUtf8)
        return NULL;

    memset(lpUtf8, 0, nLen + 1);
    nLen = WideCharToMultiByte(CP_UTF8, 0, lpszStr, -1, lpUtf8, nLen, NULL, NULL);
    if (0 == nLen)
    {
        delete[]lpUtf8;
        return NULL;
    }

    return lpUtf8;
}

wchar_t* Utf8ToUnicode(const char* lpszStr)
{
    wchar_t* lpUnicode;
    int nLen;

    if (NULL == lpszStr)
        return NULL;

    nLen = MultiByteToWideChar(CP_UTF8, 0, lpszStr, -1, NULL, 0);
    if (0 == nLen)
        return NULL;

    lpUnicode = new wchar_t[nLen + 1];
    if (NULL == lpUnicode)
        return NULL;

    memset(lpUnicode, 0, sizeof(wchar_t)* (nLen + 1));
    nLen = MultiByteToWideChar(CP_UTF8, 0, lpszStr, -1, lpUnicode, nLen);
    if (0 == nLen)
    {
        delete[]lpUnicode;
        return NULL;
    }

    return lpUnicode;
}

bool AnsiToUnicode(const char* lpszAnsi, wchar_t* lpszUnicode, int nLen)
{
    int nRet = MultiByteToWideChar(CP_ACP, 0, lpszAnsi, -1, lpszUnicode, nLen);
    return (0 == nRet) ? FALSE : TRUE;
}

bool UnicodeToAnsi(const wchar_t* lpszUnicode, char* lpszAnsi, int nLen)
{
    int nRet = WideCharToMultiByte(CP_ACP, 0, lpszUnicode, -1, lpszAnsi, nLen, NULL, NULL);
    return (0 == nRet) ? FALSE : TRUE;
}

bool AnsiToUtf8(const char* lpszAnsi, char* lpszUtf8, int nLen)
{
    wchar_t* lpszUnicode = AnsiToUnicode(lpszAnsi);
    if (NULL == lpszUnicode)
        return FALSE;

    int nRet = UnicodeToUtf8(lpszUnicode, lpszUtf8, nLen);

    delete[]lpszUnicode;

    return (0 == nRet) ? FALSE : TRUE;
}

bool Utf8ToAnsi(const char* lpszUtf8, char* lpszAnsi, int nLen)
{
    wchar_t* lpszUnicode = Utf8ToUnicode(lpszUtf8);
    if (NULL == lpszUnicode)
        return FALSE;

    int nRet = UnicodeToAnsi(lpszUnicode, lpszAnsi, nLen);

    delete[]lpszUnicode;

    return (0 == nRet) ? FALSE : TRUE;
}

bool UnicodeToUtf8(const wchar_t* lpszUnicode, char* lpszUtf8, int nLen)
{
    int nRet = WideCharToMultiByte(CP_UTF8, 0, lpszUnicode, -1, lpszUtf8, nLen, NULL, NULL);
    return (0 == nRet) ? FALSE : TRUE;
}

bool Utf8ToUnicode(const char* lpszUtf8, wchar_t* lpszUnicode, int nLen)
{
    int nRet = MultiByteToWideChar(CP_UTF8, 0, lpszUtf8, -1, lpszUnicode, nLen);
    return (0 == nRet) ? FALSE : TRUE;
}

std::wstring AnsiToUnicode(const std::string& strAnsi)
{
    std::wstring strUnicode;

    wchar_t* lpszUnicode = AnsiToUnicode(strAnsi.c_str());
    if (lpszUnicode != NULL)
    {
        strUnicode = lpszUnicode;
        delete[]lpszUnicode;
    }

    return strUnicode;
}
std::string UnicodeToAnsi(const std::wstring& strUnicode)
{
    std::string strAnsi;

    char* lpszAnsi = UnicodeToAnsi(strUnicode.c_str());
    if (lpszAnsi != NULL)
    {
        strAnsi = lpszAnsi;
        delete[]lpszAnsi;
    }

    return strAnsi;
}

std::string AnsiToUtf8(const std::string& strAnsi)
{
    std::string strUtf8;

    char* lpszUtf8 = AnsiToUtf8(strAnsi.c_str());
    if (lpszUtf8 != NULL)
    {
        strUtf8 = lpszUtf8;
        delete[]lpszUtf8;
    }

    return strUtf8;
}

std::string Utf8ToAnsi(const std::string& strUtf8)
{
    std::string strAnsi;

    char* lpszAnsi = Utf8ToAnsi(strUtf8.c_str());
    if (lpszAnsi != NULL)
    {
        strAnsi = lpszAnsi;
        delete[]lpszAnsi;
    }

    return strAnsi;
}

std::string UnicodeToUtf8(const std::wstring& strUnicode)
{
    std::string strUtf8;

    char* lpszUtf8 = UnicodeToUtf8(strUnicode.c_str());
    if (lpszUtf8 != NULL)
    {
        strUtf8 = lpszUtf8;
        delete[]lpszUtf8;
    }

    return strUtf8;
}

std::wstring Utf8ToUnicode(const std::string& strUtf8)
{
    std::wstring strUnicode;

    wchar_t* lpszUnicode = Utf8ToUnicode(strUtf8.c_str());
    if (lpszUnicode != NULL)
    {
        strUnicode = lpszUnicode;
        delete[]lpszUnicode;
    }

    return strUnicode;
}


void draw_polygon(cv::Mat& image, const json& pts, const cv::Scalar& color)
{
    std::vector<cv::Point> vpts;
    for (int i=0; i < pts.size()/2; ++i) {
        vpts.push_back(cv::Point(pts[i*2], pts[i*2+1]));
    }
    cv::polylines(image, vpts, true, color, 5, 8, 0);
}

std::vector<cv::Point2f> get_rotrect_coords(const json& xywhr, bool lt_first)
{
    double centX = xywhr[0];
    double centY = xywhr[1];
    double hw = xywhr[2];
    double hh = xywhr[3];
    double phi = xywhr[4]; // 弧度

    double ox[] = { centX - hw, centX + hw, centX + hw, centX - hw };
    double oy[] = { centY - hh, centY - hh, centY + hh, centY + hh };
    // json coords = json::array();
    std::vector<cv::Point2f> tmp_pts;
    for (int i = 0; i < 4; i++)
    {
        double x = (ox[i] - centX) * cos(-phi) - (oy[i] - centY) * sin(-phi) + centX;
        double y = (ox[i] - centX) * sin(-phi) + (oy[i] - centY) * cos(-phi) + centY;
        tmp_pts.push_back(cv::Point2f(x, y));
    }

    if (lt_first) {
        sort_rotrect_pts(tmp_pts);
    }
    return tmp_pts;
}

// std::vector<cv::Point> get_minrect_points(cv::RotatedRect rot_rect, bool lt_first)
// {
//     double x = rot_rect.center.x;
//     double y = rot_rect.center.y;
//     double w = rot_rect.size.width;
//     double h = rot_rect.size.height;
//     double r = rot_rect.angle;
//     if (std::abs(r) < 20) {
//         r = -r;
//     } else {
//         w = rot_rect.size.height;
//         h = rot_rect.size.width;
//         r = 90 - r;
//     }
//     return get_rotrect_coords(json::array({x, y, w, h, DegToRad(r)}));
// }

std::vector<cv::Point> json_to_cv_pts(const json& pts)
{
    std::vector<cv::Point> cv_pts;
    for (int i=0; i < pts.size()/2; ++i) {
        cv_pts.push_back(cv::Point(pts[i*2], pts[i*2+1]));
    }
    return cv_pts;
}

cv::Mat d6_to_cvMat(double d0, double d1, double d2, double d3, double d4, double d5) {
    cv::Mat mtx(3, 3, CV_64FC1);
    double *M = mtx.ptr<double>();
    M[0] = d0;
    M[1] = d1;
    M[2] = d2;
    M[3] = d3;
    M[4] = d4;
    M[5] = d5;
    M[6] = 0.0;
    M[7] = 0.0;
    M[8] = 1.0;
    return mtx;
}

cv::Mat cvMat6_to_cvMat9(const cv::Mat &mtx6) {
    cv::Mat mtx9(3, 3, CV_64FC1);
    double *M9 = mtx9.ptr<double>();
    const double *M6 = mtx6.ptr<double>();
    M9[0] = M6[0];
    M9[1] = M6[1];
    M9[2] = M6[2];
    M9[3] = M6[3];
    M9[4] = M6[4];
    M9[5] = M6[5];
    M9[6] = 0.0;
    M9[7] = 0.0;
    M9[8] = 1.0;
    return mtx9;
}

cv::Mat vector_angle_to_M(const LocateInfo& v1, const LocateInfo& v2)
{
    return vector_angle_to_M(v1.x, v1.y, RadToDeg(v1.angle), v2.x, v2.y, RadToDeg(v2.angle));
}

cv::Mat vector_angle_to_M(double x1, double y1, double d1, double x2, double y2, double d2)
{
    double rot_d = d2 - d1;

    cv::Point2f center(x1, y1);
    double angle = rot_d;
    cv::Mat mtx_rot = cv::getRotationMatrix2D(center, angle, 1.0);
    mtx_rot = cvMat6_to_cvMat9(mtx_rot);

    cv::Mat mtxt = d6_to_cvMat(1, 0, x2-x1, 0, 1, y2-y1);
    cv::Mat mtx_trans = mtxt * mtx_rot; // 先旋转在平移（矩阵乘法相反）
    return mtx_trans;
}

json affine_points(const cv::Mat& M, const json& pts)
{
    json points = json::array();
    return points;
}

double p2p_distance(double x1, double y1, double x2, double y2)
{
    return std::sqrt((x1-x2)*(x1-x2) + (y1-y2)*(y1-y2));
}

cv::Mat points2d_to_mat(json points)
{
    std::vector<double> v_pts;
    for (int i=0; i < points.size(); ++i) {
        v_pts.push_back(points[i]);
        if (i % 2 == 1) {
            v_pts.push_back(1);
        }
    }
	cv::Mat mat = cv::Mat(v_pts).clone(); //将vector变成单列的mat，这里需要clone(),因为这里的赋值操作是浅拷贝
	cv::Mat dest = mat.reshape(1, int(points.size()/2));
	return dest;
}

json bbox2polygon(double x1, double y1, double x2, double y2)
{
    json polygon = {
        x1, y1, x2, y1, x2, y2, x1, y2
    };
    return polygon;
}

json bbox2polygon(json bbox)
{
    return bbox2polygon(bbox[0][0], bbox[0][1], bbox[1][0], bbox[1][1]);
}

json polygon2bbox(json polygon)
{
    json out_bbox = json::array();
    std::vector<cv::Point> pts = json_to_cv_pts(polygon);
    cv::Rect rect = cv::boundingRect(pts);
    out_bbox.push_back({ rect.x, rect.y });
    out_bbox.push_back({ rect.x + rect.width, rect.y + rect.height });
    return out_bbox;
}

bool is_intersect(json bbox1, json bbox2)
{
    if (bbox1[0][0] > bbox2[1][0])
        return false;
    if (bbox1[1][0] < bbox2[0][0])
        return false;
    if (bbox1[0][1] > bbox2[1][1])
        return false;
    if (bbox1[1][1] < bbox2[0][1])
        return false;
    return true;
}

json bbox_intersect(json bbox1, json bbox2)
{
    json inter_bbox = json::array();
    if (!is_intersect(bbox1, bbox2))
        return inter_bbox;
    double xmin = std::max(bbox1[0][0], bbox2[0][0]);
    double ymin = std::max(bbox1[0][1], bbox2[0][1]);
    double xmax = std::min(bbox1[1][0], bbox2[1][0]);
    double ymax = std::min(bbox1[1][1], bbox2[1][1]);
    inter_bbox.push_back({xmin, ymin});
    inter_bbox.push_back({xmax, ymax});
    return inter_bbox;
}

void get_rotrect_size(json retrect, int& width, int& height)
{
    assert(retrect.size() == 8);
    double ltx = retrect[0];
    double lty = retrect[1];
    double rtx = retrect[2];
    double rty = retrect[3];
    double rbx = retrect[4];
    double rby = retrect[5];
    width = std::round(std::sqrt((ltx-rtx)*(ltx-rtx) + (lty-rty)*(lty-rty)));
    height = std::round(std::sqrt((rtx-rbx)*(rtx-rbx) + (rty-rby)*(rty-rby)));
}

PaperType get_paper_type(json img_param)
{
    PaperType p_type = PT_UNKOWN;
    std::string paper_name = img_param["name"];
    std::string type_str = paper_name.substr(0, 2);
    if (type_str == "11") {
        p_type = HGZ_A;
    } else if (type_str == "12") {
        p_type = HGZ_B;
    } else if (type_str == "21") {
        p_type = HBZ_A;
    } else if (type_str == "22") {
        p_type = HBZ_B_RY1; // TODO: 具体型号
    } else if (type_str == "32") {
        p_type = COC_RY;    // TODO: 具体型号
    } else if (type_str == "42") {
        p_type = RYZ_RY;    // TODO: 具体型号
    }
    return p_type;
}

cv::Mat gray_scale_image(cv::Mat img, int r1, int r2, int s1, int s2)
{
    assert(r2 > r1 && r1 >=0 && r2 <=255 && s1 >=0 && s2 <= 255, "Invalid parameter!");
    assert(s1 == 0 || (s1 > 0 && r1 > 0), "Invalid parameter!");
    assert(s2 == 255 || (s2 < 255 && r2 < 255), "Invalid parameter!");
    std::vector<double> lut(256);
    for (int i = 0; i < 256; i++)
    {
        if (i < r1) {
            if (s1 == 0) {
                lut[i] = 0;
            } else {
                double k = s1 * 1.0 / r1;
                lut[i] = k * i;
            }
        } else if (i > r2) {
            if (s2 == 255) {
                lut[i] = 255;
            } else {
                double k = (255 - s2) * 1.0 / (255 - r2);
                lut[i] = s2 + k * (i - r2);
            }
        } else {
            double k = (s2 - s1) * 1.0 / (r2 - r1);
            lut[i] = s1 + k * (i - r1);
        }
    }
    cv::Mat out_img;
    cv::LUT(img, lut, out_img);
    out_img.convertTo(out_img, CV_8U);
    return out_img;
}

cv::RotatedRect cv_rotrect_to_halcon_rotrect(cv::RotatedRect rotrect)
{
    cv::RotatedRect out_rect(rotrect);
    float r =  rotrect.angle;
    // opencv 4.5.x minAearRect to halcon rotate rect
    if (abs(rotrect.angle) < 20){ // 左下角y值最大，矩形角度= 90+abs(r)
        out_rect.angle = -rotrect.angle;
    } else {
        out_rect.size.width = rotrect.size.height;
        out_rect.size.height = rotrect.size.width;
        out_rect.angle = 90 - rotrect.angle;
    }
    return out_rect;
}

std::string get_paper_type_str(PaperType ptype)
{
    std:: array hbz_b_types = {HBZ_B_RY1, HBZ_B_RY2, HBZ_B_HD1, HBZ_B_HD2, HBZ_B_CD};
    if (ptype == HGZ_A) {
        return "HGZ_A";
    } else if (ptype == HGZ_B) {
        return "HGZ_B";
    } else if (ptype == HBZ_A) {
        return "HBZ_A";
    } else if (std::find(hbz_b_types.begin(), hbz_b_types.end(), ptype) != hbz_b_types.end()) {
        return "HBZ_B";
    } else if (ptype == RYZ_HD || ptype == RYZ_RY) {
        return "RYZ";
    } else if (ptype == COC_HD || ptype == COC_RY || ptype == COC_V4) {
        return "COC";
    }
    return "UNKNOWN";
}

void write_rgb_img(std::string fpath, cv::Mat img, bool cvtBGR)
{
    if (cvtBGR) {
        cv::Mat rgb_img;
        cv::cvtColor(img, rgb_img, cv::COLOR_RGB2BGR);
        cv::imwrite(fpath, rgb_img);
    } else {
        cv::imwrite(fpath, img);
    }
}

void write_debug_img(std::string fpath, cv::Mat img, bool cvtBGR, DebugType dbg_type)
{
    switch (dbg_type)
    {
    case DebugType::NORMAL:
#ifdef DEBUG_ON
        write_rgb_img(fpath, img, cvtBGR);
#endif
        break;
    case DebugType::CHAR_DET_OK:
#ifdef DEBUG_ON_SHOW_OK
        write_rgb_img(fpath, img, cvtBGR);
#endif
        break;
    case DebugType::CHAR_DET_NG:
#ifdef DEBUG_ON_SHOW_OK
        write_rgb_img(fpath, img, cvtBGR);
#endif
        break;
    case DebugType::COLLECT_OCR_DATA:
#ifdef COLLECT_OCR_DATA
        write_rgb_img(fpath, img, cvtBGR);
#endif
        break;
    
    default:
        break;
    }
}