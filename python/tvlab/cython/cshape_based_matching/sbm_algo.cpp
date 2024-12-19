#include<stdlib.h>
#include"line2Dup.h"
#include "cuda_icp/icp.h"
#include"sbm_algo.h"

using namespace cv;
using namespace std;


#include <chrono>
class Timer
{
public:
    Timer() : beg_(clock_::now()) {}
    void reset() { beg_ = clock_::now(); }
    double elapsed() const {
        return std::chrono::duration_cast<second_>
            (clock_::now() - beg_).count(); }
    void out(std::string message = ""){
        double t = elapsed();
        std::cout << message << "\nelasped time:" << t << "s\n" << std::endl;
        reset();
    }
private:
    typedef std::chrono::high_resolution_clock clock_;
    typedef std::chrono::duration<double, std::ratio<1> > second_;
    std::chrono::time_point<clock_> beg_;
};

namespace  cv_dnn {
namespace
{

template <typename T>
static inline bool SortScorePairDescend(const std::pair<float, T>& pair1,
                          const std::pair<float, T>& pair2)
{
    return pair1.first > pair2.first;
}

} // namespace

inline void GetMaxScoreIndex(const std::vector<float>& scores, const float threshold, const int top_k,
                      std::vector<std::pair<float, int> >& score_index_vec)
{
    for (size_t i = 0; i < scores.size(); ++i)
    {
        if (scores[i] > threshold)
        {
            score_index_vec.push_back(std::make_pair(scores[i], i));
        }
    }
    std::stable_sort(score_index_vec.begin(), score_index_vec.end(),
                     SortScorePairDescend<int>);
    if (top_k > 0 && top_k < (int)score_index_vec.size())
    {
        score_index_vec.resize(top_k);
    }
}

template <typename BoxType>
inline void NMSFast_(const std::vector<BoxType>& bboxes,
      const std::vector<float>& scores, const float score_threshold,
      const float nms_threshold, const float eta, const int top_k,
      std::vector<int>& indices, float (*computeOverlap)(const BoxType&, const BoxType&))
{
    CV_Assert(bboxes.size() == scores.size());
    std::vector<std::pair<float, int> > score_index_vec;
    GetMaxScoreIndex(scores, score_threshold, top_k, score_index_vec);

    // Do nms.
    float adaptive_threshold = nms_threshold;
    indices.clear();
    for (size_t i = 0; i < score_index_vec.size(); ++i) {
        const int idx = score_index_vec[i].second;
        bool keep = true;
        for (int k = 0; k < (int)indices.size() && keep; ++k) {
            const int kept_idx = indices[k];
            float overlap = computeOverlap(bboxes[idx], bboxes[kept_idx]);
            keep = overlap <= adaptive_threshold;
        }
        if (keep)
            indices.push_back(idx);
        if (keep && eta < 1 && adaptive_threshold > 0.5) {
          adaptive_threshold *= eta;
        }
    }
}


// copied from opencv 3.4, not exist in 3.0
template<typename _Tp> static inline
double jaccardDistance__(const Rect_<_Tp>& a, const Rect_<_Tp>& b) {
    _Tp Aa = a.area();
    _Tp Ab = b.area();

    if ((Aa + Ab) <= std::numeric_limits<_Tp>::epsilon()) {
        // jaccard_index = 1 -> distance = 0
        return 0.0;
    }

    double Aab = (a & b).area();
    // distance = 1 - jaccard_index
    return 1.0 - Aab / (Aa + Ab - Aab);
}

template <typename T>
static inline float rectOverlap(const T& a, const T& b)
{
    return 1.f - static_cast<float>(jaccardDistance__(a, b));
}

void NMSBoxes(const std::vector<Rect>& bboxes, const std::vector<float>& scores,
                          const float score_threshold, const float nms_threshold,
                          std::vector<int>& indices, const float eta=1, const int top_k=0)
{
    NMSFast_(bboxes, scores, score_threshold, nms_threshold, eta, top_k, indices, rectOverlap);
}

}

void* detector_new(int features_num, int levels_num, int *levels,
        float weak_threshold, float strong_threshold, int gaussion_kenel) {
     vector<int> T;
     for (int i=0; i < levels_num; i++) {
         T.push_back(levels[i]);
     }

     void* det = (void*)(new line2Dup::Detector(features_num, T, weak_threshold, strong_threshold, gaussion_kenel));
     return det;
}

void detector_free(void* det) {
    delete (line2Dup::Detector *)det;
}

int add_template(void* det, char *img_ptr, char *mask_ptr,
        int img_w, int img_h, int img_c, const char *class_id,
        float angle_start, float angle_stop, float angle_step,
        float scale_start, float scale_stop, float scale_step,
        int (*cbf)(int, int)) {
    line2Dup::Detector *detector = (line2Dup::Detector*)det;
    size_t num_features = detector->getModalities()->num_features;

    int img_type = CV_8UC3;
    if (img_c  == 1) {
        img_type = CV_8UC1;
    }
    Mat img(img_h, img_w, img_type, img_ptr);
    Mat mask(img_h, img_w, CV_8UC1, mask_ptr);
    shape_based_matching::shapeInfo_producer shapes(img, mask);
    if (angle_start != angle_stop) {
        shapes.angle_range = {angle_start, angle_stop};
        shapes.angle_step = angle_step;
    }

    shapes.scale_range = {1.0};
    if (scale_start != scale_stop) {
        shapes.scale_range = {scale_start, scale_stop};
        shapes.scale_step = scale_step;
    }
    shapes.produce_infos();

    shape_based_matching::shapeInfo_producer::Info info(0, 1.0);
    int templ_id = detector->addTemplate(shapes.src_of(info), class_id, shapes.mask_of(info),
                                        int(num_features*info.scale), info.angle, info.scale,
                                        img_w, img_h, 0, 0);
    if (templ_id == -1) {
        return -1;
    }

    auto temp = detector->getTemplates(class_id, templ_id);
    int real_width = temp.real_width;
    int real_height = temp.real_height;
    int cx = temp.templates[0].tl_x + temp.cx_offset;
    int cy = temp.templates[0].tl_y + temp.cy_offset;

    int i = 0;
    if (cbf(i, shapes.infos.size()) != 0) {
        return -1;
    }

    for(auto& info: shapes.infos) {
        if (info.scale == 1.0 && info.angle == 0)
            continue;
        int dst_x, dst_y;
        shapes.point_of(info, cx, cy, dst_x, dst_y);
        detector->addTemplate(shapes.src_of(info), class_id, shapes.mask_of(info),
                              int(num_features*info.scale), info.angle, info.scale,
                              real_width*info.scale, real_height*info.scale, dst_x, dst_y);
        i += 1;
        if (cbf(i, shapes.infos.size()) != 0) {
            printf("cb return error, need exit!\n");
            return -1;
        }
    }
    return 0;
}

vector<string> split(string str, string token){
    vector<string>result;
    while(str.size()){
        int index = str.find(token);
        if(index!=string::npos){
            result.push_back(str.substr(0,index));
            str = str.substr(index+token.size());
            if(str.size()==0)result.push_back(str);
        }else{
            result.push_back(str);
            str = "";
        }
    }
    return result;
}

void save_template(void *det, const char *_info_dir) {
    line2Dup::Detector *detector = (line2Dup::Detector*)det;
    string info_dir = _info_dir;
    detector->writeClasses(info_dir+"/%s.yaml");
}

void load_template(void *det, const char *_info_dir, const char *class_ids) {
    line2Dup::Detector *detector = (line2Dup::Detector*)det;
    vector<string> ids = split(string(class_ids), ",");
    string info_dir = _info_dir;
    detector->readClasses(ids, info_dir+"/%s.yaml");
}

int show_template(void * det, char *img_ptr, int img_w, int img_h, int img_c,
                   const char *class_id, int template_id) {
    line2Dup::Detector *detector = (line2Dup::Detector*)det;

    int img_type = CV_8UC3;
    if (img_c  == 1) {
        img_type = CV_8UC1;
    }
    if (detector->hasTemplates(class_id, template_id) == 0) {
        Mat img(img_h, img_w, img_type, img_ptr);
        auto templ = detector->getTemplates(class_id, template_id);
        shape_based_matching::shapeInfo_producer shapes(img);
        shape_based_matching::shapeInfo_producer::Info info(templ.angle, templ.scale);
        Mat to_show = shapes.src_of(info);
        if (img_c == 1)
            cvtColor(to_show, to_show, cv::COLOR_GRAY2RGB);
        for(int i=0; i<templ.templates[0].features.size(); i++){
            auto feat = templ.templates[0].features[i];
            cv::circle(to_show, {feat.x+templ.templates[0].tl_x,
                    feat.y+templ.templates[0].tl_y}, 3, {255, 0, 0}, -1);
        }
        if (img_c == 1)
            cvtColor(to_show, to_show, cv::COLOR_GRAY2RGB);
        to_show.copyTo(img);
        return 0;
    }
    return -1;
}

int match_template(void *det, char *img_ptr, char *debug_img_ptr,
                   int img_w, int img_h, int img_c,
                   float threshold, float iou_threshold, const char *class_ids,
                   int top_k, int subpixel,
                   float** pmatches, char **pmatches_id) {
    line2Dup::Detector *detector = (line2Dup::Detector*)det;
    int img_type = CV_8UC3;
    if (img_c  == 1) {
        img_type = CV_8UC1;
    }
    Mat img(img_h, img_w, img_type, img_ptr);
    vector<string> ids = split(string(class_ids), ",");

    auto _matches = detector->match(img, threshold, ids);
    if (_matches.size() <= 0) {
        return 0;
    }

    vector<int> idxs;
    for (auto id: ids) {
        vector<int> per_cls_idxs;
        if (iou_threshold < 1.0) {
            vector<Rect> boxes;
            vector<float> scores;
            vector<int> _per_cls_idxs;
            vector<int> ori_idxs;
            for (int i = 0; i < _matches.size(); i++) {
                auto match = _matches[i];
                if (match.class_id != id)
                    continue;
                ori_idxs.push_back(i);
                Rect box;
                box.x = match.x;
                box.y = match.y;

                auto templ = detector->getTemplates(match.class_id, match.template_id);

                box.width = templ.templates[0].width;
                box.height = templ.templates[0].height;
                boxes.push_back(box);
                scores.push_back(match.similarity);
            }
            int _top_k = top_k;
            if(top_k > scores.size())
                _top_k = scores.size();

            cv_dnn::NMSBoxes(boxes, scores, 0, iou_threshold, _per_cls_idxs, 1.0, _top_k);
            for (auto pid: _per_cls_idxs)
                per_cls_idxs.push_back(ori_idxs[pid]);
        } else {
            for (int i = 0; i < _matches.size(); i++) {
                if (_matches[i].class_id != id)
                    continue;
                per_cls_idxs.push_back(i);
                if (top_k > 0 && top_k <= (int)per_cls_idxs.size())
                    break;
            }
        }
        for (auto pid: per_cls_idxs) {
            idxs.push_back(pid);
        }
    }

    float *matches = (float*)malloc(idxs.size() * 7 * sizeof(float));
    string matches_id;
    int i = 0;
    Scene_edge scene;
    vector<::Vec2f> pcd_buffer, normal_buffer;
    if (subpixel) {
        scene.init_Scene_edge_cpu(img, pcd_buffer, normal_buffer);
    }

    for (auto idx: idxs) {
        auto match = _matches[idx];
        auto templ = detector->getTemplates(match.class_id, match.template_id);
        float x = match.x + templ.cx_offset;
        float y = match.y + templ.cy_offset;
        float scale = templ.scale;
        float angle = templ.angle;
        float real_w = templ.real_width;
        float real_h = templ.real_height;

        if (debug_img_ptr != NULL) {
            Mat debugimg(img_h, img_w, CV_8UC3, debug_img_ptr);
            for(int i=0; i<templ.templates[0].features.size(); i++){
                auto feat = templ.templates[0].features[i];
                cv::circle(debugimg, {feat.x+match.x, feat.y+match.y}, 3, {255, 0, 0}, -1);
            }
        }

        if (subpixel) {
            vector<::Vec2f> model_pcd(templ.templates[0].features.size());
            for(int i=0; i<templ.templates[0].features.size(); i++){
                auto& feat = templ.templates[0].features[i];
                model_pcd[i] = {
                    float(feat.x + match.x),
                    float(feat.y + match.y)
                };
            }
            cuda_icp::RegistrationResult result = cuda_icp::ICP2D_Point2Plane_cpu(model_pcd, scene);

            float new_x = result.transformation_[0][0]*x +
                          result.transformation_[0][1]*y +
                          result.transformation_[0][2];
            float new_y = result.transformation_[1][0]*x +
                          result.transformation_[1][1]*y +
                          result.transformation_[1][2];
            float improved_scale = std::sqrt(result.transformation_[0][0]*result.transformation_[0][0] +
                                             result.transformation_[1][0]*result.transformation_[1][0]);

            scale = scale * improved_scale;
            real_w = real_w * improved_scale;
            real_h = real_h * improved_scale;
            angle = angle + std::atan(result.transformation_[1][0]/
                                      result.transformation_[0][0])/CV_PI*180;

            x = new_x;
            y = new_y;

            if (debug_img_ptr != NULL) {
                Mat debugimg(img_h, img_w, CV_8UC3, debug_img_ptr);
                for(int i=0; i<templ.templates[0].features.size(); i++){
                    auto feat = templ.templates[0].features[i];
                    float fx = feat.x + match.x;
                    float fy = feat.y + match.y;
                    float new_fx = result.transformation_[0][0]*fx +
                                   result.transformation_[0][1]*fy +
                                   result.transformation_[0][2];
                    float new_fy = result.transformation_[1][0]*fx +
                                   result.transformation_[1][1]*fy +
                                   result.transformation_[1][2];
                    cv::circle(debugimg, {int(new_fx+0.5f), int(new_fy+0.5f)}, 3, {0, 255, 0}, -1);
                }
            }
        }
        matches_id += match.class_id + ",";
        matches[i++] = x;
        matches[i++] = y;
        matches[i++] = real_w;
        matches[i++] = real_h;
        matches[i++] = angle;
        matches[i++] = scale;
        matches[i++] = match.similarity;
    }
    char *_matches_id = (char*)malloc(matches_id.size());
    matches_id.copy(_matches_id, matches_id.size());
    _matches_id[matches_id.size()-1] = '\0';
    *pmatches = matches;
    *pmatches_id = _matches_id;
    return idxs.size();
}

void matches_free(float *matches, char *matches_id) {
    free(matches);
    free(matches_id);
}
