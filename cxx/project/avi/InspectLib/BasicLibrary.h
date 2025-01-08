
#include "opencv2\opencv.hpp"

class GuidedFilterImpl
{
public:
	virtual ~GuidedFilterImpl() {}

	cv::Mat filter(const cv::Mat &p, int depth);

protected:
	int Idepth;

private:
	virtual cv::Mat filterSingleChannel(const cv::Mat &p) const = 0;
};

class GuidedFilterMono : public GuidedFilterImpl
{
public:
	GuidedFilterMono(const cv::Mat &I, int r, double eps);

private:
	virtual cv::Mat filterSingleChannel(const cv::Mat &p) const;

private:
	int r;
	double eps;
	cv::Mat I, mean_I, var_I;
};

class GuidedFilter
{
public:
	GuidedFilter(const cv::Mat &I, int r, double eps);
	~GuidedFilter();

	cv::Mat filter(const cv::Mat &p, int depth = -1) const;

private:
	GuidedFilterImpl *impl_;
};

static cv::Mat guidedFilter(const cv::Mat &I, const cv::Mat &p, int r, double eps, int depth= -1)
{
	return GuidedFilter(I, r, eps).filter(p, depth);
}

