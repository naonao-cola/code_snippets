"""
Copyright (C) 2023 TuringVision

a vision tool  for configuring and performing blob analysis.
The blob run-time parameters include, but are not limited to,
settings for segmentation.
"""
import cv2
import math
import numpy as np
from functools import wraps


__all__ = ['BlobTool']


def _point_distance_less_thresh(point1, point2, thresh=50):
    thresh_square = thresh ** 2
    return (point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2 <= thresh_square


def _cached(f):
    @wraps(f)
    def wrapper(obj):
        cache = obj._cache
        prop = f.__name__

        if not ((prop in cache) and obj._cache_active):
            cache[prop] = f(obj)

        return cache[prop]

    return wrapper


def cal_centroid(mask, offset_x=0, offset_y=0):
    """
    Calculate centroid by blob_image
    :param mask: binary blob image
    :param x0: x axis offset
    :param y0: y axis offset
    :return centriod: tuple type(x_center, y_center)
    """
    indices = np.nonzero(mask)
    coords = np.vstack([indices[1], indices[0]]).T
    cx, cy = coords.mean(axis=0)
    return cx + offset_x, cy + offset_y


class BlobTool:
    """
    Label connected regions of an integer array and Measure properties of labeled image regions
    """

    def __init__(self, blob_image, source_image=None, cache_active=True, min_area=3):
        """
        :param blob_image: binary image(dtype: np.uint8 or np.int8), background is 0, foreground is 1 or 255
        :param source_image: the source gray image for calculate the image of region factor:np.int8
        :param cache_active:
        :param min_area: Arae threshold of blob.
        """
        blob_image = blob_image.astype(np.uint8)
        num, label_image, stats, centroids = cv2.connectedComponentsWithStats(blob_image, connectivity=8)
        self._label_image = label_image
        self._cache_active = cache_active
        self._min_area = min_area
        self._blobs = self._regionprops(num, label_image, stats, centroids, source_image)
        self._cache = {}

    def _regionprops(self, num, label_image,  stats, centroids, source_image=None):
        blobs = []
        for idx in range(1, num, 1):
            x, y, w, h, area = stats[idx]
            if area < self._min_area:
                continue

            centroid = centroids[idx]
            region = label_image[y:y+h, x:x+w] == idx

            intensity_image = None
            if source_image is not None:
                intensity_image = source_image[y:y+h, x:x+w]
            blob = Blob(region, bbox=(x, y, w, h), intensity_image=intensity_image)
            blob.area = area
            blob.centroid = centroid
            blobs.append(blob)
        return blobs

    @property
    @_cached
    def blob_count(self):
        return len(self._blobs)

    @property
    @_cached
    def blobs(self):
        """
        iterate  the region and get the property
        :return: Blob class:list[Blob]
        """
        return self._blobs

    def filter(self, func):
        '''
        filter blob according to custom function
        args:
        -----
            func: function
                have one parameter: Blob object
                return type is bool
                func return True, indicate that Blob object will be reserved, otherwise not reserve
        '''
        old_blobs = self.blobs
        new_blobs = []
        for blob in old_blobs:
            if func(blob):
                new_blobs.append(blob)
        self._cache['blobs'] = new_blobs

    def _merege_blob(self, blobs):
        imgs = []
        bboxes = []
        cnts = []
        area = 0
        perimeter = 0
        hole = 0
        for blob in blobs:
            imgs.append(blob.blob_image)
            x, y, w, h = blob.bbox
            bboxes.append([x, y, x+w, y+h])
            cnts.append(np.array(blob.contour))
            area += blob.area
            perimeter += blob.perimeter
            hole += blob.hole

        ndbox = np.array(bboxes)
        xmin, ymin, _, _ = ndbox.min(axis=0)
        _, _, xmax, ymax = ndbox.max(axis=0)
        new_img = np.zeros((ymax-ymin, xmax-xmin))

        for idx, box in enumerate(ndbox):
            x1, y1, x2, y2 = box
            new_img[y1-ymin:y2-ymin, x1-xmin:x2-xmin] = imgs[idx]

        convex_hull = cv2.convexHull(np.vstack(cnts))

        new_blob = Blob(new_img, bbox=(xmin, ymin, xmax-xmin, ymax-ymin))
        new_blob.area = area
        new_blob.centroid = cal_centroid(new_img, xmin, ymin)
        new_blob.perimeter = perimeter
        new_blob.hole = hole
        new_blob.convex_hull = convex_hull
        new_blob.contour = convex_hull

        return new_blob

    def union(self, func):
        '''
        union blobs according to custom function

        args:
        -----
            func: function, return type is bool
                have two parameters(Blob object),
                when func returns True, two blobs will be merged
                when func returns False, blobs will not be merged
        '''
        old_blobs = self.blobs
        blob_num = len(old_blobs)
        if blob_num <= 1:
            return

        old2new = {}  # collect index of similar blob
        for i in range(blob_num):
            blob1 = old_blobs[i]
            for j in range(i + 1, blob_num):
                blob2 = old_blobs[j]
                if func(blob1, blob2):
                    if i not in old2new:
                        old2new[i] = set()
                    old2new[i].add(j)
                    old2new[i].add(i)
                    old2new[j] = old2new[i]
                else:
                    if i not in old2new:
                        old2new[i] = set()
                    old2new[i].add(i)
                    if j not in old2new:
                        old2new[j] = set()
                    old2new[j].add(j)

        new_blobs_index_list = list(old2new.values())

        self._blobs = []
        red = []
        for i, blob_indexes in enumerate(new_blobs_index_list):
            if blob_indexes not in red:
                red.append(blob_indexes)
                sub_blobs = []
                for ind in blob_indexes:
                    sub_blobs.append(old_blobs[ind])
                self._blobs.append(self._merege_blob(sub_blobs))

        self._cache['blobs'] = self._blobs

    def union_by_dist(self, dist_thresh):
        self.union_by_dist_and_orientation(dist_thresh=dist_thresh)

    def union_by_orientation(self, orientation_thresh):
        self.union_by_dist_and_orientation(
            orientation_thresh=orientation_thresh)

    def union_by_dist_and_orientation(self, dist_thresh=None, orientation_thresh=None):
        if dist_thresh:
            def check_dist(blob1, blob2):
                convex_hull1 = blob1.convex_hull.reshape((-1, 2))
                convex_hull2 = blob2.convex_hull.reshape((-1, 2))

                for pt1 in convex_hull1:
                    for pt2 in convex_hull2:
                        if _point_distance_less_thresh(pt1, pt2, dist_thresh):
                            return True
                return False
        else:
            def check_dist(blob1, blob2):
                return True

        if orientation_thresh:
            def check_orientation(blob1, blob2):
                orientation1 = blob1.orientation
                orientation2 = blob2.orientation

                orientation_dist = abs(orientation1 - orientation2)
                if orientation_dist > math.pi / 2:
                    orientation_dist = math.pi - \
                        abs(orientation1) - abs(orientation2)

                if orientation_dist < orientation_thresh:
                    return True
                return False
        else:
            def check_orientation(blob1, blob2):
                return True

        def blob_is_similar(blob1, blob2):
            if dist_thresh is None and orientation_thresh is None:
                return False

            return check_dist(blob1, blob2) and check_orientation(blob1, blob2)

        self.union(blob_is_similar)


class Blob:
    """
    calculate the blob property,contains: contour,min_bbox,bbox,area,centroid,perimeter,orientation,convex_hull
    convex_area,intensity_image,min_intensity,max_intensity,mean_intensity,moments(cv2),blob_image,
    min_bbox_w,min_bbox_h,min_bbox_angle,bbox_w,bbox_h,hole,
    min_bbox_axis_aspect,bbox_axis_aspect,convexity,rectangularity,roundness,compactness. coords,get_region
    """

    def __init__(self, region, bbox=None, cache_active=True, intensity_image=None):
        self._cache_active = cache_active
        self._cache = {}
        self.region = region
        self._ndim = region.ndim
        self._area = None
        self._centroid = None
        self._perimeter = None
        self._hole = None
        self._contour = None
        self._convex_hull = None

        if bbox is not None:
            self._bbox = bbox
        else:
            self._bbox = (0, 0, region.shape[1], region.shape[0])

        if intensity_image is not None:
            self.intensity_image = intensity_image
            self.max_intensity = np.max(self.intensity_image[self.region], axis=0)
            self.min_intensity = np.mean(self.intensity_image[self.region], axis=0)
            self.mean_intensity = np.min(self.intensity_image[self.region], axis=0)

    @property
    @_cached
    def bbox(self):
        """
        :return bbox: (left, top, width, height)
        """
        return self._bbox

    @property
    def area(self):
        if self._area is None:
            self._area = np.sum(self.region)
        return self._area

    @area.setter
    def area(self, value):
        self._area = value

    @property
    def centroid(self):
        if self._centroid is None:
            self._centroid = cal_centroid(self.region, offset_x=self.bbox[0], offset_y=self.bbox[1])
        return self._centroid

    @centroid.setter
    def centroid(self, value):
        self._centroid = value

    @property
    @_cached
    def orientation(self):
        """
        link: https://en.wikipedia.org/wiki/Image_moment
        spatial moments: 'm00' 'm10' 'm01' ...
        central moments: 'mu20' 'mu11' 'mu02' ...
        Formula:
            orientation = 0.5 * arctan(2 * b / (a - c))
            a = mu20 / m00
            b = mu11 / m00
            c = mu02 / moo
        """
        M = self.moments
        a = int(M["mu20"] / M['m00'])
        b = int(M["mu11"] / M['m00'])
        c = int(M["mu02"] / M['m00'])
        if a - c == 0:
            if b < 0:
                return -math.pi / 4.
            else:
                return math.pi / 4.
        else:
            return 0.5 * math.atan2(2 * b, a - c)

    @property
    def perimeter(self):
        if self._perimeter is None:
            self._perimeter = cv2.arcLength(self.contour, True)
        return self._perimeter

    @perimeter.setter
    def perimeter(self, value):
        self._perimeter = value

    @property
    def convex_hull(self):
        """
        Get blob convex_hull contour
        :return convexhull contour np.array([[x0, y0], [x1, y1], ...])
        """
        if self._convex_hull is None:
            self._convex_hull = cv2.convexHull(self.contour).reshape(-1, 2)
        return self._convex_hull

    @convex_hull.setter
    def convex_hull(self, value):
        self._convex_hull = value

    @property
    @_cached
    def convex_area(self):
        return cv2.contourArea(self.convex_hull)

    @property
    def min_bbox_w(self):
        return self.min_bbox[1][0]

    @property
    def min_bbox_h(self):
        return self.min_bbox[1][1]

    @property
    def min_bbox_angle(self):
        return self.min_bbox[2]

    @property
    def bbox_w(self):
        return self.bbox[2]

    @property
    def bbox_h(self):
        return self.bbox[3]

    @property
    @_cached
    def blob_image(self):
        return self.region * np.uint8(255)

    @property
    def _all_contours(self):
        result = cv2.findContours(self.blob_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE,
                                        offset=(self.bbox[0], self.bbox[1]))
        if len(result) == 3:
            _, contours, _ = result
        else:
            contours, _ = result
        return contours

    @property
    def hole(self):
        if self._hole is None:
            self._hole = len(self._all_contours) - 1
        return self._hole

    @hole.setter
    def hole(self, value):
        self._hole = value

    @property
    def contour(self):
        """
        Get blob contour
        :return contour : np.array([[x0, y0], [x1, y1], ...])
        """
        if self._contour is None:
            self._contour = self._all_contours[0].reshape(-1, 2)
        return self._contour

    @contour.setter
    def contour(self, value):
        self._contour = value

    def get_region(self):
        '''
        :return: blob region:Region
        '''
        from .geometry import Region

        assert self.area >= 3

        return Region.from_polygon(self.contour)

    @property
    @_cached
    def min_bbox(self):
        """
        Get blob min area rectangle
        :return RotatedRect:((center_x, center_y), (width, height), angle)
        """
        return cv2.minAreaRect(self.contour)

    @property
    @_cached
    def moments(self):
        """
        :return cv::Moments:
            Public Attributes
                spatial moments
                    double 	m00
                    double 	m10
                    double 	m01
                    double 	m20
                    double 	m11
                    double 	m02
                    double 	m30
                    double 	m21
                    double 	m12
                    double 	m03
                central moments
                    double 	mu20
                    double 	mu11
                    double 	mu02
                    double 	mu30
                    double 	mu21
                    double 	mu12
                    double 	mu03
                central normalized moments
                    double 	nu20
                    double 	nu11
                    double 	nu02
                    double 	nu30
                    double 	nu21
                    double 	nu12
                    double 	nu03
        """
        return cv2.moments(self.blob_image)

    @property
    @_cached
    def min_bbox_axis_aspect(self):
        """
        The ratio of height to width of the smallest rectangle that completely encloses
        the blob and is aligned with the blob's principal axis.
        :return: float
        """
        if self.min_bbox_h == 0:
            return 1.0
        return self.min_bbox_w / self.min_bbox_h

    @property
    @_cached
    def bbox_axis_aspect(self):
        """
        The ratio of height to width of the smallest rectangle that completely encloses
        the blob and is aligned with the angle specified by the bbox property.
        :return:
        """
        if self.bbox_h == 0:
            return 1.0
        return self.bbox_w / self.bbox_h

    @property
    @_cached
    def convexity(self):
        """
        Shape factor for the convexity of a region
        :return: float
        """
        return self.area / self.convex_area

    @property
    @_cached
    def rectangularity(self):
        """
        Shape factor for the rectangularity of a region
        :return: float
        """
        if self.min_bbox_w == 0 or self.min_bbox_h == 0:
            return 1.0
        return self.area / (self.min_bbox_w * self.min_bbox_h)

    @property
    @_cached
    def roundness(self):
        """
        Shape factors from contour
        :return: float
        """
        contour = np.squeeze(self.contour)
        if len(contour) < 3:
            return 1.0
        dis_list = ((self.centroid[0] - contour[:, 0]) ** 2 + (self.centroid[1] - contour[:, 1]) ** 2) ** 0.5
        avg_dis = np.sum(dis_list) / len(dis_list)
        sigma = np.sum((np.array(dis_list) - avg_dis) ** 2) / len(dis_list)
        if avg_dis == 0:
            roundness = 1.0
        else:
            roundness = 1 - (sigma ** 0.5) / avg_dis
        return roundness

    @property
    @_cached
    def compactness(self):
        """
        Shape factor for the compactness of a region.
        :return: float
        """
        return self.perimeter ** 2 / (self.area * 4 * math.pi)

    @property
    @_cached
    def coords(self):
        return self.contour