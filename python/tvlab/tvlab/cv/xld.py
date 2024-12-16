"""
Copyright (C) 2023 TuringVision

a vision tool  for configuring, filtering, smoothing and calculating defect distance
 based on eXtendedLineDescriptions(XLD).
"""
import numpy as np
import warnings
import cv2
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

__all__ = ['Xld']


def _poly_fit(x, y, deg):
    '''
    Fit by polynomial

    Parameters
    ----------
        x : array_like, shape (M,)
            x-coordinates of the M sample points ``(x[i], y[i])``.
        y : array_like, shape (M,) or (M, K)
            y-coordinates of the sample points. Several data sets of sample
            points sharing the same x-coordinates can be fitted at once by
            passing in a 2D-array that contains one dataset per column.
        deg : int
            Degree of the fitting polynomial
    '''
    coef = np.polyfit(x, y, deg)
    poly_fit = np.poly1d(coef)
    y_fitted = poly_fit(x)
    return x, y_fitted


class LineBlob(list):
    def __init__(self, pts):
        '''
        get one xld contour property

        Parameters
        ----------
            pts:  type: list
                  format: [(x,y), ..., (x,y)]
        '''
        super().__init__(pts)

    def is_open(self):
        '''
        Determine whether the xld contour is open

        format:
            closed   open
             __
            |__|     |__|

        Returns：bool
        '''
        head, tail = self[0], self[-1]
        return head != tail

    def is_closed(self):
        '''
            Determine whether the xld contour is open

        Returns：bool
        '''
        return not self.is_open()

    def smooth(self, deg=1, pointsnum_per_section=5):
        '''
        smooth by piecewise

        Parameters
        ----------
            deg : int, default：1
                  Degree of the fitting polynomial
            pointsnum_per_section: int, default: 5
                  points of Number about per piecewise section
                  the larger the parameter setting, the worse the fit
        Returns：
                type: list
                format: [(fit_x1, fit_y1), ....(fit_xn, fit_yn)]
        '''
        x = [self[i][0] for i in range(len(self))]
        y = [self[i][1] for i in range(len(self))]
        final_x, final_y_fitted = [], []

        if len(x) < pointsnum_per_section:
            x, y_fitted = _poly_fit(x, y, deg)
            final_x.extend(x)
            final_y_fitted.extend(y_fitted)

        else:
            segment_x = [x[i:i + pointsnum_per_section]
                         for i in range(0, len(x), pointsnum_per_section)]
            segment_y = [y[i:i + pointsnum_per_section]
                         for i in range(0, len(y), pointsnum_per_section)]

            for i in range(len(segment_x)):
                tmp_x, tmp_y = segment_x[i], segment_y[i]
                x, y_fitted = _poly_fit(tmp_x, tmp_y, deg)
                final_x.extend(x)
                final_y_fitted.extend(y_fitted)

        return [(final_x[i], final_y_fitted[i]) for i in range(len(self))]

    def distance(self, line_blob):
        '''
        calcuate the defect distance by Euclidean distance

        Parameters
        ----------
            line_blob : list
                   format: [(x,y), ..., (x,y)]

        Returns：
                type: list
                format: [dist_1, dist_2....dist_n]
        '''
        assert len(self) == len(line_blob)

        defect_distance = []
        x1_list = [self[i][0] for i in range(len(self))]
        y1_list = [self[i][1] for i in range(len(self))]
        x2_list = [line_blob[i][0] for i in range(len(line_blob))]
        y2_list = [line_blob[i][1] for i in range(len(line_blob))]

        for i in range(len(self)):
            x1, y1 = x1_list[i], y1_list[i]
            x2, y2 = x2_list[i], y2_list[i]
            tmp_distance = np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
            defect_distance.append(tmp_distance)

        return defect_distance

    def union(self, line_blob, distance_threshold=20):
        '''
        union tow xld contours by the endpoints coordinates minimum distance

        Parameters
        ----------
            line_blob : list
                   format: [(x,y), ..., (x,y)]

            distance_threshold: int,float, default=20

        '''

        pts1_coordinate = [self[0], self[-1]]
        pts2_coordinate = [line_blob[0], line_blob[-1]]

        min_distance_list = []
        for i in range(len(pts1_coordinate)):
            for j in range(len(pts2_coordinate)):
                x1, y1 = pts1_coordinate[i][0], pts1_coordinate[i][1]
                x2, y2 = pts2_coordinate[j][0], pts2_coordinate[j][1]
                tmp_distance = np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
                min_distance_list.append(tmp_distance)

        min_distance = min(min_distance_list)
        min_distance_index = min_distance_list.index(min(min_distance_list))

        if min_distance < distance_threshold:
            # contour1, contour2
            # 0  express  connect contour1_head to contour2_head
            # 1  express  connect contour1_head to contour2_tail
            # 2  express  connect contour1_tail to contour2_head
            # 3  express  connect contour1_tail to contour2_tail
            if min_distance_index == 0:
                # print('contour1_head - contour2_head')
                pts1 = self[::-1]
                pts1.extend(line_blob)
                return pts1
            elif min_distance_index == 1:
                # print('contour1_head - contour2_tail')
                pts1 = self[::-1]
                pts2 = line_blob[::-1]
                pts1.extend(pts2)
                return pts1
            elif min_distance_index == 2:
                # print('contour1_tail - contour2_head')
                self.extend(line_blob)
                return self
            elif min_distance_index == 3:
                # print('contour1_tail - contour2_tail')
                pts2 = line_blob[::-1]
                self.extend(pts2)
                return self
            else:
                return []
        else:
            return self, line_blob


class Xld(list):
    """
    XLD: eXtended Line Descriptions
    the xld tool compare the origin contours after Canny operator processing
    and the smooth contours,then find the maximum distance between two points
    """

    def __init__(self, subpixel_group):
        super().__init__([LineBlob(line) for line in subpixel_group])

    def show(self, img=None, marker='+', figsize=None, markersize=1, linewidth=0.03):
        '''

        :param self:  xld object
        :param marker: format: 'o'
        :param img: format: gray image
        :param figsize: format:[10, 10]
        :param markersize: set the size of marker, format: 1 or 12...
        :param linewidth: set the width of line, format: 0.1 or 1...
        :return:
        '''
        plt.figure(figsize=figsize)
        if img is not None:
            plt.imshow(img, cmap='gray')
        for i, item in enumerate(self):
            color = plt.cm.Set1(i)
            x_coord, y_coord = [], []
            for xy in item:
                x_coord.append(xy[0])
                y_coord.append(xy[1])
            plt.plot(x_coord, y_coord, '-', color=color, linewidth=linewidth)
            if marker != '-':
                plt.plot(x_coord, y_coord, marker,
                         color=color, markersize=markersize)

        plt.axis('equal')
        plt.show()

    @classmethod
    def show_multi(cls, xlds, img=None, markers=None, colors=None,  figsize=None, markersize=1, linewidth=0.03):
        '''

        :param cls:
        :param xlds: format: [xld1, xld2, ...]
        :param colors: format: ['r', 'g']
        :param markers: format: ['.', 'o']
        :param img: format: gray image
        :param figsize: format:[10, 10]
        :param markersize: set the size of marker, format: 1 or 12...
        :param linewidth: set the width of line, format: 0.1 or 1...
        :return:

        :references:
            **Markers**

            =============    ===============================
            character        description
            =============    ===============================
            ``'.'``          point marker
            ``','``          pixel marker
            ``'o'``          circle marker
            ``'v'``          triangle_down marker
            ``'^'``          triangle_up marker
            ``'<'``          triangle_left marker
            ``'>'``          triangle_right marker
            ``'1'``          tri_down marker
            ``'2'``          tri_up marker
            ``'3'``          tri_left marker
            ``'4'``          tri_right marker
            ``'s'``          square marker
            ``'p'``          pentagon marker
            ``'*'``          star marker
            ``'h'``          hexagon1 marker
            ``'H'``          hexagon2 marker
            ``'+'``          plus marker
            ``'x'``          x marker
            ``'D'``          diamond marker
            ``'d'``          thin_diamond marker
            ``'|'``          vline marker
            ``'_'``          hline marker
            =============    ===============================

                **Colors**

            The supported color abbreviations are the single letter codes

            =============    ===============================
            character        color
            =============    ===============================
            ``'b'``          blue
            ``'g'``          green
            ``'r'``          red
            ``'c'``          cyan
            ``'m'``          magenta
            ``'y'``          yellow
            ``'k'``          black
            ``'w'``          white
            =============    ===============================

        '''
        default_markers = ['.', '+', 'o', 'v', '^', '<', '>', '*']
        default_colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']

        nums_xld = len(xlds)
        if colors is None:
            if nums_xld < len(default_colors):
                colors = default_colors[:nums_xld]
            else:
                raise IndexError('xld list index out of default_colors range')

        if markers is None:
            if nums_xld < len(default_markers):
                markers = default_markers[:nums_xld]
            else:
                raise IndexError('xld list index out of default_markers range')

        plt.figure(figsize=figsize)
        if img is not None:
            plt.imshow(img, cmap='gray')

        for i, xld in enumerate(xlds):
            for item in xld:
                x_coord, y_coord = [], []
                for xy in item:
                    x_coord.append(xy[0])
                    y_coord.append(xy[1])
                plt.plot(x_coord, y_coord, '-',
                         color=colors[i], linewidth=linewidth)
                plt.plot(x_coord, y_coord,
                         markers[i], color=colors[i], markersize=markersize)
        plt.axis('equal')
        plt.show()

    @classmethod
    def from_img(cls, img, sigma=0.0, th_l=0.0, th_h=0.0):
        '''
        Implementation of Canny/Devernay's sub-pixel edge detector.

        ref doc: https://iie.fing.edu.uy/publicaciones/2017/GR17a/GR17a.pdf
        paper title:    A Sub-Pixel Edge Detector: an Implementation of the Canny/Devernay
            Algorithm" by Rafael Grompone von Gioi and Gregory Randall

        Parameters
        ----------
            img: the input image
            sigma: standard deviation sigma for the Gaussian filtering
                    (if sigma=0 no filtering is performed)  Typical range of values: 0 ≤ sigma ≤ 3
            th_h: high gradient threshold in Canny's hysteresis  Typical range of values: 0 ≤ th_h ≤ 50
            th_l: low gradient threshold in Canny's hysteresis  Typical range of values: 0 ≤ th_l ≤ 50

        Returns：
            Extracted edges information
            subpixel_group：lists of sub-pixel coordinates of edge points
        '''
        from .impl.cdevernay import c_devernay
        assert len(img.shape) == 2

        if img.dtype != np.double:
            img = np.double(img)
        subpixel_group = c_devernay(img, sigma, th_l, th_h)

        return cls(subpixel_group)

    def filter(self, func):
        '''
        filter xld contours according to custom function

        Parameters
        ----------
            func: function
                have one parameter: xld object
                return type is bool
                func return True, the xld_contour will be reserved, else not reserve

        e.g.    xld1 = Xld.from_img(gray)
                xld1.filter.(lambda item: len(item) > 40)
        '''
        return Xld([item for item in self if func(item)])

    def filter_by_closed(self):
        return Xld([lineb for lineb in self if LineBlob(lineb).is_closed()])

    def filter_by_open(self):
        return Xld([lineb for lineb in self if LineBlob(lineb).is_open()])

    def smooth(self, deg=1, pointsnum_per_section=5):
        '''
        smooth all xld contours according to piecewise Least squares polynomial fit.

        it should be noted that the amount of data processed by polynomial fitting has
        certain limitations. generally, the data of higher-order polynomial fitting is
        more accurate. however, when there is too much data, you can try multiple fitting
        and splicing.

        in the function, we try to multiple fitting and splicing.

        Parameters
        ----------
            deg : int, default：1
                  Degree of the fitting polynomial

            pointsnum_per_section: int, default: 5
                  points of Number about per piecewise section
                  the larger the parameter setting, the worse the fit

        Returns：
                type: list
                format: [ [(fit1_x1, fit1_y1), ....(fit1_xn, fit1_yn)],
                            ...
                          [(fitn_x1, fitn_y1), ....(fitn_xn, fitn_yn)]
                        ]

        e.g.    xld1 = Xld.from_img(gray)
                xld1.smooth()
        '''
        return Xld([LineBlob(lineb).smooth(deg, pointsnum_per_section) for lineb in self])

    def distance(self, xld):
        '''
        calculate the defect distance  by Euclidean distance

        it should be noted that the function is a specific application scenaroi.
        which is mainly used to calculate the distance between the Xld contour
        before and after smoothing.
        if you want to use the function, you need to keep the length between the
        input xld contours set and the origin xld contours set equal.

        Parameters
        ----------
            xld : one instance of Xld()

        Returns：
                type: list
                format: [ [dist_11, dist_12, ...],
                            ...
                          [dist_n1, dist_n2, ...]
                        ]

        e.g.    xld1 = Xld.from_img(gray)
                xld1.distance(xld0)
        '''
        assert len(self) == len(xld)
        return [LineBlob(self[i]).distance(xld[i]) for i in range(len(self))]
