"""
Copyright (C) 2023 TuringVision

a vision tool that offers rapid and precise pattern detection
and location within a well-defined area of an image.
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt

__all__ = ['FindEdgePolarity', 'Caliper', 'ProjectMode']


class FindEdgePolarity:
    """
    The desired polarity of first edge. Specify DarkToLight for a dark-to-light edge,
    LightToDark for a light-to-dark edge, and DontCare to accept any edge.
    """
    DARKTOLIGHT = -1
    LIGHTTODARK = 1
    NOCARE = 0


class ProjectMode:
    PROJECT_MEAN = 0
    PROJECT_MIN = 1
    PROJECT_MAX = 2
    PROJECT_MEDIAN = 3


class Caliper:
    """
     a vision tool that offers rapid and precise pattern detection
     and location within a well-defined area of an image
    """

    def __init__(self, filter_size=2, threshold=5, polarity=FindEdgePolarity.NOCARE, debug=False,
                 sub_pix=False, project_mode=ProjectMode.PROJECT_MEAN):
        """
        :param filter_size: In general, specify a filter half size that is approximately
        equal to one half of the edge width.
        :param threshold: Minimum contrast required for an edge to be considered during
        the scoring phase
        :param polarity: The desired polarity of first edge
        :param debug:  if  debug  show the roi and result
        :param sub_pix. the result point is sub pixel:bool
        :param project_mode: The project mode, one of
                - PROJECT_MEAN
                - PROJECT_MIN
                - PROJECT_MAX
                - PROJECT_MEDIAN
        """
        self.filter_size = filter_size
        self.threshold = threshold
        self.polarity = polarity
        self.debug = debug
        self.sub_pix = sub_pix
        self.project_mode = project_mode

    def _show_debug(self, point, region, projection, gradient, find_index, input_image,
                    affine_img, pts1):

        fig = plt.figure()
        ax = fig.add_subplot(1, 2, 1)
        ax1 = fig.add_subplot(1, 2, 2)
        x, y, side_x_length, side_y_Length, rotation = region
        rect = cv2.minAreaRect(pts1)
        box = cv2.boxPoints(rect)
        s1 = pts1[1]
        e1 = pts1[0]
        ax.annotate('', s1, e1, color='red',
                    arrowprops=dict(facecolor='red', width=2, headwidth=6))
        color_img = cv2.cvtColor(affine_img, cv2.COLOR_GRAY2RGB)

        # the image of contain __affine_img,__projection_img,__gradient_img
        hs_img = None
        if hs_img is not None:
            hs_img = np.vstack((hs_img, color_img))
        else:
            hs_img = color_img

        projection_img = np.zeros((256, int(side_x_length), 3), dtype='uint8')
        for x in range(len(projection) - 1):
            y = 255 - projection[x]
            ax1.plot([x, x + 1], [y, 255 - projection[x + 1]], c='limegreen')
        hs_img = np.vstack((projection_img, hs_img))

        _max = int(gradient.max())
        _min = int(gradient.min())
        h, w = hs_img.shape[0:2]
        max_min = _max - _min
        if abs(_max) > abs(_min):
            max_h = _max
        else:
            max_h = _min
        if max_min != 0:
            norm_gradient = np.int0((gradient + max_h) / (max_h * 2) * 255)
            gradient_img = np.zeros((256, int(side_x_length), 3), dtype='uint8')
            for x in range(len(norm_gradient) - 1):
                y = norm_gradient[x]
                ax1.plot([x, x + 1], [y + h, norm_gradient[x + 1] + h], c='limegreen')
            ax1.plot([0, side_x_length],
                     [(self.threshold + max_h) / (max_h * 2) * 255 + h,
                      (self.threshold + max_h) / (max_h * 2) * 255 + h],
                     c='blue')
            ax1.plot([0, side_x_length],
                     [(-self.threshold + max_h) / (max_h * 2) * 255 + h,
                      (-self.threshold + max_h) / (max_h * 2) * 255 + h],
                     c='blue')
            hs_img = np.vstack((hs_img, gradient_img))
        h, w = hs_img.shape[0:2]

        ax1.plot([find_index, find_index], [0, h], c='red')
        poly = plt.Polygon(box, fill=False, color='limegreen', linewidth=1)
        ax.add_patch(poly)
        ax.imshow(input_image, cmap="gray"), ax.set_title('Result')
        if point is not None:
            ax.scatter([point[0]], [point[1]], marker='+', c='limegreen')
        ax1.imshow(hs_img, cmap="gray")
        ax1.set_title('hs_image')
        plt.show()

    def run(self, image, region):
        """
        run the method by input image (only support 1 channel image)
        :param image: the  image for processing : np.array (uint8)
        :param region:xywhr(x,y,w,h,r)
        affine rectangle from its origin point :(x,y)
        w: The length of side x (the side along the x-axis) :int
        h: The length of side y (the side along the y-axis) :int
        r: The The angle of rotation expressed in terms of angles :float
        :return:(find,(x,y)):find :if find the edge point is true:bool
        (x,y):a 2-D position of this result in the input image:(x,y) float
        """
        from .impl.c_caliper import c_find_point
        ret = c_find_point(image, region, self.project_mode, self.filter_size,
                self.polarity, self.threshold, self.sub_pix, self.debug)
        if self.debug:
            find, point, debug_info = ret
            ret = (find, point)
            pts1, affine_img, projection, gradient, find_index = debug_info
            self._show_debug(point, region, projection, gradient, find_index, image,
                             affine_img, pts1)
        return ret
