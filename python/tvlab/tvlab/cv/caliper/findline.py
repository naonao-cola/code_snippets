"""
Copyright (C) 2023 TuringVision

a vision tool that runs a series of Caliper tools over a specified region
of an image to locate multiple edge points, supplies these edge points
to an underlying Fit Line tool, and ultimately returns the line that
best fits those input points.
"""
import cv2

from ..geometry import Line, gen_rotate_rect
from .caliper import *
import numpy as np
from math import *
import matplotlib.pyplot as plt

__all__ = ['FindLine']


def cal_angle(start_x, start_y, end_x, end_y):
    """
    calculate the line angle
    :param start_x:  start_x of line
    :param start_y: start_y of line
    :param end_x: end_x of line
    :param end_y: end_y of line
    :return:  the line ,s angle
    """
    import math
    angle = (90 + math.atan2((end_y - start_y), (end_x - start_x)) * 180 / math.pi)
    if angle > 180:
        angle = angle - 360
    return angle


class FindLine:
    """
    a vision tool that runs a series of Caliper tools over a specified region
    of an image to locate multiple edge points, supplies these edge points
    to an underlying Fit Line tool, and ultimately returns the line that
    best fits those input points.
    """

    def __init__(self, sub_pix, debug, project_mode=ProjectMode.PROJECT_MEAN):
        """
        :param debug: if debug is True will show debug image :bool
        :param sub_pix. the result point is sub pixel:bool
        """
        self.debug = debug
        self.sub_pix = sub_pix
        self.project_mode = project_mode

    @staticmethod
    def _show_debug(input_image, result_calipers, input_line, result_line, rotate, angle,
                    side_x_length):
        if len(input_image.shape) == 3:
            output_image = input_image.copy()
        else:
            output_image = cv2.cvtColor(input_image, cv2.COLOR_GRAY2BGR)
        fig = plt.figure()
        ax = fig.add_subplot()
        for caliper_result in result_calipers:
            rect = cv2.minAreaRect(caliper_result['roi'])
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            if caliper_result['used']:
                ax.scatter([caliper_result['point'][0]], [caliper_result['point'][1]],
                           c='limegreen', marker='+')
                # cv2.drawContours(image, [box], 0, (0, 255, 0), 1)
                poly = plt.Polygon(box, fill=False, color='limegreen', linewidth=1)
                ax.add_patch(poly)
            else:
                if caliper_result['find']:
                    plt.scatter([caliper_result['point'][0]], [caliper_result['point'][1]], c='red',
                                marker='+')
                poly = plt.Polygon(box, fill=False, color='red', linewidth=1)
                ax.add_patch(poly)

        start_xy, end_xy = input_line.to_xy()
        ax.annotate('', end_xy, start_xy, color='limegreen',
                    arrowprops=dict(facecolor='limegreen', width=2, headwidth=6))
        mid_x = (start_xy[0] + end_xy[0]) / 2
        mid_y = (start_xy[1] + end_xy[1]) / 2

        pts1_dir = gen_rotate_rect((mid_x, mid_y, side_x_length, 1, angle))
        pts1_dir_n = np.int0(pts1_dir)
        s1_dir = (pts1_dir_n[1][0], pts1_dir_n[1][1])
        e1_dir = (pts1_dir_n[0][0], pts1_dir_n[0][1])
        if rotate == 1:
            ax.annotate('', e1_dir, s1_dir, color='red',
                        arrowprops=dict(facecolor='red', width=2, headwidth=6))
        else:
            ax.annotate('', s1_dir, e1_dir, color='red',
                        arrowprops=dict(facecolor='red', width=2, headwidth=6))

        if result_line is not None:
            xy = result_line.coords()
            ax.plot(xy[:, 0], xy[:, 1], c='blue')
        ax.imshow(output_image)
        ax.set_title('Result')
        plt.show()

    def run(self, input_image, line, caliper_param, return_ext_points):
        """
        :param return_ext_points: if True return the
        :param input_image: inputimage for findline :np.array
        :param line: Line object
        :param caliper_param: look for the set_caliper_param() method
        :return: line: Line object
        """
        [[start_x, start_y], [end_x, end_y]] = line.coords()
        (filter_size, threshold, polarity, direction, num_caliper, side_x_length,
         side_y_length, filter_num_point) = caliper_param

        from .impl.c_caliper import c_find_line

        if len(input_image.shape) == 3:
            gray_img = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)
        else:
            gray_img = input_image

        ret = c_find_line(gray_img, (start_x, start_y, end_x, end_y), self.project_mode,
                filter_size, polarity, threshold, self.sub_pix, direction, num_caliper,
                side_x_length, side_y_length, filter_num_point, return_ext_points or self.debug)
        if self.debug:
            result_line, result_calipers = ret
            if not return_ext_points:
                ret = result_line
            angle = cal_angle(start_x, start_y, end_x, end_y)
            self._show_debug(input_image, result_calipers, line, result_line, direction, angle,
                             side_x_length)
        return ret
