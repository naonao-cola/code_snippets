"""
Copyright (C) 2023 TuringVision

a vision tool that runs a series of Caliper tools over a specified,
circular region of an image to locate multiple edge points, supplies
these edge points to an underlying Fit Circle tool, and ultimately
returns the circle that best fits those input points .
"""
import cv2
from ..geometry import gen_rotate_rect
from .caliper import *
import numpy as np
from math import *
import matplotlib.pyplot as plt

__all__ = ['FindCircle']


class FindCircle:
    """
    a vision tool that runs a series of Caliper tools over a specified,
    circular region of an image to locate multiple edge points, supplies
    these edge points to an underlying Fit Circle tool, and ultimately
    returns the circle that best fits those input points .
    """

    def __init__(self, sub_pix=False, debug=False, project_mode=ProjectMode.PROJECT_MEAN):
        """
        :param debug: if debug is True will show debug image :bool
        :param sub_pix. the result point is sub pixel:bool
        """
        self.debug = debug
        self.sub_pix = sub_pix
        self.project_mode = project_mode

    @staticmethod
    def _show_debug(input_image, result_calipers, circle, radian_param,
                    direction):
        if len(input_image.shape) == 3:
            output_image = input_image.copy()
        else:
            output_image = cv2.cvtColor(input_image, cv2.COLOR_GRAY2BGR)
        fig = plt.figure()
        ax = fig.add_subplot()
        center_x, center_y, radius, angle, angel_range = radian_param
        pts1 = gen_rotate_rect((center_x, center_y, radius * 2, 1, angle))
        s1 = (pts1[0][0], pts1[0][1])
        e1 = (center_x, center_y)
        if not direction:
            s1, e1 = e1, s1
        ax.annotate('', e1, s1, color='blue',
                    arrowprops=dict(facecolor='red', width=2, headwidth=6))

        circ0 = plt.Circle((center_x, center_y), radius, color='limegreen', fill=False)
        ax.add_patch(circ0)

        center_x, center_y, radius = 0, 0, 0
        if circle is not None:
            center_x, center_y, radius = circle

        for caliper_result in result_calipers:
            rect = cv2.minAreaRect(caliper_result['roi'])
            box = cv2.boxPoints(rect)
            if caliper_result['used']:
                ax.scatter([caliper_result['point'][0]], [caliper_result['point'][1]],
                           c='limegreen', marker='+')
                poly = plt.Polygon(box, fill=False, color='limegreen', linewidth=1)
                ax.add_patch(poly)
            else:
                if caliper_result['find']:
                    plt.scatter([caliper_result['point'][0]], [caliper_result['point'][1]], c='red',
                                marker='+')
                poly = plt.Polygon(box, fill=False, color='limegreen', linewidth=1)
                ax.add_patch(poly)
        circ1 = plt.Circle((center_x, center_y), radius, color='b', fill=False)
        ax.add_patch(circ1)
        plt.scatter([center_x], [center_y], c='b',
                    marker='+')
        plt.imshow(output_image)
        plt.title('Result')
        plt.show()

    def run(self, input_image, radian_param, caliper_param, return_ext_points):
        """
        :param return_ext_points:
        :param input_image: input_image for findcircle :np.array
        :param radian_param:  param=center_x, center_y, radius,
        start_angle, range_angle
        :param caliper_param: look for the set_caliper_param() method
        :return: center_x, center_y, radius
        """
        (filter_size, threshold, polarity, direction, num_caliper, side_x_length,
         side_y_length, filter_num_point) = caliper_param

        from .impl.c_caliper import c_find_circle

        if len(input_image.shape) == 3:
            gray_img = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)
        else:
            gray_img = input_image

        ret = c_find_circle(gray_img, radian_param, self.project_mode,
                filter_size, polarity, threshold, self.sub_pix, direction, num_caliper,
                side_x_length, side_y_length, filter_num_point, return_ext_points or self.debug)
        if self.debug:
            circle, result_calipers = ret
            if not return_ext_points:
                ret = circle
            self._show_debug(input_image, result_calipers, circle, radian_param, direction)
        return ret
