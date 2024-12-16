"""
Copyright (C) 2023 TuringVision
Image find point,find line,find circle interface
"""

__all__ = ['CaliperTool']


class CaliperTool:
    DARKTOLIGHT = -1
    LIGHTTODARK = 1
    NOCARE = 0

    PROJECT_MEAN = 0
    PROJECT_MIN = 1
    PROJECT_MAX = 2
    PROJECT_MEDIAN = 3

    @classmethod
    def find_point(cls, image, xywhr,
                   filter_size=2,
                   threshold=60,
                   polarity=NOCARE,
                   sub_pix=False,
                   project_mode=PROJECT_MEAN,
                   debug=False):
        """
        :param sub_pix: the result point is sub pixel:bool
        :param image: the  image for processing : np.array
        :param xywhr(x,y,w,h,r)

        eg1：x,y,w,h,r = 5, 8, 6, 3, -60

            O--------> x
            |
            |     1\ h
            |    /  2      /                /| (search direction)
            |   /E /      /                /
            |  0  / w    / r = -60        /
            |   `3      -------          /
            |
            v y


        eg2：x,y,w,h,r = 5, 8, 3, 6, 30

            O--------> x
            |
            |     0\ w
            |    /  1                     \
            |   /E /                       \
            |  3  / h                       \
            |   `2      ---------            \| (search direction)
            |            \  r = 30
            |              \
            v y

        affine rectangle from its origin point :(x,y)
        w: The length of side x (the side along the x-axis) :int
        h: The length of side y (the side along the y-axis) :int
        r: The The angle of rotation expressed in terms of angles :float
        :param filter_size:  In general, specify a filter half size that is approximately
        equal to one half of the edge width. :int
        :param threshold:Minimum contrast required for an edge to be considered during
        the scoring phase
        :param polarity:The desired polarity of first edge, one of
                - CaliperTool.DARKTOLIGHT
                - CaliperTool.LIGHTTODARK
                - CaliperTool.NOCARE
        :param project_mode: The project mode, one of
                - CaliperTool.PROJECT_MEAN
                - CaliperTool.PROJECT_MIN
                - CaliperTool.PROJECT_MAX
                - CaliperTool.PROJECT_MEDIAN
        :param debug: if debug is True show the roi and result
        """
        from .caliper import Caliper
        caliper1 = Caliper(sub_pix=sub_pix, debug=debug)
        caliper1.filter_size = filter_size
        caliper1.threshold = threshold
        caliper1.polarity = polarity
        caliper1.project_mode = project_mode
        find, xy = caliper1.run(image, xywhr)
        if not find:
            xy = None
        return xy

    @classmethod
    def find_line(cls, image, line,
                  filter_size=2,
                  threshold=60,
                  polarity=NOCARE,
                  direction=0,
                  num_caliper=100,
                  side_x_length=100,
                  side_y_length=10,
                  filter_num_point=1,
                  sub_pix=False,
                  return_ext_points=False,
                  project_mode=PROJECT_MEAN,
                  debug=False):
        """
        :param return_ext_points: if True return the all found point
        :param sub_pix: the result point is sub pixel:bool
        :param filter_size:  In general, specify a filter half size that is approximately
        equal to one half of the edge width. :int
        :param threshold:Minimum contrast required for an edge to be considered during
        the scoring phase
        :param polarity:The desired polarity of first edge, one of
                - CaliperTool.DARKTOLIGHT
                - CaliperToolLIGHTTODARK
                - CaliperTool.NOCARE
        :param direction:   1 rotate 180° ,0 keep angle:int:
        :param num_caliper: using num caliper tool fitline or fitcircle :int
        :param side_x_length:The length of side x (the side along the x-axis) :int
        :param side_y_length:The length of side y (the side along the y-axis) :int

        eg1：line = ((2, 10), (20, 10)), num_caliper = 4
             side_x_length=7, side_y_length=2, direction=0
                         \      |
            O--------> x   \    |               |
            |     _      _   \  _      _        |
            |    | |    | |   \| |    | |       |
            |    | |    | |    | |    | |       |
            |  s ------------------------> e    |
            |    | |    | |    | |    | |       V
            |    |_|    |_|    |_|    |_|     search direction
            |
            v y

        :param filter_num_point: The number of points that will be ignored in
        the fitting operation. :int
        :param project_mode: The project mode, one of
                - CaliperTool.PROJECT_MEAN
                - CaliperTool.PROJECT_MIN
                - CaliperTool.PROJECT_MAX
                - CaliperTool.PROJECT_MEDIAN
        :param debug: if debug is True show the roi and result
        :param image: the  image for processing : np.array
        :param line: Line object
        :return: Line: Line object,if return_ext_points is True
        return contains list[ caliper_result]:
        caliper_result = {'find': find, "point": point,
                              'roi': pts1,
                              'disptol': [],
                              'used': find}
        """
        from .findline import FindLine
        param = (
            filter_size, threshold, polarity, direction, num_caliper, side_x_length, side_y_length,
            filter_num_point)
        find_line = FindLine(sub_pix=sub_pix, debug=debug, project_mode=project_mode)
        line = find_line.run(image, line, param, return_ext_points)
        return line

    @classmethod
    def find_circle(cls, image, roi,
                    filter_size=2,
                    threshold=60,
                    polarity=NOCARE,
                    direction=0,
                    num_caliper=100,
                    side_x_length=100,
                    side_y_length=10,
                    filter_num_point=1,
                    sub_pix=False,
                    return_ext_points=False,
                    project_mode=PROJECT_MEAN,
                    debug=False):
        """
        :param return_ext_points: if True return the all found point
        :param sub_pix: the result point is sub pixel:bool
        :param filter_size:  In general, specify a filter half size that is approximately
        equal to one half of the edge width. :int
        :param threshold:Minimum contrast required for an edge to be considered during
        the scoring phase
        :param polarity:The desired polarity of first edge, one of
                - CaliperTool.DARKTOLIGHT
                - CaliperToolLIGHTTODARK
                - CaliperTool.NOCARE
        :param direction:   1 rotate 180° ,0 keep angle:int:
        :param num_caliper: using num caliper tool fitline or fitcircle :int
        :param side_x_length:The length of side x (the side along the x-axis) :int
        :param side_y_length:The length of side y (the side along the y-axis) :int
        :param filter_num_point: The number of points that will be ignored in
        the fitting operation. :int
        :param project_mode: The project mode, one of
                - CaliperTool.PROJECT_MEAN
                - CaliperTool.PROJECT_MIN
                - CaliperTool.PROJECT_MAX
                - CaliperTool.PROJECT_MEDIAN
        :param debug: if debug is True show the roi and result
        :param image: input_image for findcircle :np.array
        :param roi: x, y, r, start_angle, range_angle

        eg1：roi = (10, 10, 5, 0, 180), num_caliper=2
             side_x_length = 8, side_y_length=2, direction=0

            O--------> x
            |              --
            |             |  |           |
            |          ---------         |
            |        /    |  |   \       | (search direction)
            |    -- |--    --     |      |
            |   |   |  |   ....r..|      V
            |    -- |--  (x,y)    |
            |        \           /
            |          ---------
            v y

        :return: center_x, center_y, radius ,if return_ext_points is True
        return contains list[ caliper_result]:
        caliper_result = {'find': find, "point": point,
                              'roi': pts1,
                              'disptoc': [],
                              'used': find}
        """
        from .findcircle import FindCircle
        param = (
            filter_size, threshold, polarity, direction, num_caliper, side_x_length, side_y_length,
            filter_num_point)
        find_circle = FindCircle(sub_pix=sub_pix, debug=debug, project_mode=project_mode)
        circle = find_circle.run(image, roi, param, return_ext_points)
        return circle
