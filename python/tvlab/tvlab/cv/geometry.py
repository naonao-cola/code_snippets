"""
Copyright (C) 2023 TuringVision
"""
from math import *
import numpy as np
import cv2

__all__ = ['Point', 'Line', 'Region']

def gen_rotate_rect(xywhr):
    """

    :param xywhr: xywhr(x,y,w,h,r) affine rectangle from its origin point :(x,y)
     w: The length of side x (the side along the x-axis) :int
     h: The length of side y (the side along the y-axis) :int
     r: The The angle of rotation expressed in terms of angles :float
    :return:  pts1 :the affine point :np.array (x,y)

    eg1：x,y,w,h,r = 5, 8, 6, 3, -60

        O--------> x
        |
        |     1\ h
        |    /  2      /
        |   /E /      /
        |  0  / w    / angle = -60
        |   `3      -------
        |
        v y


    eg2：x,y,w,h,r = 5, 8, 3, 6, 30

        O--------> x
        |
        |     0\ w
        |    /  1    ---------
        |   /E /      \  angle = 30
        |  3  / h       \
        |   `2
        |
        v y

    """
    x, y, w, h, r = xywhr
    return cv2.boxPoints(((x,y), (w,h), r))[[1,2,3,0]]


def get_cross_angle(l1, l2):
    arr_0 = np.array([(l1[1].x - l1[0].x), (l1[1].y - l1[0].y)])
    arr_1 = np.array([(l2[1].x - l2[0].x), (l2[1].y - l2[0].y)])
    cos_value = (float(arr_0.dot(arr_1)) / (np.sqrt(arr_0.dot(arr_0)) * np.sqrt(arr_1.dot(arr_1))))
    return np.arccos(cos_value) * (180 / np.pi)


def get_foot_point(point, line):
    """
    @point, line_p1, line_p2 : [x, y]
    """
    x0 = point.x
    y0 = point.y

    x1 = line[0].x
    y1 = line[0].y

    x2 = line[1].x
    y2 = line[1].y

    k = -((x1 - x0) * (x2 - x1) + (y1 - y0) * (y2 - y1)) / \
        ((x2 - x1) ** 2 + (y2 - y1) ** 2) * 1.0

    xn = k * (x2 - x1) + x1
    yn = k * (y2 - y1) + y1

    return xn, yn


def collinear_points(points):
    s = (1 / 2) * (
            points[0][0] * points[1][1] + points[1][0] * points[2].y + points[2].x * points[0][
        1] - points[0][0] * points[2].y - points[1][0] * points[0][1] - points[2].x * points[1][
                1])
    if s == 0:
        return True
    return False


def line_intersection(line1, line2):
    '''
    In:
        line1: (x1,x2,y1,y2)
        line2: (x1,x2,y1,y2)
    Out:
         x,y
    '''
    line1 = np.array(line1).reshape(-1, 2)
    line2 = np.array(line2).reshape(-1, 2)
    xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
    ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])

    def det(a, b):
        return a[0] * b[1] - a[1] * b[0]

    div = det(xdiff, ydiff)
    if div == 0:
        return None

    d = (det(*line1), det(*line2))
    x = det(d, xdiff) / div
    y = det(d, ydiff) / div
    return x, y


class Region:
    def __init__(self, points_xy):
        from shapely.geometry import Polygon as spy_Polygon
        self.region = spy_Polygon(points_xy)

    @classmethod
    def from_bbox(cls, ltrb):
        """
        :param ltrb: (l, t, r, b)
        :return:
        """
        l, t, r, b = ltrb
        points = [[l, t], [r, t], [r, b], [l, b]]
        return cls(points)

    def bbox(self):
        """
        :return: Region (l, t, r, b)
        """
        return Region.from_bbox(self.region.bounds)

    def to_bbox(self):
        """
        :return: (l, t, r, b)
        """
        return self.region.bounds

    @classmethod
    def from_rot_bbox(cls, xywhr):
        """
        :param xywhr: [x, y, w, h, r]
        eg1：x,y,w,h,r = 5, 8, 6, 3, -60

            O--------> x
            |
            |     1\ h
            |    /  2      /
            |   /E /      /
            |  0  / w    / angle = -60
            |   `3      -------
            |
            v y


        eg2：x,y,w,h,r = 5, 8, 3, 6, 30

            O--------> x
            |
            |     0\ w
            |    /  1    ---------
            |   /E /      \  angle = 30
            |  3  / h       \
            |   `2
            |
            v y

        :return:
        """
        points = gen_rotate_rect(xywhr)
        return cls(points)

    def min_bbox(self):
        """
        :return: Region
        """
        first_pt = self.coords()[0]
        pts = np.array(self.region.minimum_rotated_rectangle.exterior)
        idx = ((pts - first_pt) ** 2).sum(axis=1).argmin()
        idxs = [(i+idx)%4 for i in range(4)]
        return Region.from_polygon(pts[idxs])

    def to_rot_bbox(self):
        """
        :return: (x,y,w,h,r)
        """
        points = self.coords()
        if len(points) != 4:
            raise Exception('the len of points must be 4,you can xxx.min_bbox().to_rot_bbox()')
        cx = np.sum(points[:, 0]) / 4
        cy = np.sum(points[:, 1]) / 4
        w = Point(points[0]).distance(Point(points[1]), model='point')
        h = Point(points[1]).distance(Point(points[2]), model='point')
        vc1 = (points[1][0] - points[0][0], points[1][1] - points[0][1])
        l1 = Line([points[0][0], points[0][1]], [points[1][0], points[1][1]])
        l2 = Line([points[1][0], points[1][1]], [points[2][0], points[2][1]])
        l3 = Line([points[2][0], points[2][1]], [points[3][0], points[3][1]])
        p12, angle12 = l1.intersection(l2, segment=False)
        p23, angle23 = l2.intersection(l3, segment=False)
        import math
        if math.fabs(angle12 - 90) > 1.0e-1 or math.fabs(angle23 - 90) > 1.0e-1:
            raise Exception('the shape must be rectangle,you can use xxx.min_bbox().to_rot_bbox()')
        r = math.atan2(vc1[1], vc1[0]) * 180 / math.pi
        return cx, cy, w, h, r

    @classmethod
    def from_mask(cls, mask_image, only_max_area=True, area_threshold=25):
        '''
        :param area_threshold:
        :param mask_image (np.uint8): binary image, background is 0
        :param only_max_area:
        if True return max area region;
        if False return all region with  blob area>area_threshold
        :return: Region
        '''
        from .blob import BlobTool
        blobtools = BlobTool(mask_image)
        if only_max_area:
            areas = np.array([blob.area for blob in blobtools.blobs])
            blob = blobtools.blobs[areas.argmax()]
            return blob.get_region()
        else:
            if area_threshold < 2:
                area_threshold = 2
            return [blob.get_region() for blob in blobtools.blobs if blob.area > area_threshold]

    def to_mask(self, image_size=None):
        '''
        :param image_size:mask image size
        :return:(np.uint8): binary image, background is 0,foreground is 255
        '''
        import cv2
        if image_size is not None:
            zero_image = np.zeros(image_size)
            cv2.fillPoly(zero_image, [np.int0(self.coords())], 255)
        else:
            l, t, r, b = self.to_bbox()
            w = int(r - l)
            h = int(b - t)
            zero_image = np.zeros((h, w), dtype=np.uint8)
            xy = self.coords() - np.array([[l,t]])

            cv2.fillPoly(zero_image, [np.int0(xy)], 255)
        return zero_image

    @classmethod
    def from_polygon(cls, polygon):
        """
        :param polygon: [(x1,y1), (x2,y2), ... ] or [x1,y1, x2,y2, ...]
        :return:
        """
        points = np.float32(polygon).reshape(-1, 2)
        return cls(points)

    def to_polygon(self):
        """
        :return: [x1, y1, x2, y2, ...]
        """
        coords = self.coords()
        return coords.flatten().tolist()

    @classmethod
    def from_circle_xyr(cls, center_xy, radius):
        from shapely.geometry import Point as spy_Point
        return cls(spy_Point(center_xy).buffer(radius))

    def area(self):
        """
        the area of the region :float
        :return:
        """
        return self.region.area

    def intersection(self, dst_shape):
        """
        Returns a representation of the intersection of this object with the
        other geometric object
        :param dst_shape:other geometric object
        :return:intersection of this collection object
        only return multipolygon or polygon :list[Region]
        """
        from shapely import geometry
        region_collection = []
        ir = self.region.intersection(dst_shape.region)
        if not ir.is_empty:
            if isinstance(ir, geometry.MultiPolygon):
                for ply in ir.geoms:
                    if isinstance(ply, geometry.Polygon):
                        region_collection.append(Region(ply))
            elif isinstance(ir, geometry.Polygon):
                region_collection.append(Region(ir))
        return region_collection

    def union(self, dst_shape):
        """
        Returns a representation of the union of points from this
        object and the other geometric object.
        :param dst_shape: the other geometric object.
        :return: union of this collection object
        only return multipolygon or polygon :list[Region]
        """
        from shapely import geometry

        region_collection = []
        ir = self.region.union(dst_shape.region)
        if isinstance(ir, geometry.MultiPolygon):
            for ply in ir.geoms:
                if isinstance(ply, geometry.Polygon):
                    region_collection.append(Region(ply))
        elif isinstance(ir, geometry.Polygon):
            region_collection.append(Region(ir))

        return region_collection

    def transform(self, center=None, offset=(0, 0), rotate=0, scale=(1, 1)):
        """
        transform the region by offset,rotate,scale, around the center
        :param center: The point of origin can be a keyword 'center' for the
        2D bounding box center (default), 'centroid' for the geometry's 2D
        centroid, a Point object or a coordinate tuple (x0, y0)
        :param offset:offsets along each dimension(xoff,yoff)
        :param rotate:The angle of rotation :degrees
        :param scale:scaled by factors along each dimension(xfact,yfact)
        :return: the region after transformed
        """
        from shapely import affinity
        translate_x = offset[0]
        translate_y = offset[1]
        scale_x = scale[0]
        scale_y = scale[1]

        if center is None:
            new_center = 'center'
        else:
            new_center = (center[0] + offset[0], center[1] + offset[1])

        translate_points = affinity.translate(
            self.region, translate_x, translate_y)
        rotate_points = affinity.rotate(
            translate_points, rotate, origin=new_center)
        scale_points = affinity.scale(
            rotate_points, scale_x, scale_y, origin=new_center)

        return Region(scale_points)

    def coords(self):
        """
        :return:  the coords of the region :np.array([x,y])
        """
        return np.float32(self.region.exterior.coords)[:-1]


class Point:
    def __init__(self, xy):
        """
        The Point constructor takes positional coordinate values or
        point tuple parameters
        :param xy: point (x,y) or [x,y]
        """
        from shapely.geometry import Point as spy_Point
        self.points = spy_Point(xy)

    def to_xy(self):
        return self.coords().flatten().tolist()

    def distance(self, dst_shape, model='point'):
        """
        :param dst_shape: point or line
        :param model: one of the ('point','line'])other point :Point
        :return:  distance :float
        """
        point = dst_shape.points
        if model == 'point':
            dis = self.points.distance(point)
        elif model == 'line':
            foot_point = get_foot_point(self.points, point)
            dis = ((foot_point[0] - self.points.x) ** 2 + (
                    foot_point[1] - self.points.y) ** 2) ** 0.5
        else:
            raise Exception('the model must be one of point or line.')
        return dis

    def transform(self, center=None, offset=(0, 0), rotate=0, scale=(1, 1)):
        """
        transform the region by offset,rotate,scale, around the center
        :param center: The point of origin can be a keyword 'center' for the
        2D bounding box center (default), 'centroid' for the geometry's 2D
        centroid, a Point object or a coordinate tuple (x0, y0)
        :param offset:offsets along each dimension(xoff,yoff)
        :param rotate:The angle of rotation :degrees
        :param scale:scaled by factors along each dimension(xfact,yfact)
        :return: the region after transformed
        """
        from shapely import affinity
        translate_x = offset[0]
        translate_y = offset[1]
        scale_x = scale[0]
        scale_y = scale[1]

        if center is None:
            new_center = 'center'
        else:
            new_center = (center[0] + offset[0], center[1] + offset[1])

        translate_points = affinity.translate(
            self.points, translate_x, translate_y)
        rotate_points = affinity.rotate(
            translate_points, rotate, origin=new_center)
        scale_points = affinity.scale(
            rotate_points, scale_x, scale_y, origin=new_center)

        points = np.array(scale_points)
        return Point(points)

    def coords(self):
        return np.float32(self.points)


class Line:
    def __init__(self, start_xy, end_xy):
        """
        The Line constructor takes an ordered sequence of 2 (x, y) point tuples.
        :param start_xy: start point(x,y) or [x,y]
        :param end_xy: end point(x,y) or [x,y]
        """
        from shapely.geometry import MultiPoint as spy_MultiPoint
        self.points = spy_MultiPoint([start_xy, end_xy])

    def to_xy(self):
        start_xy, end_xy = self.coords()
        return start_xy.flatten().tolist(), end_xy.flatten().tolist()

    def coords(self):
        """
        :return:  the coords of the region :np.array([x,y])
        """
        return np.float32(self.points)

    def parallel_line(self, point=None, distance=None, side='left'):
        """
        :return parallel from point
        :param point: Point object
        :return: line:(x1,y1),(x2,y2)
        """
        if point is not None:
            dis = point.distance(self)
            ppl_left = self.points.convex_hull.parallel_offset(dis, side='left')
            ppl_right = self.points.convex_hull.parallel_offset(dis, side='right')
            if collinear_points([ppl_left.coords[0], ppl_left.coords[1], point.points]):
                return Line(ppl_left.coords[0], ppl_left.coords[1])
            return Line(ppl_right.coords[0], ppl_right.coords[1])

        ppl = self.points.convex_hull.parallel_offset(distance, side=side)
        return Line(ppl.coords[0], ppl.coords[1])

    def perpendicular_line(self, point):
        """
        :return perpendicular from point
        :param point:  Point object
        :return:  line:(x1,y1),(x2,y2)
        """
        foot_point = get_foot_point(point.points, self.points)
        return Line((point.points.x, point.points.y), foot_point)

    def intersection(self, dst_shape, segment=True):
        """
        :return point of intersection and angle from line and line1
        :param dst_shape: other line: Line
        :param segment: if  True return the point of segment intersect,
        if False return the point of line intersect : bool
        :return: (x,y),angle :(float,float),float
        """
        angle = get_cross_angle(self.points, dst_shape.points)
        if angle != 0:
            if segment:
                coords = \
                    self.points.convex_hull.intersection(dst_shape.points.convex_hull).coords
                if len(coords) != 0:
                    intersect_point = coords[0]
                else:
                    intersect_point = None
            else:
                intersect_point = line_intersection(self.points, dst_shape.points)
        else:
            intersect_point = None
        return intersect_point, angle

    def transform(self, center=None, offset=(0, 0), rotate=0, scale=(1, 1)):
        """
        transform the region by offset,rotate,scale, around the center
        :param center: The point of origin can be a keyword 'center' for the
        2D bounding box center (default), 'centroid' for the geometry's 2D
        centroid, a Point object or a coordinate tuple (x0, y0)
        :param offset:offsets along each dimension(xoff,yoff)
        :param rotate:The angle of rotation :degrees
        :param scale:scaled by factors along each dimension(xfact,yfact)
        :return: the region after transformed
        """
        from shapely import affinity
        translate_x = offset[0]
        translate_y = offset[1]
        scale_x = scale[0]
        scale_y = scale[1]

        if center is None:
            new_center = 'center'
        else:
            new_center = (center[0] + offset[0], center[1] + offset[1])

        translate_points = affinity.translate(
            self.points, translate_x, translate_y)
        rotate_points = affinity.rotate(
            translate_points, rotate, origin=new_center)
        scale_points = affinity.scale(
            rotate_points, scale_x, scale_y, origin=new_center)

        points = np.array(scale_points)
        return Line(points[0], points[1])
