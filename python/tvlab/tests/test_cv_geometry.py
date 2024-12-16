from tvlab import *
import unittest
import cv2
import numpy as np

SHOW = False


class TestLine(unittest.TestCase):
    def test_init(self):
        line = Line((0, 0), (2, 2))
        l_c = line.coords().tolist()
        l_p = line.points
        re = line.transform(offset=(1, 0), rotate=45, scale=(0, 0))
        re_c = np.around(re.coords(), 2).tolist()
        parl = line.parallel_line(Point((1, 0)))
        perl = line.perpendicular_line(Point((1, 0)))
        self.assertEqual(l_c, [[0.0, 0.0], [2.0, 2.0]])
        self.assertEqual(str(l_p), 'MULTIPOINT (0 0, 2 2)')
        self.assertEqual(re_c, [[2.0, 1.000000000000000], [2.0, 1.000000000000000]])


class TestPoint(unittest.TestCase):
    def test_init(self):
        point = Point((1, 2))
        p_c = point.coords()
        p_p = point.points
        re = point.transform(offset=(0, 1), rotate=15, scale=(1, 2))
        re_c = re.coords()
        re_p = re.points
        self.assertEqual(str(p_c), '[1. 2.]')
        self.assertEqual(str(p_p), 'POINT (1 2)')
        self.assertEqual(str(re_c), '[1. 3.]')
        self.assertEqual(str(re_p), 'POINT (1 3)')


class TestRectangele(unittest.TestCase):
    def test_init(self):
        rectangel = Region.from_bbox((0, 0, 10, 10))
        r_a = rectangel.area()
        r_c = rectangel.coords().tolist()
        r_r = rectangel.region.type
        self.assertEqual(r_a, 100.0)
        self.assertEqual(r_c, [[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]])
        self.assertEqual(str(r_r), 'Polygon')

    def test_intersection(self):
        rectangle = Region.from_bbox((0, 0, 10, 10))
        tmp = Region.from_bbox((5, 5, 15, 15))
        re_src = rectangle.region
        tmp_src = tmp.region
        r_inter = re_src.intersection(tmp_src)
        self.assertEqual(r_inter.area, 25.0)

    def test_union(self):
        rectangle = Region.from_bbox((0, 0, 10, 10))
        tmp = Region.from_bbox((5, 5, 15, 15))
        re_src = rectangle.region
        tmp_src = tmp.region
        r_union = re_src.union(tmp_src)
        self.assertEqual(r_union.area, 175.0)

    def test_shape_transform(self):
        rectangle = Region.from_bbox((0, 0, 10, 10))
        rec = rectangle.transform(offset=(0, 1), rotate=45, scale=(2, 1))
        rec_a = round(rec.area())
        self.assertEqual(rec_a, 200)


class TestRotationRectangle(unittest.TestCase):
    def test_init(self):
        ro_rec = Region.from_polygon([[0, 5], [5, 10], [10, 5], [5, 0]])
        self.assertEqual(ro_rec.area(), 50.0)
        self.assertEqual(ro_rec.coords().tolist(),
                         [[0.0, 5.0], [5.0, 10.0], [10.0, 5.0], [5.0, 0.0]])
        self.assertEqual(str(ro_rec.region.type), 'Polygon')

    def test_from_xyrwh(self):
        ro_rec = Region.from_polygon([[0, 5], [5, 10], [10, 5], [5, 0]])
        re = ro_rec.from_rot_bbox((5, 5, 5, 5, 90))
        self.assertEqual(re.area(), 25.0)

    def test_shape_transform(self):
        ro_rec = Region.from_rot_bbox((5, 5, 10, 5, 90))
        rec = ro_rec.transform(rotate=-45)
        rec_a = rec.to_rot_bbox()
        rec_ai = tuple(np.around(rec_a, 2))
        self.assertEqual(rec_ai, (5, 5, 10, 5, 45))


class TestPoygon(unittest.TestCase):
    def test_init(self):
        polygon = Region.from_polygon([[0, 5], [5, 10], [10, 5], [5, 0], [0, 0]])

        mask_image = cv2.imread('./data/pet_mask.jpg', 0)
        region = Region.from_mask(mask_image, only_max_area=False)
        mask = region[0].to_mask(image_size=(577, 998))

        if SHOW:
            import pylab

            def plot_line(g, o):
                a = np.asarray(g)
                pylab.plot(a[:, 0], a[:, 1], o)

            coords = region[0].coords()
            plt.figure()
            plt.subplot(2, 2, 1)
            plot_line(coords, 1)
            plt.subplot(2, 2, 1)
            pylab.imshow(mask_image, cmap='gray')
            plt.subplot(2, 2, 2)
            pylab.imshow(mask, cmap='gray')
            pylab.show()

        self.assertEqual(polygon.area(), 62.5)
        self.assertEqual(polygon.coords().tolist(),
                         [[0.0, 5.0], [5.0, 10.0], [10.0, 5.0], [5.0, 0.0], [0.0, 0.0]])
        self.assertEqual(str(polygon.region.type), 'Polygon')


class TestCircle(unittest.TestCase):
    def test_init(self):
        circle = Region.from_circle_xyr((5.0, 5.0), 5.0)
        self.assertEqual(round(circle.area(), 2), 78.41)
        self.assertEqual(len(circle.coords().tolist()), 65)
        self.assertEqual(str(circle.region.type), 'Polygon')


if __name__ == '__main__':
    unittest.main()
