import unittest
from tvlab.detection import pascal_voc_io
import os.path as osp


class TestPascalVocIo(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(TestPascalVocIo, self).__init__(*args, **kwargs)
        self.XML_PATH = osp.normpath('./data/coco_instance/detection_xml/000000067213.xml')

    def test_copy_copy(self):
        a = pascal_voc_io.copy.copy('a')
        self.assertEqual(a, 'a')

    def test_copy_deepcopy(self):
        a = pascal_voc_io.copy.deepcopy('a')
        self.assertEqual(a, 'a')

    @unittest.skip("skip test_copy_dispatch_table_keys")
    def test_copy_dispatch_table_keys(self):
        a = pascal_voc_io.copy.dispatch_table.keys()

    @unittest.skip("skip test_copy_dispatch_table_values")
    def test_copy_dispatch_table_values(self):
        a = pascal_voc_io.copy.dispatch_table.values()

    def test_codes_ascii(self):
        a = pascal_voc_io.codecs.ascii_encode("ab")
        self.assertEqual(a[1], 2)

    def test_codes_encode_decode(self):
        a = pascal_voc_io.codecs.encode('ab')
        a = pascal_voc_io.codecs.decode(a)
        self.assertEqual(a, 'ab')

    def test_DEFAULT_CONTRAST_bit_length(self):
        a = pascal_voc_io.DEFAULT_CONTRAST.bit_length()
        self.assertEqual(a, 0)

    def test_PascalVocReader(self):
        a = pascal_voc_io.PascalVocReader(self.XML_PATH).parseXML()
        self.assertTrue(a, 'True')

    @unittest.skip("skip test_PascalVocWriter")
    def test_PascalVocWriter(self):
        pascal_voc_io.PascalVocWriter(osp.normpath('./XML/test'), 'test.xml', [1, 10, 10])

    def test_ElementTree_ElementTree(self):
        a = pascal_voc_io.ElementTree.ElementTree(self.XML_PATH)
        self.assertEqual(a.parse(self.XML_PATH).tag, 'annotation')


if __name__ == '__main__':
    unittest.main()
