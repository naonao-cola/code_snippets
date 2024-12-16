import sys
import copy
from xml.etree import ElementTree
from xml.etree.ElementTree import Element, SubElement
from lxml import etree
import codecs

XML_EXT = '.xml'
ENCODE_METHOD = 'utf-8'
ustr = str
DEFAULT_CONTRAST = 0
DEFAULT_LUMINANCE = 0


class PascalVocWriter:

    def __init__(self, foldername, filename, imgSize, databaseSrc='Unknown', localImgPath=None):
        self.foldername = foldername
        self.filename = filename
        self.databaseSrc = databaseSrc
        self.imgSize = imgSize
        self.boxlist = []
        self.localImgPath = localImgPath
        self.verified = False

    def prettify(self, elem):
        """
            Return a pretty-printed XML string for the Element.
        """
        rough_string = ElementTree.tostring(elem, 'utf8')
        root = etree.fromstring(rough_string)
        return etree.tostring(root, pretty_print=True, encoding=ENCODE_METHOD).replace("  ".encode(), "\t".encode())
        # minidom does not support UTF-8
        '''reparsed = minidom.parseString(rough_string)
        return reparsed.toprettyxml(indent="\t", encoding=ENCODE_METHOD)'''

    def genXML(self):
        """
            Return XML root
        """
        # Check conditions
        if self.filename is None or \
                self.foldername is None or \
                self.imgSize is None:
            return None

        top = Element('annotation')
        if self.verified:
            top.set('verified', 'yes')

        folder = SubElement(top, 'folder')
        folder.text = self.foldername

        filename = SubElement(top, 'filename')
        filename.text = self.filename

        if self.localImgPath is not None:
            localImgPath = SubElement(top, 'path')
            localImgPath.text = self.localImgPath

        source = SubElement(top, 'source')
        database = SubElement(source, 'database')
        database.text = self.databaseSrc

        size_part = SubElement(top, 'size')
        width = SubElement(size_part, 'width')
        height = SubElement(size_part, 'height')
        depth = SubElement(size_part, 'depth')
        width.text = str(self.imgSize[0])
        height.text = str(self.imgSize[1])
        if len(self.imgSize) == 3:
            depth.text = str(self.imgSize[2])
        else:
            depth.text = '1'

        segmented = SubElement(top, 'segmented')
        segmented.text = '0'
        return top

    def addBndBox(self, xmin, ymin, xmax, ymax, name, difficult, extern_info=dict()):
        extern_info = copy.deepcopy(extern_info)
        bndbox = {'xmin': int(xmin), 'ymin': int(ymin), 'xmax': int(xmax), 'ymax': int(ymax)}
        bndbox['name'] = name
        if "difficult" not in extern_info:
            extern_info["difficult"] = "1" if difficult else "0"
        if "pose" not in extern_info:
            extern_info['pose'] = 'Unspecifed'
        if "truncated" not in extern_info:
            extern_info['truncated'] = '0'
        self.boxlist.append([bndbox, extern_info])

    def appendObjects(self, top):
        for each_object, extern_info in self.boxlist:
            object_item = SubElement(top, 'object')
            name = SubElement(object_item, 'name')
            name.text = ustr(each_object['name'])
            bndbox = SubElement(object_item, 'bndbox')
            xmin = SubElement(bndbox, 'xmin')
            xmin.text = str(each_object['xmin'])
            ymin = SubElement(bndbox, 'ymin')
            ymin.text = str(each_object['ymin'])
            xmax = SubElement(bndbox, 'xmax')
            xmax.text = str(each_object['xmax'])
            ymax = SubElement(bndbox, 'ymax')
            ymax.text = str(each_object['ymax'])

            for ext_key in extern_info:
                ext_key_element = SubElement(object_item, ext_key)

                if ext_key == "truncated":
                    if int(float(each_object['ymax'])) == int(float(self.imgSize[0])) or (
                            int(float(each_object['ymin'])) == 1):
                        ext_key_element.text = "1"  # max == height or min
                    elif (int(float(each_object['xmax'])) == int(float(self.imgSize[1]))) or (
                            int(float(each_object['xmin'])) == 1):
                        ext_key_element.text = "1"  # max == width or min
                    else:
                        ext_key_element.text = "0"
                else:
                    ext_key_element.text = extern_info[ext_key]

    def save(self, targetFile=None):
        root = self.genXML()
        self.appendObjects(root)
        out_file = None
        if targetFile is None:
            out_file = codecs.open(
                self.filename + XML_EXT, 'w', encoding=ENCODE_METHOD)
        else:
            out_file = codecs.open(targetFile, 'w', encoding=ENCODE_METHOD)

        prettifyResult = self.prettify(root)
        out_file.write(prettifyResult.decode('utf8'))
        out_file.close()


class PascalVocReader:

    def __init__(self, filepath):
        # shapes type:
        # [labbel, [(x1,y1), (x2,y2), (x3,y3), (x4,y4)], color, color, difficult]
        self.shapes = []
        self.extern_infos = []
        self.default_shape_keys = ['name', 'bndbox']
        self.filepath = filepath
        self.verified = False
        self.filename = None
        try:
            self.parseXML()
        except:
            pass

    def getShapes(self):
        return self.shapes

    def getExternInfos(self):
        return self.extern_infos

    def addShape(self, shape_info):
        shape_info = copy.deepcopy(shape_info)
        label = shape_info["name"]
        bndbox = shape_info["bndbox"]
        xmin = int(float(bndbox.get('xmin')))
        ymin = int(float(bndbox.get('ymin')))
        xmax = int(float(bndbox.get('xmax')))
        ymax = int(float(bndbox.get('ymax')))
        points = [(xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin, ymax)]
        shape_info["points"] = points
        self.shapes.append(shape_info)

    def addExternInfo(self, shape_info):
        exter_info = {k: copy.deepcopy(v) for k, v in shape_info.items() if k not in self.default_shape_keys}
        self.extern_infos.append(exter_info)

    def parseXML(self):
        assert self.filepath.endswith(XML_EXT), "Unsupport file format"
        parser = etree.XMLParser(encoding=ENCODE_METHOD)
        xmltree = ElementTree.parse(self.filepath, parser=parser).getroot()
        filename = xmltree.find('filename').text
        self.filename = filename
        try:
            verified = xmltree.attrib['verified']
            if verified == 'yes':
                self.verified = True
        except KeyError:
            self.verified = False

        labels_element = xmltree.findall("objectList")
        if len(labels_element):
            # For ts marker tool xml style
            labels_element_iters = labels_element[0].findall('LabelElement')
        else:
            # For labelimg xml style
            labels_element_iters = xmltree.findall('object')
        for object_iter in labels_element_iters:
            shape_info = {}
            for item in object_iter.getchildren():
                child_item_list = item.getchildren()
                if len(child_item_list) == 0:
                    shape_info[item.tag] = item.text
                else:
                    shape_info[item.tag] = {c_item.tag: c_item.text for c_item in child_item_list}
            self.addShape(shape_info)
            self.addExternInfo(shape_info)
        return True
