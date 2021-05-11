import os
from xml.etree import ElementTree


def parse_xml(xml_file, file):
    tree = ElementTree.parse(xml_file)
    root = tree.getroot()
    objects = root.findall("object")
    for object in objects:
        cls = object.find("name").text
        bbox = object.find("bndbox")
        xmin = bbox.find("xmin").text
        ymin = bbox.find("ymin").text
        xmax = bbox.find("xmax").text
        ymax = bbox.find("ymax").text
        info = [str(clss.index(cls)), xmin, ymin, xmax, ymax]
        file.write(' '.join(info) + '\n')


if __name__ == '__main__':
    clss = ['airport', 'baseballfield', 'groundtrackfield', 'tenniscourt', 'windmill', 'vehicle', 'overpass',
            'Expressway-toll-station', 'storagetank', 'harbor', 'ship', 'airplane', 'dam', 'golffield', 'stadium',
            'Expressway-Service-area', 'bridge', 'chimney', 'trainstation', 'basketballcourt']
    xml_path = '/workspace/lyf/detect/DIOR/annotations'
    txt_path = '/workspace/lyf/detect/DIOR/label'
    for xml_ in os.listdir(xml_path):
        with open(os.path.join(txt_path, xml_.replace('xml', 'txt')), 'w+') as file:
            parse_xml(os.path.join(xml_path, xml_), file)
