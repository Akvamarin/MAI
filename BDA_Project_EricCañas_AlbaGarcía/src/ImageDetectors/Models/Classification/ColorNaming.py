from sklearn.neighbors import KNeighborsClassifier
from matplotlib.colors import to_rgb, rgb_to_hsv
from colormath.color_diff import delta_e_cie2000
from math import sqrt
from colormath.color_objects import LabColor, sRGBColor, HSVColor
from colormath.color_conversions import convert_color
from matplotlib import colors as colors
import numpy as np

BASIC_COLOR_NAMES = ['red', 'orange', 'yellow', 'green', 'cyan', 'blue', 'purple', 'pink', 'brown', 'black', 'white']
BASIC_RGB_COLORS = [to_rgb(color) for color in BASIC_COLOR_NAMES]

# XKCD colors are extracted from a survey of colors that were perceptually differentiable from 140,000 users.
XKCD_PREFIX = 'xkcd:'
XKCD_COLOR_NAMES = [color[len(XKCD_PREFIX):] for color in colors.XKCD_COLORS.keys() if 'again' not in color]
XKCD_RGB_COLORS = [to_rgb(XKCD_PREFIX + color) for color in XKCD_COLOR_NAMES]

class ColorClassifier:
    """
    This classifier takes a color in rgb and returns two simplified color names corresponding with the two colors
    more perceptually similar. A basic color name: among the 11 basic color names. A XKCD color name, a detailed color
    name among the 942 color names defined by 140,000 users into the XKCD color survey (https://xkcd.com/color/rgb.txt).

    For returning the most perceptually similar colors, it uses a 1-Nearest Neighbor model working on the CIE L*A*B
    colorspace, that uses deltaE CIE2000 for measuring distances. The CIE L*A*B color space fits the criteria that the
    distance that separates two points, when measuring through a deltaE distance function, will be directly related
    with the perceptual difference between those two colors perceived by the Human Visual System.
    """
    def __init__(self):
        """
        Initialize the two 1 Nearest Neighbors models with all the points within both simplified color spaces
        """
        # PretrainedModel containing the 11 color names
        basic_lab_colors = [rgb_to_lab(color) for color in BASIC_RGB_COLORS]
        self.basic_color_classifier = KNeighborsClassifier(n_neighbors=1, metric=deltaECIE2000)
        self.basic_color_classifier.fit(basic_lab_colors, BASIC_COLOR_NAMES)

        # PretrainedModel containing the 942 XKCD color names
        complex_lab_colors = [rgb_to_lab(color) for color in XKCD_RGB_COLORS]
        self.complex_color_classifier = KNeighborsClassifier(n_neighbors=1, metric=deltaECIE2000)
        self.complex_color_classifier.fit(complex_lab_colors, XKCD_COLOR_NAMES)

    def get_basic_color_name(self, rgb_color):
        """
        Takes an RGB color and return the name of its 11 color names nearest color (into the CIE2000 L*A*B space)
        :param rgb_color: 3 values iterable representing a RGB color. Color to convert to a color name into the
        11 basic color names space.
        :return: String. Color Name of the perceptually nearest color to rgb_color into the 11 Color Name space.
        """
        return self.basic_color_classifier.predict(np.array(rgb_to_lab(rgb_color)).reshape(1,-1))[0]

    def get_xkcd_color_name(self, rgb_color):
        """
        Takes an RGB color and return the name of its nearest color (into the CIE2000 L*A*B space) among XKCD colors.
        :param rgb_color: 3 values iterable representing a RGB color. Color to convert to a color name into the
        XKCD color name space.
        :return: String. Color Name of the perceptually nearest color to rgb_color into the XKCD Color Name space.
        """
        return self.complex_color_classifier.predict(np.array(rgb_to_lab(rgb_color)).reshape(1,-1))[0]

def deltaECIE2000(LAB_color1, LAB_color2):
    """
    Gives the deltaE CIE2000 distance between two colors into the L*A*B space. As longer this measure will be,
    greater will be the perceptual difference among those 2 colors
    :param LAB_color1: values iterable representing a L*A*B color. Color1 to which measure the distance
    :param LAB_color2: values iterable representing a L*A*B color. Color2 against to which measure the distance
    :return: float. deltaE CIE2000 distance among LAB_color1 and LAB_color2.
    """
    l1, a1, b1 = LAB_color1
    l2, a2, b2 = LAB_color2
    return delta_e_cie2000(color1=LabColor(l1, a1, b1), color2=LabColor(l2, a2, b2))

def hsv_distance(hsv_color1, hsv_color2):
    (h1, s1, v1), (h2, s2, v2) = hsv_color1, hsv_color2
    h_dist = min(abs(h2 - h1), 360. - abs(h2 - h1)) / 180. # (Since hue space is circular.)
    s_dist = (s2 - s1)
    v_dist = (v2 - v1) / 255.
    return sqrt(h_dist * h_dist + s_dist * s_dist + v_dist * v_dist)

def rgb_to_hsv(rgb_color):
    """
    Returns rgb_color into the hsv space
    :param rgb_color: 3 values iterable representing a RGB color. Color which convert to the hsv color space.
    :return: 3 values iterable represention of the rgb_color into the hsv space.
    """
    r, g, b = rgb_color
    hsv_color = convert_color(sRGBColor(r, g, b), HSVColor)
    return hsv_color.hsv_h, hsv_color.hsv_s, hsv_color.hsv_v

def hsv_to_rgb(hsv_color):
    """
    Returns rgb_color into the hsv space
    :param rgb_color: 3 values iterable representing a RGB color. Color which convert to the hsv color space.
    :return: 3 values iterable represention of the rgb_color into the hsv space.
    """
    h, s, v = hsv_color
    rgb_color = convert_color(HSVColor(h, s, v), sRGBColor)
    return rgb_color.rgb_r, rgb_color.rgb_g, rgb_color.rgb_b

def lab_to_hsv(lab_color):
    """
    Returns lab_color into the hsv space
    :param lab_color: 3 values iterable representing a lab color. Color which convert to the hsv color space.
    :return: 3 values iterable represention of the lab_color into the hsv space.
    """
    l, a, b = lab_color
    hsv_color = convert_color(LabColor(l, a, b), HSVColor)
    return hsv_color.hsv_h, hsv_color.hsv_s, hsv_color.hsv_v

def rgb_to_lab(rgb_color):
    """
    Converts an RGB color to the L*A*B space, which will be greatly useful for measuring perceptual distances.
    :param rgb_color: 3 values iterable representing a RGB color. Color which convert to the L*A*B color space.
    :return: 3 values iterable represention of the rgb_color into the L*A*B space.
    """
    r, g, b = rgb_color
    lab_color = convert_color(sRGBColor(r, g, b), LabColor)
    return lab_color.lab_l, lab_color.lab_a, lab_color.lab_b

def lab_to_rgb(lab_color):
    """
    Returns lab_color into the RGB space
    :param lab_color: 3 values iterable representing a lab color. Color which convert to the hsv color space.
    :return: 3 values iterable represention of the lab_color into the rgb space.
    """
    l, a, b = lab_color
    rgb_color = convert_color(LabColor(l, a, b), sRGBColor)
    return rgb_color.rgb_r, rgb_color.rgb_g, rgb_color.rgb_b

def name_to_rgb(name_color, return_as_uint8 = False):
    """
    Transform a basic color name or a XKCD color name into its RGB value.
    :param name_color: String. String representing a Basic Color Name or a XKCD color name
    :param return_as_uint8: Boolean. If True return the result into the uint8 range [0 - 255] otherwise return
    a value into the range [0., 1.]. Default: False.
    :return: 3 values iterable represention of the correspondent RGB color
    """
    rgb_color =  to_rgb(name_color) if name_color in BASIC_COLOR_NAMES else to_rgb(XKCD_PREFIX+name_color)
    if return_as_uint8:
        rgb_color = [int(channel*np.iinfo(np.uint8).max) for channel in rgb_color]
    return rgb_color

def name_to_lab(name_color):
    """
    Transform a basic color name or a XKCD color name into its LAB value.
    :param name_color: String. String representing a Basic Color Name or a XKCD color name
    :return: 3 values iterable represention of the correspondent L*a*B color
    """
    return rgb_to_lab(rgb_color=name_to_rgb(name_color=name_color, return_as_uint8 = False))

def name_to_hsv(name_color):
    """
    Transform a basic color name or a XKCD color name into its HSV value.
    :param name_color: String. String representing a Basic Color Name or a XKCD color name
    :return: 3 values iterable represention of the correspondent HSV color
    """
    return rgb_to_hsv(rgb_color=name_to_rgb(name_color=name_color, return_as_uint8 = False))