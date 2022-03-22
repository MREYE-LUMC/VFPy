import collections as _co
import itertools

import cv2
import numpy as np
import pytesseract as pyt

from vfpy.core.errors import ValueWarning, warn
from vfpy.core.utils import misc, imageutils

pyt.pytesseract.tesseract_cmd = r''

ROI = _co.namedtuple('ROI', ['x', 'y', 'w', 'h'])


def set_path_to_tesseract(exepath):
    pyt.pytesseract.tesseract_cmd = exepath

class OCRPreProcessor:
    """Preprocessor used to process an opencv image instance for optical caracter recognition.

    Args:
        img (np.ndarray): An opencv image.
        """

    _ScreenResolution = misc.get_screen_resolution()

    def __init__(self, img):
        self.OriginalImage = img.copy()
        self.ModifiedImage = img.copy()
        self.DrawingBoard = np.zeros((self.OriginalImage.shape[0], self.OriginalImage.shape[1], 3), np.uint8)

    def apply_on_image(self, inline, func, *args, **kwargs):
        """Applies function on the Modified image.

        Args:
            inline (bool): Defines if the self.ModifiedImage should be changed.
            func (function): the function to apply.
            *args: Variable length argument list to parse to func
            **kwargs: Arbitrary keyword arguments to parse to func

        Returns:
            np.ndarray: The modified image.
        """
        ret = func(self.ModifiedImage, *args, **kwargs)

        if isinstance(ret, np.ndarray):
            out = ret
        else:
            try:
                for item in ret:
                    if isinstance(item, np.ndarray):
                        out = item
                        break
                else:
                    raise ValueError('The applied function did not return an cv2 image as expected')
            except TypeError:
                raise ValueError('The applied function did not return an cv2 image as expected')

        if inline:
            self.ModifiedImage = out

        return out

    def select_stored_image(self, target):
        """Selects one of the stores images.

        Args:
            target (str): The image to select. Currently, 'Original', 'Modified' and 'DrawingBoard' are supported

        returns:
            np.ndarray: The selected image
        """
        if target == 'Modified':
            img = self.ModifiedImage
        elif target == 'Original':
            img = self.OriginalImage
        elif target == 'DrawingBoard':
            img = self.DrawingBoard
        else:
            img = self.ModifiedImage
            warn('Parameter target only takes "Original" or "Modified" as argument. '
                 'The modified image is returned', ValueWarning)

        return img

    def get_image(self, target):
        """Returns image based on the target type supplied.

        Args:
            target (str or np.ndarray): The image to use. Eiter a string pointing to a stored image,
                or an image opencv image instance.

        Returns:
            np.ndarray: an opencv image instance
        """
        if isinstance(target, str):
            img = self.select_stored_image(target=target)
        elif isinstance(target, np.ndarray):
            img = target
        else:
            warn('The selected target is not know. Please supply a known string command or '
                 'a np.ndarray to target. The default (self.ModifiedImage) will be used', ValueWarning)
            img = self.select_stored_image(target='Modified')

        return img

    def show_image(self, target='Modified', waitkey=1):
        """Shows one of the stored image objects.

        Args:
            target (str): Defines which image is to be shown, either takes 'bOriginal' or 'Modified'.
            waitkey (int): the argument passed to cv2.waitKey(). Choose 0 to wait for user action.
        """
        img = self.select_stored_image(target=target)

        wdwname = '{}Image'.format(target)

        sf = 0.85  # shrink factor
        wdwscale = self._ScreenResolution[0]/img.shape[0]
        wdwheight = int(sf * self._ScreenResolution[0])
        wdwwidth = int(sf * img.shape[1] * wdwscale)

        cv2.namedWindow(wdwname, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(wdwname, wdwwidth, wdwheight)
        cv2.imshow(wdwname, img)
        cv2.waitKey(waitkey)

    def identify_lines(self, target='Modified', **kwargs):
        """Identifies lines on an image using cv2.pHoughLines.

        Args:
            target (str or np.ndarray): The image to use. Eiter a string pointing to a stored image,
                or an image opencv image instance.
            **kwargs: Keyword arguments, see below.

        Keyword Arguments:
            rho (int or float): The rho to use in the detection.
            theta (float): The theta to use in the detection.
            threshold (int): The threshold to use in the detection.
            minlinelength (int): The minimal line length in pixels.
            maxlinegap (int): The maximal line gap in pixels.

        Returns:
            np.ndarray: an array containing all identified lines
            """

        img = self.get_image(target=target)

        rho = kwargs.get('rho', 1)
        theta = kwargs.get('theta', np.pi / 180)
        threshold = kwargs.get('threshold', 1000)
        minlinelength = kwargs.get('minlinelength', int(0.05*img.shape[0]))
        maxlinegap = kwargs.get('maxlinegap', 0.25*minlinelength)

        print('Searching for lines on image using the following settings:\n\t'
              'Rho: {}\n\tTheta: {}\n\tThreshold: {}\n\tMin line length: {}\n\tMax line gap: {}'
              .format(rho, theta, threshold, minlinelength, maxlinegap))
        lines = cv2.HoughLinesP(img, rho=rho, theta=theta, threshold=threshold,
                                minLineLength=minlinelength, maxLineGap=maxlinegap)

        # remove an additional dimension that is arbitrary (see:
        # https://stackoverflow.com/questions/40362942/why-does-the-houghlinesp-output-a-3d-array-instead-of-a-2d-array)
        lines = lines.squeeze()
        print('{} lines identified'.format(len(lines)))
        return lines

    def draw_lines(self, linearlinesegmentarray, target='DrawingBoard', **kwargs):
        img = self.get_image(target)
        for linesegment in linearlinesegmentarray:
            imageutils.draw_linearlinesegment(img, linesegment, **kwargs)

    def draw_intersections(self, intersectionarray, target='DrawingBoard', **kwargs):
        img = self.get_image(target)
        for isct in intersectionarray:
            imageutils.draw_point(img, isct.rounded_coordinate, **kwargs)

    def draw_clusters(self, clusterarray, target='DrawingBoard', **kwargs):
        img = self.get_image(target)
        kwargs['fill'] = False
        for clstr in clusterarray:
            if clstr is not None:
                imageutils.draw_point(img, clstr.rounded_center, **kwargs)
                imageutils.draw_plus(img, clstr.rounded_center, 60, **kwargs)

    def draw_point(self, coordinate, target='DrawingBoard', **kwargs):
        img = self.get_image(target)
        imageutils.draw_point(img, coordinate, **kwargs)

    def overlay_drawingboard(self, target='Modified', waitkey=1):
        img = self.select_stored_image(target=target)

        wdwname = 'DrawingBoardOverlay'

        sf = 0.85  # shrink factor
        wdwscale = self._ScreenResolution[0] / img.shape[0]
        wdwheight = int(sf * self._ScreenResolution[0])
        wdwwidth = int(sf * img.shape[1] * wdwscale)

        inv_mask = np.any(self.DrawingBoard, axis=2)
        img[inv_mask] = 0

        bottomim = imageutils.convert_to_gbr(img)

        overlay = cv2.add(bottomim, self.DrawingBoard)

        cv2.namedWindow(wdwname, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(wdwname, wdwwidth, wdwheight)
        cv2.imshow(wdwname, overlay)
        cv2.waitKey(waitkey)

    def clear_drawingboard(self):
        self.DrawingBoard = np.zeros(self.DrawingBoard.shape, self.DrawingBoard.dtype)


def houghlinep_to_linearlinesegmets(houghlines):
    """Converts an array of cv2.HoughLineP results to an array of LinearLineSegment instances.
    Args:
        houghlines (np.ndarray | list): An array containing the houghline results as obtained from cv2.HoughLineP.

    Returns:
        np.array: An array containing LinearLineSegment instances
    """
    out = []
    for item in houghlines:
        out.append(misc.LinearLineSegment(item[0], item[1], item[2], item[3]))
    out = np.array(out)

    return out


def calc_isct(line1, line2=None):
    if isinstance(line1, tuple):
        line1, line2 = line1

    interval_1 = [min(line1.x1, line1.x2), max(line1.x1, line1.x2)]
    interval_2 = [min(line2.x1, line2.x2), max(line2.x1, line2.x2)]
    interval_x = [min(interval_1[1], interval_2[1]), max(interval_1[0], interval_2[0])]

    interval_3 = [min(line1.y1, line1.y2), max(line1.y1, line1.y2)]
    interval_4 = [min(line2.y1, line2.y2), max(line2.y1, line2.y2)]
    interval_y = [min(interval_3[1], interval_4[1]), max(interval_3[0], interval_4[0])]

    if interval_1[1] < interval_2[0]:
        # print('No Intersections')
        return None

    elif interval_3[1] < interval_4[0]:
        # print('No Intersections')
        return None

    D = line1.A * line2.B - line1.B * line2.A
    Dx = line1.C * line2.B - line1.B * line2.C
    Dy = line1.A * line2.C - line1.C * line2.A

    if D != 0:
        x = Dx / D
        y = Dy / D

        if x < interval_x[1] or x > interval_x[0]:
            # print('Intersection out of bound')
            return None
        elif y < interval_y[1] or y > interval_y[0]:
            # print('Intersection out of bound')
            return None
        else:
            return misc.LineSegmentIntersection(x, y, line1, line2)

    else:
        # print('Not intersecting')
        return None


def calc_perpendicular_intersections(linesegments, hlineslope_th = 0.25, vlineslope_th=4):
    # search for Intersections

    hlines = [line for line in linesegments if (line.Slope is not None and abs(line.Slope) <= hlineslope_th)]
    vlines = [line for line in linesegments if (line.Slope is None or abs(line.Slope) >= vlineslope_th)]

    print('Calculation intersections for {} horizontal and {} vertical lines'.format(len(hlines), len(vlines)))

    intersections = [calc_isct(line1, line2) for line1, line2 in itertools.product(hlines, vlines)]
    intersections = [isct for isct in intersections if isct is not None]

    print('{} intersections found'.format(len(intersections)))
    return intersections


def intersections_to_clusters(intersections, cluster_radius, tune_cluster_radius=False):
    print('Identifying intersection clusters')
    clusters = []
    for isct in intersections:
        for clstr in clusters:
            if clstr.is_member(isct):
                clstr.add_member(isct)
                break
        else:
            clusters.append(misc.IntersectionCluster(isct,
                                                     radius=cluster_radius,
                                                     tune_radius=tune_cluster_radius))
    print('{} clusters identified'.format(len(clusters)))
    return clusters


def create_text_patch(label):
    font = cv2.FONT_HERSHEY_DUPLEX
    font_scale = 1.0
    font_thickness = 1

    (label_width, label_height), baseline = cv2.getTextSize(label, font, font_scale, font_thickness)
    label_patch = np.full((label_height + baseline, label_width), 255)

    cv2.putText(label_patch, label, (0, label_height-1), font, font_scale, 0, font_thickness)
    return label_patch


class OCRoi:
    """A baseclass for ocr regions of interest."""

    def __init__(self, roi, roiname=''):
        self.ROI = roi
        self.ROIName = roiname
        self.Processed = False
        self.Result = None


class OCResult:
    """A baseclass for ocr results."""

    def __init__(self, text=None, confidence=np.NaN):
        self.Text = text
        self.Confidence = confidence
        self.OCRDataFrame = None

    def df_min_confidence(self):
        if self.OCRDataFrame is None:
            return np.NaN
        else:
            return self.OCRDataFrame.loc[:, 'conf'].min()

class OCRProcessor:
    def __init__(self, img):
        self.Image = img.copy()
        self.OCRois = []

    def process(self):
        for ocroi in self.OCRois:
            pass # perform ocr at location


def levenshtein(string1, string2, case_sensitive=True, ignore_whitespaces=False):
    """"Calculate levenshtein difference between two words"""
    lev_mat = np.full((len(string1)+1, len(string2)+1), 0)

    if not case_sensitive:
        string1 = string1.lower()
        string2 = string2.lower()

    if ignore_whitespaces:
        string1 = ''.join(string1.split())
        string2 = ''.join(string2.split())

    for ii in range(1, len(string1)+1):
        lev_mat[ii, 0] = ii

    for jj in range(1, len(string2)+1):
        lev_mat[0, jj] = jj

    for ii in range(1, len(string1)+1):
        for jj in range(1, len(string2)+1):

            if string1[ii-1] == string2[jj-1]:
                cost = 0
            else:
                cost = 1

            lev_mat[ii, jj] = min((lev_mat[ii-1, jj] + 1),
                                  (lev_mat[ii, jj-1] + 1),
                                  (lev_mat[ii-1, jj-1] + cost))

    return int(lev_mat[len(string1), len(string2)])
