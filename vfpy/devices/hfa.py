import collections as _co
import datetime as dt
import itertools as it
import operator
import re
import os

import cv2
import numpy as np
import pandas as pd
import pytesseract as pyt

from vfpy.core.dtypes.vf_dtypes import StaticPerimetry
from vfpy.core.utils import ocrutils, imageutils, dicomutils, misc
from vfpy.core.utils.imageutils import crop_to_bounding_box, remove_frame, add_boarder
from vfpy.core.utils.ocrutils import ROI, OCResult, levenshtein

hfa_landmarks = _co.namedtuple('HFA_Landmarks',
                               ['Top1', 'Top2', 'Bottom1', 'Bottom2', 'ImageCenter',
                                'Cross1', 'Cross2', 'Cross3', 'Cross4', 'Cross5'])

HFA_60_4_MASK = np.array([
    [False, False, True, True, True, True, True, True, False, False],
    [True, True, True, True, True, True, True, True, True, True],
    [True, True, True, False, False, False, False, True, True, True],
    [True, True, True, False, False, False, False, True, True, True],
    [True, True, True, False, False, False, False, True, True, True],
    [True, True, True, False, False, False, False, True, True, True],
    [False, True, True, True, True, True, True, True, True, False],
    [False, True, True, True, True, True, True, True, True, False],
    [False, False, False, True, True, True, True, False, False, False],
])

def from_dicom(filepath, resolution=600):
    dicomdata = dicomutils.load_dicom(filepath)
    wandimage = imageutils.bytearray_to_wandimage(dicomdata['0042', '0011'].value, resolution=resolution)
    opencvimage = imageutils.wandimage_to_opencvimage(wandimage)

    ret = HFAScan(opencvimage)
    ret.DicomData = dicomdata

    wandimage.destroy()
    del wandimage
    del dicomdata

    return ret


def from_pdf(filepath, resolution=600):
    # TODO Test
    wandimage = imageutils.bytearray_to_wandimage(filepath, resolution=resolution)
    opencvimage = imageutils.wandimage_to_opencvimage(wandimage)

    ret = HFAScan(opencvimage)

    wandimage.destroy()
    del wandimage

    return ret


def unstring_maps(stringmap, setbelow0to=-99):
    ret = np.full(stringmap.shape, np.NaN).ravel()
    for num, item in enumerate(stringmap.ravel()):
        if item == '':
            ret[num] = np.NaN
        elif item == '<0':
            ret[num] = setbelow0to
        else:
            try:
                ret[num] = np.float(item)
            except ValueError:
                ret[num] = np.NaN
    return ret.reshape(stringmap.shape).astype(np.float64)


def get_transitions(img, axis, only_white_to_black=False):
    """Returns a series of tupels containing (gapsize, transitionline, transitionnumber)"""
    transits = np.apply_along_axis(lambda x: len(list(it.accumulate([len(list(items))
                                                                     for item, items
                                                                     in it.groupby(x)]))[:-1]),
                                   axis, img)
    emptylines = transits == 0
    transitems = [0] + list([item for item, items in it.groupby(emptylines)])
    trans = [0] + list(it.accumulate([len(list(items)) for item, items in it.groupby(emptylines)]))
    gaps = [(jj, trans[ii+1], ii) for ii, jj in enumerate(np.diff(trans))]
    if not only_white_to_black:
        selection = sorted([item for item in gaps], reverse=True)
    else:
        selection = sorted([item for item in gaps if not transitems[item[2]]], reverse=True)
    selection = sorted(selection, key=lambda x: x[1])
    return selection


def highlight_conf(ser):
    ret = []
    for val in ser:
        if pd.isnull(val):
            ret.append('')
        elif isinstance(val, str):
            ret.append('')
        elif val <= 25:
            ret.append('background-color: red')
        elif 25 < val <= 75:
            ret.append('background-color: yellow')
        elif val > 75:
            ret.append('background-color: green')
        else:
            ret.append('')
    return ret


class HFAScanParameters:
    def __init__(self):
        self.AnalysisType = None
        self.Eye = None
        self.Name = None
        self.ID = None
        self.DoB = None
        self.TestType = None
        self.FixationMonitor = None
        self.FixationTarget = None
        self.FixationLosses = None
        self._FixationLosses_Numerator = None
        self._FixationLosses_Denominator = None
        self.FalsePosErrors = None
        self._FalsePosErrors_Numerator = None
        self._FalsePosErrors_Denominator = None
        self._FalsePosErrors_Unit = None
        self.FalseNegErrors = None
        self._FalseNegErrors_Numerator = None
        self._FalseNegErrors_Denominator = None
        self._FalseNegErrors_Unit = None
        self.TestDuration = None
        self.Fovea = None
        self.Stimulus = None
        self.Background = None
        self._Background_Unit = None
        self.Strategy = None
        self.PupilDiameter = None
        self._PupilDiameter_Unit = None
        self.VisualAcuity = None
        self.Refraction = None
        self.Date = None
        self.Time = None
        self.Age = None


class HFAScan(StaticPerimetry):
    __BELOW0 = -99
    __MAPTODATAFRAMEPROTOCOL = 'nt'

    def __init__(self, image, protocol=None, dicomdata=None, **measurement_details):
        super().__init__(**measurement_details)
        self.Protocol = protocol
        self.Image = image

        self.Landmarks = None

        self.OCRScanParameters = HFAScanParameters()
        self.ScanParameters = HFAScanParameters()
        self.ScanParameterDf = None

        self.DefectDepthRois = []
        self.DefectDepthMap = None
        self.DefectDepthStringMap = None
        self.DefectDepthConfidenceMap = None
        self.DefectDepthDf = None
        self.DefectDepthStringDf = None
        self.DefectDepthConfidenceDf = None

        self.ThresholdRois = []
        self.ThresholdMap = None
        self.ThresholdStringMap = None
        self.ThresholdConfidenceMap = None
        self.ThresholdDf = None
        self.ThresholdStringDf = None
        self.ThresholdConfidenceDf = None

        self.DicomData = dicomdata

        self.isProcessed = False

    def guess_protocol(self):
        stringblob = pyt.image_to_string(self.Image[:int(self.Image.shape[0] / 2), :int(self.Image.shape[1] / 2)])
        treshtest = re.search(r'\d{2}-\d Threshold Test', stringblob, flags=re.IGNORECASE)
        if treshtest:
            self.Protocol = treshtest.group().split(' ')[0]

    def identify_landmarks(self, show_process=True):
        if self.Protocol is None:
            self.guess_protocol()

        if self.Protocol == '60-4':
            settings = {}
            self.Landmarks = hfa_60_4_get_landmarks(self.Image, show_process=show_process, **settings)
        else:
            print('The analysis protocol (30-2, 60-4, etc.) should be known for landmark identification')

    def obtain_gridocrois(self):
        if self.Landmarks is None:
            self.identify_landmarks()

        if self.Protocol == '60-4':
            # TODO think about shifting left or right
            self.DefectDepthRois.extend(hfa_60_4_get_gridocrois(self.Image, self.Landmarks.Cross2, 'DDRoi'))
            self.ThresholdRois.extend(hfa_60_4_get_gridocrois(self.Image, self.Landmarks.Cross3, 'ThRoi'))
        else:
            print('The analysis protocol (30-2, 60-4, etc.) should be known for landmark identification')

    def read_scanparameters(self):
        if not self.Landmarks:
            self.identify_landmarks()

        cv2.namedWindow('OCR Input')
        #################
        # Above top bar #
        #################

        crop = self.Image[0:int(self.Landmarks.Top1 + 1), :]
        crop = crop_to_bounding_box(crop)
        crop = remove_frame(crop)

        # remove optional bottom line
        if not np.all(crop[-1, :]):
            rowtrans = get_transitions(crop, 1)
            crop = crop[0:rowtrans[0][1] + 1, :]

        crop = crop_to_bounding_box(crop)

        # split in two
        coltrans = get_transitions(crop, 0)
        maxcoltrans_right = sorted(coltrans, key=lambda x: x[0], reverse=True)[0]  # large whitespace
        maxcoltrans_left = sorted(coltrans, key=lambda x: x[2], reverse=True)[maxcoltrans_right[2]-1]

        # OCR Analysis type
        ocr_im = crop[:, :maxcoltrans_left[1]]
        ocr_im = crop_to_bounding_box(ocr_im)
        ocr_im = add_boarder(ocr_im)
        cv2.imshow('OCR Input', ocr_im.astype(np.uint8))
        cv2.waitKey(1)

        txt = pyt.image_to_string(ocr_im)
        txt_data = pyt.image_to_data(ocr_im, output_type=pyt.Output.DATAFRAME)

        res = OCResult(text=txt)
        res.OCRDataFrame = txt_data
        res.OCRDataFrame.dropna(axis=0, inplace=True)
        self.OCRScanParameters.AnalysisType = res

        # OCR laterality
        ocr_im = crop[:, maxcoltrans_right[1]:]
        ocr_im = crop_to_bounding_box(ocr_im)
        ocr_im = add_boarder(ocr_im)
        cv2.imshow('OCR Input', ocr_im.astype(np.uint8))
        cv2.waitKey(1)

        txt = pyt.image_to_string(ocr_im)
        txt_data = pyt.image_to_data(ocr_im, output_type=pyt.Output.DATAFRAME)
        param = txt.split(':')[0]
        if levenshtein(param, 'eye', case_sensitive=False, ignore_whitespaces=True) <= 1:
            res = OCResult(text=txt)
            res.OCRDataFrame = txt_data
            res.OCRDataFrame.dropna(axis=0, inplace=True)
            self.OCRScanParameters.Eye = res

        ##################
        # Within top bar #
        ##################

        crop = self.Image[int(self.Landmarks.Top1):int(self.Landmarks.Top2+1), :]
        crop = crop_to_bounding_box(crop)
        crop = remove_frame(crop)
        crop = crop_to_bounding_box(crop)

        # split in two
        coltrans = get_transitions(crop, 0)
        maxcoltrans_right = sorted(coltrans, key=lambda x: x[0], reverse=True)[0]  # large whitespace
        maxcoltrans_left = sorted(coltrans, key=lambda x: x[2], reverse=True)[maxcoltrans_right[2] - 1]

        # OCR Name and ID
        ocr_im = crop[:, :maxcoltrans_left[1]]
        ocr_im = crop_to_bounding_box(ocr_im)
        ocr_im = add_boarder(ocr_im)
        cv2.imshow('OCR Input', ocr_im.astype(np.uint8))
        cv2.waitKey(1)

        txt = pyt.image_to_string(ocr_im)
        txt_data = pyt.image_to_data(ocr_im, output_type=pyt.Output.DATAFRAME)
        for item in txt.split('\n'):
            param = item.split(':')[0]
            if levenshtein(param, 'name', case_sensitive=False, ignore_whitespaces=True) <= 1:
                res = OCResult(text=item)
                res.OCRDataFrame = txt_data[txt_data['text'].isin(item.split())].copy()
                res.OCRDataFrame = \
                    res.OCRDataFrame[res.OCRDataFrame['par_num'] ==
                                     txt_data[txt_data['text'].isin(item.split())].groupby('par_num').size().idxmax()]
                res.OCRDataFrame.dropna(axis=0, inplace=True)
                self.OCRScanParameters.Name = res
            elif levenshtein(param, 'id', case_sensitive=False, ignore_whitespaces=True) <= 1:
                res = OCResult(text=item)
                res.OCRDataFrame = txt_data[txt_data['text'].isin(item.split())].copy()
                res.OCRDataFrame = \
                    res.OCRDataFrame[res.OCRDataFrame['par_num'] ==
                                     txt_data[txt_data['text'].isin(item.split())].groupby('par_num').size().idxmax()]
                res.OCRDataFrame.dropna(axis=0, inplace=True)
                self.OCRScanParameters.ID = res

        # OCR DoB
        ocr_im = crop[:, maxcoltrans_right[1]:]
        ocr_im = crop_to_bounding_box(ocr_im)
        ocr_im = add_boarder(ocr_im)
        cv2.imshow('OCR Input', ocr_im.astype(np.uint8))
        cv2.waitKey(1)

        txt = pyt.image_to_string(ocr_im)
        txt_data = pyt.image_to_data(ocr_im, output_type=pyt.Output.DATAFRAME)
        param = txt.split(':')[0]
        if levenshtein(param, 'dob', case_sensitive=False, ignore_whitespaces=True) <= 1:
            res = OCResult(text=txt)
            res.OCRDataFrame = txt_data[txt_data['text'].isin(txt.split())].copy()
            res.OCRDataFrame.dropna(axis=0, inplace=True)
            self.OCRScanParameters.DoB = res

        #################
        # Below top bar #
        #################
        if self.Landmarks.Cross1 is not None:
            btm = self.Landmarks.Cross1.Center[1]
        else:
            btm = self.Landmarks.ImageCenter[1]

        crop = self.Image[int(self.Landmarks.Top2):int(btm), :]
        crop = crop_to_bounding_box(crop)
        crop = remove_frame(crop)

        # remove optional partial line at top
        if not np.all(crop[0, :]):
            rowtrans = get_transitions(crop, 1)
            crop = crop[rowtrans[0][1]:, :]

        crop = crop_to_bounding_box(crop)

        # split in 4
        coltrans = get_transitions(crop[:int(crop.shape[0]/3), :], 0, True)
        largest_gaps = sorted(coltrans, key=lambda x: x[0], reverse=True)[:3]  # large whitespace
        g1r, g2r, g3r = sorted(largest_gaps, key=lambda x: x[1])
        coltrans = sorted(get_transitions(crop[:int(crop.shape[0] / 3), :], 0, False), key=lambda x: x[2])
        g1l = coltrans[g1r[2] - 1]
        g2l = coltrans[g2r[2] - 1]
        g3l = coltrans[g3r[2] - 1]

        # First column
        ocr_im = crop[:, :int(g1l[1] + 1)]
        ocr_im = crop_to_bounding_box(ocr_im)
        ocr_im = add_boarder(ocr_im)
        cv2.imshow('OCR Input', ocr_im.astype(np.uint8))
        cv2.waitKey(1)

        txt = pyt.image_to_string(ocr_im)
        txt_data = pyt.image_to_data(ocr_im, output_type=pyt.Output.DATAFRAME)
        for item in txt.split('\n'):
            param = item.split(':')[0]
            if levenshtein(param, 'Fixation Monitor', case_sensitive=False, ignore_whitespaces=True) <= 2:
                res = OCResult(text=item)
                res.OCRDataFrame = txt_data[txt_data['text'].isin(item.split())].copy()
                res.OCRDataFrame = \
                    res.OCRDataFrame[res.OCRDataFrame['par_num'] ==
                                     txt_data[txt_data['text'].isin(item.split())].groupby('par_num').size().idxmax()]
                res.OCRDataFrame.dropna(axis=0, inplace=True)
                self.OCRScanParameters.FixationMonitor = res
            elif levenshtein(param, 'Fixation Target', case_sensitive=False, ignore_whitespaces=True) <= 2:
                res = OCResult(text=item)
                res.OCRDataFrame = txt_data[txt_data['text'].isin(item.split())].copy()
                res.OCRDataFrame = \
                    res.OCRDataFrame[res.OCRDataFrame['par_num'] ==
                                     txt_data[txt_data['text'].isin(item.split())].groupby('par_num').size().idxmax()]
                res.OCRDataFrame.dropna(axis=0, inplace=True)
                self.OCRScanParameters.FixationTarget = res
            elif levenshtein(param, 'Fixation Losses', case_sensitive=False, ignore_whitespaces=True) <= 2:
                res = OCResult(text=item)
                res.OCRDataFrame = txt_data[txt_data['text'].isin(item.split())].copy()
                res.OCRDataFrame = \
                    res.OCRDataFrame[res.OCRDataFrame['par_num'] ==
                                     txt_data[txt_data['text'].isin(item.split())].groupby('par_num').size().idxmax()]
                res.OCRDataFrame.dropna(axis=0, inplace=True)
                self.OCRScanParameters.FixationLosses = res
            elif levenshtein(param, 'False POS Errors', case_sensitive=False, ignore_whitespaces=True) <= 2:
                res = OCResult(text=item)
                res.OCRDataFrame = txt_data[txt_data['text'].isin(item.split())].copy()
                res.OCRDataFrame = \
                    res.OCRDataFrame[res.OCRDataFrame['par_num'] ==
                                     txt_data[txt_data['text'].isin(item.split())].groupby('par_num').size().idxmax()]
                res.OCRDataFrame.dropna(axis=0, inplace=True)
                self.OCRScanParameters.FalsePosErrors = res
            elif levenshtein(param, 'False NEG Errors', case_sensitive=False, ignore_whitespaces=True) <= 2:
                res = OCResult(text=item)
                res.OCRDataFrame = txt_data[txt_data['text'].isin(item.split())].copy()
                res.OCRDataFrame = \
                    res.OCRDataFrame[res.OCRDataFrame['par_num'] ==
                                     txt_data[txt_data['text'].isin(item.split())].groupby('par_num').size().idxmax()]
                res.OCRDataFrame.dropna(axis=0, inplace=True)
                self.OCRScanParameters.FalseNegErrors = res
            elif levenshtein(param, 'Test Duration', case_sensitive=False, ignore_whitespaces=True) <= 2:
                res = OCResult(text=item)
                res.OCRDataFrame = txt_data[txt_data['text'].isin(item.split())].copy()
                res.OCRDataFrame = \
                    res.OCRDataFrame[res.OCRDataFrame['par_num'] ==
                                     txt_data[txt_data['text'].isin(item.split())].groupby('par_num').size().idxmax()]
                res.OCRDataFrame.dropna(axis=0, inplace=True)
                self.OCRScanParameters.TestDuration = res
            elif levenshtein(param, 'Fovea', case_sensitive=False, ignore_whitespaces=True) <= 2:
                res = OCResult(text=item)
                res.OCRDataFrame = txt_data[txt_data['text'].isin(item.split())].copy()
                res.OCRDataFrame = \
                    res.OCRDataFrame[res.OCRDataFrame['par_num'] ==
                                     txt_data[txt_data['text'].isin(item.split())].groupby('par_num').size().idxmax()]
                res.OCRDataFrame.dropna(axis=0, inplace=True)
                self.OCRScanParameters.Fovea = res
            else:
                if levenshtein(item, 'Peripheral 60-4 Threshold Test') <= 5:
                    res = OCResult(text=item)
                    res.OCRDataFrame = txt_data[txt_data['text'].isin(txt.split())]
                    res.OCRDataFrame = \
                        res.OCRDataFrame[res.OCRDataFrame['par_num'] ==
                                         txt_data[txt_data['text'].isin(item.split())].groupby(
                                             'par_num').size().idxmax()]
                    res.OCRDataFrame.dropna(axis=0, inplace=True)
                    self.OCRScanParameters.TestType = res

        # Second column
        ocr_im = crop[:, int(g1r[1]):int(g2l[1] + 1)]
        ocr_im = crop_to_bounding_box(ocr_im)
        ocr_im = add_boarder(ocr_im)
        cv2.imshow('OCR Input', ocr_im.astype(np.uint8))
        cv2.waitKey(1)

        txt = pyt.image_to_string(ocr_im)
        txt_data = pyt.image_to_data(ocr_im, output_type=pyt.Output.DATAFRAME)
        for item in txt.split('\n'):
            param = item.split(':')[0]
            if levenshtein(param, 'Stimulus', case_sensitive=False, ignore_whitespaces=True) <= 2:
                res = OCResult(text=item)
                res.OCRDataFrame = txt_data[txt_data['text'].isin(item.split())].copy()
                res.OCRDataFrame = \
                    res.OCRDataFrame[res.OCRDataFrame['par_num'] ==
                                     txt_data[txt_data['text'].isin(item.split())].groupby('par_num').size().idxmax()]
                res.OCRDataFrame.dropna(axis=0, inplace=True)
                self.OCRScanParameters.Stimulus = res
            elif levenshtein(param, 'Background', case_sensitive=False, ignore_whitespaces=True) <= 2:
                res = OCResult(text=item)
                res.OCRDataFrame = txt_data[txt_data['text'].isin(item.split())].copy()
                res.OCRDataFrame = \
                    res.OCRDataFrame[res.OCRDataFrame['par_num'] ==
                                     txt_data[txt_data['text'].isin(item.split())].groupby('par_num').size().idxmax()]
                res.OCRDataFrame.dropna(axis=0, inplace=True)
                self.OCRScanParameters.Background = res
            elif levenshtein(param, 'Strategy', case_sensitive=False, ignore_whitespaces=True) <= 2:
                res = OCResult(text=item)
                res.OCRDataFrame = txt_data[txt_data['text'].isin(item.split())].copy()
                res.OCRDataFrame = \
                    res.OCRDataFrame[res.OCRDataFrame['par_num'] ==
                                     txt_data[txt_data['text'].isin(item.split())].groupby('par_num').size().idxmax()]
                res.OCRDataFrame.dropna(axis=0, inplace=True)
                self.OCRScanParameters.Strategy = res

        # Third column
        ocr_im = crop[:, int(g2r[1]):int(g3l[1] + 1)]
        ocr_im = crop_to_bounding_box(ocr_im)
        ocr_im = add_boarder(ocr_im)
        cv2.imshow('OCR Input', ocr_im.astype(np.uint8))
        cv2.waitKey(1)

        txt = pyt.image_to_string(ocr_im)
        txt_data = pyt.image_to_data(ocr_im, output_type=pyt.Output.DATAFRAME)
        for item in txt.split('\n'):
            param = item.split(':')[0]
            if levenshtein(param, 'Pupil Diameter', case_sensitive=False, ignore_whitespaces=True) <= 2:
                res = OCResult(text=item)
                res.OCRDataFrame = txt_data[txt_data['text'].isin(item.split())].copy()
                res.OCRDataFrame = \
                    res.OCRDataFrame[res.OCRDataFrame['par_num'] ==
                                     txt_data[txt_data['text'].isin(item.split())].groupby('par_num').size().idxmax()]
                res.OCRDataFrame.dropna(axis=0, inplace=True)
                self.OCRScanParameters.PupilDiameter = res
            elif levenshtein(param, 'Visual Acuity', case_sensitive=False, ignore_whitespaces=True) <= 2:
                res = OCResult(text=item)
                res.OCRDataFrame = txt_data[txt_data['text'].isin(item.split())].copy()
                res.OCRDataFrame = \
                    res.OCRDataFrame[res.OCRDataFrame['par_num'] ==
                                     txt_data[txt_data['text'].isin(item.split())].groupby('par_num').size().idxmax()]
                res.OCRDataFrame.dropna(axis=0, inplace=True)
                self.OCRScanParameters.VisualAcuity = res
            elif levenshtein(param, 'RX', case_sensitive=False, ignore_whitespaces=True) <= 1:
                res = OCResult(text=item)
                res.OCRDataFrame = txt_data[txt_data['text'].isin(item.split())].copy()
                res.OCRDataFrame = \
                    res.OCRDataFrame[res.OCRDataFrame['par_num']  ==
                                     txt_data[txt_data['text'].isin(item.split())].groupby('par_num').size().idxmax()]
                res.OCRDataFrame.dropna(axis=0, inplace=True)
                self.OCRScanParameters.Refraction = res

        # Fourth column
        ocr_im = crop[:, int(g3r[1]):]
        ocr_im = crop_to_bounding_box(ocr_im)
        ocr_im = add_boarder(ocr_im)
        cv2.imshow('OCR Input', ocr_im.astype(np.uint8))
        cv2.waitKey(1)

        txt = pyt.image_to_string(ocr_im)
        txt_data = pyt.image_to_data(ocr_im, output_type=pyt.Output.DATAFRAME)
        for item in txt.split('\n'):
            param = item.split(':')[0]
            if levenshtein(param, 'Date', case_sensitive=False, ignore_whitespaces=True) <= 1:
                res = OCResult(text=item)
                res.OCRDataFrame = txt_data[txt_data['text'].isin(item.split())].copy()
                res.OCRDataFrame = \
                    res.OCRDataFrame[res.OCRDataFrame['par_num'] ==
                                     txt_data[txt_data['text'].isin(item.split())].groupby('par_num').size().idxmax()]
                res.OCRDataFrame.dropna(axis=0, inplace=True)
                self.OCRScanParameters.Date = res
            elif levenshtein(param, 'Time', case_sensitive=False, ignore_whitespaces=True) <= 1:
                res = OCResult(text=item)
                res.OCRDataFrame = txt_data[txt_data['text'].isin(item.split())].copy()
                res.OCRDataFrame = \
                    res.OCRDataFrame[res.OCRDataFrame['par_num'] ==
                                     txt_data[txt_data['text'].isin(item.split())].groupby('par_num').size().idxmax()]
                res.OCRDataFrame.dropna(axis=0, inplace=True)
                self.OCRScanParameters.Time = res
            elif levenshtein(param, 'Age', case_sensitive=False, ignore_whitespaces=True) <= 1:
                res = OCResult(text=item)
                res.OCRDataFrame = txt_data[txt_data['text'].isin(item.split())].copy()
                res.OCRDataFrame = \
                    res.OCRDataFrame[res.OCRDataFrame['par_num'] ==
                                     txt_data[txt_data['text'].isin(item.split())].groupby('par_num').size().idxmax()]
                res.OCRDataFrame.dropna(axis=0, inplace=True)
                self.OCRScanParameters.Age = res

        cv2.destroyWindow('OCR Input')

    def unpack_scanparameters(self):
        if not self.OCRScanParameters:
            self.read_scanparameters()

        # Analysis Type
        self.ScanParameters.AnalysisType = self.OCRScanParameters.AnalysisType.Text

        # Eye
        txt = self.OCRScanParameters.Eye.Text
        try:
            val = txt.split(':', 1)[1].strip()
            if levenshtein(val, 'Right', case_sensitive=False, ignore_whitespaces=True) <= 1:
                self.ScanParameters.Eye = 'Right'
            elif levenshtein(val, 'Left', case_sensitive=False, ignore_whitespaces=True) <= 1:
                self.ScanParameters.Eye = 'Left'
            else:
                print('Cannot determine eye from {}'.format(txt))
        except IndexError:
            print('Cannot determine eye from {}'.format(txt))

        # Name
        txt = self.OCRScanParameters.Name.Text
        try:
            val = txt.split(':', 1)[1].strip()
            self.ScanParameters.Name = val
        except IndexError:
            print('Cannot determine name from {}'.format(txt))

        # ID
        txt = self.OCRScanParameters.ID.Text
        try:
            val = txt.split(':', 1)[1].strip()
            self.ScanParameters.ID = int(''.join([nn for nn in val if nn.isdigit()]))
        except IndexError:
            print('Cannot determine ID from {}'.format(txt))

        # DoB
        txt = self.OCRScanParameters.DoB.Text
        try:
            val = re.search(r'\d{2}-\d{2}-\d{4}', txt).group()
            self.ScanParameters.DoB = dt.datetime.strptime(val, '%d-%m-%Y').date()
        except AttributeError:
            print('Cannot determine DoB from {}'.format(txt))

        # Test type
        txt = self.OCRScanParameters.TestType.Text
        if levenshtein(txt, 'Peripheral 60-4 Threshold Test', case_sensitive=False, ignore_whitespaces=True) <= 5:
            self.ScanParameters.TestType = 'Peripheral 60-4 Threshold Test'
        elif levenshtein(txt, 'Central 30-2 Threshold Test', case_sensitive=False, ignore_whitespaces=True) <= 5:
            self.ScanParameters.TestType = 'Central 30-2 Threshold Test'
        else:
            print('Cannot determine Test Type from {}'.format(txt))

        # Fixation Monitor
        txt = self.OCRScanParameters.FixationMonitor.Text
        try:
            val = txt.split(':', 1)[1].strip()
            self.ScanParameters.FixationMonitor = val
        except IndexError:
            print('Cannot determine Fixation Monitor from {}'.format(txt))

        # Fixation Target
        txt = self.OCRScanParameters.FixationTarget.Text
        try:
            val = txt.split(':', 1)[1].strip()
            self.ScanParameters.FixationTarget = val
        except IndexError:
            print('Cannot determine Fixation Target from {}'.format(txt))

        # Fixation Losses
        txt = self.OCRScanParameters.FixationLosses.Text
        try:
            val = txt.split(':', 1)[1].strip()
            self.ScanParameters.FixationLosses = val

            self.ScanParameters._FixationLosses_Numerator = int(re.search(r'\d+', (val.split('/')[0])).group())
            self.ScanParameters._FixationLosses_Denominator = int(re.search(r'\d+', (val.split('/')[1])).group())
        except IndexError:
            print('Cannot determine Fixation Losses from {}'.format(txt))
        except AttributeError:
            print('Cannot determine Numerator and Denominator for Fixation Losses from {}'.format(txt))

        # False Pos Errors
        txt = self.OCRScanParameters.FalsePosErrors.Text
        try:
            val = txt.split(':', 1)[1].strip()
            if '/' in val:
                self.ScanParameters.FalsePosErrors = val

                self.ScanParameters._FalsePosErrors_Numerator = int(re.search(r'\d+', (val.split('/')[0])).group())
                self.ScanParameters._FalsePosErrors_Denominator = int(re.search(r'\d+', (val.split('/')[1])).group())
                self.ScanParameters._FalsePosErrors_Unit = ''
            elif '%' in val:
                self.ScanParameters.FalsePosErrors = int(re.search(r'\d+', val).group())
                self.ScanParameters._FalsePosErrors_Numerator = np.NaN
                self.ScanParameters._FalsePosErrors_Denominator = np.NaN
                self.ScanParameters._FalsePosErrors_Unit = '%'
        except IndexError:
            print('Cannot determine False Pos Errors from {}'.format(txt))
        except AttributeError:
            print('Cannot determine Numerator and Denominator for False Pos Errors from {}'.format(txt))

        # False Neg Errors
        txt = self.OCRScanParameters.FalseNegErrors.Text
        try:
            val = txt.split(':', 1)[1].strip()
            if '/' in val:
                self.ScanParameters.FalseNegErrors = val

                self.ScanParameters._FalseNegErrors_Numerator = int(re.search(r'\d+', (val.split('/')[0])).group())
                self.ScanParameters._FalseNegErrors_Denominator = int(re.search(r'\d+', (val.split('/')[1])).group())
                self.ScanParameters._FalseNegErrors_Unit = ''
            elif '%' in val:
                self.ScanParameters.FalseNegErrors = int(re.search(r'\d+', val).group())
                self.ScanParameters._FalseNegErrors_Numerator = np.NaN
                self.ScanParameters._FalseNegErrors_Denominator = np.NaN
                self.ScanParameters._FalseNegErrors_Unit = '%'

        except IndexError:
            print('Cannot determine False Neg Errors from {}'.format(txt))
        except AttributeError:
            print('Cannot determine Numerator and Denominator for False Neg Errors from {}'.format(txt))

        # Test Duration
        txt = self.OCRScanParameters.TestDuration.Text
        try:
            val = re.search(r'\d{2}:\d{2}', txt).group()
            self.ScanParameters.TestDuration = dt.datetime.strptime(val, '%M:%S').time()
        except AttributeError:
            print('Cannot determine Test Duration from {}'.format(txt))

        # Fovea
        txt = self.OCRScanParameters.Fovea.Text
        try:
            val = txt.split(':', 1)[1].strip()
            self.ScanParameters.Fovea = val
        except AttributeError:
            print('Cannot determine Fovea from {}'.format(txt))

        # Stimulus
        txt = self.OCRScanParameters.Stimulus.Text
        try:
            val = txt.split(':', 1)[1].strip()
            self.ScanParameters.Stimulus = val
        except AttributeError:
            print('Cannot determine Stimulus from {}'.format(txt))

        # Background
        txt = self.OCRScanParameters.Background.Text
        try:
            val = txt.split(':', 1)[1].strip()
            self.ScanParameters.Background = float(re.search(r'\d+(.\d+)?', val).group())
            self.ScanParameters._Background_Unit = re.search(r'[a-zA-Z]+', val).group()
        except AttributeError:
            print('Cannot determine Background from {}'.format(txt))

        # Strategy
        txt = self.OCRScanParameters.Strategy.Text
        try:
            val = txt.split(':', 1)[1].strip()
            self.ScanParameters.Strategy = val
        except AttributeError:
            print('Cannot determine Strategy from {}'.format(txt))

        # Pupil Diameter
        txt = self.OCRScanParameters.PupilDiameter.Text
        try:
            val = txt.split(':', 1)[1].strip()
            self.ScanParameters.PupilDiameter = float(re.search(r'\d+(.\d+)?', val).group())
            self.ScanParameters._PupilDiameter_Unit = re.search(r'[a-zA-Z]+', val).group()
        except AttributeError:
            print('Cannot determine Pupil Diameter from {}'.format(txt))

        # Visual Acuity
        txt = self.OCRScanParameters.VisualAcuity.Text
        try:
            val = txt.split(':', 1)[1].strip()
            self.ScanParameters.VisualAcuity = val
        except AttributeError:
            print('Cannot determine Visual Acuity from {}'.format(txt))

        # Refraction
        txt = self.OCRScanParameters.Refraction.Text
        try:
            val = txt.split(':', 1)[1].strip()
            self.ScanParameters.Refraction = val
        except AttributeError:
            print('Cannot determine Refraction from {}'.format(txt))

        # Date
        txt = self.OCRScanParameters.Date.Text
        try:
            val = re.search(r'\d{2}-\d{2}-\d{4}', txt).group()
            self.ScanParameters.Date = dt.datetime.strptime(val, '%d-%m-%Y').date()
        except AttributeError:
            print('Cannot determine Date from {}'.format(txt))

        # Time
        txt = self.OCRScanParameters.Time.Text
        try:
            val = re.search(r'\d{2}:\d{2}', txt).group()
            self.ScanParameters.Time = dt.datetime.strptime(val, '%M:%S').time()
        except AttributeError:
            print('Cannot determine Time from {}'.format(txt))

        # Age
        txt = self.OCRScanParameters.Age.Text
        try:
            val = txt.split(':', 1)[1].strip()
            self.ScanParameters.Age = float(val)
        except AttributeError:
            print('Cannot determine Age from {}'.format(txt))
    
    def mask_maps(self):
        if self.Protocol == '60-4':
            self.ThresholdMap[~HFA_60_4_MASK] = np.NaN
            self.ThresholdStringMap[~HFA_60_4_MASK] = ''
            self.ThresholdConfidenceMap[~HFA_60_4_MASK] = np.NaN

            self.DefectDepthMap[~HFA_60_4_MASK] = np.NaN
            self.DefectDepthStringMap[~HFA_60_4_MASK] = ''
            self.DefectDepthConfidenceMap[~HFA_60_4_MASK] = np.NaN

    def convert_maps_to_dataframes(self):
        if self.Protocol == '60-4':
            self.ThresholdDf = hfa_60_4_map_to_dataframe(self.ThresholdMap, self.ScanParameters.Eye, 
                                                         self.__MAPTODATAFRAMEPROTOCOL)
            self.ThresholdStringDf = hfa_60_4_map_to_dataframe(self.ThresholdStringMap, self.ScanParameters.Eye,
                                                               self.__MAPTODATAFRAMEPROTOCOL)
            self.ThresholdConfidenceDf = hfa_60_4_map_to_dataframe(self.ThresholdConfidenceMap, self.ScanParameters.Eye,
                                                                   self.__MAPTODATAFRAMEPROTOCOL)

            self.DefectDepthDf = hfa_60_4_map_to_dataframe(self.DefectDepthMap, self.ScanParameters.Eye,
                                                           self.__MAPTODATAFRAMEPROTOCOL)
            self.DefectDepthStringDf = hfa_60_4_map_to_dataframe(self.DefectDepthStringMap, self.ScanParameters.Eye,
                                                                 self.__MAPTODATAFRAMEPROTOCOL)
            self.DefectDepthConfidenceDf = hfa_60_4_map_to_dataframe(self.DefectDepthConfidenceMap,
                                                                     self.ScanParameters.Eye,
                                                                     self.__MAPTODATAFRAMEPROTOCOL)
        else:
            raise NotImplementedError('Map To Dataframes is not implemented for {}'.format(self.Protocol))

    def convert_scanparameters_to_dataframe(self):
        res = pd.DataFrame(columns=['Value', 'Confidence'])

        for item in self.ScanParameters.__dict__.items():
            if not item[0].startswith('_'):
                res.loc[item[0], 'Value'] = item[1]

                try:
                    res.loc[item[0], 'Confidence'] = self.OCRScanParameters.__dict__[item[0]].df_min_confidence()
                except (IndexError, AttributeError):
                    res.loc[item[0], 'Confidence'] = np.NaN
            else:
                ind, col = item[0].lstrip('_').split('_')
                res.loc[ind, col] = item[1]

        if 'Unit' in res.columns:
            nullunit = pd.isnull(res['Unit'])
            res.loc[nullunit, 'Unit'] = ''

        self.ScanParameterDf = res

    def save(self, savedir):
        if not self.isProcessed:
            raise ValueError('HFA scan should be processed before saving')

        sjname = re.search(r'Dysfo\d{3}', ''.join(self.ScanParameters.Name.split())).group()
        testdir = os.path.join(savedir, '{}_{}_{}_OCROutput'.format(sjname,
                                                                    self.ScanParameters.Date.strftime('%Y%m%d'),
                                                                    self.ScanParameters.Eye))
        csvdir = os.path.join(testdir, 'csvs')
        dfdir = os.path.join(testdir, 'dataframes')
        
        os.makedirs(savedir, exist_ok=True)
        os.makedirs(testdir, exist_ok=True)
        os.makedirs(csvdir, exist_ok=True)
        os.makedirs(dfdir, exist_ok=True)
        
        # Scan Parameters
        self.ScanParameterDf.to_pickle(os.path.join(dfdir, 
                                                    '{}_{}'.format(sjname, 'ScanParameters.df')))
        self.ScanParameterDf.to_csv(os.path.join(csvdir,
                                                 '{}_{}'.format(sjname, 'ScanParameters.csv')))

        # Defect depth
        self.DefectDepthDf.to_pickle(os.path.join(dfdir,
                                                  '{}_{}'.format(sjname, 'DefectDepth.df')))
        self.DefectDepthDf.to_csv(os.path.join(csvdir,
                                               '{}_{}'.format(sjname, 'DefectDepth.csv')))
        self.DefectDepthStringDf.to_pickle(os.path.join(dfdir,
                                                        '{}_{}'.format(sjname,
                                                                       'DefectDepthString.df')))
        self.DefectDepthStringDf.to_csv(os.path.join(csvdir,
                                                     '{}_{}'.format(sjname, 'DefectDepthString.csv')))
        self.DefectDepthConfidenceDf.to_pickle(os.path.join(dfdir,
                                                            '{}_{}'.format(sjname,
                                                                           'DefectDepthConfidence.df')))
        self.DefectDepthConfidenceDf.to_csv(os.path.join(csvdir,
                                                         '{}_{}'.format(sjname,
                                                                        'DefectDepthConfidence.csv')))
                
        # Threshold Mesurements
        self.ThresholdDf.to_pickle(os.path.join(dfdir,
                                                '{}_{}'.format(sjname, 'Threshold.df')))
        self.ThresholdDf.to_csv(os.path.join(csvdir,
                                             '{}_{}'.format(sjname, 'Threshold.csv')))
        self.ThresholdStringDf.to_pickle(os.path.join(dfdir,
                                                      '{}_{}'.format(sjname, 'ThresholdString.df')))
        self.ThresholdStringDf.to_csv(os.path.join(csvdir,
                                                   '{}_{}'.format(sjname, 'ThresholdString.csv')))
        self.ThresholdConfidenceDf.to_pickle(os.path.join(dfdir,
                                                          '{}_{}'.format(sjname, 
                                                                         'ThresholdConfidence.df')))
        self.ThresholdConfidenceDf.to_csv(os.path.join(csvdir,
                                                       '{}_{}'.format(sjname, 
                                                                      'ThresholdConfidence.csv')))
        

        #to excel
        writer = pd.ExcelWriter(os.path.join(testdir, '{}_OCRResult.xlsx'.format(sjname)),
                                engine='xlsxwriter')
        s1 = self.ScanParameterDf.style.apply(highlight_conf, subset=['Confidence'])
        s1.to_excel(writer, sheet_name='ScanParameters')

        self.ThresholdDf.to_excel(writer, sheet_name='Threshold')
        self.ThresholdConfidenceDf.style.apply(highlight_conf).to_excel(writer, sheet_name='ThresholdConfidence')
        self.ThresholdStringDf.to_excel(writer, sheet_name='ThresholdString')
        self.DefectDepthDf.to_excel(writer, sheet_name='DefectDepth')
        self.DefectDepthConfidenceDf.style.apply(highlight_conf).to_excel(writer, sheet_name='DefectDepthConfidence')
        self.DefectDepthStringDf.to_excel(writer, sheet_name='DefectDepthString')

        writer.save()
        writer.close()

    def process(self):
        self.guess_protocol()
        self.identify_landmarks()
        self.read_scanparameters()
        self.unpack_scanparameters()

        self.obtain_gridocrois()
        for item in self.ThresholdRois:
            item.recognize(self.Image, character_whitelist='0123456789<{(.,')
        for item in self.DefectDepthRois:
            item.recognize(self.Image, character_whitelist='0123456789<{(.,-')

        self.ThresholdStringMap = np.array([item.Result.Text for item in self.ThresholdRois]).reshape((9, 10))
        self.ThresholdMap = unstring_maps(self.ThresholdStringMap, setbelow0to=self.__BELOW0)
        self.ThresholdConfidenceMap = np.array([item.Result.Confidence for item in self.ThresholdRois]).reshape((9, 10))

        self.DefectDepthStringMap = np.array([item.Result.Text for item in self.DefectDepthRois]).reshape((9, 10))
        self.DefectDepthMap = unstring_maps(self.DefectDepthStringMap, setbelow0to=self.__BELOW0)
        self.DefectDepthConfidenceMap = \
            np.array([item.Result.Confidence for item in self.DefectDepthRois]).reshape((9, 10))

        self.mask_maps()
        self.convert_scanparameters_to_dataframe()
        self.convert_maps_to_dataframes()

        self.isProcessed = True

    def close(self):
        cv2.destroyAllWindows()


def hfa_60_4_map_to_dataframe(hfamap, lateraliy, protocol):
    """

    :param hfamap:
    :param lateraliy: either 'Left' or 'Right'
    :param protocol: Either 'lr' or 'nt'.
    :return:
    """

    index = ['S4', 'S3', 'S2', 'S1', 'I1', 'I2', 'I3', 'I4', 'I5']
    
    if protocol.lower() == 'nt':
        cols = ['N5', 'N4', 'N3', 'N2', 'N1', 'T1', 'T2', 'T3', 'T4', 'T5']
        if lateraliy.lower() == 'right':
            res = pd.DataFrame(index=index, columns=cols, data=hfamap)
        elif lateraliy.lower() == 'left':
            res = pd.DataFrame(index=index, columns=cols[::-1], data=hfamap)
        else:
            raise ValueError('Laterality should be "Left" or "Right"')
    elif protocol.lower() == 'lr':
        cols = ['L5', 'L4', 'L3', 'L2', 'L1', 'R1', 'R2', 'R3', 'R4', 'R5']
        res = pd.DataFrame(index=index, columns=cols, data=hfamap)
    else:
        raise ValueError('Protocol should be "nt" or "lr"')
    return res


def hfa_60_4_get_landmarks(img, show_process=True, **kwargs):
    preproc = ocrutils.OCRPreProcessor(img=img)
    if show_process:
        preproc.overlay_drawingboard()

    preproc.apply_on_image(True, cv2.threshold, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    if show_process:
        preproc.overlay_drawingboard()

    lines = preproc.identify_lines(**kwargs)
    linesegments = ocrutils.houghlinep_to_linearlinesegmets(lines)
    # preproc.apply_on_image(True, cv2.cvtColor, cv2.COLOR_GRAY2BGR)
    preproc.apply_on_image(True, imageutils.convert_to_gbr)
    if show_process:
        preproc.draw_lines(linesegments, thickness=3)
        preproc.overlay_drawingboard()

    toplines, bottomlines = hfa_identify_top_and_bottom_lines(linesegments)
    top1 = min([tl.Center[1] for tl in toplines])
    top2 = max([tl.Center[1] for tl in toplines])
    bottom1 = min([bl.Center[1] for bl in bottomlines])
    bottom2 = max([bl.Center[1] for bl in bottomlines])
    image_center = (np.average([bl.Center[0] for bl in bottomlines] + [tl.Center[0] for tl in toplines]),
                    (bottom1 - top2) / 2)
    if show_process:
        preproc.draw_lines(toplines, color=(0, 255, 0), thickness=5)
        preproc.draw_lines(bottomlines, color=(0, 255, 0), thickness=5)
        preproc.draw_point((int(image_center[0]), int(image_center[1])),
                           color=(0, 255, 0), thickness=15)
        preproc.overlay_drawingboard()

    linesegments = [item for item in linesegments if
                    (bottom1 >
                     item.Center[1] >
                     top2)]

    iscts = ocrutils.calc_perpendicular_intersections(linesegments, hlineslope_th=0.1, vlineslope_th=10)
    if show_process:
        preproc.draw_intersections(iscts, thickness=20, radius=30, fill=False)
        preproc.overlay_drawingboard()

    clusters = ocrutils.intersections_to_clusters(iscts, cluster_radius=60, tune_cluster_radius=True)
    if show_process:
        preproc.draw_clusters(clusters, color=(255, 0, 0), thickness=10, radius=30)
        preproc.overlay_drawingboard()

    cross1, cross2, cross3 = \
        hfa_60_4_identify_landmarks_from_clusters(clusters=clusters, image_center=image_center)

    if show_process:
        preproc.draw_clusters([cross1, cross2, cross3],
                              color=(0, 255, 0), thickness=10, radius=30)
        preproc.overlay_drawingboard()

    lms = hfa_landmarks._make([top1, top2, bottom1, bottom2, image_center, cross1, cross2, cross3, None, None])

    del preproc
    return lms


def hfa_60_4_identify_landmarks_from_clusters(clusters, image_center=None):

    if image_center is None:  # if there is no center specified, make an arbitrary center
        center = (np.average([cluster.Center[0]for cluster in clusters]),
                  np.average([cluster.Center[1]for cluster in clusters]))
    else:
        center = image_center

    l1_candidates = [cluster for cluster in clusters if cluster.Center[1] <= center[1]]
    l2_candidates = [cluster for cluster in clusters if
                     (cluster.Center[0] < center[0] and cluster.Center[1] > center[1])]
    l3_candidates = [cluster for cluster in clusters if
                     (cluster.Center[0] >= center[0] and cluster.Center[1] > center[1])]

    l1 = sorted(l1_candidates,
                key=operator.attrgetter('largest_member_length'),
                reverse=True)[0] if len(l1_candidates) > 0 else None
    l2 = sorted(l2_candidates,
                key=operator.attrgetter('largest_member_length'),
                reverse=True)[0] if len(l2_candidates) > 0 else None
    l3 = sorted(l3_candidates,
                key=operator.attrgetter('largest_member_length'),
                reverse=True)[0] if len(l3_candidates) > 0 else None

    return l1, l2, l3


def hfa_identify_top_and_bottom_lines(linearlinesegmentarray):
    maxhlinelength = max([line.Length for line in linearlinesegmentarray if (line.Slope is not None
                                                                             and abs(line.Slope) < 0.5)],
                         default=np.nan)

    vcenter = np.average([line.Center[1] for line in linearlinesegmentarray])

    toplines = []
    bottomlines = []
    for line in linearlinesegmentarray:
        if line.Length < 0.9 * maxhlinelength:
            continue

        if line.Slope is not None and abs(line.Slope) < 0.5:
            if line.Center[1] < vcenter:
                toplines.append(line)
            else:
                bottomlines.append(line)

    return toplines, bottomlines


def hfa_60_4_define_readout_grid(img, cross, show_process=True, show_zoom=True):
    """ Defines a readout grid to read all items around the cross landmark.
    
    Returns:
        tuple: the horizontal and vertical line pairs stored in lists
    """
    preproc = ocrutils.OCRPreProcessor(img=img)
    if show_process:
        preproc.overlay_drawingboard()

    preproc.apply_on_image(True, cv2.threshold, 0, 255, cv2.THRESH_BINARY_INV)
    if show_process:
        preproc.overlay_drawingboard()

    ll = cross.largest_member.shortest_line
    hl = int(ll/2)

    xoffs, yoffs = cross.rounded_center

    masked_img = imageutils.inverse_circular_mask(preproc.ModifiedImage, cross.rounded_center, hl)

    x1 = int(xoffs - hl)
    x2 = int(xoffs + hl)
    y1 = int(yoffs - hl)
    y2 = int(yoffs + hl)

    roi = masked_img[y1:y2+1, x1:x2+1]
    cycler = it.cycle([np.min(roi), np.max(roi)])

    rroi = roi.copy()
    rroi[:,
         int(rroi.shape[1] / 2) - int(0.025 * rroi.shape[1]):int(rroi.shape[1] / 2) + int(0.025 * rroi.shape[1]) + 1] \
        = np.min(rroi)

    # np.array([[next(cycler) for _ in range(len(roi[0]))]]).transpose()
    rroi[int(rroi.shape[0]/2) - int(0.025 * rroi.shape[0]):int(rroi.shape[0]/2) + int(0.025 * rroi.shape[0]) + 1, :] = \
        np.array([next(cycler) for _ in range(len(rroi))])

    rowtransitions = np.apply_along_axis(lambda x: len(list(it.accumulate([len(list(items))
                                                                           for item, items
                                                                           in it.groupby(x)]))[:-1]),
                                         1, rroi)

    # check upper half
    upperhalf = rowtransitions[:hl+1]
    urowemptylines = upperhalf == 0
    urowitems = list([item for item, items in it.groupby(urowemptylines)])
    urowtrans = [0] + list(it.accumulate([len(list(items)) for item, items in it.groupby(urowemptylines)]))

    urowgaps = [(jj, ii) for ii, jj in enumerate(np.diff(urowtrans))]
    urowselection = sorted([item for item in urowgaps if urowitems[item[1]]], reverse=True)[:4]  # major black lines
    urowselection = sorted(urowselection, key=lambda x: x[1])
    urowpairs = [np.array([urowtrans[ii+1], urowtrans[ii+2]]) for _, ii in urowselection]  # following boundries
    urowpairs[-1] = np.array([urowpairs[-1][0], hl])  # set last point to center of cross

    # Check lower half
    lowerhalf = rowtransitions[hl:]
    lrowemptylines = lowerhalf == 0
    lrowitems = list([item for item, items in it.groupby(lrowemptylines)])
    lrowtrans = list(it.accumulate([len(list(items)) for item, items in it.groupby(lrowemptylines)]))
    lrowtrans = [hl] + [item + hl for item in lrowtrans[:-1]]  # add offset and remove last transition

    lrowgaps = [(jj, ii) for ii, jj in enumerate(np.diff(lrowtrans))]
    lrowselection = sorted([item for item in lrowgaps if lrowitems[item[1]]], reverse=True)[:4]  # major black lines
    lrowselection = sorted(lrowselection, key=lambda x: x[1])
    lrowpairs = [np.array([lrowtrans[ii + 1], lrowtrans[ii + 2]]) for _, ii in lrowselection]  # following boundries
    lrowpairs = [np.array([hl, lrowtrans[lrowselection[0][1]]])] + lrowpairs  # set first point to center of cross

    croi = roi.copy()
    croi[int(croi.shape[0] / 2) - int(0.025 * rroi.shape[0]):int(rroi.shape[0] / 2) + int(0.025 * rroi.shape[0]) + 1,
         :] = np.min(roi)

    croi[:,
         int(croi.shape[0] / 2) - int(0.025 * croi.shape[0]):int(croi.shape[0] / 2) + int(0.025 * croi.shape[0]) + 1] \
        = np.array([next(cycler) for _ in range(len(croi[:, 0]))]).reshape(len(croi[:, 0]), 1)

    # check columns
    coltransitions = np.apply_along_axis(lambda x: len(list(it.accumulate([len(list(items))
                                                                           for item, items
                                                                           in it.groupby(x)]))[:-1]),
                                         0, croi)

    # check left half
    lefthalf = coltransitions[:hl+1]
    lcolemptylines = lefthalf == 0
    lcolitems = list([item for item, items in it.groupby(lcolemptylines)])
    lcoltrans = [0] + list(it.accumulate([len(list(items)) for item, items in it.groupby(lcolemptylines)]))

    lcolgaps = [(jj, ii) for ii, jj in enumerate(np.diff(lcoltrans))]
    lcolselection = sorted([item for item in lcolgaps if lcolitems[item[1]]], reverse=True)[:5]  # major black lines
    lcolselection = sorted(lcolselection, key=lambda x: x[1])

    lcolpairsindex = [item[1] for item in lcolselection] + [len(lcoltrans) - 1]
    lcolpairs = [np.array([lcoltrans[lcolpairsindex[ii] + 1], lcoltrans[lcolpairsindex[ii + 1]]]) for ii in
                 range(len(lcolselection))]

    # check right half
    righthalf = coltransitions[hl:]
    rcolemptylines = righthalf == 0
    rcolitems = list([item for item, items in it.groupby(rcolemptylines)])
    rcoltrans = list(it.accumulate([len(list(items)) for item, items in it.groupby(rcolemptylines)]))
    rcoltrans = [hl] + [item + hl for item in rcoltrans]  # add offset

    rcolgaps = [(jj, ii) for ii, jj in enumerate(np.diff(rcoltrans))]
    rcolselection = sorted([item for item in rcolgaps if rcolitems[item[1]]], reverse=True)[:5]  # major black lines
    rcolselection = sorted(rcolselection, key=lambda x: x[1])

    rcolpairsindex = [0] + [item[1] for item in rcolselection]
    rcolpairs = [np.array([rcoltrans[rcolpairsindex[ii] + 1], rcoltrans[rcolpairsindex[ii+1]]]) for ii in
                 range(len(rcolselection))]
    rcolpairs[0] = np.array([hl, rcolpairs[0][1]])

    if show_process:
        preproc.apply_on_image(True, imageutils.convert_to_gbr)

        hls = []
        for item in urowpairs:
            hls.append(misc.LinearLineSegment(x1, item[0] + y1, x2, item[0] + y1))
            hls.append(misc.LinearLineSegment(x1, item[1] + y1, x2, item[1] + y1))
        for item in lrowpairs:
            hls.append(misc.LinearLineSegment(x1, item[0] + y1, x2, item[0] + y1))
            hls.append(misc.LinearLineSegment(x1, item[1] + y1, x2, item[1] + y1))
        preproc.draw_lines(hls, thickness=5)

        vls = []
        for item in lcolpairs:
            vls.append(misc.LinearLineSegment(item[0] + x1, y1, item[0] + x1, y2))
            vls.append(misc.LinearLineSegment(item[1] + x1, y1, item[1] + x1, y2))
        for item in rcolpairs:
            vls.append(misc.LinearLineSegment(item[0] + x1, y1, item[0] + x1, y2))
            vls.append(misc.LinearLineSegment(item[1] + x1, y1, item[1] + x1, y2))
        preproc.draw_lines(vls, thickness=5)

        preproc.overlay_drawingboard()

        if show_zoom:
            roi = imageutils.convert_to_gbr(roi)
            for item in urowpairs:
                cv2.line(roi, (0, item[0]), (len(roi), item[0]), (0, 255, 0), thickness=4)
                cv2.line(roi, (0, item[1]), (len(roi), item[1]), (0, 255, 0), thickness=4)
            for item in lrowpairs:
                cv2.line(roi, (0, item[0]), (len(roi), item[0]), (0, 255, 0), thickness=4)
                cv2.line(roi, (0, item[1]), (len(roi), item[1]), (0, 255, 0), thickness=4)

            for item in lcolpairs:
                cv2.line(roi, (item[0], 0), (item[0], len(roi[:, 0])), (0, 255, 0), thickness=4)
                cv2.line(roi, (item[1], 0), (item[1], len(roi[:, 0])), (0, 255, 0), thickness=4)
            for item in rcolpairs:
                cv2.line(roi, (item[0], 0), (item[0], len(roi[:, 0])), (0, 255, 0), thickness=4)
                cv2.line(roi, (item[1], 0), (item[1], len(roi[:, 0])), (0, 255, 0), thickness=4)
            cv2.namedWindow('Region of interest', cv2.WINDOW_NORMAL)
            cv2.imshow('Region of interest', roi)

            cv2.waitKey(1)

    rowpairs = np.add(np.concatenate([urowpairs, lrowpairs]), y1)  # add offset to match original image
    colpairs = np.add(np.concatenate([lcolpairs, rcolpairs]), x1)  # add offset to match original image

    return colpairs, rowpairs


def hfa_60_4_get_gridocrois(img, landmark_cross, ocroi_basename=''):
    grows, gcols = hfa_60_4_define_readout_grid(img, landmark_cross)

    grc = it.product(gcols, grows)
    rois = list(map(lambda x: ROI(x=int(x[1][0]), y=int(x[0][0]), w=int(x[1][1]-x[1][0]), h=int(x[0][1] - x[0][0])),
                    grc))

    ocrois = [HFAGridOCRoi(ii, loc, '{}_{}'.format(ocroi_basename, loc)) for loc, ii in enumerate(rois)]

    return ocrois


class HFAGeneralOCRoi(ocrutils.OCRoi):
    def __init__(self, ocr_roi, roiname=''):
        super().__init__(ocr_roi, roiname)
        self.HFAParamType = 'General'


class HFAStatOCRoi(ocrutils.OCRoi):
    def __init__(self, ocr_roi, roiname=''):
        super().__init__(ocr_roi, roiname)
        self.HFAParamType = 'Statistical'


class HFAGridOCRoi(ocrutils.OCRoi):
    def __init__(self, ocr_roi, gridloc, roiname=''):
        super().__init__(ocr_roi, roiname)
        self.HFAParamType = 'Gridpoint'
        self.GridLoc = gridloc

    def recognize(self, img, character_whitelist='0123456789<{(.,'):

        crop = img[self.ROI.y:self.ROI.y+self.ROI.h+1,
                   self.ROI.x:self.ROI.x+self.ROI.w+1]

        crop_inv = cv2.bitwise_not(crop)
        # optionally remove frame
        xsel = ~np.all(crop_inv.astype(np.bool), axis=0)
        ysel = ~np.all(crop_inv.astype(np.bool), axis=1)

        crop = crop[ysel][:, xsel]
        crop_inv = crop_inv[ysel][:, xsel]
        cv2.namedWindow('OCR Input')
        if not np.all(crop.astype(bool)):

            contours, _ = cv2.findContours(crop_inv, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE)
            xbox, ybox, wbox, hbox = cv2.boundingRect(np.concatenate(contours))

            crop_2 = cv2.bitwise_not(crop_inv)[ybox:ybox+hbox, xbox:xbox+wbox]
            crop_3 = np.full((int(crop_2.shape[0] * 1.4), int(crop_2.shape[1] * 1.4)), 255)
            offsety3 = int(crop_2.shape[0] * 0.2)
            offsetx3 = int((crop_2.shape[1] * 0.2))
            crop_3[offsety3:offsety3 + crop_2.shape[0], offsetx3:offsetx3 + crop_2.shape[1]] = crop_2

            crop_3 = crop_3.astype(np.uint8)

            canny = cv2.Canny(crop_3, 0, 1, apertureSize=3)

            blur = cv2.GaussianBlur(canny, (3, 3), 0)

            add = cv2.bitwise_not(cv2.add(cv2.bitwise_not(crop_3), blur))

            gb = cv2.GaussianBlur(add, (3, 3), 0)

            crop_3_proc = cv2.resize(gb, tuple(np.array(crop_3.shape)[::-1] * 2), interpolation=cv2.INTER_LINEAR)

            brackets = pyt.image_to_string(crop_3_proc, config=r'--psm 11 -c tessedit_char_whitelist=()',
                                           output_type=pyt.Output.DICT)

            if ')' in brackets['text']:
                snip = crop_3_proc[:int(crop_3_proc.shape[0]*0.5) + 1, :]

                contours, _ = cv2.findContours(np.bitwise_not(snip), mode=cv2.RETR_EXTERNAL,
                                               method=cv2.CHAIN_APPROX_SIMPLE)
                xbox, ybox, wbox, hbox = cv2.boundingRect(np.concatenate(contours))
                sel = snip[ybox:ybox + hbox, xbox:xbox + wbox]
                crop_4 = np.full((int(sel.shape[0] * 1.4), int(sel.shape[1] * 1.4)), 255)
                offsety4 = int(sel.shape[0] * 0.2)
                offsetx4 = int((sel.shape[1] * 0.2))
                crop_4[offsety4:offsety4 + sel.shape[0], offsetx4:offsetx4 + sel.shape[1]] = sel
                crop_4 = crop_4.astype(crop_3_proc.dtype)
            else:
                crop_4 = crop_3_proc.copy()

            cv2.imshow('OCR Input', crop_4)
            cv2.waitKey(1)

            res1 = pyt.image_to_string(crop_4,
                                       config='--psm 6 -c tessedit_char_whitelist={}'.format(character_whitelist),
                                       output_type=pyt.Output.STRING)
            res2 = pyt.image_to_data(crop_4,
                                     config='--psm 6 -c tessedit_char_whitelist={}'.format(character_whitelist),
                                     output_type=pyt.Output.DICT)

            try:
                conf = int(res2['conf'][res2['text'].index(res1)])
            except ValueError:
                conf = 0
            if conf == -1:
                conf = 0

            res1 = res1.replace('{', '<')
            matches = re.findall(r'^<|^-|\d+', res1)
            if len(matches) == 0:
                res1 = ''
            elif matches[0] == '<':
                if len(matches) == 2:
                    res1 = matches[0] + matches[1]
                else:
                    res1 = ''
            elif matches[0] == '-':
                if len(matches) > 1:
                    res1 = ''.join([item for item in matches])
                else:
                    res1 = ''
            else:
                res1 = ''.join([item for item in matches])

            self.Result = OCResult(text=res1.replace('{', '<'), confidence=conf)

            cv2.destroyWindow('OCR Input')

        else:
            cv2.imshow('OCR Input', crop)
            cv2.waitKey(1)
            self.Result = OCResult(text='', confidence=np.NaN)
