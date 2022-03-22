import pydicom as dcm
from datetime import datetime

_patient_info_tags = {
    'PatientName': ('0010', '0010'),
    'PatientID': ('0010', '0020'),
    'PatientDoB': ('0010', '0030'),
    'PatientSex': ('0010', '0040'),
}

_general_measurement_info_tags = {
    'MeasurementDate': ('0008', '0020'),
    'MeasurementTime': ('0008', '0030'),
}

def load_dicom(filepath):
    """Loads a dicom or dicomdir file"""
    dcmdata = dcm.dcmread(fp=filepath)
    return dcmdata


def get_patient_info(dicomdata):
    """Obtains patient info from the Dicom data."""
    ret = {}
    for key, tag in _patient_info_tags.items():
        if _patient_info_tags[key] in dicomdata:
            ret[key] = dicomdata[tag].value
        else:
            ret[key] = None
    return ret


def get_general_measurement_info(dicomdata, datefmt='%Y%m%d', timefmt='%H%M%S'):
    """Obtains general measurement info from the Dicom data."""
    ret = {}
    for key, tag in _general_measurement_info_tags.items():
        if _general_measurement_info_tags[key] in dicomdata:
            if tag == ('0008', '0020'):
                ret[key] = datetime.strptime(dicomdata[tag].value, datefmt).date()
            elif tag == ('0008', '0030'):
                ret[key] = datetime.strptime(dicomdata[tag].value, timefmt).time()
            else:
                ret[key] = dicomdata[tag].value
        else:
            ret[key] = None
    return ret