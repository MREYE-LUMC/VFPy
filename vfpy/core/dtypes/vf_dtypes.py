class Measurement:

    _coreAttr = [
        'PatientName',
        'PatientID',
        'PatientDoB',
        'PatientSex',
        'MeasurementType',
        'MeasurementDate',
        'MeasurementTime',
    ]

    def __init__(self, **measurement_details):

        for key in self._coreAttr:
            self.__setattr__(key, measurement_details.get(key, None))


class Perimetry(Measurement):
    """Default class for a perimetry test """

    def __init__(self, **measurement_details):
        super().__init__(**measurement_details)
        self.MeasurementType = 'Perimetry'


class StaticPerimetry(Perimetry):
    """Static perimetry test."""

    def __init__(self, **measurement_details):
        super().__init__(**measurement_details)
        self.MeasurementSubType = 'Static'


class KineticPerimetry(Perimetry):
    """Kinetic perimetry test."""

    def __init__(self, **measurement_details):
        super().__init__(**measurement_details)
        raise NotImplementedError('Kinetic Perimetry is not yet implemented')


if __name__ == '__main__':
    aa = StaticPerimetry(**{'PatientName': 'Jan'})
    print(1)