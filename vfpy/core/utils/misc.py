import operator
import tkinter as tk
import os
import numpy as np


def get_screen_resolution():
    """Obtains the current screen resolution.

    Returns:
        tuple: (screenheight, screenwidth)
    """

    root = tk.Tk()
    width = root.winfo_screenwidth()
    height = root.winfo_screenheight()

    del root

    return height, width


class LinearLineSegment:
    """A linear  line segment (Ax + By = C).

    Initiates a line segment based on two sets of coordinates (x1, y1) and (x2, y2).

    Args:
        x1 (int | float): The first x-coordinate.
        y1 (int | float): The first y-coordinate.
        x2 (int | float): The second x-coordinate.
        y2 (int | float): The second y-coordinate.

    Attributes:
        Point1 (tuple): The first line coordinate, formed by x1 and y1.
        Point2 (tuple): The second line coordinate, formed by x2 and y2.
        Center (tuple): The center coordinate of the line
    """

    def __init__(self, x1, y1, x2, y2):

        self.Point1 = (x1, y1)
        self.Point2 = (x2, y2)
        self.Center = ((x1 + x2)/2, (y1 + y2)/2)

        self.A = np.subtract(y1, y2, dtype=np.float)
        self.B = np.subtract(x2, x1, dtype=np.float)
        self.C = -np.subtract((x1 * y2), (x2 * y1), dtype=np.float)

        self.Length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

        if self.B != 0:
            self.Slope = -self.A / self.B
            self.Intercept = self.C / self.B
        else:
            self.Slope = None
            self.Intercept = y1

    def get_x_coordinate(self, y):
        """Obtains x-coordinate corresponding to the supplied y-coordinate.

        If the line is vertical, the function returns None
        Args:
            y (int | float): The y-coordinate

        Returns:
            int | None: The corresponding x-coordinate
        """
        if self.A != 0:
            x = (self.C - (self.B * y)) / self.A
            return int(x)
        else:  # the line is vertical
            return None

    def get_y_coordinate(self, x):
        """Obtains y-coordinate corresponding to the supplied x-coordinate.

        If the line is horizontal, the function returns None
        Args:
            x (int | float): The y-coordinate

        Returns:
            int | None: The corresponding y-coordinate
        """
        if self.B != 0:
            y = (self.C - (self.A * x)) / self.B
            return int(y)
        else:  # the line is horizontal
            return None

    @property
    def x1(self):
        return self.Point1[0]

    @property
    def y1(self):
        return self.Point1[1]

    @property
    def x2(self):
        return self.Point2[0]

    @property
    def y2(self):
        return self.Point2[1]


class LineSegmentIntersection:

    def __init__(self, x, y, line1=None, line2=None):
        self.Coordinate = (x, y)
        self._line1 = line1
        self._line2 = line2

    @property
    def x(self):
        return self.Coordinate[0]

    @property
    def y(self):
        return self.Coordinate[1]

    @property
    def rounded_coordinate(self):
        return int(round(self.Coordinate[0], 0)), int(round(self.Coordinate[1], 0))

    @property
    def total_line_length(self):
        if self._line1 is None or self._line2 is None:
            return np.nan
        else:
            return self._line1.Length + self._line2.Length

    @property
    def longest_line(self):
        return max([self._line1.Length, self._line2.Length])

    @property
    def shortest_line(self):
        return min([self._line1.Length, self._line2.Length])


class IntersectionCluster:
    def __init__(self, linesegmentintersection, radius=30, tune_radius=False):
        self.Center = linesegmentintersection.Coordinate
        self.Intersections = [linesegmentintersection]
        self.Radius = radius

        self.TuneRadius = tune_radius

    def add_member(self, linesegmentintersection):
        self.Intersections.append(linesegmentintersection)
        self.Center = (np.average([item.Coordinate[0] for item in self.Intersections]),
                       np.average([item.Coordinate[1] for item in self.Intersections]))

    def is_member(self, linesegmentintersection):
        distance = np.sqrt((linesegmentintersection.Coordinate[0] - self.Center[0]) ** 2 +
                           (linesegmentintersection.Coordinate[1] - self.Center[1]) ** 2)

        th = self.radius_threshold

        if distance <= th:
            return True
        else:
            return False

    @property
    def n_members(self):
        return len(self.Intersections)

    @property
    def largest_member_length(self):
        return max([item.total_line_length for item in self.Intersections])

    @property
    def largest_member(self):
        return sorted(self.Intersections,
                      key=operator.attrgetter('longest_line'),
                      reverse=True)[0] if len(self.Intersections) > 0 else None

    @property
    def total_member_length(self):
        return sum([item.total_line_length for item in self.Intersections])

    @property
    def radius_threshold(self):
        if not self.TuneRadius:
            return self.Radius
        else:
            return self.Radius / np.sqrt(self.n_members)

    @property
    def rounded_center(self):
        return int(round(self.Center[0], 0)), int(round(self.Center[1], 0))


def get_all_files_in_dir(dirpath, includesubdirs=False):
    if not includesubdirs:
        return [os.path.join(dirpath, f) for f in os.listdir(dirpath) if os.path.isfile(os.path.join(dirpath, f))]
    else:
        ret = []
        for path, subdirs, files in os.walk(dirpath):
            for name in files:
                ret.append(os.path.join(path, name))
        return ret