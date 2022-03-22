import cv2
import numpy as np
import wand.color
import wand.image


def pdf_to_wandimage(filepath, resolution):
    """Loads a PDF file as wand.image.Image instance."""
    return wand.image.Image(filename=filepath, resolution=resolution)


def bytearray_to_wandimage(bytearray, resolution=800):
    """Converts a byte array to a wand.image.Image instance.

    Note:
        As the wand image does not destruct itself (no use of with...), make sure sure to call .destroy() on the image
        when finished.
    """
    return wand.image.Image(blob=bytearray, resolution=resolution)


def wandimage_to_opencvimage(wandimage, **kwargs):
    """Converts a wand.image.Image instance to a cv2 image"""
    wandimage.background_color = kwargs.get('background_color', wand.color.Color('white'))
    wandimage.format = kwargs.get('format', 'tif')
    wandimage.alpha_channel = kwargs.get('alpha_channel', 'remove')

    buffer = np.asarray(bytearray(wandimage.make_blob()), dtype=np.uint8)

    opencvimage = cv2.imdecode(buffer, cv2.IMREAD_UNCHANGED)

    return opencvimage


def draw_linearlinesegment(image, linearlinesegment, **kwargs):
    cv2.line(img=image,
             pt1=linearlinesegment.Point1,
             pt2=linearlinesegment.Point2,
             color=kwargs.get('color', (0, 0, 255)),
             thickness=kwargs.get('thickness', 1),
             lineType=kwargs.get('lineType', 8),
             shift=kwargs.get('shift', 0))

def draw_point(image, center, **kwargs):
    cv2.circle(img=image,
               center=center,
               radius=kwargs.get('radius', 1),
               color=kwargs.get('color', (204, 204, 0)),
               thickness=kwargs.get('thickness', 1) if not kwargs.get('filled', True)
               else -1 * kwargs.get('thickness', 1),
               lineType=kwargs.get('lineType', 8),
               shift=kwargs.get('shift', 0))


def draw_plus(image, center, rad=5, **kwargs):
    color = kwargs.get('color', (255, 0, 0))
    thickness = kwargs.get('thickness', 1)
    lineType = kwargs.get('lineType', 8)
    shift = kwargs.get('shift', 0)
    cv2.line(img=image, pt1=(int(center[0] - rad), int(center[1])), pt2=(int(center[0] + rad), int(center[1])),
             color=color, thickness=thickness, lineType=lineType, shift=shift)
    cv2.line(img=image, pt1=(int(center[0]), int(center[1] - rad)), pt2=(int(center[0]), int(center[1] + rad)),
             color=color, thickness=thickness, lineType=lineType, shift=shift)


def estimage_colorspace(img):
    shape = img.shape
    if len(shape) == 2:
        cspace = 'GRAY'
    elif len(shape) == 3:
        ncolors = shape[-1]

        if ncolors == 3:
            cspace = 'BGR'
        else:
            cspace = None
    else:
        cspace = None

    return cspace


def convert_to_gbr(img):

    cspace = estimage_colorspace(img)

    if cspace == 'GRAY':
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        return img
    elif cspace == 'BGR':
        return img
    else:
        raise TypeError('Cannot convert image with unknown colorspace')


def inverse_circular_mask(img, circle_centre, radius, fill=0):
    """Only works on image represented by 2D np_array"""

    x1 = int(circle_centre[0] - radius)
    x2 = int(circle_centre[0] + radius)
    y1 = int(circle_centre[1] - radius)
    y2 = int(circle_centre[1] + radius)

    localimg = img[y1:y2+1, x1:x2+1]

    localmask = np.full(localimg.shape, True)

    coordvec = np.arange(-radius, radius + 1, 1, dtype=int)
    coordvec = np.square(coordvec)
    xx, yy = np.meshgrid(coordvec, coordvec)
    dist = np.sqrt(xx + yy)
    del coordvec, xx, yy

    localmask[dist <= radius] = False
    del dist

    localimg[localmask] = fill
    retim = np.full(img.shape, fill, dtype=img.dtype)
    retim[y1:y2+1, x1:x2+1] = localimg

    return retim


def crop_to_bounding_box(img, invert_for_countrous=True):
    if invert_for_countrous:
        contours, _ = cv2.findContours(cv2.bitwise_not(img), mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE)
    else:
        contours, _ = cv2.findContours(img, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE)
    xbox, ybox, wbox, hbox = cv2.boundingRect(np.concatenate(contours))

    return img[ybox:ybox + hbox, xbox:xbox + wbox]


def remove_frame(img):
    img_inv = cv2.bitwise_not(img)
    # optionally remove frame
    xsel = ~np.all(img_inv.astype(np.bool), axis=0)
    ysel = ~np.all(img_inv.astype(np.bool), axis=1)

    return img[ysel][:, xsel]


def add_boarder(img):
    board = np.full((int(img.shape[0] * 1.4), int(img.shape[1] * 1.4)), 255)
    offsety = int(img.shape[0] * 0.2)
    offsetx = int((img.shape[1] * 0.2))
    board[offsety:offsety + img.shape[0], offsetx:offsetx + img.shape[1]] = img
    return board
