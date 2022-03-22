# Python 3.6

dependencies = ['cv2', 'numpy', 'pydicom', 'pytesseract', 'wand']
# Currently, ImageMagick version 9.6.3.1 and Wand version 4.5 are used.

missing_dependencies = []
for module in dependencies:
    try:
        __import__(module)
    except ImportError:
        missing_dependencies.append(module)
    del module

if missing_dependencies:
    raise ImportError('Missing required dependencies: {}. Please install'.format(missing_dependencies))

del dependencies, missing_dependencies

