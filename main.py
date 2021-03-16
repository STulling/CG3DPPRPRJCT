import sys
from skimage import data, color, io
from skimage.transform import resize, rotate
from numpy import asarray
from scipy import ndimage
import numpy as np
import matplotlib.pyplot as plt
from skimage.draw import rectangle

rgb_weights = [0.2989, 0.5870, 0.1140]

mag = None
dir = None
blur = None
data = None

def blend(a, b):
    # Extract the RGB channels
    srcRGB = a[..., :3]
    dstRGB = b[..., :3]

    # Extract the alpha channels and normalise to range 0..1
    srcA = a[..., 3] / 255.0
    dstA = b[..., 3] / 255.0

    outA = srcA + dstA * (1 - srcA)
    outRGB = (srcRGB * srcA[..., np.newaxis] + dstRGB * dstA[..., np.newaxis] * (1 - srcA[..., np.newaxis])) / outA[..., np.newaxis]

    outRGBA = np.dstack((outRGB, outA * 255)).astype(np.uint8)

    return outRGBA


def paint(canvas, scale, mag_range, density, opactiy, stroke):
    randomSample = np.random.uniform(0, 1, blur.shape) < density
    indices = np.nonzero(randomSample * (mag >= mag_range[0]) * (mag < mag_range[1]))
    for x, y in zip(indices[0], indices[1]):
        if mag[x, y] < 80:
            rotation = (blur[x, y] / 255 * 180)
        else:
            rotation = (dir[x, y] * 180 / 3.1415)
        coloredStroke = stroke.copy()
        coloredStroke = coloredStroke * np.append(data[x, y] / 255, [opactiy])
        coloredStroke = resize(coloredStroke, (int(coloredStroke.shape[0] * scale), int(coloredStroke.shape[1] * scale)), anti_aliasing=True)
        coloredStroke = rotate(coloredStroke, rotation, resize=True)
        if x + coloredStroke.shape[0] < canvas.shape[0] and y + coloredStroke.shape[1] < canvas.shape[1]:
            template = np.zeros_like(canvas)
            template[x: x + coloredStroke.shape[0], y: y + coloredStroke.shape[1], :] = coloredStroke
            canvas = blend(canvas, template)

def main(image):
    global mag
    global dir
    global blur
    global data
    data = image

    stroke = io.imread('brush.png')
    greyscale = np.dot(data, rgb_weights)
    blur = ndimage.uniform_filter(greyscale, size=(3, 3), mode='nearest')
    dx = ndimage.sobel(blur, 0)
    dy = ndimage.sobel(blur, 1)
    mag = np.hypot(dx, dy)
    mag *= 255.0 / np.max(mag)
    dir = np.arctan(dy / dx)

    background = np.ones((image.shape[0], image.shape[1], 4), dtype=np.uint8) * 255
    background2 = np.zeros((image.shape[0], image.shape[1], 4), dtype=np.uint8)
    background3 = np.zeros((image.shape[0], image.shape[1], 4), dtype=np.uint8)

    print('coarse')
    paint(background, 5, [0, 23], 0.05, 0.58, stroke)
    print('medium')
    paint(background2, 2.5, [23, 92], 0.18, 0.81, stroke)
    print('fine')
    paint(background3, 1, [92, 255], 0.78, 0.92, stroke)

    fig, axs = plt.subplots(2, 3)
    plt.gray()
    axs[0, 0].imshow(mag)
    axs[0, 1].imshow(dir)
    #axs[0, 2].imshow()
    axs[1, 0].imshow(background)
    axs[1, 1].imshow(background2)
    axs[1, 2].imshow(background3)
    plt.show()


if __name__ == "__main__":
    image = io.imread(sys.argv[1])
    main(image)
