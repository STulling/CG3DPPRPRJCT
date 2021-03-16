import sys
from PIL import Image
import PIL
from numpy import asarray
from scipy import ndimage
import numpy as np
import matplotlib.pyplot as plt

rgb_weights = [0.2989, 0.5870, 0.1140]


def main(image):
    stroke = Image.open('brush.png', 'r')
    stroke = np.array(stroke)
    data = asarray(image)
    greyscale = np.dot(data, rgb_weights)
    blur = ndimage.uniform_filter(greyscale, size=(3, 3), mode='nearest')
    dx = ndimage.sobel(blur, 0)
    dy = ndimage.sobel(blur, 1)
    mag = np.hypot(dx, dy)
    mag *= 255.0 / np.max(mag)
    dir = np.arctan(dy / dx)
    #randomSample = np.random.uniform(0, 1, blur.shape) < 0.01

    background = Image.new('RGBA', (blur.shape[1], blur.shape[0]), (255, 255, 255, 255))
    background2 = Image.new('RGBA', (blur.shape[1], blur.shape[0]), (255, 255, 255, 255))
    background3 = Image.new('RGBA', (blur.shape[1], blur.shape[0]), (255, 255, 255, 255))

    randomSample = np.random.uniform(0, 1, blur.shape) < 0.01
    indices = np.nonzero(randomSample * (mag < 20))
    for x in zip(indices[1], indices[0]):
        if mag[x[1], x[0]] < 80:
            rotation = (blur[x[1], x[0]] / 255 * 180)
        else:
            rotation = (dir[x[1], x[0]] * 180 / 3.1415)
        coloredStroke = stroke.copy()
        coloredStroke = coloredStroke * np.append(data[x[1], x[0]] / 255, [1])
        tmpStroke = Image.fromarray(coloredStroke.astype(np.uint8))
        background.paste(tmpStroke.rotate(rotation, expand=True), x, tmpStroke.rotate(rotation, expand=True))

    fig, axs = plt.subplots(2, 3)
    plt.gray()
    axs[0, 0].imshow(mag)
    axs[0, 1].imshow(dir)
    #axs[0, 2].imshow(background + background2 + background3)
    axs[1, 0].imshow(background)
    axs[1, 1].imshow(background2)
    axs[1, 2].imshow(background3)
    plt.show()


if __name__ == "__main__":
    image = Image.open(sys.argv[1])
    main(image)
