import sys

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from numpy import asarray
from scipy import ndimage

rgb_weights = [0.2989, 0.5870, 0.1140]

mag = []
blur = []
data = []
dir = []


def progress_bar(cur, max):
    max = max - 1
    b = '\u2588'
    sys.stdout.write(f"\r{b * int((cur / max) * 20)}{' ' * int(((max - cur) / max) * 20)}  {str(cur)}/{str(max)}")
    if cur >= max:
        print()


def paint(canvas, scale, mag_range, density, opacity, stroke):
    randomSample = np.random.uniform(0, 1, (canvas.size[1], canvas.size[0])) < density
    indices = np.nonzero(randomSample * (mag >= (mag_range[0]) * (mag < mag_range[1])))
    for i, x in enumerate(zip(indices[1], indices[0])):
        progress_bar(i, len(indices[0]))
        if mag[x[1], x[0]] < 0.1:
            rotation = (blur[x[1], x[0]] * 180)
        else:
            rotation = (dir[x[1], x[0]] * 180 / 3.1415)
        coloredStroke = stroke.copy()
        coloredStroke = 1.5 * coloredStroke * np.append(data[x[1], x[0]], [opacity])
        coloredStroke = np.clip(coloredStroke, 0, 255)
        tmpStroke = Image.fromarray(coloredStroke.astype(np.uint8))
        tmpStroke = tmpStroke.resize((int(tmpStroke.width * scale[0]), int(tmpStroke.height * scale[1])))
        tmp = Image.new('RGBA', (512, 512))
        tmp.paste(tmpStroke.rotate(rotation, expand=True), (x[0] - tmpStroke.width // 2, x[1] - tmpStroke.height // 2),
                  tmpStroke.rotate(rotation, expand=True))
        canvas = Image.alpha_composite(tmp, canvas)

    return canvas


def main(image):
    global mag, blur, dir, data
    image = image.resize((512, 512))

    stroke = Image.open('b3.png', 'r').convert('RGBA').resize((15, 7))
    stroke = np.array(stroke)
    data = asarray(image) / 255
    greyscale = np.dot(data, rgb_weights)
    blur = ndimage.uniform_filter(greyscale, size=(3, 3), mode='nearest')
    dx = ndimage.sobel(blur, 0)
    dy = ndimage.sobel(blur, 1)
    mag = np.hypot(dx, dy)
    dir = np.arctan(dy / dx)
    # randomSample = np.random.uniform(0, 1, blur.shape) < 0.01

    background = Image.new('RGBA', (blur.shape[1], blur.shape[0]), (255, 255, 255, 0))
    background2 = Image.new('RGBA', (blur.shape[1], blur.shape[0]), (255, 255, 255, 0))
    background3 = Image.new('RGBA', (blur.shape[1], blur.shape[0]), (255, 255, 255, 0))

    background = paint(background, scale=[1, 2], mag_range=[0, 0.09], density=0.03, opacity=0.67, stroke=stroke)
    background2 = paint(background2, [0.47, 0.47], [0.09, 0.36], 0.18, 0.81, stroke)
    background3 = paint(background3, [0.23, 0.23], [0.36, 1], 0.78, 0.92, stroke)

    fig, axs = plt.subplots(2, 3)
    plt.gray()
    axs[0, 0].imshow(mag)
    axs[0, 1].imshow(dir)
    axs[0, 2].imshow(Image.alpha_composite(background, Image.alpha_composite(background2, background3)))
    axs[1, 0].imshow(background)
    axs[1, 1].imshow(background2)
    axs[1, 2].imshow(background3)
    plt.show()


if __name__ == "__main__":
    image = Image.open(sys.argv[1])
    main(image)
