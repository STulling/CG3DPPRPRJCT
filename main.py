import sys

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from numpy import asarray
from scipy import ndimage

DEBUG = True

rgb_weights = [0.2989, 0.5870, 0.1140]
resolution = (512, 512)
hue_threshold = 0.1
stroke_template = asarray(Image.open('brush.png', 'r').convert('RGBA').resize((15, 7)))


class StrokeProperties:
    def __init__(self, template=stroke_template, density=1.0, opacity=1.0, scale=(1, 1)):
        self.template = template
        self.density = density
        self.opacity = opacity

        self.x_scale = scale[0]
        self.y_scale = scale[1]


def progress_bar(cur, max):
    max = max - 1
    b = '\u2588'
    sys.stdout.write(f"\r{b * int((cur / max) * 20)}{' ' * int(((max - cur) / max) * 20)}  {str(cur)}/{str(max)}")
    if cur >= max:
        print()


def load_image(path):
    im = Image.open(path)
    im = im.resize(resolution)
    return asarray(im) / 255


def image_processing(image):
    image = box_blur(image)
    return gradient_extraction(image)


def stroke_processing(magnitudes, angles, color_data, stroke_properties, thresholds=(0.09, 0.36), size=resolution):
    coarse, medium, fine = layer_separation(magnitudes, thresholds=thresholds)

    coarse = coarse * random_texture(size=size, density=stroke_properties['coarse'].density)
    medium = medium * random_texture(size=size, density=stroke_properties['medium'].density)
    fine = fine * random_texture(size=size, density=stroke_properties['fine'].density)

    coarse = stroke_placement(coarse, magnitudes, angles, color_data, size=size, properties=stroke_properties['coarse'])
    medium = stroke_placement(medium, magnitudes, angles, color_data, size=size, properties=stroke_properties['medium'])
    fine = stroke_placement(fine, magnitudes, angles, color_data, size=size, properties=stroke_properties['fine'])

    return coarse, medium, fine


def random_texture(size=resolution, density=1.0):
    return np.random.uniform(0, 1, (size[1], size[0])) < density


def layer_separation(in_image, thresholds):
    layers = []
    last_t = 0
    for t in list(thresholds) + [1.0]:
        layers.append(in_image * (last_t <= in_image) * (in_image < t))
        last_t = t
    return layers


def stroke_placement(layer, magnitudes, angles, color_data, size=resolution, properties=StrokeProperties()):
    indices = np.nonzero(layer)
    canvas = Image.new('RGBA', size, (255, 255, 255, 0))
    for i, (x, y) in enumerate(zip(indices[1], indices[0])):
        if DEBUG:
            progress_bar(i, len(indices[0]))

        rotation = stroke_rotation(x, y, magnitudes, angles, color_data)
        stroke = create_stroke(x, y, rotation, color_data, size=size, properties=properties)
        canvas = Image.alpha_composite(stroke, canvas)
    return canvas


def stroke_rotation(x, y, magnitudes, angles, color_data):
    if magnitudes[y, x] < hue_threshold:
        return hue(*color_data[y, x])
    return angles[y, x] * 180 / np.pi


def hue(r, g, b):
    minimum = min(r, g, b)
    maximum = max(r, g, b)

    if minimum == maximum:
        return 0

    if maximum == r:
        h = (g - b) / (maximum - minimum)
    elif maximum == g:
        h = 2 + (b - r) / (maximum - minimum)
    else:
        h = 4 + (r - g) / (maximum - minimum)

    h = h * 60
    if h < 0:
        h = h + 360
    return round(h)


def create_stroke(x, y, rotation, color_data, size=resolution, properties=StrokeProperties()):
    stroke = 1.5 * properties.template.copy() * np.append(color_data[y, x], [properties.opacity])
    stroke = np.clip(stroke, 0, 255)
    stroke = Image.fromarray(stroke.astype(np.uint8))
    stroke = stroke.resize((int(stroke.width * properties.x_scale), int(stroke.height * properties.y_scale)))

    canvas = Image.new('RGBA', size)
    canvas.paste(stroke.rotate(rotation, expand=True), (x - stroke.width // 2, y - stroke.height // 2),
                 stroke.rotate(rotation, expand=True))
    return canvas


def to_greyscale(in_image):
    return np.dot(asarray(in_image), rgb_weights)


def box_blur(in_image, kernel=(3, 3)):
    return ndimage.uniform_filter(in_image, size=kernel, mode='nearest')


def gradient_extraction(in_image):
    dx = ndimage.sobel(in_image, 0)
    dy = ndimage.sobel(in_image, 1)
    return np.hypot(dx, dy), np.arctan(dy / dx)


def demo_gradient_extraction():
    fig, axs = plt.subplots(3, 3)
    plt.gray()

    for i in range(3):
        path = f'preview/gradient{i + 1}.png'
        original_image = load_image(path)
        image = to_greyscale(original_image)
        magnitudes, angles = image_processing(image)

        axs[i, 0].imshow(original_image)
        axs[i, 1].imshow(magnitudes)
        axs[i, 2].imshow(angles)

    plt.show()


def demo_layer_separation():
    fig, axs = plt.subplots(3, 4)
    plt.gray()

    for i in range(3):
        path = f'preview/separation{i + 1}.png'
        original_image = load_image(path)
        image = to_greyscale(original_image)
        magnitudes, _ = image_processing(image)
        c, m, f = layer_separation(magnitudes, [0.1, 0.15])

        axs[i, 0].imshow(original_image)
        axs[i, 1].imshow(1-c)
        axs[i, 2].imshow(1-m)
        axs[i, 3].imshow(1-f)

    plt.show()


def demo_complete():
    props = {
        'coarse': StrokeProperties(template=stroke_template,
                                   density=0.03,
                                   opacity=0.67,
                                   scale=[1, 2]),
        'medium': StrokeProperties(template=stroke_template,
                                   density=0.18,
                                   opacity=0.81,
                                   scale=[0.47, 0.47]),
        'fine': StrokeProperties(template=stroke_template,
                                 density=0.78,
                                 opacity=0.92,
                                 scale=[0.23, 0.23]),
    }

    path = 'preview/complete1.jpg'
    original_image = load_image(path)
    image = to_greyscale(original_image)
    magnitudes, angles = image_processing(image)
    coarse, medium, fine = stroke_processing(magnitudes, angles, original_image, size=resolution,
                                             stroke_properties=props)

    fig, axs = plt.subplots(2, 3)
    plt.gray()
    axs[0, 0].imshow(original_image)
    axs[0, 1].imshow(magnitudes)
    axs[0, 2].imshow(Image.alpha_composite(coarse, Image.alpha_composite(medium, fine)))
    axs[1, 0].imshow(coarse)
    axs[1, 1].imshow(medium)
    axs[1, 2].imshow(fine)
    plt.show()


if __name__ == "__main__":
    demo_complete()
