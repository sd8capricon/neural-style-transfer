import PIL
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


# load image
def load_img(img_path):
    img_size = 224  # 512
    # Read image file
    img = tf.io.read_file(img_path)
    # Decode the image into 3 channels RGB
    img = tf.image.decode_image(img, channels=3)
    # convert from uint8 to float32
    img = tf.image.convert_image_dtype(img, tf.float32)
    shape = tf.cast(img.shape[:-1], tf.float32)
    long_dim = max(img.shape)
    # Calulate scaling factor for resizing
    scale = img_size / long_dim
    # resize
    new_shape = tf.cast(tf.round(shape * scale), tf.int32)
    img = tf.image.resize(img, new_shape)
    img = img[tf.newaxis, :]
    return img


def tensor_to_image(tensor):
    tensor = tensor * 255
    tensor = np.array(tensor, dtype=np.uint8)
    if np.ndim(tensor) > 3:
        assert tensor.shape[0] == 1
        tensor = tensor[0]
    return PIL.Image.fromarray(tensor)


# Display a tensor image
def imshow(img: tf.Tensor, title=None):
    if len(img.shape) > 3:
        img = tf.squeeze(img, axis=0)

    plt.imshow(img)
    if title:
        plt.title(title)
    plt.show()
