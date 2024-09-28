import time
import tensorflow as tf
from config import style_layers, content_layers
from util import load_img, imshow
from model import Extractor, train_step

content_image = load_img("images/content/dancing.jpg")
style_image = load_img("images/styles/picasso.jpg")

extractor = Extractor(style_layers, content_layers)

style_targets = extractor(style_image)["style"]
content_targets = extractor(content_image)["content"]

image = tf.Variable(content_image)


start = time.time()

epochs = 10
steps_per_epoch = 100

step = 0
for n in range(epochs):
    for m in range(steps_per_epoch):
        step += 1
        train_step(extractor, image, style_targets, content_targets)
        print(".", end="", flush=True)
    print("Train step: {}".format(step))

end = time.time()
print("Total time: {:.1f}".format(end - start))
imshow(image)
