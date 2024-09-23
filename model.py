import tensorflow as tf
from config import num_style_layers, num_content_layers, style_weight, content_weight
from helpers import gram_matrix, clip_0_1


# for layer in vgg.layers:
#     print(layer)


def vgg_layers(layer_names):
    vgg = tf.keras.applications.VGG19(include_top=False, weights="imagenet")
    vgg.trainable = False

    outputs = [vgg.get_layer(layer_name).output for layer_name in layer_names]

    model = tf.keras.Model([vgg.input], outputs)

    return model


class Extractor(tf.keras.models.Model):
    def __init__(self, style_layers, content_layers):
        super(Extractor, self).__init__()
        self.vgg = vgg_layers(style_layers + content_layers)
        self.style_layers = style_layers
        self.content_layers = content_layers
        self.num_style_layers = len(style_layers)
        self.vgg.trainable = False

    def call(self, inputs):
        inputs = inputs * 255.0
        # RGB to BGR
        preprocessed_img = tf.keras.applications.vgg19.preprocess_input(inputs)
        outputs = self.vgg(preprocessed_img)
        # Get style and content output
        style_outputs = outputs[: self.num_style_layers]
        content_outputs = outputs[self.num_style_layers :]

        style_outputs = [gram_matrix(output) for output in style_outputs]

        content_dict = {
            content_name: value
            for content_name, value in zip(self.content_layers, content_outputs)
        }

        style_dict = {
            style_name: value
            for style_name, value in zip(self.style_layers, style_outputs)
        }

        return {"content": content_dict, "style": style_dict}


opt = tf.keras.optimizers.Adam(learning_rate=0.02, beta_1=0.99, epsilon=1e-1)


def style_content_loss(outputs, style_targets, content_targets):
    style_outputs = outputs["style"]
    content_outputs = outputs["content"]
    style_loss = tf.add_n(
        [
            tf.reduce_mean((style_outputs[name] - style_targets[name]) ** 2)
            for name in style_outputs.keys()
        ]
    )
    style_loss *= style_weight / num_style_layers

    content_loss = tf.add_n(
        [
            tf.reduce_mean((content_outputs[name] - content_targets[name]) ** 2)
            for name in content_outputs.keys()
        ]
    )
    content_loss *= content_weight / num_content_layers
    loss = style_loss + content_loss
    return loss


#
# @tf.function
def train_step(extractor, image: tf.Variable, style_targets, content_targets):
    with tf.GradientTape() as tape:
        outputs = extractor(image)
        loss = style_content_loss(outputs, style_targets, content_targets)

    grad = tape.gradient(loss, image)
    opt.apply_gradients([(grad, image)])
    image.assign(clip_0_1(image))
