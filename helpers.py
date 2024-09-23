import tensorflow as tf


# G = A^T*A
# def gram_matrix(input_tensor):
#     # Get the shape of the input tensor
#     batch_size, height, width, channels = tf.shape(input_tensor)

#     # Reshape the tensor to (batch_size, num_pixels, channels)
#     # where num_pixels = height * width
#     num_pixels = height * width
#     reshaped_tensor = tf.reshape(input_tensor, (batch_size, num_pixels, channels))

#     # Compute the Gram matrix for each image in the batch
#     # Compute the Gram matrix G = A^T * A
#     # where A is the reshaped tensor
#     # G will have shape (batch_size, channels, channels)
#     gram_matrix = tf.linalg.matmul(reshaped_tensor, reshaped_tensor, transpose_a=True)

#     # Normalize the Gram matrix by the number of locations (pixels)
#     gram_matrix /= tf.cast(num_pixels, tf.float32)

#     return gram_matrix


# Google's Implementation
def gram_matrix(input_tensor):
    result = tf.linalg.einsum("bijc,bijd->bcd", input_tensor, input_tensor)
    input_shape = tf.shape(input_tensor)
    num_locations = tf.cast(input_shape[1] * input_shape[2], tf.float32)
    return result / (num_locations)


# Since Image is  flot
def clip_0_1(image):
    return tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=1.0)
