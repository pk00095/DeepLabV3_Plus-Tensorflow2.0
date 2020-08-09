import tensorflow as tf
import matplotlib.pyplot as plt
import cv2
import numpy as np

img = cv2.imread('./mini_ADE20K/images/ADE_val_00000029.jpg')
image = tf.cast(img, tf.uint8)
image = tf.image.resize(image,(H, W))
image = tf.cast(image, tf.keras.backend.floatx())
normalized_image = image-[103.939, 116.779, 123.68]
normalized_image = np.expand_dims(normalized_image, axis=0)
y = model.predict(normalized_image)

final = np.argmax(y, axis=-1)[0]

plt.imshow(final)