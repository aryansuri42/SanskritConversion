# import tensorflow as tf
# tf.compat.v1.enable_resource_variables()
# path = r"A:\Projects\Sanskrit Text Conversion\Sanskrit-Text-Conversion\c3_attention\1"
# model = tf.saved_model.load(path)

import tensorflow as tf
from PIL import Image
import io

def bytes_to_image(raw_bytes):
    # Convert the raw bytes into a BytesIO object
    image_stream = io.BytesIO(raw_bytes)
    # Open the image stream with Pillow
    image = Image.open(image_stream)
    return image

from tensorflow.python.saved_model import tag_constants
path = r"A:\Projects\Sanskrit Text Conversion\Sanskrit-Text-Conversion\c3_attention\1"
tf.compat.v1.reset_default_graph()

def load_image_as_bytes(image_path):
    with open(image_path, "rb") as f:
        image_bytes = f.read()
    return image_bytes


# Load the SavedModel in a TensorFlow 1.x session
with tf.compat.v1.Session() as sess:
    meta_graph_def  = tf.compat.v1.saved_model.loader.load(sess, [tf.saved_model.SERVING], path)
    # Perform inference or further operations
    # signature_def = meta_graph_def.signature_def
    # print("Available signatures:", list(signature_def.keys()))
    # serving_signature = signature_def["serving_default"]
    # print("Input Tensor Info:", serving_signature.inputs)
    # print("Output Tensor Info:", serving_signature.outputs)
    # input_tensor = sess.graph.get_tensor_by_name("input_image_as_bytes:0")  # Replace with your tensor name
    # output_tensor = sess.graph.get_tensor_by_name("prediction:0")  # Replace with your tensor name
    path_image = r"A:\Projects\Sanskrit Text Conversion\Sanskrit-Text-Conversion\images\sampleimage.jpeg"
    image_bytes = load_image_as_bytes(path_image)
    predictions = sess.run(
    ["probability:0", "prediction:0"],  # Replace with your desired output tensor names
    feed_dict={"input_image_as_bytes:0": [image_bytes]}  # Input tensor name and data
    )
    probabilities, output = predictions
    print("Probabilities:", probabilities)
    print("Prediction Output:", output)
    image = bytes_to_image(output)
    image.show()
