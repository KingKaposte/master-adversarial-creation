import tensorflow as tf


# save the audio_file into the given path
def save_audio_file(output, filename):
    with open(filename, 'wb') as fh:
        fh.write(output)


# return the audio_file with the given path
def load_audio_file(path):
    with open(path, 'rb') as fh:
        return fh.read()


# load TensorFlow Graph
def load_graph(path):
    with tf.gfile.FastGFile(path, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        tf.import_graph_def(graph_def, name='')