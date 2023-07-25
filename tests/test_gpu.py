import tensorflow as tf


def is_gpu_available():
    no_gpu = len(tf.config.list_physical_devices('GPU'))
    assert no_gpu > 0
