import tensorflow as tf
import sys
import logging

def decode_liver_example(serialized_example):
    features = {}

    features["meta/size"] = tf.FixedLenFeature([3], tf.int64)
    features["meta/spacing"] = tf.FixedLenFeature([3], tf.float32)
    features["meta/origin"] = tf.FixedLenFeature([3], tf.float32)
    features["meta/direction"] = tf.FixedLenFeature([9], tf.float32)
    features["meta/class"] = tf.FixedLenFeature([1], tf.int64)
    features["meta/number_of_classes"] = tf.FixedLenFeature([1], tf.int64)

    features['image/ct_image'] = tf.FixedLenFeature([], tf.string)
    features['image/label'] = tf.FixedLenFeature([], tf.string)

    parsed = tf.parse_single_example(
        serialized_example,
        features=features)

    shape = parsed["meta/size"]

    image = tf.decode_raw(parsed['image/ct_image'], tf.as_dtype(tf.float32))
    image = tf.reshape(image, shape)
    label = tf.decode_raw(parsed['image/label'], tf.as_dtype(tf.float32))
    label = tf.reshape(label, tf.concat([shape, [-1]], -1))

    return image, label


if __name__ == "__main__":
    tf_filename = sys.argv[1]
    logging.info(f"loading {tf_filename}")
    options = tf.python_io.TFRecordOptions(tf.python_io.TFRecordCompressionType.GZIP)
    tf.enable_eager_execution()

    record_iterator = tf.python_io.tf_record_iterator(path=tf_filename, options=options)
    for string_record in record_iterator:
        image, label = decode_liver_example(string_record)
        assert image.shape == (128,128,128)
        assert label.shape == (128,128,128,3)
