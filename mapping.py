import tensorflow as tf
import sys

def decode_liver_example(serialized_example):
    features = {
        'meta/size': tf.FixedLenFeature([3], tf.int64),

        'image': tf.FixedLenFeature([], tf.string),
        'label': tf.FixedLenFeature([], tf.string)
    }
    parsed = tf.parse_single_example(
        serialized_example,
        features=features)

    shape = parsed["meta/size"]

    image = tf.decode_raw(parsed['image'], tf.as_dtype(tf.float32))
    image = tf.reshape(image, shape)
    label = tf.decode_raw(parsed['label'], tf.as_dtype(tf.int8))
    label = tf.reshape(label, tf.concat([shape, [-1]], -1))

    return image, label


if __name__ == "__main__":
    tf_filename = sys.argv[0]
    options = tf.python_io.TFRecordOptions(tf.python_io.TFRecordCompressionType.GZIP)
    tf.enable_eager_execution()

    record_iterator = tf.python_io.tf_record_iterator(path=tf_filename, options=options)
    for string_record in record_iterator:
        image, label = decode_liver_example(string_record)
        assert image.shape == (1, 2, 3)
        assert label.shape == (4, 5, 6)
