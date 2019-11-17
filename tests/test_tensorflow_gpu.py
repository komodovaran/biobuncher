import tensorflow as tf
assert tf.test.is_gpu_available()
assert tf.test.is_built_with_cuda()

a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape = [2, 3], name = 'a')
b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape = [3, 2], name = 'b')
c = tf.matmul(a, b)
print(c)