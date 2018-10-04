import timeit

if __name__ == "__main__":
    setup = "import os; import tensorflow as tf; x = tf.get_variable('x', dtype=tf.float32, initializer=tf.random_normal((1000,4096))); a = tf.get_variable('a', dtype=tf.float32, initializer=tf.random_normal((4096,4096))); d = tf.matmul(x,a); sess = tf.Session(); sess.run(tf.global_variables_initializer())"
    print("Linear: ", timeit.timeit("sess.run(d)", setup=setup, number=100))
