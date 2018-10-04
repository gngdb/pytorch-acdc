
import tensorflow as tf

if __name__ == "__main__":
    x = tf.random_normal((1000,4096))
    d = tf.spectral.dct(x)    

    # execute
    sess = tf.Session()
    sess.run(d)

    import timeit
    #setup = "import os; import tensorflow as tf; x = tf.random_normal((1000,4096)); d = tf.spectral.dct(x); sess = tf.Session()"
    setup = "import os; import tensorflow as tf; x = tf.get_variable('x', dtype=tf.float32, initializer=tf.random_normal((1000,4096))); d = tf.spectral.dct(x); sess = tf.Session(); sess.run(tf.global_variables_initializer())"
    print("DCT: ", timeit.timeit("sess.run(d)", setup=setup, number=100))
