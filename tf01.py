# -*- coding: utf-8 -*-
"""
Created on Mon Feb 27 22:38:44 2017

@author: diz
"""

import tensorflow as tf

# Create TensorFlow object called hello_constant
hello_constant = tf.constant('Hello World!')

with tf.Session() as sess:
    # Run the tf.constant operation in the session
    output = sess.run(hello_constant)
    print(output)
    
x = tf.placeholder(tf.string)

with tf.Session() as sess:
    output = sess.run(x, 
                      feed_dict={x: 'Hello World'})
    