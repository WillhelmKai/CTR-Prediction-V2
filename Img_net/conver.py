def max_pool_2x2(x):
    # stride [1, x_movement, y_movement, 1] the first and last digit must be 1
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

def conv2d(x, W):
    #x is the input, W is the weight, 
    # stride [1, x_movement, y_movement, 1] the first and last digit must be 1
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def norm(input, size):
    fc_mean, fc_var = tf.nn.moments(input, axes = [0])
    scale = tf.Variable(tf.ones([size]))
    shift = tf.Variable(tf.zeros([size]))
    epsilon = 0.001
    out = tf.nn.batch_normalization(input,fc_mean, fc_var, shift, scale,epsilon )
    return out

def converlutional(x_image):
    W_conv1 = weight_variable([5,5,3,32]) # patch 5x5, in size 3, out size 32
    b_conv1 = bias_variable([32])
    # h_conv1_nom = norm(conv2d(x_image, W_conv1) + b_conv1,32)
    # h_conv1 = tf.nn.relu(h_conv1_nom) # output size 112*112*32
    h_conv1 = tf.nn.leaky_relu(conv2d(x_image, W_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)  # output size 56*56*32

    ########################
    #4 conv layers
        #conv1.1 in 56*56 out 56*56
    W_conv2 = weight_variable([3,3,32,32]) # patch 3*3, in size 32, out size 32
    b_conv2 = bias_variable([32])
    # h_conv2_nom = norm(conv2d(h_pool1, W_conv2) + b_conv2,32)
    # h_conv2 = tf.nn.relu(h_conv2_nom) # output size 56*56*32
    h_conv2 = tf.nn.leaky_relu(conv2d(h_pool1, W_conv2) + b_conv2)

        #conv1.2 in 56*56 out 56*56
    W_conv3 = weight_variable([3,3,32,32]) # patch 3*3, in size 32, out size 32
    b_conv3 = bias_variable([32])
    # h_conv3_nom = norm(conv2d(h_conv2, W_conv3) + b_conv3,32)
    # h_conv3 = tf.nn.relu(h_conv3_nom) # output size 56*56*32
    h_conv3 = tf.nn.leaky_relu(conv2d(h_conv2, W_conv3) + b_conv3)


        #conv1.3 in 56*56 out 56*56
    W_conv4 = weight_variable([3,3,32,32]) # patch 3*3, in size 32, out size 32
    b_conv4 = bias_variable([32])
    h_conv4_nom = norm(conv2d(h_conv3, W_conv4) + b_conv4,32)
    h_conv4 = tf.nn.releaky_relulu(h_conv4_nom) # output size 56*56*32
    # h_conv4 = tf.nn.leaky_relu(conv2d(h_conv3, W_conv4) + b_conv4)

        #conv1.4 in 56*56 out 56*56
    W_conv5 = weight_variable([3,3,32,32]) # patch 3*3, in size 32, out size 32
    b_conv5 = bias_variable([32])
    # h_conv5_nom = norm(conv2d(h_conv4, W_conv5) + b_conv5,32)
    # h_conv5 = tf.nn.relu(h_conv5_nom) # output size 56*56*32
    h_conv5 = tf.nn.leaky_relu(conv2d(h_conv4, W_conv5) + b_conv5)
    #####################


    #4 conv layers
        #conv2.1 in 56*56 out 
    W_conv6 = weight_variable([3,3,32,64]) # patch 3*3, in size 32, out size 64
    b_conv6 = bias_variable([64])
    # h_conv6_nom = norm(conv2d(h_conv5, W_conv6) + b_conv6,64)
    # h_conv6 = tf.nn.relu(h_conv6_nom) # output size 56*56*64
    h_conv6 = tf.nn.leaky_relu(conv2d(h_conv5, W_conv6) + b_conv6)

        #conv2.2 in  out 
    W_conv7 = weight_variable([3,3,64,64]) # patch 3*3, in size 64, out size 64
    b_conv7 = bias_variable([64])
    # h_conv7_nom = norm(conv2d(h_conv6, W_conv7) + b_conv7,64)
    # h_conv7 = tf.nn.relu(h_conv7_nom) # output size 56*56*64
    h_conv7 = tf.nn.leaky_relu(conv2d(h_conv6, W_conv7) + b_conv7)

        #conv2.3 in  out 
    W_conv8 = weight_variable([3,3,64,64]) # patch 3*3, in size 64, out size 64
    b_conv8 = bias_variable([64])
    h_conv8_nom = norm(conv2d(h_conv7, W_conv8) + b_conv8,64)
    h_conv8 = tf.nn.leaky_relu(h_conv8_nom) # output size 56*56*64
    # h_conv8 = tf.nn.leaky_relu(conv2d(h_conv7, W_conv8) + b_conv8)    

        #conv2.4 in  out 28*28
    W_conv9 = weight_variable([3,3,64,64]) # patch 3*3, in size 64, out size 64
    b_conv9 = bias_variable([64])
    # h_conv9_nom = norm(conv2d(h_conv8, W_conv9) + b_conv9,64)
    # h_conv9 = tf.nn.relu(h_conv9_nom) # output size 56*56*64
    h_conv9 = tf.nn.leaky_relu(conv2d(h_conv8, W_conv9) + b_conv9)
    h_pool9 = max_pool_2x2(h_conv9)  # output size 28*28*64
    #######################


    #4 conv layers
        #conv3.1 in 28*28 out 
    W_conv10 = weight_variable([3,3,64,128]) # patch 3*3, in size 64, out size 128
    b_conv10 = bias_variable([128])
    # h_conv10_nom = norm(conv2d(h_pool9, W_conv10) + b_conv10,128)
    # h_conv10 = tf.nn.relu(h_conv10_nom) # output size 28*28*128
    h_conv10 = tf.nn.leaky_relu(conv2d(h_pool9, W_conv10) + b_conv10)

        #conv3.2 in  out 
    W_conv11 = weight_variable([3,3,128,128]) # patch 3*3, in size 128, out size 128
    b_conv11 = bias_variable([128])
    # h_conv11_nom = norm(conv2d(h_conv10, W_conv11) + b_conv11,128)
    # h_conv11 = tf.nn.relu(h_conv11_nom) # output size 28*28*128
    h_conv11 = tf.nn.leaky_relu(conv2d(h_conv10, W_conv11) + b_conv11)

        #conv3.3 in  out 
    W_conv12 = weight_variable([3,3,128,128]) # patch 3*3, in size 128, out size 128
    b_conv12 = bias_variable([128])
    h_conv12_nom = norm(conv2d(h_conv11, W_conv12) + b_conv12,128)
    h_conv12 = tf.nn.leaky_relu(h_conv12_nom) # output size 28*28*128
    # h_conv12 = tf.nn.leaky_relu(conv2d(h_conv11, W_conv12) + b_conv12)

        #conv3.4 in  out 14*14
    W_conv13 = weight_variable([3,3,128,128]) # patch 3*3, in size 128, out size 128
    b_conv13 = bias_variable([128])
    # h_conv13_nom = norm(conv2d(h_conv12, W_conv13) + b_conv13,128)
    # h_conv13 = tf.nn.relu(h_conv13_nom) # output size 28*28*128
    h_conv13 = tf.nn.leaky_relu(conv2d(h_conv12, W_conv13) + b_conv13)
    h_pool13 = max_pool_2x2(h_conv13)  # output size 14*14*128
    ################################

    #4 conv layers
        #conv4.1 in 56*56 out 
    W_conv14 = weight_variable([3,3,128,256]) # patch 3*3, in size 128, out size 256
    b_conv14 = bias_variable([256])
    # h_conv14_nom = norm(conv2d(h_pool13, W_conv14) + b_conv14,256)
    # h_conv14 = tf.nn.relu(h_conv14_nom) # output size 14*14*256
    h_conv14 = tf.nn.leaky_relu(conv2d(h_pool13, W_conv14) + b_conv14)

        #conv4.2 in  out 
    W_conv15 = weight_variable([3,3,256,256]) # patch 3*3, in size 256, out size 256
    b_conv15 = bias_variable([256])
    # h_conv15_nom = norm(conv2d(h_conv14, W_conv15) + b_conv15,256)
    # h_conv15 = tf.nn.relu(h_conv15_nom) # output size 14*14*256
    h_conv15 = tf.nn.leaky_relu(conv2d(h_conv14, W_conv15) + b_conv15)

        #conv4.3 in  out 
    W_conv16 = weight_variable([3,3,256,256]) # patch 3*3, in size 256, out size 256
    b_conv16 = bias_variable([256])
    # h_conv16_nom = norm(conv2d(h_conv15, W_conv16) + b_conv16,256)
    # h_conv16 = tf.nn.relu(h_conv16_nom) # output size 14*14*256
    h_conv16 = tf.nn.leaky_relu(conv2d(h_conv15, W_conv16) + b_conv16)

        #conv4.4 in  out 7*7
    W_conv17 = weight_variable([3,3,256,256]) # patch 3*3, in size 256, out size 256
    b_conv17 = bias_variable([256])
    h_conv17_nom = norm(conv2d(h_conv16, W_conv17) + b_conv17,256)
    h_conv17 = tf.nn.leaky_relu(h_conv17_nom) # output size 14*14*256
    # h_conv17 = tf.nn.leaky_relu(conv2d(h_conv16, W_conv17) + b_conv17)
    h_pool17 = max_pool_2x2(h_conv17)  # output size 7*7*256

    #flaten and fully connect layer
    h_pool17_flat = tf.reshape(h_pool17, [-1, 7*7*256])
        #function layer1 
    W_fc1 = weight_variable([7*7*256, 500])
    b_fc1 = bias_variable([500])
    h_fc1 = tf.nn.tanh(tf.matmul(h_pool17_flat, W_fc1)+b_fc1)
    h_fc1_drop = tf.nn.dropout(h_fc1, 0.5)
    #out [1*1024]
    return h_fc1_drop