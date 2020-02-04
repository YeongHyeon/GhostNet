import random
import tensorflow as tf

class GhostNet(object):

    def __init__(self, height, width, channel, num_class, leaning_rate=1e-3):

        print("\nInitializing Neural Network...")
        self.height, self.width, self.channel = height, width, channel
        self.num_class, self.k_size = num_class, 3
        self.leaning_rate = leaning_rate

        self.x = tf.compat.v1.placeholder(tf.float32, [None, self.height, self.width, self.channel])
        self.y = tf.compat.v1.placeholder(tf.float32, [None, self.num_class])
        self.batch_size = tf.compat.v1.placeholder(tf.int32, shape=[])
        self.training = False

        self.weights, self.biasis = [], []
        self.w_names, self.b_names = [], []

        self.y_hat = self.build_model(input=self.x)

        self.smce = tf.nn.softmax_cross_entropy_with_logits(labels=self.y, logits=self.y_hat)
        self.loss = tf.compat.v1.reduce_mean(self.smce)

        #default: beta1=0.9, beta2=0.999
        self.optimizer = tf.compat.v1.train.AdamOptimizer( \
            self.leaning_rate, beta1=0.9, beta2=0.999).minimize(self.loss)

        self.score = tf.nn.softmax(self.y_hat)
        self.pred = tf.argmax(self.score, 1)
        self.correct_pred = tf.equal(self.pred, tf.argmax(self.y, 1))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_pred, tf.float32))

        tf.compat.v1.summary.scalar('softmax_cross_entropy', self.loss)
        self.summaries = tf.compat.v1.summary.merge_all()

    def set_training(self): self.training = True

    def set_test(self): self.training = False

    def build_model(self, input):

        conv1 = self.conv2d(input=input, stride=1, padding='SAME', \
            filter_size=[self.k_size, self.k_size, self.channel, 16], \
            activation="relu", name="conv1")

        ghost1_1 = self.ghost_block(input=conv1, \
            num_inputs=16, num_expention=16, num_outputs=16, type=1, name="ghost1_1")
        ghost1_2 = self.ghost_block(input=ghost1_1, \
            num_inputs=16, num_expention=48, num_outputs=24, type=2, name="ghost1_2")

        ghost2_1 = self.ghost_block(input=ghost1_2, \
            num_inputs=24, num_expention=72, num_outputs=24, type=1, name="ghost2_1")
        ghost2_2 = self.ghost_block(input=ghost2_1, \
            num_inputs=24, num_expention=72, num_outputs=40, type=2, name="ghost2_2")

        ghost3_1 = self.ghost_block(input=ghost2_2, \
            num_inputs=40, num_expention=120, num_outputs=40, type=1, name="ghost3_1")
        ghost3_2 = self.ghost_block(input=ghost3_1, \
            num_inputs=40, num_expention=240, num_outputs=80, type=2, name="ghost3_2")

        [n, h, w, c] = ghost3_2.shape
        fullcon_in = tf.compat.v1.reshape(ghost3_2, shape=[self.batch_size, h*w*c], name="fullcon_in")
        fullcon1 = self.fully_connected(input=fullcon_in, num_inputs=int(h*w*c), \
            num_outputs=512, activation="relu", name="fullcon1")
        fullcon2 = self.fully_connected(input=fullcon1, num_inputs=512, \
            num_outputs=self.num_class, activation=None, name="fullcon2")

        return fullcon2

    def ghost_block(self, input, num_inputs=16, num_expention=48, num_outputs=16, type=1, name=""):

        [n, h, w, c] = input.shape

        # expention
        ghost1 = self.conv2d(input=input, stride=1, padding='SAME', \
            filter_size=[self.k_size, self.k_size, num_inputs, num_expention], \
            activation="relu", name="%s_1" %(name))

        if(type != 1):
            ghost1 = self.dwconv2d(input=ghost1, stride=2, padding='SAME', \
                filter_size=[self.k_size, self.k_size, num_expention, 1], \
                activation="relu", name="%s_1_1" %(name))
            [gn, gh, gw, gc] = ghost1.shape
            input = tf.image.resize(images=input, size=[gh, gw])

        # reduction
        ghost2 = self.conv2d(input=ghost1, stride=1, padding='SAME', \
            filter_size=[self.k_size, self.k_size, num_expention, num_outputs], \
            activation="relu", name="%s_2" %(name))

        if(c != num_outputs):
            input = self.conv2d(input=input, stride=1, padding='SAME', \
                filter_size=[1, 1, num_inputs, num_outputs], \
                activation="relu", name="%s_2_1" %(name))

        return input + ghost2

    def initializer(self):
        return tf.compat.v1.initializers.variance_scaling(distribution="untruncated_normal", dtype=tf.dtypes.float32)

    def maxpool(self, input, ksize, strides, padding, name=""):

        out_maxp = tf.compat.v1.nn.max_pool(value=input, \
            ksize=ksize, strides=strides, padding=padding, name=name)
        print("Max-Pool", input.shape, "->", out_maxp.shape)

        return out_maxp

    def activation_fn(self, input, activation="relu", name=""):

        if("sigmoid" == activation):
            out = tf.compat.v1.nn.sigmoid(input, name='%s_sigmoid' %(name))
        elif("tanh" == activation):
            out = tf.compat.v1.nn.tanh(input, name='%s_tanh' %(name))
        elif("relu" == activation):
            out = tf.compat.v1.nn.relu(input, name='%s_relu' %(name))
        elif("lrelu" == activation):
            out = tf.compat.v1.nn.leaky_relu(input, name='%s_lrelu' %(name))
        elif("elu" == activation):
            out = tf.compat.v1.nn.elu(input, name='%s_elu' %(name))
        else: out = input

        return out

    def variable_maker(self, var_bank, name_bank, shape, name=""):

        try:
            var_idx = name_bank.index(name)
        except:
            variable = tf.compat.v1.get_variable(name=name, \
                shape=shape, initializer=self.initializer())

            var_bank.append(variable)
            name_bank.append(name)
        else:
            variable = var_bank[var_idx]

        return var_bank, name_bank, variable

    def batch_normalization(self, input, name=""):

        bnlayer = tf.compat.v1.keras.layers.BatchNormalization(
            axis=-1,
            momentum=0.99,
            epsilon=0.001,
            center=True,
            scale=True,
            beta_initializer='zeros',
            gamma_initializer='ones',
            moving_mean_initializer='zeros',
            moving_variance_initializer='ones',
            renorm_momentum=0.99,
            trainable=True,
            name="%s_bn" %(name),
        )

        bn = bnlayer(inputs=input, training=self.training)
        print("BN (%s)" %(name))
        return bn

    def conv2d(self, input, stride, padding, \
        filter_size=[3, 3, 16, 32], dilations=[1, 1, 1, 1], activation="relu", name=""):

        self.weights, self.w_names, weight = self.variable_maker(var_bank=self.weights, name_bank=self.w_names, \
            shape=filter_size, name='%s_w' %(name))
        self.biasis, self.b_names, bias = self.variable_maker(var_bank=self.biasis, name_bank=self.b_names, \
            shape=[filter_size[-1]], name='%s_b' %(name))

        out_conv = tf.compat.v1.nn.conv2d(
            input=input,
            filter=weight,
            strides=[1, stride, stride, 1],
            padding=padding,
            use_cudnn_on_gpu=True,
            data_format='NHWC',
            dilations=dilations,
            name='%s_conv' %(name),
        )
        out_bias = tf.math.add(out_conv, bias, name='%s_add' %(name))
        out_bias = self.batch_normalization(input=out_bias, name=name)

        print("Conv (%s)" %(name), input.shape, "->", out_bias.shape)
        return self.activation_fn(input=out_bias, activation=activation, name=name)

    def dwconv2d(self, input, stride, padding, \
        filter_size=[3, 3, 16, 32], dilations=[1, 1], activation="relu", name=""):

        self.weights, self.w_names, weight = self.variable_maker(var_bank=self.weights, name_bank=self.w_names, \
            shape=filter_size, name='%s_w' %(name))
        self.biasis, self.b_names, bias = self.variable_maker(var_bank=self.biasis, name_bank=self.b_names, \
            shape=[filter_size[-1]], name='%s_b' %(name))

        out_conv = tf.nn.depthwise_conv2d(
            input=input,
            filter=weight,
            strides=[1, stride, stride, 1],
            padding=padding,
            data_format='NHWC',
            rate=dilations,
            name='%s_dwconv' %(name),
        )
        out_bias = tf.math.add(out_conv, bias, name='%s_add' %(name))
        out_bias = self.batch_normalization(input=out_bias, name=name)

        print("Depth-Wise-Conv (%s)" %(name), input.shape, "->", out_bias.shape)
        return self.activation_fn(input=out_bias, activation=activation, name=name)

    def fully_connected(self, input, num_inputs, num_outputs, activation="relu", name=""):

        self.weights, self.w_names, weight = self.variable_maker(var_bank=self.weights, name_bank=self.w_names, \
            shape=[num_inputs, num_outputs], name='%s_w' %(name))
        self.biasis, self.b_names, bias = self.variable_maker(var_bank=self.biasis, name_bank=self.b_names, \
            shape=[num_outputs], name='%s_b' %(name))

        out_mul = tf.compat.v1.matmul(input, weight, name='%s_mul' %(name))
        out_bias = tf.math.add(out_mul, bias, name='%s_add' %(name))
        out_bias = self.batch_normalization(input=out_bias, name=name)

        print("Full-Con (%s)" %(name), input.shape, "->", out_bias.shape)
        return self.activation_fn(input=out_bias, activation=activation, name=name)
