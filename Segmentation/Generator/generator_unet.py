import os
from tfcore.interfaces.IModel import IModel, IModel_Params
from tfcore.core.layer import *
from tfcore.core.activations import *
from tfcore.core.loss import *
from tfcore.utilities.utils import pad_borders, get_patches


class Generator_UNet_Params(IModel_Params):
    """
    Parameter class for ExampleModel
    """

    def __init__(self,
                 activation='relu',
                 normalization='IN',
                 filter_dim=16,
                 loss_name='dice',
                 depth=3,
                 generator='Segmentation',
                 scope='Generator_UNet',
                 name='Generator_UNet'):
        super().__init__(scope=scope + '_' + generator, name=name + '_' + generator + '_' + str(depth))

        self.activation = activation
        self.normalization = normalization
        self.filter_dim = filter_dim
        self.depth = depth
        self.loss_name = loss_name
        self.path = os.path.realpath(__file__)


class Generator_UNet_Model(IModel):
    """
    Example of a simple 3 layer generator model for super-resolution
    """

    def __init__(self, sess, params, global_steps, is_training):
        """
        Init of Example Class

        # Arguments
            sess: Tensorflow-Session
            params: Instance of ExampleModel_Params
            global_steps: Globel steps for optimizer
        """
        super().__init__(sess, params, global_steps)
        self.model_name = self.params.name
        self.activation_down = get_activation(name='relu')
        self.activation = get_activation(name=self.params.activation)
        self.normalization = get_normalization(self.params.normalization)
        self.max_images = 25
        self.is_training = is_training

        self.summary_vis_unknown = []
        self.summary_unknown = []
        self.summary_vis_one_unknown = []

    def build_model(self, input, is_train=False, reuse=False):
        """
        Build model and create summary

        # Arguments
            input: Input-Tensor
            is_train: Bool
            reuse: Bool

        # Return
            Tensor of dimension 4D
        """
        self.reuse = reuse
        self.inputs = input
        self.G = super().build_model(self.inputs, is_train, reuse)

        return self.G

    def model(self, input, is_train=False, reuse=False):
        """
        Create generator model

        # Arguments
            input: Input-Tensor
            is_train: Bool
            reuse: Bool

        # Return
            Tensor of dimension 4D
        """

        def down_block(input, scope, f_out, k_size=3, activation=tf.nn.relu, normalization=None, is_training=False, is_last_block=False):

            with tf.variable_scope(scope):
                net = pad_borders(input, k_size, mode="REFLECT")
                net = conv2d(net,
                             f_out=f_out,
                             k_size=3,
                             activation=activation,
                             normalization=normalization,
                             padding='VALID',
                             is_training=is_training,
                             reuse=self.reuse,
                             use_bias=True,
                             name='conv_1')

                net = pad_borders(net, k_size, mode="REFLECT")
                net = conv2d(net,
                             f_out=f_out,
                             k_size=3,
                             stride=1,
                             activation=activation,
                             normalization=normalization,
                             padding='VALID',
                             is_training=is_training,
                             reuse=self.reuse,
                             use_bias=True,
                             name='conv_2')

                if f_out >= 512:
                    net = dropout(net, 0.5, is_training)

                net_conv = net
                if not is_last_block:
                    net = max_pool(net)

            return net, net_conv

        def up_block(net, net_down, scope, f_out, k_size=3, activation=tf.nn.relu, normalization=None, is_training=False):

            with tf.variable_scope(scope):

                net = deconv2d(net,
                               f_out=f_out,
                               k_size=2,
                               stride=2,
                               activation=activation,
                               normalization=normalization,
                               padding='SAME',
                               is_training=is_training,
                               use_bias=True,
                               name='deconv_1')

                for layer in net_down:
                    if net.shape == layer.shape:
                        net = tf.concat([net, layer], axis=3)
                        print("Concat")

                net = pad_borders(net, k_size, mode="REFLECT")
                net = conv2d(net,
                             f_out=f_out,
                             k_size=3,
                             activation=activation,
                             normalization=normalization,
                             padding='VALID',
                             is_training=is_training,
                             reuse=self.reuse,
                             use_bias=True,
                             name='conv_1')

                net = pad_borders(net, k_size, mode="REFLECT")
                net = conv2d(net,
                             f_out=f_out,
                             k_size=3,
                             activation=activation,
                             normalization=normalization,
                             padding='VALID',
                             is_training=is_training,
                             reuse=self.reuse,
                             use_bias=True,
                             name='conv_2')

            return net

        with tf.variable_scope(self.params.scope, reuse=tf.AUTO_REUSE):

            net = input

            layer = []
            f_out_max = self.params.filter_dim
            for n in range(0, self.params.depth - 1):
                net, net_conv = down_block(net,
                                           scope='down_block' + str(n + 1),
                                           f_out=f_out_max,
                                           k_size=3,
                                           activation=self.activation,
                                           normalization=self.normalization,
                                           is_training=self.is_training)
                f_out_max *= 2
                layer.append(net_conv)

            net, net_conv = down_block(net,
                                       scope='down_block' + str(self.params.depth),
                                       f_out=f_out_max,
                                       k_size=3,
                                       activation=self.activation,
                                       normalization=self.normalization,
                                       is_training=self.is_training,
                                       is_last_block=True)

            f_out_max /= 2
            layer_up = []
            for n in range(self.params.depth - 1):
                net = up_block(net,
                               layer,
                               scope='up_block' + str(n + 1),
                               f_out=f_out_max,
                               k_size=3,
                               activation=self.activation,
                               normalization=self.normalization,
                               is_training=self.is_training)
                f_out_max /= 2
                layer_up.append(net)

            net = conv2d(net,
                         f_out=2,
                         k_size=1,
                         stride=1,
                         is_training=self.is_training,
                         reuse=self.reuse,
                         use_bias=True,
                         padding='VALID',
                         name='conv_out')

            self.logits = net
            self.probs = tf.nn.softmax(net)
            self.label = tf.cast(tf.argmin(self.probs, 3), dtype=tf.float32)
            self.prediction = tf.expand_dims(self.label, -1)
            self.mask_image = (self.inputs - 0.5) + (self.inputs - 0.5) * self.prediction

        print(' [*] U-Net loaded...')
        return self.logits

    def loss(self, Y, normalize=False, name='cross_entropy'):

        Y_true = tf.concat([tf.multiply(Y, 1.0), tf.multiply((1.0 - Y), 1.0)], axis=3)

        if self.params.loss_name is 'dice':
            loss = dice_loss = loss_dice(Y_true, self.probs)
            self.summary.append(tf.summary.scalar("dice_loss", dice_loss))
        elif self.params.loss_name is 'pixelwise_softmax':
            loss = pixel_wise_loss = cross_entropy(label=Y_true,
                                                   probs=pixel_wise_softmax(self.logits))
            self.summary.append(tf.summary.scalar("pixelwise_cross_entropy_loss", pixel_wise_loss))
        else:
            loss = cross_entropy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=tf.layers.flatten(self.logits),
                                                                                                  labels=tf.layers.flatten(Y_true)))
            self.summary.append(tf.summary.scalar("cross_entropy_loss", cross_entropy_loss))

        # Accuracy for train and test set
        self.summary.append(tf.summary.scalar("accuracy", accuracy(Y, self.prediction)))
        self.summary.append(tf.summary.scalar("iou", intersection_of_union(labels=Y, predictions=self.prediction)))
        self.summary_val.append(tf.summary.scalar("accuracy_test", accuracy(Y, self.prediction)))
        self.summary_val.append(tf.summary.scalar("iou_test", intersection_of_union(labels=Y, predictions=self.prediction)))

        self.summary.append(tf.summary.scalar("LR_UNET", self.learning_rate))
        self.summary_vis.append(tf.summary.image('Prediction', tf.cast(self.prediction, dtype=tf.float32), max_outputs=self.max_images))
        self.summary_vis.append(tf.summary.image('Mask', self.mask_image, max_outputs=self.max_images))

        self.summary_vis_unknown.append(tf.summary.image('Prediction_Unknown', tf.cast(self.prediction, dtype=tf.float32), max_outputs=self.max_images))
        self.summary_vis_unknown.append(tf.summary.image('Mask_Unknown', self.mask_image, max_outputs=self.max_images))

        self.total_loss = loss

        return self.total_loss
