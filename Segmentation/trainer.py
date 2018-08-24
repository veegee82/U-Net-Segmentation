import time
import math
from tfcore.interfaces.ITraining import *
from tfcore.utilities.preprocessing import *
#from Segmentation.Generator.generator_unet import Generator_UNet_Model, Generator_UNet_Params
from Generator.generator_unet import *

class Trainer_Params(ITrainer_Params):
    """
    Parameter-Class for Example-Trainer
    """

    def __init__(self,
                 image_size,
                 params_path='',
                 loss=0,
                 gpu=0,
                 load=True,
                 ):
        """ Constructor

        # Arguments
            image_size: Image size (int)
            params_path: Parameter-Path for loading and saving (str)
            load: Load parameter (boolean)
        """
        super().__init__()

        self.image_size = image_size
        self.epoch = 10000
        self.batch_size = 8
        self.decay = 0.999
        self.step_decay = 100
        self.beta1 = 0.9
        self.learning_rate_G = 0.0001
        self.use_tensorboard = True
        self.gpus = [gpu]
        self.cyclic_learning_rate = True

        self.use_NN = True
        self.normalization_G = 'IN'

        # U-Net Params
        self.depth = 5
        self.filter_dim = 64
        self.loss = loss

        self.use_pretrained_generator = True
        self.pretrained_generator_dir = "../../../../pretrained_models/generator/"

        self.experiment_name = "SRResNET_GAN_" + self.loss
        self.checkpoint_restore_dir = ''
        self.sample_dir = 'samples'
        self.load_checkpoint = False

        self.use_validation_set = True
        self.evals_per_iteration = 250
        self.save_checkpoint = False

        if params_path is not '':
            if load:
                if self.load(params_path):
                    self.save(params_path)
            else:
                self.save(params_path)

        self.root_dir = "/mnt/datadrive/silvio"
        if not os.path.exists(self.root_dir):
            self.root_dir = "../../../Results/Local"
        else:
            self.use_tensorboard = False

    def load(self, path):
        """ Load Parameter

        # Arguments
            path: Path of json-file
        # Return
            Parameter class
        """
        return super().load(os.path.join(path, "Trainer_Params"))

    def save(self, path):
        """ Save parameter as json-file

        # Arguments
            path: Path to save
        """
        if not os.path.exists(path):
            os.makedirs(path)
        super().save(os.path.join(path, "Trainer_Params"))
        return


class UNET_Trainer(ITrainer):
    """ A example class to train a generator neural-network
    """

    def __init__(self, trainer_params):
        """
        # Arguments
            trainer_params: Parameter from class Example_Trainer_Params
        """

        #   Initialize the abstract Class ITrainer
        super().__init__(trainer_params)

        self.is_training = tf.placeholder(tf.bool, shape=[], name='is_training')

        unet_params = Generator_UNet_Params(activation='relu',
                                            normalization=self.params.normalization_G,
                                            filter_dim=self.params.filter_dim,
                                            depth=self.params.depth,
                                            loss_name=self.params.loss,
                                            generator='UNET')

        unet_params.decay = self.params.decay
        unet_params.step_decay = self.params.step_decay
        unet_params.beta1 = self.params.beta1
        unet_params.learning_rate = self.params.learning_rate_G

        self.generator_UNET = Generator_UNet_Model(self.sess, unet_params, self.global_step, self.is_training)

        #   Create the directorys for logs, checkpoints and samples
        self.prepare_directorys()
        #   Save the hole dl_core library as zip
        save_experiment(self.checkpoint_dir)
        #   Save the Trainer_Params as json
        self.params.save(self.checkpoint_dir)

        #   Placeholder for input x
        self.all_X = tf.placeholder(tf.float32,
                                    [None,
                                     self.params.image_size,
                                     self.params.image_size, 3],
                                    name='all_X')

        #   Placeholder for ground-truth Y
        self.all_Y = tf.placeholder(tf.float32,
                                    [None,
                                     self.params.image_size,
                                     self.params.image_size, 1],
                                    name='all_Y')

        #   Build Pipeline
        self.build_pipeline()

        #   Initialize all variables
        self.sess.run(tf.global_variables_initializer())
        self.sess.run(tf.local_variables_initializer())
        print(' [*] All variables initialized...')

        self.saver = tf.train.Saver()

        #   Load pre-trained model
        if self.params.use_pretrained_generator or self.params.new is False:
            self.generator_UNET.load(self.params.pretrained_generator_dir)

        #   Load checkpoint
        if self.params.load_checkpoint:
            load(self.sess, self.params.checkpoint_restore_dir)

        return

    def prepare_directorys(self):
        """ Prepare the directorys for logs, samples and checkpoints
        """
        self.model_dir = "%s__%s" % (
            self.generator_UNET.params.name, self.batch_size)
        self.sample_dir = os.path.join(self.params.root_dir, 'samples',
                                       self.params.experiment_name + '_GPU' +
                                       str(self.params.gpus),
                                       self.sample_dir + '_' + self.model_dir)
        self.checkpoint_dir = os.path.join(self.params.root_dir, 'checkpoints',
                                           self.params.experiment_name + '_GPU' +
                                           str(self.params.gpus),
                                           self.checkpoint_dir)
        self.log_dir = os.path.join(self.params.root_dir, 'logs',
                                    self.params.experiment_name + '_GPU' +
                                    str(self.params.gpus))

        if self.params.new:
            if os.path.exists(self.checkpoint_dir):
                shutil.rmtree(self.checkpoint_dir)
            if os.path.exists(self.sample_dir):
                shutil.rmtree(self.sample_dir)
            if os.path.exists(self.log_dir):
                shutil.rmtree(self.log_dir)

        if not os.path.exists(self.sample_dir):
            os.makedirs(self.sample_dir)
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

    def set_validation_set(self, batch_valid_X, batch_valid_Y, batch_valid_unknown_X):
        """ Set the validation set for x and Y which same batch-size like training-examples

        # Arguments
            batch_valid_X: samples for x
            batch_valid_Y: samples for Y
        """
        self.image_X = normalize(batch_valid_X, normalization_type='tanh')
        self.image_Y = normalize(batch_valid_Y, normalization_type='sigmoid')

        self.unknown_X = normalize(batch_valid_unknown_X, normalization_type='tanh')

        print(' [*] Unknown_X ' + str(self.unknown_X.shape))
        print(' [*] Image_X ' + str(self.image_X.shape))
        print(' [*] Image_Y ' + str(self.image_Y.shape))

    def validate(self, epoch, iteration, idx):
        """ Validate the validation-set

        # Arguments
            epoch:   Current epoch
            iteration: Current interation
            idx: Index in current epoch
        """

        #   Validate Samples

        samples, g_loss_val, g_summery, g_summery_vis = self.sess.run([self.generator_UNET.G,
                                                                       self.generator_UNET.total_loss,
                                                                       self.summary_val,
                                                                       self.summary_vis],
                                                                      feed_dict={self.all_X: self.image_X,
                                                                                 self.all_Y: self.image_Y,
                                                                                 self.epoch: epoch,
                                                                                 self.generator_UNET.learning_rate: self.params.learning_rate_G,
                                                                                 self.is_training: False})

        self.writer.add_summary(g_summery, iteration)
        self.writer.add_summary(g_summery_vis, iteration)

        samples, g_summery, g_summery_vis = self.sess.run([self.generator_UNET.G,
                                                           self.summary_unknown,
                                                           self.summary_vis_unknown],
                                                          feed_dict={self.all_X: self.unknown_X,
                                                                     self.all_Y: self.image_Y,
                                                                     self.epoch: epoch,
                                                                     self.generator_UNET.learning_rate: self.params.learning_rate_G,
                                                                     self.is_training: False})

        self.writer.add_summary(g_summery, iteration)
        self.writer.add_summary(g_summery_vis, iteration)

        if iteration == 0:
            _, g_loss_val, g_summery = self.sess.run([self.generator_UNET.G,
                                                      self.generator_UNET.total_loss,
                                                      self.summary_vis_one],
                                                     feed_dict={self.all_X: self.image_X,
                                                                self.all_Y: self.image_Y,
                                                                self.epoch: epoch,
                                                                self.is_training: False})

            self.writer.add_summary(g_summery, iteration)

            _, g_summery = self.sess.run([self.generator_UNET.G,
                                          self.summary_vis_one_unknown],
                                         feed_dict={self.all_X: self.unknown_X,
                                                    self.all_Y: self.image_Y,
                                                    self.epoch: epoch,
                                                    self.is_training: False})

            self.writer.add_summary(g_summery, iteration)

        print("[Sample] g_loss: %.8f" % g_loss_val)

    def make_summarys(self, gradient_list):
        """ Calculate some metrics and add it to the summery for the validation-set

        # Arguments
            gradient_list: Gradients to store in log-file as histogram
        """

        super().make_summarys(gradient_list)

        self.summary_vis_unknown = tf.summary.merge([self.generator_UNET.summary_vis_unknown])
        self.summary_vis_one_unknown = tf.summary.merge([self.generator_UNET.summary_vis_one_unknown])
        self.summary_unknown = tf.summary.merge([self.generator_UNET.summary_unknown])

    def set_losses(self, Y):

        self.g_loss = self.generator_UNET.loss(Y)

    def build_model(self, tower_id):
        """ Build models for U-Net

        Paper:

        # Arguments
            tower_id: Tower-ID
        # Return
            List of all existing models witch should trained
        """

        #   Create generator model
        self.generator_UNET.build_model(self.all_X)

        self.set_losses(self.all_Y)

        #   Append all models with should be optimized
        model_list = [self.generator_UNET]

        return model_list

    def train_online(self, batch_X, batch_Y, epoch=0, counter=1, idx=0, batch_total=0):
        """ Training, validating and saving of the generator model

        # Arguments
            batch_X: Training-Examples for input x
            batch_Y: Training-Examples for ground-truth Y
            epoch: Current epoch
            counter: Current iteration
            idx: Current batch
            batch_total: Total batch size
        """

        #   Fill batch randomly if images are missing
        if batch_X.shape[0] != self.params.batch_size or batch_Y.shape[0] != self.params.batch_size:
            batch_X_list = batch_X.tolist()
            batch_Y_list = batch_Y.tolist()

            for i in range(len(batch_X), self.params.batch_size):
                index = randint(0, len(batch_X) - 1)
                batch_X_list.append(batch_X[index])
                batch_Y_list.append(batch_Y[index])

            batch_X = np.asarray(batch_X_list)
            batch_Y = np.asarray(batch_Y_list)
            print(' [!] Batch filled')

        start_time = time.time()

        if self.params.cyclic_learning_rate:
            self.params.learning_rate_G = self.generator_UNET.crl.get_learning_rate(counter)

        #   Data augumentaion
        pre_processing = Preprocessing()
        pre_processing.add_function_xy(Preprocessing.Rotate(steps=15).function)
        pre_processing.add_function_xy(Preprocessing.Flip().function)

        for i in range(len(batch_X)):
            batch_X[i], batch_Y[i] = pre_processing.run(batch_X[i], batch_Y[i])

        #   Normalize input images between -1 and 1
        self.input_X = normalize(batch_X, normalization_type='tanh')
        batch_Y = batch_Y.reshape((batch_Y.shape[0], batch_Y.shape[1], batch_Y.shape[2], 1))
        self.input_Y = normalize(batch_Y, normalization_type='sigmoid')

        #   Validate after N iterations
        if np.mod(counter, self.params.evals_per_iteration) == 0:
            self.validate(epoch, counter, idx)

        # Optimize Generator

        feed_dict = {self.all_X: self.input_X,
                     self.all_Y: self.input_Y,
                     self.epoch: epoch,
                     self.generator_UNET.learning_rate: self.params.learning_rate_G,
                     self.is_training: True}

        _, g_loss, summary_G = self.sess.run([self.generator_UNET.optimizer, self.generator_UNET.total_loss, self.summary],
                                             feed_dict=feed_dict)

        self.writer.add_summary(summary_G, counter)

        print("Train UNET: Epoch: [%2d] [%4d/%4d] time: %4.4f, g_loss: %.8f"
              % (epoch, idx, batch_total, time.time() - start_time, g_loss))

        #   Save model and checkpoint
        if np.mod(counter + 1, 50) == 0:  # int(batch_total / 2)
            self.generator_UNET.save(self.sess, self.checkpoint_dir, self.global_step)
