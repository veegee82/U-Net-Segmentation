from tfcore.interfaces.IInferencing import *
from Generator.generator_unet import *


class Inferencer_Params(IInferencer_Params):
    def __init__(self,
                 image_size,
                 params_path='',
                 model_path='',
                 load=False,
                 ):
        super().__init__()

        self.root_dir = "../"
        self.image_size = image_size
        self.gpus = [0]
        self.input_norm = 'tanh'
        self.normalization_G = 'IN'

        # U-Net Params
        self.depth = 5
        self.filter_dim = 64

        self.use_pretrained_generator = True
        self.pretrained_generator_dir = model_path

        if params_path is not '':
            if load:
                if self.load(params_path):
                    self.save(params_path)
            else:
                self.save(params_path)

        if params_path is not '':
            if load:
                if self.load(params_path):
                    self.save(params_path)
            else:
                self.save(params_path)

    def load(self, path):
        return super().load(os.path.join(path, "Trainer_Params.json"))

    def save(self, path):
        if not os.path.exists(path):
            os.makedirs(path)
        super().save(os.path.join(path, "Trainer_Params.json"))
        return


class Inferencer(IInferencing):

    def __init__(self, params):
        super().__init__(params)

        self.global_step = tf.Variable(0, trainable=False)

        self.is_training = tf.placeholder(tf.bool, shape=[], name='is_training')

        # Load Model
        classifier_params = Generator_UNet_Params(activation='relu',
                                                  normalization=self.params.normalization_G,
                                                  depth=self.params.depth,
                                                  filter_dim=self.params.filter_dim,
                                                  generator='UNET')

        self.unet = Generator_UNet_Model(self.sess, classifier_params, self.global_step, self.is_training)

        if self.build_model_inference():
            print(' [*] Build model pass...')

        tf.global_variables_initializer().run(session=self.sess)
        print(' [*] All variables initialized...')

        self.unet.load(self.params.pretrained_generator_dir)

    def build_model_inference(self):
        with tf.device("/gpu:0"):
            self.inputs = tf.placeholder(tf.float32,
                                         [None,
                                          self.params.image_size,
                                          self.params.image_size,
                                          3],
                                         name='inputs')

            self.model = self.unet.build_model(self.inputs)

        print('Model loaded...')
        return

    def inference(self, input):
        input = normalize(input).astype(np.float)
        w, h, c = input.shape

        sample = np.asarray(input.reshape(1, w, h, c))

        predictions = self.unet.mask_image.eval(feed_dict={self.inputs: sample,
                                                           self.is_training: False}, session=self.sess)

        return predictions
