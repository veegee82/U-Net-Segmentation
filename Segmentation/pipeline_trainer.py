from trainer import *
#from Segmentation.trainer import *
from tfcore.interfaces.IPipeline_Trainer import *
from tfcore.utilities.preprocessing import Preprocessing
import gflags
import os
import sys

class Pipeline_Params(IPipeline_Trainer_Params):
    """ Simple example for Pipeline_Params

    """

    def __init__(self,
                 data_dir_y,
                 data_dir_x,
                 validation_dir_x,
                 validation_dir_y,
                 output_dir,
                 convert=True,
                 epochs=25,
                 batch_size=16,
                 shuffle=True,
                 cache_size=1,
                 interp='bicubic'):
        super().__init__(data_dir_x=data_dir_x,
                         data_dir_y=data_dir_y,
                         validation_dir_x=validation_dir_x,
                         validation_dir_y=validation_dir_y,
                         output_dir=output_dir,
                         convert=convert,
                         epochs=epochs,
                         batch_size=batch_size,
                         shuffle=shuffle,
                         cache_size=cache_size,
                         interp=interp)
        self.validation_dir_x = validation_dir_x


class Training_Pipeline(IPipeline_Trainer):
    """ Simple example of inherent from IPipeline and to create an class

    # Arguments
        trainer: Implementation of meta class ITrainer
        params: Implementation of meta class IPipeline_Params
        pre_processing: Implementation of class Preprocessing
    """

    def __init__(self, trainer, params, pre_processing):

        super().__init__(trainer, params, pre_processing)

    def get_element(self, idx):

        try:
            img_y = imageio.imread(self.files_y[idx])
        except FileNotFoundError:
            raise FileNotFoundError(' [!] File not found of data-set Y')

        if not self.params.convert:
            try:
                img_x = imageio.imread(self.files_x[idx])
            except FileNotFoundError:
                raise FileNotFoundError(' [!] File not found of data-set X')
        else:
            img_x = img_y

        if self.pre_processing is not None:
            img_x, img_y = self.pre_processing.run(img_x, img_y)

        return img_x, img_y

    def set_validation(self):

        if self.params.validation_dir_x is not None:
            files_val_x = sorted(get_img_paths(self.params.data_dir_x))
            if len(files_val_x) == 0:
                raise FileNotFoundError(' [!] No files in validation data-set')

        if self.params.validation_dir_y is not None:
            files_val_y = sorted(get_img_paths(self.params.data_dir_y))
            if len(files_val_y) == 0:
                raise FileNotFoundError(' [!] No files in validation data-set')

        if self.params.validation_dir_x is not None:
            files_unknown_x = sorted(get_img_paths(self.params.validation_dir_x))
            if len(files_unknown_x) == 0:
                raise FileNotFoundError(' [!] No files in validation data-set')

        batch_val_x = np.asarray([imageio.imread(file) for file in files_val_x])
        batch_val_y = np.asarray([imageio.imread(file) for file in files_val_y])
        batch_val_y = batch_val_y.reshape((batch_val_y.shape[0],batch_val_y.shape[1],batch_val_y.shape[2], 1))

        batch_unknown_x = np.asarray([imageio.imread(file) for file in files_unknown_x])

        try:
            self.trainer.set_validation_set(np.asarray(batch_val_x), np.asarray(batch_val_y), np.asarray(batch_unknown_x))
        except Exception as err:
            print(' [!] Error in Trainer on set_validation():', err)
            raise


#   Flaks to configure from shell
flags = gflags.FLAGS
gflags.DEFINE_string("config_path", '', "Path for config files")
gflags.DEFINE_string("dataset", "../Data/", "Dataset path")
gflags.DEFINE_string("loss", "pixelwise_softmax", "Dataset path")
gflags.DEFINE_integer("gpu", 0, "Dataset path")

def main():
    flags(sys.argv)

    #   Trainer_Params witch inherits from ITrainer_Params
    model_params = Trainer_Params(image_size=256,
                                  params_path=flags.config_path,
                                  loss=flags.loss,
                                  gpu=flags.gpu)
    #   Trainer witch inherits from ITrainer
    model_trainer = UNET_Trainer(model_params)

    #   Pre-processing Pipeline
    pre_processing = Preprocessing()
    pre_processing.add_function_xy(Preprocessing.Flip(direction=('horizontal', 'vertical')).function)
    pre_processing.add_function_xy(Preprocessing.Rotate(steps=1).function)

    #   Pipeline_Params witch inherits from IPipeline_Params
    pipeline_params = Pipeline_Params(data_dir_x=os.path.join(flags.dataset, 'train_X'),
                                      data_dir_y=os.path.join(flags.dataset, 'train_Y'),
                                      validation_dir_x=os.path.join(flags.dataset, 'test_X'),
                                      validation_dir_y=os.path.join(flags.dataset, 'train_Y'),
                                      batch_size=model_params.batch_size,
                                      epochs=model_params.epoch,
                                      convert=False,
                                      shuffle=True,
                                      output_dir=None)

    #   Pipeline witch inherits from IPipeline
    pipeline = Training_Pipeline(trainer=model_trainer, params=pipeline_params, pre_processing=pre_processing)

    pipeline.set_validation()

    #   Start Training
    pipeline.run()


if __name__ == "__main__":
    main()
