from inferencer import Inferencer, Inferencer_Params
from tfcore.interfaces.IPipeline_Inferencer import IPipeline_Inferencer_Params, IPipeline_Inferencer
from tfcore.utilities.preprocessing import Preprocessing
import gflags
import os
import sys
import imageio
import numpy as np


def get_filename(idx, filename='', decimals=5):
    for n in range(decimals, -1, -1):
        if idx < pow(10, n):
            filename += '0'
        else:
            filename += str(idx)
            break
    return filename + '.png'

class Pipeline_Inferencer_Params(IPipeline_Inferencer_Params):

    def __init__(self,
                 data_dir_y,
                 data_dir_x=''):
        super().__init__(data_dir_y=data_dir_y, data_dir_x=data_dir_x)


class Pipeline_Inferencer(IPipeline_Inferencer):

    def __init__(self, inferencer, params, pre_processing):
        super().__init__(inferencer, params, pre_processing)

    def get_element(self, idx):

        try:
            img_x = imageio.imread(self.files_x[idx])
            img_y = self.files_x[idx].count('good')
        except FileNotFoundError:
            raise FileNotFoundError(' [!] File not found of data-set x')

        if self.pre_processing is not None:
            img_x, _ = self.pre_processing.run(img_x, None)

        return img_x, np.asarray(img_y)


# flags = tf.app.flags
flags = gflags.FLAGS
gflags.DEFINE_string("dataset", "../Data/", "Dataset path")
gflags.DEFINE_string("outdir", "../Data/predictions", "Output path")
gflags.DEFINE_string("model_dir", "../Model_dida", "Model directory")

def main():
    flags(sys.argv)

    model_params = Inferencer_Params(image_size=256,
                                     model_path=flags.model_dir)
    model_inferencer = Inferencer(model_params)

    pipeline_params = Pipeline_Inferencer_Params(data_dir_x=os.path.join(flags.dataset, 'test_X'),
                                                 data_dir_y=None)
    pipeline = Pipeline_Inferencer(inferencer=model_inferencer, params=pipeline_params, pre_processing=None)

    count = 0
    first_pass = True
    while first_pass or img_out is not None:
        if first_pass:
            first_pass = False
            if not os.path.exists(flags.outdir):
                os.makedirs(flags.outdir)

        img_out = pipeline.run()
        if img_out is not None:
            filename = get_filename(count, 'image_')
            imageio.imwrite(os.path.join(flags.outdir, filename), img_out)
            print(' [*] save file ' + filename)
        count += 1

if __name__ == "__main__":
    main()
