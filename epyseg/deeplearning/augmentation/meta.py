from matplotlib import pyplot as plt
from epyseg.deeplearning.augmentation.generators.data import DataGenerator
from epyseg.deeplearning.augmentation.generators.meta import MetaGenerator
import numpy as np

# logging
from epyseg.tools.logger import TA_logger

logger = TA_logger()

ALL_AUGMENTATIONS_BUT_INVERT_AND_HIGH_NOISE = [{'type': None}, {'type': None}, {'type': 'zoom'}, {'type': 'blur'},
                                               {'type': 'translate'},
                                               {'type': 'shear'}, {'type': 'flip'}, {'type': 'rotate'},
                                               {'type': 'low noise'}, {'type': 'high noise'}, {'type': 'stretch'}]

ALL_AUGMENTATIONS_BUT_INVERT = [{'type': None}, {'type': None}, {'type': 'zoom'}, {'type': 'blur'},
                                {'type': 'translate'},
                                {'type': 'shear'}, {'type': 'flip'}, {'type': 'rotate'}, {'type': 'low noise'},
                                {'type': 'high noise'}, {'type': 'stretch'}]

ALL_AUGMENTATIONS_BUT_INVERT_AND_NOISE = [{'type': None}, {'type': 'zoom'}, {'type': 'blur'},
                                          {'type': 'translate'}, {'type': 'shear'},
                                          {'type': 'flip'}, {'type': 'rotate'}, {'type': 'stretch'},
                                          {'type': 'rotate (interpolation free)'},
                                          {'type': 'rotate (interpolation free)'},
                                          {'type': 'rotate (interpolation free)'}]

ALL_AUGMENTATIONS = [{'type': None}, {'type': None}, {'type': 'zoom'}, {'type': 'blur'}, {'type': 'translate'},
                     {'type': 'shear'}, {'type': 'flip'}, {'type': 'rotate'}, {'type': 'invert'}, {'type': 'low noise'},
                     {'type': 'high noise'}, {'type': 'stretch'}]

ALL_AUGMENTATIONS_BUT_HIGH_NOISE = [{'type': None}, {'type': None}, {'type': 'zoom'}, {'type': 'blur'},
                                    {'type': 'translate'},
                                    {'type': 'shear'}, {'type': 'flip'}, {'type': 'rotate'}, {'type': 'invert'},
                                    {'type': 'low noise'}, {'type': 'stretch'}]

STRETCHED_AUG_EPITHELIA = [{'type': None}, {'type': 'stretch'}, {'type': 'stretch'},
                       {'type': 'stretch'}, {'type': 'invert'}, {'type': 'flip'}, {'type': 'translate'},
                       {'type': 'zoom'}, {'type': 'blur'}, {'type': 'shear'}, {'type': 'rotate'}, {'type': 'low noise'}]

STRETCHED_AUG_EPITHELIA_2 = [{'type': None}, {'type': None}, {'type': None}, {'type': None}, {'type': 'stretch'}, {'type': 'stretch'},
                       {'type': 'stretch'}, {'type': 'invert'},{'type': 'invert'},{'type': 'invert'},{'type': 'invert'}, {'type': 'flip'}, {'type': 'translate'},
                       {'type': 'zoom'}, {'type': 'blur'}, {'type': 'shear'}, {'type': 'rotate'}, {'type': 'low noise'}]

STRETCHED_AUG_EPITHELIA_3 = [{'type': None}, {'type': None}, {'type': None}, {'type': None}, {'type': 'stretch'}, {'type': 'stretch'},
                       {'type': 'stretch'}, {'type': 'flip'}, {'type': 'translate'},
                       {'type': 'zoom'}, {'type': 'blur'}, {'type': 'shear'}, {'type': 'rotate'}, {'type': 'low noise'},{'type': 'rotate (interpolation free)'}, {'type': 'rotate (interpolation free)'}, {'type': 'rotate (interpolation free)'}]

STRETCHED_AUG_EPITHELIA_4 = [{'type': None}, {'type': None}, {'type': 'stretch'}, {'type': 'stretch'},
                       {'type': 'stretch'}, {'type': 'flip'}, {'type': 'translate'},{'type': 'flip'},  {'type': 'zoom'}, {'type': 'shear'}, {'type': 'rotate'}, {'type': 'rotate'}, {'type': 'rotate'}, {'type': 'rotate'},
                       {'type': 'zoom'}, {'type': 'blur'}, {'type': 'shear'}, {'type': 'rotate'}, {'type': 'low noise'},{'type': 'rotate (interpolation free)'}, {'type': 'rotate (interpolation free)'}, {'type': 'rotate (interpolation free)'}]


TRAINING_FOR_BEGINNING_LITTLE_INTERPOLATION =  [{'type': 'rotate (interpolation free)'}, {'type': 'rotate (interpolation free)'}, {'type': 'rotate (interpolation free)'}, {'type': None}, {'type': 'flip'}, {'type': 'translate'}, {'type': 'blur'}]

NO_AUGMENTATION = [{'type': None}]

TEST_AUGMENTATION = [{'type': 'invert'}]

SAFE_AUGMENTATIONS_FOR_SINGLE_PIXEL_WIDE = [{'type': None}, {'type': 'blur'}, {'type': 'translate'}, {'type': 'flip'}]

SAFE_AUGMENTATIONS_FOR_SINGLE_PIXEL_WIDE_PLUS_INVERT_AND_NOISE = [{'type': None}, {'type': 'blur'},
                                                                  {'type': 'translate'}, {'type': 'flip'},
                                                                  {'type': 'invert'}, {'type': 'low noise'}]


class MetaAugmenter:

    def __init__(self, inputs=None, outputs=None, output_folder=None, input_shape=(None, None, None, 1),
                 output_shape=(None, None, None, 1), input_channel_of_interest=None, output_channel_of_interest=None,
                 input_channel_reduction_rule='copy channel of interest to all channels',
                 input_channel_augmentation_rule='copy channel of interest to all channels',
                 output_channel_reduction_rule='copy channel of interest to all channels',
                 output_channel_augmentation_rule='copy channel of interest to all channels',
                 augmentations=None, crop_parameters=None, mask_dilations=None, infinite=False,
                 default_input_tile_width=128, default_input_tile_height=128,
                 default_output_tile_width=128, default_output_tile_height=128,
                 keep_original_sizes=False,
                 input_normalization={'method': 'Rescaling (min-max normalization)', 'range': [0, 1],
                                      'individual_channels': True},
                 output_normalization={'method': 'Rescaling (min-max normalization)', 'range': [0, 1],
                                       'individual_channels': True},
                 validation_split=0, test_split=0,
                 shuffle=True, clip_by_frequency=None, is_predict_generator=False, overlap_x=0, overlap_y=0,
                 batch_size=None, batch_size_auto_adjust=False, invert_image=False, remove_n_border_mask_pixels=None,
                 is_output_1px_wide=False, rebinarize_augmented_output=False, **kwargs):

        self.augmenters = []

        self.inputs = inputs
        self.outputs = outputs
        self.output_folder = output_folder
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.input_channel_of_interest = input_channel_of_interest
        self.output_channel_of_interest = output_channel_of_interest

        self.input_channel_reduction_rule = input_channel_reduction_rule
        self.input_channel_augmentation_rule = input_channel_augmentation_rule
        self.output_channel_reduction_rule = output_channel_reduction_rule
        self.output_channel_augmentation_rule = output_channel_augmentation_rule

        self.augmentations = augmentations
        self.crop_parameters = crop_parameters
        self.batch_size = batch_size
        self.batch_size_auto_adjust = batch_size_auto_adjust
        self.invert_image = invert_image
        self.remove_n_border_mask_pixels = remove_n_border_mask_pixels
        self.is_output_1px_wide = is_output_1px_wide
        self.rebinarize_augmented_output = rebinarize_augmented_output
        self.mask_dilations = mask_dilations
        self.infinite = infinite
        self.default_input_tile_width = default_input_tile_width
        self.default_input_tile_height = default_input_tile_height
        self.default_output_tile_width = default_output_tile_width
        self.default_output_tile_height = default_output_tile_height
        self.keep_original_sizes = keep_original_sizes
        self.input_normalization = input_normalization
        self.output_normalization = output_normalization
        self.validation_split = validation_split
        self.test_split = test_split
        self.shuffle = shuffle
        self.clip_by_frequency = clip_by_frequency
        self.is_predict_generator = is_predict_generator
        self.overlap_x = overlap_x
        self.overlap_y = overlap_y

        if inputs is not None:
            for i, inp in enumerate(inputs):
                if outputs is not None:
                    cur_output = outputs[i]
                else:
                    cur_output = None
                self.augmenters.append(
                    DataGenerator(inputs=inp, outputs=cur_output, output_folder=output_folder, input_shape=input_shape,
                                  output_shape=output_shape, input_channel_of_interest=input_channel_of_interest,
                                  output_channel_of_interest=output_channel_of_interest,
                                  input_channel_reduction_rule=input_channel_reduction_rule,
                                  input_channel_augmentation_rule=input_channel_augmentation_rule,
                                  output_channel_reduction_rule=output_channel_reduction_rule,
                                  output_channel_augmentation_rule=output_channel_augmentation_rule,
                                  augmentations=augmentations, crop_parameters=crop_parameters,
                                  mask_dilations=mask_dilations,
                                  infinite=infinite, default_input_tile_width=default_input_tile_width,
                                  default_input_tile_height=default_input_tile_height,
                                  default_output_tile_width=default_output_tile_width,
                                  default_output_tile_height=default_output_tile_height,
                                  keep_original_sizes=keep_original_sizes,
                                  input_normalization=input_normalization,
                                  output_normalization=output_normalization,
                                  validation_split=validation_split, test_split=test_split,
                                  shuffle=shuffle,
                                  clip_by_frequency=clip_by_frequency,
                                  is_predict_generator=is_predict_generator, overlap_x=overlap_x, overlap_y=overlap_y,
                                  invert_image=invert_image, remove_n_border_mask_pixels=remove_n_border_mask_pixels,
                                  is_output_1px_wide=is_output_1px_wide,
                                  rebinarize_augmented_output=rebinarize_augmented_output))

    def _get_significant_parameter(self, local_param, global_param):
        if local_param is not None:
            return local_param
        else:
            return global_param

    def appendDatasets(self, datasets=None, augmentations=None, **kwargs):

        logger.debug('datasets ' + str(datasets))
        logger.debug('augs ' + str(augmentations))

        if datasets is None:
            return

        # parse and handle inputs
        for dataset in datasets:
            fused = {**dataset, 'augmentations': augmentations}

            print('fused', fused)

            self.append(**fused)

    def append(self, inputs=None, outputs=None, output_folder=None, input_shape=None, output_shape=None,
               input_channel_of_interest=None, output_channel_of_interest=None,
               input_channel_reduction_rule=None, input_channel_augmentation_rule=None,
               output_channel_reduction_rule=None, output_channel_augmentation_rule=None,
               augmentations=None, crop_parameters=None, mask_dilations=None, infinite=None,
               default_input_tile_width=None, default_input_tile_height=None, default_output_tile_width=None,
               default_output_tile_height=None, keep_original_sizes=None, input_normalization=None,
               output_normalization=None, validation_split=None, test_split=None,
               shuffle=None, clip_by_frequency=None,
               is_predict_generator=None, overlap_x=None, overlap_y=None, invert_image=None,
               remove_n_border_mask_pixels=None, is_output_1px_wide=None, rebinarize_augmented_output=None, **kwargs):

        print('debug 123', inputs, outputs, self.inputs, self.outputs)

        self.augmenters.append(
            DataGenerator(inputs=self._get_significant_parameter(inputs, self.inputs),
                          ouputs=self._get_significant_parameter(outputs, self.outputs),
                          output_folder =self._get_significant_parameter(output_folder, self.output_folder),
                          input_shape=self._get_significant_parameter(input_shape, self.input_shape),
                          output_shape=self._get_significant_parameter(output_shape, self.output_shape),
                          input_channel_of_interest=self._get_significant_parameter(input_channel_of_interest,
                                                                                    self.input_channel_of_interest),
                          output_channel_of_interest=self._get_significant_parameter(output_channel_of_interest,
                                                                                     self.output_channel_of_interest),
                          input_channel_reduction_rule=self._get_significant_parameter(input_channel_reduction_rule,
                                                                                       self.input_channel_reduction_rule),
                          input_channel_augmentation_rule=self._get_significant_parameter(
                              input_channel_augmentation_rule, self.input_channel_augmentation_rule),
                          output_channel_reduction_rule=self._get_significant_parameter(output_channel_reduction_rule,
                                                                                        self.output_channel_reduction_rule),
                          output_channel_augmentation_rule=self._get_significant_parameter(
                              output_channel_augmentation_rule, self.output_channel_augmentation_rule),
                          augmentations=self._get_significant_parameter(augmentations, self.augmentations),
                          crop_parameters=self._get_significant_parameter(crop_parameters, self.crop_parameters),
                          mask_dilations=self._get_significant_parameter(mask_dilations, self.mask_dilations),
                          infinite=self._get_significant_parameter(infinite, self.infinite),
                          default_input_tile_width=self._get_significant_parameter(default_input_tile_width,
                                                                                   self.default_input_tile_width),
                          default_input_tile_height=self._get_significant_parameter(default_input_tile_height,
                                                                                    self.default_input_tile_height),
                          default_output_tile_width=self._get_significant_parameter(default_output_tile_width,
                                                                                    self.default_output_tile_width),
                          default_output_tile_height=self._get_significant_parameter(default_output_tile_height,
                                                                                     self.default_output_tile_height),
                          keep_original_sizes=self._get_significant_parameter(keep_original_sizes,
                                                                              self.keep_original_sizes),
                          validation_split=self._get_significant_parameter(validation_split, self.validation_split),
                          test_split=self._get_significant_parameter(test_split, self.test_split),
                          shuffle=self._get_significant_parameter(shuffle, self.shuffle),
                          clip_by_frequency=self._get_significant_parameter(clip_by_frequency, self.clip_by_frequency),
                          is_predict_generator=self._get_significant_parameter(is_predict_generator,
                                                                               self.is_predict_generator),
                          overlap_x=self._get_significant_parameter(overlap_x, self.overlap_x),
                          overlap_y=self._get_significant_parameter(overlap_y, self.overlap_y),
                          invert_image=self._get_significant_parameter(invert_image, self.invert_image),
                          remove_n_border_mask_pixels=self._get_significant_parameter(remove_n_border_mask_pixels,
                                                                                      self.remove_n_border_mask_pixels),
                          input_normalization=self._get_significant_parameter(input_normalization,
                                                                              self.input_normalization),
                          output_normalization=self._get_significant_parameter(output_normalization,
                                                                               self.output_normalization),
                          is_output_1px_wide=self._get_significant_parameter(is_output_1px_wide,
                                                                             self.is_output_1px_wide),
                          rebinarize_augmented_output=self._get_significant_parameter(rebinarize_augmented_output,
                                                                                      self.rebinarize_augmented_output)
                          ))

    def validation_generator(self, infinite=False):
        if infinite:
            while True:
                for orig, label in self._validation_generator(skip_augment=True):
                    yield orig, label
        else:
            for orig, label in self._validation_generator(skip_augment=True):
                yield orig, label

    def train_generator(self, infinite=False):
        if infinite:
            while True:
                for orig, label in self._train_generator(skip_augment=False):
                    yield orig, label
        else:
            for orig, label in self._train_generator(skip_augment=False):
                yield orig, label

    def test_generator(self, infinite=False):
        if infinite:
            while True:
                for orig, label in self._test_generator(skip_augment=True):
                    yield orig, label
        else:
            for orig, label in self._test_generator(skip_augment=True):
                yield orig, label

    def _train_generator(self, skip_augment):
        train = MetaGenerator(self.augmenters, shuffle=self.shuffle, batch_size=self.batch_size, gen_type='train')
        for out in train.generator(skip_augment):
            try:
                yield out
            except:
                # failed to generate output --> continue
                continue

    def _test_generator(self, skip_augment):
        test = MetaGenerator(self.augmenters, shuffle=False, batch_size=self.batch_size, gen_type='test')
        for out in test.generator(skip_augment):
            yield out

    def _validation_generator(self, skip_augment):
        valid = MetaGenerator(self.augmenters, shuffle=self.shuffle, batch_size=self.batch_size, gen_type='valid')
        for out in valid.generator(skip_augment):
            yield out

    def predict_generator(self):  # TODO can use datagen for now
        pass

    def __len__(self):
        # returns the nb of datasets
        if not self.augmenters:
            return 0
        return len(self.augmenters)

    # returns the real nb of batches with the current parameters...
    def get_train_length(self):
        # need run the train algo once with real tiled data to get the counts
        train_generator = self._train_generator(skip_augment=True)
        nb_batches = 0
        for _, _ in train_generator:
            nb_batches += 1
        return nb_batches

    def get_test_length(self):
        # need run the train algo once with real tiled data to get the counts
        test_generator = self._test_generator(skip_augment=True)
        nb_batches = 0
        for _, _ in test_generator:
            nb_batches += 1
        return nb_batches

    def get_validation_length(self):
        # need run the train algo once with real tiled data to get the counts
        validation_generator = self._validation_generator(skip_augment=True)
        nb_batches = 0
        for _, _ in validation_generator:
            nb_batches += 1
        return nb_batches


if __name__ == '__main__':
    pass
