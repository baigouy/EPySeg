from matplotlib import pyplot as plt
from epyseg.deeplearning.augmentation.generators.data import DataGenerator
from epyseg.deeplearning.augmentation.generators.meta import MetaGenerator
import numpy as np
from epyseg.tools.logger import TA_logger # logging

logger = TA_logger()


# MINIMAL_AUGMENTATIONS = [{'type': None}, {'type': None},{'type': None}, {'type': None}, {'type': 'zoom'}, {'type': 'blur'}, {'type': 'translate'}, {'type': 'rotate'}]
# added intensity shifts to the minimal augmentation --> should make it more robust for masking
# {'type': None},

# I have recently modified the augmentation so that it takes into account the elastic deformation which is very good training I assume
MINIMAL_AUGMENTATIONS = [{'type': None}, {'type': None},{'type': None},  {'type': 'zoom'}, {'type': 'blur'}, {'type': 'translate'}, {'type': 'rotate'},{'type': 'random_intensity_gamma_contrast'}, {'type': 'intensity'}, {'type': 'random_intensity_gamma_contrast'}, {'type': 'intensity'}, {'type':'elastic'}, {'type':'elastic'}]

# MINIMAL_AUGMENTATIONS_WITH_ELASTIC= [{'type': None}, {'type': None},{'type': None}, {'type': 'zoom'}, {'type': 'blur'}, {'type': 'translate'}, {'type': 'rotate'},{'type': 'random_intensity_gamma_contrast'}, {'type': 'intensity'}, {'type': 'random_intensity_gamma_contrast'}, {'type': 'intensity'}, {'type':'elastic'}, {'type':'elastic'}, {'type':'elastic'}]

ALL_AUGMENTATIONS_BUT_INVERT_AND_HIGH_NOISE = [{'type': None}, {'type': None}, {'type': 'zoom'}, {'type': 'blur'},
                                               {'type': 'translate'}, {'type':'elastic'},
                                               {'type': 'shear'}, {'type': 'flip'}, {'type': 'rotate'},
                                               {'type': 'low noise'}, {'type': 'high noise'}, {'type': 'stretch'}]

ALL_AUGMENTATIONS_BUT_INVERT = [{'type': None}, {'type': None}, {'type': 'zoom'}, {'type': 'blur'},
                                {'type': 'translate'}, {'type':'elastic'},
                                {'type': 'shear'}, {'type': 'flip'}, {'type': 'rotate'}, {'type': 'low noise'},
                                {'type': 'high noise'}, {'type': 'stretch'}]

ALL_AUGMENTATIONS_BUT_INVERT_AND_NOISE = [{'type': None}, {'type': 'zoom'}, {'type': 'blur'},
                                          {'type': 'translate'}, {'type': 'shear'}, {'type':'elastic'},
                                          {'type': 'flip'}, {'type': 'rotate'}, {'type': 'stretch'},
                                          {'type': 'rotate (interpolation free)'},
                                          {'type': 'rotate (interpolation free)'},
                                          {'type': 'rotate (interpolation free)'}]

ALL_AUGMENTATIONS = [{'type': None}, {'type': None}, {'type': 'zoom'}, {'type': 'blur'}, {'type': 'translate'},
                     {'type': 'shear'}, {'type': 'flip'}, {'type': 'rotate'}, {'type': 'invert'}, {'type': 'low noise'},
                     {'type': 'high noise'}, {'type': 'stretch'}, {'type':'elastic'}]

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

TEST_AUGMENTATION = [{'type': 'elastic'}]#[{'type': 'invert'}]

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
                 augmentations=None, treat_some_inputs_as_outputs=False,
                 crop_parameters=None, mask_dilations=None, infinite=False,
                 default_input_tile_width=128, default_input_tile_height=128,
                 default_output_tile_width=128, default_output_tile_height=128,
                 keep_original_sizes=False,
                 input_normalization={'method': 'Rescaling (min-max normalization)', 'range': [0, 1],
                                      'individual_channels': True},
                 output_normalization={'method': 'Rescaling (min-max normalization)', 'range': [0, 1],
                                       'individual_channels': True},
                 validation_split=0, test_split=0,
                 shuffle=True, clip_by_frequency=None, is_predict_generator=False, overlap_x=0, overlap_y=0,
                 batch_size=None, batch_size_auto_adjust=False, invert_image=False, input_bg_subtraction=None, create_epyseg_style_output=None, remove_n_border_mask_pixels=None,
                 is_output_1px_wide=False, rebinarize_augmented_output=False,
                 rotate_n_flip_independently_of_augmentation=False,
                 mask_lines_and_cols_in_input_and_mask_GT_with_nans=None, # should be 'id' or 'noid' and requires a custom loss and metrics --> can only be applied with some losses
                 z_frames_to_add=None,
                 **kwargs):

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
        self.treat_some_inputs_as_outputs= treat_some_inputs_as_outputs
        self.crop_parameters = crop_parameters
        self.batch_size = batch_size
        self.batch_size_auto_adjust = batch_size_auto_adjust
        self.invert_image = invert_image
        self.input_bg_subtraction = input_bg_subtraction
        self.create_epyseg_style_output=create_epyseg_style_output
        self.remove_n_border_mask_pixels = remove_n_border_mask_pixels
        self.is_output_1px_wide = is_output_1px_wide
        self.rebinarize_augmented_output = rebinarize_augmented_output
        self.rotate_n_flip_independently_of_augmentation = rotate_n_flip_independently_of_augmentation
        self.mask_lines_and_cols_in_input_and_mask_GT_with_nans = mask_lines_and_cols_in_input_and_mask_GT_with_nans
        self.z_frames_to_add = z_frames_to_add
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
                                  augmentations=augmentations,
                                  treat_some_inputs_as_outputs=treat_some_inputs_as_outputs,
                                  crop_parameters=crop_parameters,
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
                                  invert_image=invert_image, input_bg_subtraction=input_bg_subtraction, create_epyseg_style_output=create_epyseg_style_output, remove_n_border_mask_pixels=remove_n_border_mask_pixels,
                                  is_output_1px_wide=is_output_1px_wide,
                                  rebinarize_augmented_output=rebinarize_augmented_output,
                                  rotate_n_flip_independently_of_augmentation=rotate_n_flip_independently_of_augmentation,
                                  mask_lines_and_cols_in_input_and_mask_GT_with_nans=mask_lines_and_cols_in_input_and_mask_GT_with_nans,
                                  z_frames_to_add = z_frames_to_add
                                  ))

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

            # print('fused', fused)

            self.append(**fused)

    # TODO add a way to treat input as mask and maybe ouput as input from the point of view of data aug
    def append(self, inputs=None, outputs=None, output_folder=None, input_shape=None, output_shape=None,
               input_channel_of_interest=None, output_channel_of_interest=None,
               input_channel_reduction_rule=None, input_channel_augmentation_rule=None,
               output_channel_reduction_rule=None, output_channel_augmentation_rule=None,
               augmentations=None, treat_some_inputs_as_outputs=None,
               crop_parameters=None, mask_dilations=None, infinite=None,
               default_input_tile_width=None, default_input_tile_height=None, default_output_tile_width=None,
               default_output_tile_height=None, keep_original_sizes=None, input_normalization=None,
               output_normalization=None, validation_split=None, test_split=None,
               shuffle=None, clip_by_frequency=None,
               is_predict_generator=None, overlap_x=None, overlap_y=None, invert_image=None, input_bg_subtraction=None,create_epyseg_style_output=None,
               remove_n_border_mask_pixels=None, is_output_1px_wide=None, rebinarize_augmented_output=None,
               rotate_n_flip_independently_of_augmentation=None,mask_lines_and_cols_in_input_and_mask_GT_with_nans=None,
               z_frames_to_add = None,
               **kwargs):

        # print('debug 123', inputs, outputs, self.inputs, self.outputs)
        # inputs and outputs are ok --> why is there a bug then????

        self.augmenters.append(
            DataGenerator(inputs=self._get_significant_parameter(inputs, self.inputs),
                          outputs=self._get_significant_parameter(outputs, self.outputs),
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
                          treat_some_inputs_as_outputs=self._get_significant_parameter(treat_some_inputs_as_outputs, self.treat_some_inputs_as_outputs),
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
                          input_bg_subtraction=self._get_significant_parameter(input_bg_subtraction, self.input_bg_subtraction),
                          create_epyseg_style_output=self._get_significant_parameter(create_epyseg_style_output, self.create_epyseg_style_output),
                          remove_n_border_mask_pixels=self._get_significant_parameter(remove_n_border_mask_pixels,
                                                                                      self.remove_n_border_mask_pixels),
                          input_normalization=self._get_significant_parameter(input_normalization,
                                                                              self.input_normalization),
                          output_normalization=self._get_significant_parameter(output_normalization,
                                                                               self.output_normalization),
                          is_output_1px_wide=self._get_significant_parameter(is_output_1px_wide,
                                                                             self.is_output_1px_wide),
                          rebinarize_augmented_output=self._get_significant_parameter(rebinarize_augmented_output,
                                                                                      self.rebinarize_augmented_output),
                          rotate_n_flip_independently_of_augmentation=self._get_significant_parameter(rotate_n_flip_independently_of_augmentation,
                                                                                      self.rotate_n_flip_independently_of_augmentation),
                          mask_lines_and_cols_in_input_and_mask_GT_with_nans=self._get_significant_parameter(mask_lines_and_cols_in_input_and_mask_GT_with_nans,
                                                                                      self.mask_lines_and_cols_in_input_and_mask_GT_with_nans),
                          z_frames_to_add=self._get_significant_parameter(z_frames_to_add, self.z_frames_to_add),
                          ))

    def validation_generator(self, infinite=False):
        if infinite:
            while True:
                for orig, label in self._validation_generator(skip_augment=True):
                    # bug fix for recent tensorflow that really needs true and pred to be unpacked if single input and output
                    if len(orig) == 1:
                        orig = orig[0]
                    if len(label) == 1:
                        label = label[0]
                    yield orig, label
        else:
            for orig, label in self._validation_generator(skip_augment=True):
                # bug fix for recent tensorflow that really needs true and pred to be unpacked if single input and output
                if len(orig) == 1:
                    orig = orig[0]
                if len(label) == 1:
                    label = label[0]
                yield orig, label

    def train_generator(self, infinite=False):
        if infinite:
            while True:
                for orig, label in self._train_generator(skip_augment=False):
                    # bug fix for recent tensorflow that really needs true and pred to be unpacked if single input and output
                    if len(orig) == 1:
                        orig = orig[0]
                    if len(label) == 1:
                        label = label[0]
                    yield orig, label
        else:
            for orig, label in self._train_generator(skip_augment=False):
                # bug fix for recent tensorflow that really needs true and pred to be unpacked if single input and output
                if len(orig) == 1:
                    orig = orig[0]
                if len(label) == 1:
                    label = label[0]
                yield orig, label

    def test_generator(self, infinite=False):
        if infinite:
            while True:
                for orig, label in self._test_generator(skip_augment=True):
                    # bug fix for recent tensorflow that really needs true and pred to be unpacked if single input and output
                    if len(orig) == 1:
                        orig = orig[0]
                    if len(label) == 1:
                        label = label[0]
                    yield orig, label
        else:
            for orig, label in self._test_generator(skip_augment=True):
                # bug fix for recent tensorflow that really needs true and pred to be unpacked if single input and output
                if len(orig) == 1:
                    orig = orig[0]
                if len(label) == 1:
                    label = label[0]
                yield orig, label


    def angular_yielder(self, orig, mask, count):
        # mask = self.extra_watershed_mask(mask) # shrink mask to 1 px wide irrespective of transfo
        # NB could do here the generations of the nine stacks --> TODO --> would increase size by 9 but it is a good idea I think
        # can also copy the code of the other stuff

        if count == 0:
            # rot 180
            return np.rot90(orig, 2, axes=(-3, -2)), np.rot90(mask, 2, axes=(-3, -2))

        if count == 1:
            # flip hor
            return np.flip(orig, -2), np.flip(mask, -2)

        if count == 2:
            # flip ver
            return np.flip(orig, -3), np.flip(mask, -3)

        # make it yield the original and the nine versions of it
        # --> TODO
        # ça marche ça me genere les 9 versions du truc dans tous les sens --> probablement ce que je veux --> tt mettre ici
        if count == 3:
            # yield np.rot90(orig, axes=(-3, -2)), np.rot90(mask, axes=(-3, -2))

            # rot 90
            return np.rot90(orig, axes=(-3, -2)), np.rot90(mask, axes=(-3, -2))

        if count == 4:
            # rot 90_flipped_hor or ver
            return np.flip(np.rot90(orig, axes=(-3, -2)), -2), np.flip(np.rot90(mask, axes=(-3, -2)), -2)

        if count == 5:
            # rot 90_flipped_hor or ver
            return np.flip(np.rot90(orig, axes=(-3, -2)), -3), np.flip(np.rot90(mask, axes=(-3, -2)), -3)

        if count == 6:
            # rot 270
            return np.rot90(orig, 3, axes=(-3, -2)), np.rot90(mask, 3, axes=(-3, -2))

    def _train_generator(self, skip_augment, first_run=False):
        train = MetaGenerator(self.augmenters, shuffle=self.shuffle, batch_size=self.batch_size, gen_type='train')
        for out in train.generator(skip_augment, first_run):
            try:
                # # print(len(out))
                # #  that works check that all are there and all are possible otherwise skip
                # # --> need ensure that width = height
                # # need set a parameter to be sure to use it or not and need remove rotation and flip from augmentation list (or not in fact)
                # orig, mask = out
                # augmentations = 7
                # if orig[0].shape[-2] != orig[0].shape[-3]:
                #     augmentations = 3
                # for aug in range(augmentations):
                #     yield self.angular_yielder(orig, mask, aug)
                # yield orig, mask
                yield out
            except GeneratorExit:
                # bug fix for the GeneratorExit error, see https://stackoverflow.com/questions/46542147/elegant-way-for-breaking-a-generator-loop-generatorexit-error
                pass
                break
            except Exception:
                # failed to generate output --> continue
                continue

    def _test_generator(self, skip_augment, first_run=False):
        test = MetaGenerator(self.augmenters, shuffle=False, batch_size=self.batch_size, gen_type='test')
        for out in test.generator(skip_augment, first_run):
            try:
                # # yield out
                # # print(len(out))
                # #  that works check that all are there and all are possible otherwise skip
                # # --> need ensure that width = height
                # # need set a parameter to be sure to use it or not and need remove rotation and flip from augmentation list (or not in fact)
                # orig, mask = out
                # augmentations = 7
                # if orig[0].shape[-2] != orig[0].shape[-3]:
                #     augmentations = 3
                # for aug in range(augmentations):
                #     yield self.angular_yielder(orig, mask, aug)
                # yield orig, mask
                yield out
            except GeneratorExit:
                # bug fix for the GeneratorExit error, see https://stackoverflow.com/questions/46542147/elegant-way-for-breaking-a-generator-loop-generatorexit-error
                pass
                break
            except Exception:
                # failed to generate output --> continue
                continue

    def _validation_generator(self, skip_augment, first_run=False):
        valid = MetaGenerator(self.augmenters, shuffle=self.shuffle, batch_size=self.batch_size, gen_type='valid')
        for out in valid.generator(skip_augment, first_run):
            try:
                # # yield out
                # # print(len(out))
                # #  that works check that all are there and all are possible otherwise skip
                # # --> need ensure that width = height
                # # need set a parameter to be sure to use it or not and need remove rotation and flip from augmentation list (or not in fact)
                # orig, mask = out
                # augmentations = 7
                # if orig[0].shape[-2] != orig[0].shape[-3]:
                #     augmentations = 3
                # for aug in range(augmentations):
                #     yield self.angular_yielder(orig, mask, aug)
                # yield orig, mask
                yield out
            except GeneratorExit:
                # bug fix for the GeneratorExit error, see https://stackoverflow.com/questions/46542147/elegant-way-for-breaking-a-generator-loop-generatorexit-error
                pass
                break
            except Exception:
                # failed to generate output --> continue
                continue

    def predict_generator(self):  # TODO can use datagen for now
        pass

    def __len__(self):
        # returns the nb of datasets
        if not self.augmenters:
            return 0
        return len(self.augmenters)

    # returns the real nb of batches with the current parameters...
    def get_train_length(self, first_run=False):
        # need run the train algo once with real tiled data to get the counts
        train_generator = self._train_generator(skip_augment=True, first_run=first_run)
        nb_batches = 0
        for _, _ in train_generator:
            nb_batches += 1
        return nb_batches

    def get_test_length(self, first_run=False):
        # need run the train algo once with real tiled data to get the counts
        test_generator = self._test_generator(skip_augment=True, first_run=first_run)
        nb_batches = 0
        for _, _ in test_generator:
            nb_batches += 1
        return nb_batches


    def get_validation_length(self, first_run=False):
        # need run the train algo once with real tiled data to get the counts
        validation_generator = self._validation_generator(skip_augment=True, first_run=first_run)
        nb_batches = 0
        for _, _ in validation_generator:
            nb_batches += 1
        return nb_batches


if __name__ == '__main__':
    pass
