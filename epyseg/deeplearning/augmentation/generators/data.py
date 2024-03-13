# TODO clean and annotate this class
# finally added support for adding images to it -−> THE ONLY THING IF DIRECTLY LOADING IMAGES IS THAT THE NB OF CHANNELS AND GENERAL SHAPE OF THE IMAGE SHOULD BE THE ONE TO BE USED
# PRACTICALLY THE IMAGE WILL BE REPLACED BY A SINGLE STACK IMAGE AND THE SOFT WILL LOOP OVER THE FIRST DIMENSION -> REALLY GOOD IN A WAY AND EASIER TO USE!!

from builtins import enumerate
from functools import partial

from natsort import natsorted
import traceback

import epyseg.img
from epyseg.binarytools.cell_center_detector import get_seeds
from skimage.measure import label
from epyseg.deeplearning.augmentation.generators.data_augmentation_2 import graded_intensity_modification, elastic, \
    random_intensity_gamma_contrast_changer, stretch, high_noise, low_noise, shuffleZ, rollZ, invert, translate, \
    change_image_intensity_and_shift_range, blur, flip, rotate_interpolation_free, rotate, zoom, shear, \
    execute_chained_augmentations, elastic_with_zoom_and_translation, strong_elastic, \
    strong_elastic_with_zoom_and_translation, crop_augmentation
from epyseg.img import Img, white_top_hat, black_top_hat, pad_border_xy, get_2D_tiles_with_overlap, tiles_to_linear, \
    tiles_to_batch, clip_by_frequency, normalization_methods, normalization, to_stack, stack_to_imgs
import random
from scipy import ndimage
import numpy as np
import glob
import os
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt

from epyseg.ta.tracking.tools import smart_name_parser
from epyseg.utils.loadlist import loadlist
from epyseg.tools.logger import TA_logger # logging

# TODO add augmentation randomly swap channels --> can learn by itself which channel is containing epithelia

logger = TA_logger()

class DataGenerator:
    # TODO should I also store increment there ???
    shear_range = [0, 0.99, 0.25]
    zoom_range = [0.05, 0.5, 0.25]
    stretch_range = [1.5, 3.5, 3.5]
    blur_range = [0.1, 1., 2.]
    width_height_shift_range = [0.01, 0.30, 0.075]
    rotation_range = [0., 1., 1.]
    intensity_range = [0.1, 1.,
                       0.1]  # default is intensity * 0.1 --> divided by up to 10 --> since it's a range its intensity divided by something between 10 and 1 --> cool
    augmentation_types_and_ranges = {'None': None, 'shear': shear_range, 'zoom': zoom_range, 'rotate': rotation_range,
                                     'rotate (interpolation free)': None,
                                     'flip': None, 'blur': blur_range, 'intensity': intensity_range,
                                     'translate': width_height_shift_range,
                                     'invert': None, 'low noise': None,
                                     'roll along Z (2D + GT ignored)': None,
                                     'shuffle images along Z (2D + GT ignored)': None,
                                     'random_intensity_gamma_contrast': None,
                                     'high noise': None, 'stretch': stretch_range,
                                     'elastic': None,
                                     'strong_elastic': None,
                                     'elastic_with_zoom_and_translation': None,
                                     'strong_elastic_with_zoom_and_translation': None,
                                     'graded_intensity_modification':None}

    augmentation_types_and_values = {'None': None, 'shear': shear_range[2], 'zoom': zoom_range[2],
                                     'rotate': rotation_range[2],
                                     'rotate (interpolation free)': None,
                                     'flip': None, 'blur': blur_range[2],
                                     'intensity': intensity_range[2],
                                     'translate': width_height_shift_range[2],
                                     'invert': None,
                                     'roll along Z (2D + GT ignored)': None,
                                     'shuffle images along Z (2D + GT ignored)': None,
                                     'low noise': None,
                                     'high noise': None,
                                     'random_intensity_gamma_contrast': None,
                                     'stretch': 3.,
                                     'elastic': None,
                                     'strong_elastic': None,
                                     'elastic_with_zoom_and_translation': None,
                                     'strong_elastic_with_zoom_and_translation': None,
                                     'graded_intensity_modification': None
                                     }

    augmentation_types_and_methods = {'None': None, 'shear': shear, 'zoom': zoom,
                                      'rotate': rotate,
                                      'rotate (interpolation free)': rotate_interpolation_free,
                                      'flip': flip, 'blur': blur,
                                      'intensity': change_image_intensity_and_shift_range,
                                      'translate': translate, 'invert': invert,
                                      'roll along Z (2D + GT ignored)': rollZ,
                                      'shuffle images along Z (2D + GT ignored)': shuffleZ,
                                      'low noise': low_noise, 'high noise': high_noise,
                                      'stretch': stretch,
                                      'random_intensity_gamma_contrast': random_intensity_gamma_contrast_changer,
                                      'elastic': elastic,
                                      'strong_elastic': strong_elastic,
                                      'elastic_with_zoom_and_translation': elastic_with_zoom_and_translation,
                                      'strong_elastic_with_zoom_and_translation': strong_elastic_with_zoom_and_translation,
                                      'graded_intensity_modification': graded_intensity_modification}  # added elastic deformation because can be very useful

    # TODO add also the elastic deforms
    # TODO add the possibility to tranform inputs to masks
    # z_frames_to_add --> an int, or a tuple or a list of len == 2 --> can be used to add black images above or below...
    def __init__(self, inputs=None, outputs=None, output_folder=None, input_shape=(None, None, None, 1),
                 output_shape=(None, None, None, 1), input_channel_of_interest=None, output_channel_of_interest=None,
                 input_channel_reduction_rule='copy channel of interest to all channels',
                 input_channel_augmentation_rule='copy channel of interest to all channels',
                 output_channel_reduction_rule='copy channel of interest to all channels',
                 output_channel_augmentation_rule='copy channel of interest to all channels',
                 augmentations=None, treat_some_inputs_as_outputs=False,
                 crop_parameters=None, mask_dilations=None, infinite=False,
                 default_input_tile_width=64, default_input_tile_height=64,
                 default_output_tile_width=64, default_output_tile_height=64,
                 keep_original_sizes=False,
                 input_normalization={'method': 'Rescaling (min-max normalization)', 'range': [0, 1],
                                      'individual_channels': True},
                 output_normalization={'method': 'Rescaling (min-max normalization)', 'range': [0, 1],
                                       'individual_channels': True},
                 validation_split=0, test_split=0,
                 shuffle=True, clip_by_frequency=None, is_predict_generator=False, overlap_x=0, overlap_y=0,
                 invert_image=False, input_bg_subtraction=None, pre_processing_input__or_output=None, create_epyseg_style_output=None,
                 remove_n_border_mask_pixels=None, is_output_1px_wide=False,
                 rebinarize_augmented_output=False, rotate_n_flip_independently_of_augmentation=False,
                 mask_lines_and_cols_in_input_and_mask_GT_with_nans=None,
                 # if none does nothing if noid --> cannot learn from masked pixels at all reducedid --> can learn from masked pixels --> requires specific losses to work --> find an easy way to do that ...
                 z_frames_to_add=None,
                 **kwargs):

        # NB Keep IF USING DIRECTLY IMAGES AS INPUT THEN ONE MUST ABSOLUTELY MAKE SURE IT HAS THE RIGHT NB OF CHANNELS !!! --> BE VERY CAREFUL WITH THAT

        logger.debug('clip by freq' + str(clip_by_frequency))

        self.random_seed = datetime.now()

        self.EXTRAPOLATION_MASKS = 1  # bicubic 0 # nearest # TODO keep like that because 3 makes really weird results for binary images
        self.is_predict_generator = is_predict_generator

        # convert single input string or image to list
        if isinstance(inputs, (str,  np.ndarray)):
            inputs = [inputs]
        if isinstance(outputs,  (str,  np.ndarray)):
            outputs = [outputs]

        # convert single images to list
        # if isinstance(inputs, np.ndarray):
        #     inputs=[inputs]
        # if isinstance(outputs, np.ndarray):
        #     outputs=[outputs]


        if self.is_predict_generator:
            self.overlap_x = overlap_x
            self.overlap_y = overlap_y
        else:
            self.overlap_x = 0
            self.overlap_y = 0

        if self.overlap_x is None:
            self.overlap_x = 0
        if self.overlap_y is None:
            self.overlap_y = 0

        self.invert_image = invert_image  # first thing to do. Should not be applied to the output.
        self.input_bg_subtraction = input_bg_subtraction  # bg subtraction for input
        self.pre_processing_input__or_output = pre_processing_input__or_output
        self.create_epyseg_style_output = create_epyseg_style_output  # to be used only for pre trained models

        self.clip_by_frequency = clip_by_frequency
        self.shuffle = shuffle
        self.output_folder = output_folder  # save folder --> if specified save images as .npi # TODO finalize

        if self.is_predict_generator:
            self.augmentations = [None]
            # self.invert_in_augs = False
        else:
            self.augmentations = augmentations
            # self.invert_in_augs = False
            # if self.augmentations is not None and len(self.augmentations) > 0:
            #     # if invert is in augs --> remove it and apply it to input image with frequency = 0.5 (random)
            #     if {'type': 'invert'} in self.augmentations:
            #         self.invert_in_augs = True
            #         self.augmentations.remove({'type': 'invert'})

        self.treat_some_inputs_as_outputs = treat_some_inputs_as_outputs

        self.mask_dilations = mask_dilations
        self.infinite = infinite  # useful to create an infinite generator

        self.input_shape = input_shape
        if isinstance(self.input_shape, tuple):
            self.input_shape = [self.input_shape]

        # print(self.input_shape, inputs)
        if len(self.input_shape) < len(inputs):
            logger.error('Please specify input shapes the model has ' + str(
                len(self.input_shape)) + ' inputs whereas the inputs len is ' + str(
                len(inputs)) + ' --> soft will crash')

        self.output_shape = output_shape
        if isinstance(self.output_shape, tuple):
            self.output_shape = [self.output_shape]

        try:
            if outputs and len(self.output_shape) < len(outputs):
                logger.error('Please specify output shapes the model has ' + str(
                    len(self.output_shape)) + ' outputs whereas the outputs len is ' + str(
                    len(outputs)) + ' --> soft will crash')
        except:
            traceback.print_exc()

        self.input_normalization = input_normalization

        if self.input_normalization is None:
            self.input_normalization = {'method': None}
        self.output_normalization = output_normalization
        if self.output_normalization is None:
            self.output_normalization = {'method': None}
        self._index = -1
        self.validation_split = validation_split  # TODO
        self.test_split = test_split

        self.default_input_tile_width = default_input_tile_width
        self.default_input_tile_height = default_input_tile_height
        self.default_output_tile_width = default_output_tile_width
        self.default_output_tile_height = default_output_tile_height
        self.keep_original_sizes = keep_original_sizes
        self.remove_n_border_mask_pixels = remove_n_border_mask_pixels
        self.is_output_1px_wide = is_output_1px_wide
        self.rebinarize_augmented_output = rebinarize_augmented_output
        self.rotate_n_flip_independently_of_augmentation = rotate_n_flip_independently_of_augmentation
        self.mask_lines_and_cols_in_input_and_mask_GT_with_nans = mask_lines_and_cols_in_input_and_mask_GT_with_nans

        # can be used to add black frames above or below --> can be useful for denoising and surface projection
        self.z_frames_to_add = z_frames_to_add
        if z_frames_to_add is not None:
            if isinstance(z_frames_to_add, int):
                self.z_frames_to_add_above = z_frames_to_add
                self.z_frames_to_add_below = z_frames_to_add
            else:
                self.z_frames_to_add_above = z_frames_to_add[0]
                self.z_frames_to_add_below = z_frames_to_add[1]

        # TODO need create an incrementer per input and output
        self.input_incr = 0
        self.output_incr = 0

        self.input_channel_of_interest = input_channel_of_interest
        self.output_channel_of_interest = output_channel_of_interest

        self.input_channel_reduction_rule = input_channel_reduction_rule
        self.input_channel_augmentation_rule = input_channel_augmentation_rule
        self.output_channel_reduction_rule = output_channel_reduction_rule
        self.output_channel_augmentation_rule = output_channel_augmentation_rule

        self.crop_parameters = crop_parameters
        if inputs is not None:
            self.full_set_inputs = []
            for path in inputs:
                lst = DataGenerator.get_list_of_images(path)
                self.full_set_inputs.append(lst)

        if outputs is not None:
            self.full_set_outputs = []
            for path in outputs:
                lst = DataGenerator.get_list_of_images(path)
                self.full_set_outputs.append(lst)
        elif not is_predict_generator:
            # Auto load masks from TA
            self.full_set_outputs = []
            # assume TA organisation --> check in TA folder for mask
            for liste in self.full_set_inputs:
                lst = []
                for file in liste:

                    if isinstance(file, str):
                        # get path without ext
                        mask_folder = os.path.splitext(file)[0]
                        file = Path(mask_folder + '/handCorrection.tif')  # TODO do path join instead...
                        if file.exists():
                            mask = mask_folder + '/handCorrection.tif'
                        else:
                            mask = mask_folder + '/handCorrection.png'
                        lst.append(mask)
                    else:
                        # if it is already an nd array --> just add it directly
                        lst.append(file)
                self.full_set_outputs.append(lst)

        # for training and validation I need inputs and outputs
        self.train_inputs = []
        self.train_outputs = []

        self.validation_inputs = []
        self.validation_outputs = []

        # for test I just need the inputs TODO
        self.test_inputs = []  # here I just need inputs
        self.test_outputs = []  # here I just need inputs

        fullset_size = len(self.full_set_inputs[0])

        if not is_predict_generator:
            if test_split is not None and test_split != 0:
                size = int(test_split * fullset_size)

                if size == 0:
                    logger.error('not enough data in list to generate a test dataset for input ' + str(
                        inputs))
                else:
                    for lst in self.full_set_inputs:
                        sub_lst = lst[::int(len(lst) / size)]
                        self.test_inputs.append(sub_lst)
                        for i in sub_lst:
                            lst.remove(i)
                    for lst in self.full_set_outputs:
                        sub_lst = lst[::int(len(lst) / size)]
                        self.test_outputs.append(sub_lst)
                        for i in sub_lst:
                            lst.remove(i)

            if validation_split is not None and validation_split != 0:
                size = int(validation_split * fullset_size)

                if validation_split >= 1:
                    # assume percentage
                    validation_split /= 100
                # print('test dsqdqsd', size, validation_split, fullset_size, validation_split*fullset_size, int(len(lst)/size), len(lst)/size)

                if size == 0:
                    logger.error('not enough data in list to generate a validation dataset for input ' + str(inputs))
                else:
                    for lst in self.full_set_inputs:
                        sub_lst = lst[::int(len(lst) / size)]
                        self.validation_inputs.append(sub_lst)
                        for i in sub_lst:
                            lst.remove(i)
                    for lst in self.full_set_outputs:
                        sub_lst = lst[::int(len(lst) / size)]
                        self.validation_outputs.append(sub_lst)
                        for i in sub_lst:
                            lst.remove(i)

            # take the remaining inputs/outputs as train inputs/outputs
            self.train_inputs = self.full_set_inputs
            self.train_outputs = self.full_set_outputs

            logger.debug('full set ' + str(fullset_size))
            if self.train_inputs:
                logger.debug('train lst ' + str(len(self.train_inputs[0])) + ' ' + str(len(self.train_inputs)))
            if self.train_outputs:
                logger.debug('train lst ' + str(len(self.train_outputs[0])) + ' ' + str(len(self.train_outputs)))
            if self.test_inputs:
                logger.debug('test lst ' + str(len(self.test_inputs[0])) + ' ' + str(len(self.test_inputs)))
            if self.test_outputs:
                logger.debug('test lst ' + str(len(self.test_outputs[0])) + ' ' + str(len(self.test_outputs)))
            if self.validation_inputs:
                logger.debug(
                    'validation lst ' + str(len(self.validation_inputs[0])) + ' ' + str(len(self.validation_inputs)))
            if self.validation_outputs:
                logger.debug(
                    'validation lst ' + str(len(self.validation_outputs[0])) + ' ' + str(len(self.validation_outputs)))
        else:
            self.predict_inputs = self.full_set_inputs
            if self.predict_inputs:
                logger.debug('predict lst' + str(self.predict_inputs))

    def has_train_set(self):
        """
        Check if the generator has a train set.

        Returns:
            bool: True if a train set exists, False otherwise.
        """
        return self.train_inputs and len(self.train_inputs) != 0

    def has_validation_set(self):
        """
        Check if the generator has a validation set.

        Returns:
            bool: True if a validation set exists, False otherwise.
        """
        return self.validation_inputs and len(self.validation_inputs) != 0

    def has_test_set(self):
        """
        Check if the generator has a test set.

        Returns:
            bool: True if a test set exists, False otherwise.
        """
        return self.test_inputs and len(self.test_inputs) != 0

    def is_train_set_size_coherent(self):
        """
        Check if the train set size is coherent.

        Returns:
            bool: True if the train set size is coherent, False otherwise.
        """
        return len(self.train_inputs) == len(self.train_outputs)

    def is_test_set_size_coherent(self):
        """
        Check if the test set size is coherent.

        Returns:
            bool: True if the test set size is coherent, False otherwise.
        """
        return len(self.test_inputs) == len(self.test_outputs)

    def is_validation_set_size_coherent(self):
        """
        Check if the validation set size is coherent.

        Returns:
            bool: True if the validation set size is coherent, False otherwise.
        """
        return len(self.validation_inputs) == len(self.validation_outputs)

    def get_validation_set_length(self):
        """
        Get the length of the validation set.

        Returns:
            int: Length of the validation set.
        """
        if not self.validation_inputs:
            return 0
        return len(self.validation_inputs[0])

    def get_test_set_length(self):
        """
        Get the length of the test set.

        Returns:
            int: Length of the test set.
        """
        if not self.test_inputs:
            return 0
        return len(self.test_inputs[0])

    def get_train_set_length(self):
        """
        Get the length of the train set.

        Returns:
            int: Length of the train set.
        """
        if not self.train_inputs:
            return 0
        return len(self.train_inputs[0])

    def train_generator(self, skip_augment, first_run, __DEBUG__=False):
        """
        Generator function for training data.

        Args:
            skip_augment (bool): Flag indicating whether to skip data augmentation.
            first_run (bool): Flag indicating whether it is the first run.
            __DEBUG__ (bool, optional): Debug flag. Defaults to False.

        Yields:
            tuple: A tuple containing the original input and the corresponding mask.

        Raises:
            GeneratorExit: Raised when the generator is exited.
        """
        if self.shuffle:
            indices = random.sample(range(len(self.train_inputs[0])), len(self.train_inputs[0]))
        else:
            indices = list(range(len(self.train_inputs[0])))


        for idx in indices:
            try:
                if __DEBUG__:
                    print('self.train_inputs', self.train_inputs)
                    print('self.train_outputs', self.train_outputs)

                orig, mask = self.generate(self.train_inputs, self.train_outputs, idx,
                                           skip_augment, first_run)

                yield orig, mask
            except GeneratorExit:
                break
            except Exception:
                traceback.print_exc()
                continue

    def angular_yielder(self, orig, count=None):
        """
        Generate augmented versions of the input.

        Args:
            orig (ndarray): Original input.
            count (int, optional): Number of augmentations. Defaults to None.

        Returns:
            ndarray: Augmented version of the input.
        """
        random.seed(self.random_seed)

        if count is None:
            augmentations = 8
            if orig[0].shape[-2] != orig[0].shape[-3]:
                augmentations = 4
            count = random.choice(range(augmentations))

        if count == 0:
            return orig
        elif count == 1:
            # rot 180
            return np.rot90(orig, 2, axes=(-3, -2))
        elif count == 2:
            # flip hor
            return np.flip(orig, -2)
        elif count == 3:
            # flip ver
            return np.flip(orig, -3)
        elif count == 4:
            # rot 90
            return np.rot90(orig, axes=(-3, -2))
        elif count == 5:
            # rot 90_flipped_hor or ver
            return np.flip(np.rot90(orig, axes=(-3, -2)), -2)
        elif count == 6:
            # rot 90_flipped_hor or ver
            return np.flip(np.rot90(orig, axes=(-3, -2)), -3)
        elif count == 7:
            # rot 270
            return np.rot90(orig, 3, axes=(-3, -2))

    def test_generator(self, skip_augment, first_run):
        """
        Generator function for test data.

        Args:
            skip_augment (bool): Flag indicating whether to skip data augmentation.
            first_run (bool): Flag indicating whether it is the first run.

        Yields:
            tuple: A tuple containing the generated inputs and outputs for testing.
        """
        for idx in range(len(self.test_inputs[0])):
            try:
                yield self.generate(self.test_inputs, self.test_outputs, idx, skip_augment, first_run)
            except GeneratorExit:
                break
            except Exception:
                traceback.print_exc()
                continue

    def validation_generator(self, skip_augment, first_run):
        """
        Generator function for validation data.

        Args:
            skip_augment (bool): Flag indicating whether to skip data augmentation.
            first_run (bool): Flag indicating whether it is the first run.

        Yields:
            tuple: A tuple containing the generated inputs and outputs for validation.
        """
        for idx in range(len(self.validation_inputs[0])):
            try:
                yield self.generate(self.validation_inputs, self.validation_outputs, idx, skip_augment, first_run)
            except GeneratorExit:
                break
            except Exception:
                continue

    def predict_generator(self, skip_augment=True):
        """
        Generator function for prediction data.

        Args:
            skip_augment (bool, optional): Flag indicating whether to skip data augmentation. Defaults to True.

        Yields:
            ndarray: The generated inputs for prediction.
        """
        for idx in range(len(self.predict_inputs[0])):
            try:
                yield self.generate(self.predict_inputs, None, idx, skip_augment)
            except GeneratorExit:
                break
            except Exception:
                traceback.print_exc()
                continue

    def _get_from(self, input_list, idx):
        """
        Get data from the input list at the specified index.

        Args:
            input_list (list): List of inputs.
            idx (int): Index.

        Returns:
            list: Data at the specified index from the input list.
        """

        if input_list is None:
            return None
        data = []
        for lst in input_list:
            if lst and lst is not None:
                data.append(lst[idx])
            else:
                data.append(None)
        return data

    def generate(self, inputs, outputs, cur_idx, skip_augment, first_run=False):
        """
        Generates augmented inputs and outputs for a given index.

        Args:
            inputs (list): List of input images.
            outputs (list): List of output images.
            cur_idx (int): Current index.
            skip_augment (bool): Flag to skip augmentation.
            first_run (bool, optional): Flag indicating the first run. Defaults to False.

        Returns:
            tuple: Augmented inputs and outputs.

        Raises:
            AssertionError: If the length of inputs and outputs are not equal.

        # Examples:
        #     >>> generator = MyGenerator()
        #     >>> inputs = [...]
        #     >>> outputs = [...]
        #     >>> cur_idx = 0
        #     >>> skip_augment = False
        #     >>> first_run = True
        #     >>> result = generator.generate(inputs, outputs, cur_idx, skip_augment, first_run)
        """
        # Preprocessing and augmenting inputs and outputs
        inp, out = self.augment(self._get_from(inputs, cur_idx), self._get_from(outputs, cur_idx), skip_augment,
                                first_run)




        if self.rebinarize_augmented_output:
            # Rebinarize the augmented output
            for p, o in enumerate(out):
                o[o > o.min()] = o.max()
                out[p] = o

        if self.invert_image:
            # Invert the images if invert_image flag is True
            for idx, img in enumerate(inp):
                inp[idx] = epyseg.img.invert(img)

        inputs = []
        outputs = []
        if self.keep_original_sizes:
            # Generate inputs and outputs without resizing
            for img in inp:
                inputs.append(img)
                if self.is_predict_generator:
                    outputs.append(None)
                    return inputs, outputs
            if not self.is_predict_generator:
                for img in out:
                    outputs.append(img)
            return inputs, outputs
        else:
            # Generate inputs and outputs with resizing
            for idx, img in enumerate(inp):
                input_shape = self.input_shape[idx]
                dimension_h = 1
                dimension_w = 2
                if len(input_shape) == 5:
                    dimension_h = 2
                    dimension_w = 3

                width = self.default_input_tile_width
                height = self.default_input_tile_height
                if input_shape[-2] is not None:
                    width = input_shape[-2]
                if input_shape[-3] is not None:
                    height = input_shape[-3]
                if width is None:
                    width = img.shape[-2]
                if height is None:
                    height = img.shape[-3]

                # Generate tiles with overlap
                crop_parameters, tiles2D_inp = get_2D_tiles_with_overlap(
                    img, width=width, height=height, overlap_x=self.overlap_x, overlap_y=self.overlap_y,
                    overlap=0, dimension_h=dimension_h, dimension_w=dimension_w, force_to_size=True)
                inputs.append(tiles2D_inp)
                if self.is_predict_generator:
                    outputs.append(crop_parameters)
            if not self.is_predict_generator:
                for idx, img in enumerate(out):
                    output_shape = self.output_shape[idx]
                    dimension_h = 1
                    dimension_w = 2
                    if len(output_shape) == 5:
                        dimension_h = 2
                        dimension_w = 3
                    width = self.default_output_tile_width
                    height = self.default_output_tile_height
                    if input_shape[-2] is not None:
                        width = input_shape[-2]
                    if input_shape[-3] is not None:
                        height = input_shape[-3]
                    if width is None:
                        width = img.shape[-2]
                    if height is None:
                        height = img.shape[-3]

                    # Generate tiles with overlap
                    _, tiles2D_out = get_2D_tiles_with_overlap(
                        img, width=width, height=height, overlap_x=self.overlap_x, overlap_y=self.overlap_y,
                        overlap=0, dimension_h=dimension_h, dimension_w=dimension_w, force_to_size=True)
                    outputs.append(tiles2D_out)

            bckup_input_incr = self.input_incr
            bckup_output_incr = self.output_incr
            if self.output_folder is not None and not not self.is_predict_generator:
                # Save input and output tiles to files
                for idx, tiles2D_inp in enumerate(inputs):
                    self.input_incr = bckup_input_incr
                    tiles2D_inp = tiles_to_linear(tiles2D_inp)
                    for idx2, inp in enumerate(tiles2D_inp):
                        if len(np.squeeze(inp).shape) != 1:
                            Img(inp, dimensions='hw').save(
                                self.output_folder + '/input_' + str(idx) + '_' + str(self.output_incr) + '.npz')
                            self.input_incr += 1
                        else:
                            print('error size')

                for idx1, tiles2D_out in enumerate(outputs):
                    self.output_incr = bckup_output_incr
                    tiles2D_out = tiles_to_linear(tiles2D_out)
                    for idx2, inp in enumerate(tiles2D_out):
                        if len(np.squeeze(inp).shape) != 1:
                            Img(inp, dimensions='hw').save(
                                self.output_folder + '/output_' + str(idx) + '_' + str(self.output_incr) + '.npz')
                            self.output_incr += 1
                        else:
                            print('error size')
                return
            else:
                final_inputs = []
                for tiles2D_inp in inputs:
                    final_inputs.append(tiles_to_batch(tiles2D_inp))

                if self.is_predict_generator:
                    return final_inputs, outputs

                final_outputs = []
                for tiles2D_out in outputs:
                    final_outputs.append(tiles_to_batch(tiles2D_out))
                return final_inputs, final_outputs

    def increase_or_reduce_nb_of_channels(self, img, desired_shape, channel_of_interest, increase_rule=None,
                                          decrease_rule=None):
        """
        Adjusts the number of channels in an image to match the desired shape.

        Args:
            img (np.ndarray): Input image.
            desired_shape (tuple): Desired shape of the image.
            channel_of_interest (int): Channel index to copy when increasing or reducing channels.
            increase_rule (str, optional): Rule for increasing the number of channels. Defaults to None.
            decrease_rule (str, optional): Rule for reducing the number of channels. Defaults to None.

        Returns:
            np.ndarray: Image with adjusted number of channels.

        Raises:
            ValueError: If the increase_rule or decrease_rule is not recognized.

        # Examples:
        #     >>> generator = MyGenerator()
        #     >>> img = np.zeros((100, 100, 3), dtype=np.uint8)
        #     >>> desired_shape = (100, 100, 4)
        #     >>> channel_of_interest = 0
        #     >>> increase_rule = 'copy_all'
        #     >>> decrease_rule = 'copy'
        #     >>> result = generator.increase_or_reduce_nb_of_channels(img, desired_shape, channel_of_interest, increase_rule, decrease_rule)
        """


        try:
            input_has_channels = img.has_c()
            if input_has_channels:
                input_channels = img.get_dimension('c')
            else:
                input_channels = None
            if input_channels is None:
                input_channels = 1

            if not input_has_channels:
                img = np.reshape(img, (*img.shape, 1))  # Reshape to add a channel dimension
        except:
            logger.debug('image fed directly by user and not loaded from disk --> assuming it contains the right dimensions')
            # shall I copy the image to be on the safe side ?? --> probably yes so that the nb of pixels are not changed !!!
            img = img.copy()
            input_channels = img.shape[-1] #  if that is the case then the nb of channels is coming from the final stuff

        if desired_shape[-1] != input_channels:
            if input_channels < desired_shape[-1]:
                # Too few channels in the image compared to the desired shape --> need to add channels
                multi_channel_img = np.zeros((*img.shape[:-1], desired_shape[-1]), dtype=img.dtype)
                channel_to_copy = channel_of_interest if channel_of_interest is not None else 0

                if increase_rule and 'copy' in increase_rule:
                    if 'all' in increase_rule:
                        logger.debug('Increasing nb of channels by copying COI to all available channels')
                        # Copy the channel of interest to all available channels
                        for c in range(desired_shape[-1]):
                            multi_channel_img[..., c] = img[..., channel_to_copy]
                        img = multi_channel_img
                    elif 'missing' in increase_rule:
                        logger.debug('Increasing nb of channels by copying COI to extra channels only')
                        # Copy the channel of interest to missing channels, other channels are kept unchanged
                        for c in range(img.shape[-1]):
                            multi_channel_img[..., c] = img[..., c]
                        for c in range(img.shape[-1], desired_shape[-1]):
                            multi_channel_img[..., c] = img[..., channel_to_copy]
                        img = multi_channel_img
                    else:
                        logger.error('Unknown channel number increase rule: ' + str(increase_rule))
                elif increase_rule and 'add' in increase_rule:
                    logger.debug('Increasing nb of channels by adding empty (black) channels')
                    # Copy just the existing channel and keep the rest black
                    for c in range(img.shape[-1]):
                        multi_channel_img[..., c] = img[..., c]
                    img = multi_channel_img
            elif input_channels > desired_shape[-1]:
                # Too many channels in the image compared to the desired shape --> need to reduce channels
                reduced_channel_img = np.zeros((*img.shape[:-1], desired_shape[-1]), dtype=img.dtype)
                channel_to_copy = channel_of_interest if channel_of_interest is not None else 0

                if decrease_rule and 'copy' in decrease_rule:
                    logger.debug('Decreasing nb of channels by copying COI to available channels')
                    # Copy the channel of interest to available channels
                    for c in range(desired_shape[-1]):
                        reduced_channel_img[..., c] = img[..., channel_to_copy]
                    img = reduced_channel_img
                elif decrease_rule and 'remove' in decrease_rule:
                    logger.debug('Decreasing nb of channels by removing extra channels')
                    # Remove extra channels
                    for c in range(desired_shape[-1]):
                        reduced_channel_img[..., c] = img[..., c]
                    img = reduced_channel_img
                else:
                    logger.error('Unknown channel number decrease rule: ' + str(decrease_rule))
            else:  # img.shape[-1] == desired_shape[-1]
                if 'force' in increase_rule or 'force' in decrease_rule:
                    logger.debug('Force copy COI to all channels')
                    # Force copy channel of interest to all channels
                    channel_to_copy = channel_of_interest if channel_of_interest is not None else 0
                    for c in range(desired_shape[-1]):
                        img[..., c] = img[..., channel_to_copy]

        return img

    @staticmethod
    def get_list_of_images(path):
        """
        Retrieves a list of image file paths from a given path.

        Args:
            path (str): Path to a directory containing images, a single image file, or a text file listing image paths.

        Returns:
            list: List of image file paths.

        # Examples:
        #     >>> generator = MyGenerator()
        #     >>> path = 'images/'
        #     >>> image_list = generator.get_list_of_images(path)
        """


        if isinstance(path, np.ndarray):
            # if it is an ndarray return it as a list of images along the first dimension
            return stack_to_imgs(path)

        if path is None or not path:
            return []

        if isinstance(path, list):
            return path  # If the path is already a list, return it

        folderpath = path
        if not folderpath.endswith('/') and not folderpath.endswith(
                '\\') and not '*' in folderpath and not os.path.isfile(folderpath):
            folderpath += '/'  # Add a trailing slash to the folder path if necessary

        list_of_files = []

        if folderpath.lower().endswith('.lst') or folderpath.lower().endswith('.txt'):
            # Load list of files from a text file
            list_of_files = loadlist(folderpath)
        elif '*' in folderpath:
            # Load files using glob pattern matching
            list_of_files = natsorted(glob.glob(folderpath))
        elif os.path.isdir(folderpath):
            # Load image files from a directory
            extensions = ('.png', '.jpg', '.jpeg', '.tif', '.tiff', '.lsm', '.czi', '.lif')
            list_of_files = glob.glob(folderpath + "*")  # Get all files in the directory
            list_of_files = [f for f in list_of_files if
                             f.lower().endswith(extensions)]  # Filter by supported extensions
            list_of_files = natsorted(list_of_files)  # Sort the file list
        elif os.path.isfile(folderpath):
            # Single image file --> convert it to a list
            list_of_files.append(folderpath)

        return list_of_files

    # TODO remove
    def get_augment_method(self):
        if self.augmentations is not None and len(self.augmentations) > 0:
            # print('augmenytations here',self.augmentations)
            # print('random', random.randint(0,100000))
            method = random.choice(self.augmentations)
            # print('picked', method)
            if 'value' in method:
                # change range of augmentation
                self.augmentation_types_and_values[method['type']] = method['value']
            # define augmentation method
            if method['type'] is None:
                return None
            meth = method['type']
            # hack to support passing directly methods or chained methods to the augmenter
            if isinstance(meth,str):
                method = self.augmentation_types_and_methods[method['type']]
            else:
                method=meth
            return method
        else:
            return None

    # get the first not None input to get width and height and return a black image of the appropriate shape
    # TODO maybe make this optional and that really makes no sense I guess for output unless I set it to Nan with an approprite loss --> just think about it
    def _rescue_missing_items(self, items, idx):
        for item in items:
            if item is not None:
                img = Img(item)
                width = img.get_width()
                height = img.get_height()
                inp_shape = self.input_shape[idx]
                im_shape = list(inp_shape)
                # set the image the apppriate width and height and use the model to get the nb of channels
                im_shape[-2]=width
                im_shape[-3]=height
                # print('desired shape', inp_shape, im_shape)
                black_image_for_rescue = Img(np.zeros(tuple(im_shape[1:]),dtype=img.dtype), dimensions='hwc')# fake image just for rescue we don't care about its real dims it should just have a c channel
                # print('black_image_for_rescue',black_image_for_rescue.has_c())
                return black_image_for_rescue
        return None

    def augment(self, inputs, outputs, skip_augment, first_run):
        augmented_inputs = []
        augmented_outputs = []

        if not skip_augment:
            method = self.get_augment_method()
            # print('augmentation method picked', method) #--> ok till there but the lost... --> why
        else:
            method = None

        random_sync_input=0

        inputs_outputs_to_skip = []
        for idx, input in enumerate(inputs):
            if idx == 0:
                # we set the new seed for the first input and keep it for all others
                # set seed here so that random is the same for all inputs
                self.random_seed = datetime.now()
                random.seed(self.random_seed)
            if input is None:
                input = self._rescue_missing_items(inputs, idx)
            # en fait c'est ici qu'il faut que je permette de convert des inputs en outpus --> TODO
            if not self.treat_some_inputs_as_outputs or not self.treat_some_inputs_as_outputs[idx]:
                augmented = self.augment_input(input, self.input_shape[idx],
                                                                                method,
                                                                                skip_augment, first_run)
            else:
                augmented = self.augment_output(input, self.input_shape[idx],
                                                                                 method,
                                                                                 first_run)
            if augmented is None:
                inputs_outputs_to_skip.append(idx)
            augmented_inputs.append(augmented)
        # print('random value at the end of augmentation of input',random.random())
        random_sync_input = random.random()


        if outputs is not None:
            for idx, output in enumerate(outputs):
                augmented = self.augment_output(output, self.output_shape[idx],
                                                                                 method,
                                                                                 first_run)

                if augmented is None:
                    inputs_outputs_to_skip.append(idx)
                augmented_outputs.append(augmented)
            # print('random value at the end of augmentation of output', random.random())
            # if random_sync_input != random.random()
            try:
                assert random_sync_input == random.random() , logger.error("NB Random is NOT IN SYNC, please stop immediately the training and check your augmentations")
                # print('EVERYTHING OK: Random is in SYNC')
            except AssertionError as e:
               print(e)
        else:
            augmented_outputs = None

        for el in sorted(inputs_outputs_to_skip, reverse=True):
            del augmented_inputs[el]
            if augmented_outputs is not None:
                del augmented_outputs[el]
        return augmented_inputs, augmented_outputs

    # NB if there are several inputs they need to have the same random seed --> see how I can do that need to and also same is true for
    def augment_input(self, input, input_shape, method,  skip_augment, first_run):
        logger.debug('method input ' + str(method) + ' ' + str(input))
        # print('method used', method)

        if isinstance(input, str):
            try:
                # logger.info('processing', input)
                input = Img(input)

                # can be the first thing to be done in a way then need copy parameters of the image and increase the value for the Z if that worked

                # should work but need to test it !!! # the
                # add black frames above and below the z channel
                try:
                    logger.debug('adding Z frames to image')
                    if self.z_frames_to_add is not None:
                        if input.has_d():
                            # copy the metadata
                            # print('added Z frame in', input.shape)
                            meta = input.metadata
                            input = self.add_z_frames(input, [], False)[1]
                            # print('added Z frame 2', input.shape)
                            # print(input.metadata)
                            input = Img(input,
                                        metadata=meta)  # reinject metadata!!! # see if I can do that in a cleaner way ???
                            # print(input.metadata)
                            # print('added Z frame', input.shape)
                except:
                    traceback.print_exc()
                    logger.error('could not add Z frames to image')

                # TODO add         if self.z_frames_to_add is not None:
                #             input =

                # pb is that it will require the image to have shape dhwc and c may not have been added yet... and need be done before normalization... --> really not good think how to do and really clean the code
                # best is to hack the image so that it always generate image with the same shape... + a few controls to handle pbs such as display and save with numpy squeeze!!!
                # in fact all is ok...


            except:
                logger.error('Missing/corrupt/unsupported file \'' + input + '\'')
                return None, None
                # so it will fail --> need continue

        input = self.increase_or_reduce_nb_of_channels(input, input_shape, self.input_channel_of_interest,
                                                       self.input_channel_augmentation_rule,
                                                       self.input_channel_reduction_rule)


        # shall I do it here or after channel reduction ??? --> both can be good ideas --> think about it --> simpler here because keeps has_c and alike !!!
        # added code for general preprocessing of input and or output
        if self.pre_processing_input__or_output is not None:
            if isinstance(self.pre_processing_input__or_output, list) and self.pre_processing_input__or_output:
                pre_proc_fn = self.pre_processing_input__or_output[0]
            else:
                pre_proc_fn = self.pre_processing_input__or_output
            if pre_proc_fn is not None:
                input = pre_proc_fn(input)

        # TODO in fact that would make more sense to clip by freq before --> indeed clipping should be done before any data augmentation
        if self.clip_by_frequency is not None:
            logger.debug('Clipping image prior to any processing')
            if isinstance(self.clip_by_frequency, float):
                # for idx, img in enumerate(inp):
                input = clip_by_frequency(input, upper_cutoff=self.clip_by_frequency, channel_mode=True)
                # TODO should I add clipping here for GT too ???
            elif isinstance(self.clip_by_frequency, tuple):
                if len(self.clip_by_frequency) == 2:
                    # for idx, img in enumerate(inp):
                    # print('clip by freq here', self.clip_by_frequency[0], self.clip_by_frequency[1])
                    input = clip_by_frequency(input, lower_cutoff=self.clip_by_frequency[0],
                                                  upper_cutoff=self.clip_by_frequency[1], channel_mode=True)
                # TODO should I add clipping here for GT too ???
                # if not self.is_predict_generator:
                #     # clipping output...
                #     print('clipping output...')
                #     for idx, img in enumerate(out):
                #         out[idx] = clip_by_frequency(img, lower_cutoff=self.clip_by_frequency[0],
                #                                          upper_cutoff=self.clip_by_frequency[1], channel_mode=True)
                else:
                    # for idx, img in enumerate(inp):
                    input = clip_by_frequency(input, upper_cutoff=self.clip_by_frequency[0], channel_mode=True)
                    # TODO should I add clipping here for GT too ???
            elif isinstance(self.clip_by_frequency, dict):
                # for idx, img in enumerate(inp):
                # print('clip by freq here', self.clip_by_frequency['lower_cutoff'], self.clip_by_frequency['upper_cutoff'])
                # print('bef', img.min(), img.max())

                input = clip_by_frequency(input, lower_cutoff=self.clip_by_frequency['lower_cutoff'],
                                              upper_cutoff=self.clip_by_frequency['upper_cutoff'],
                                              channel_mode=self.clip_by_frequency['channel_mode'])
                # TODO should I add clipping here for GT too ???
                # if not self.is_predict_generator:
                #     # clipping output...
                #     print('clipping output...')
                #     for idx, img in enumerate(out):
                #         out[idx] = clip_by_frequency(img, lower_cutoff=self.clip_by_frequency['lower_cutoff'],
                #                                          upper_cutoff=self.clip_by_frequency['upper_cutoff'],
                #                                          channel_mode=self.clip_by_frequency['channel_mode'])
                # print('aft', img.min(), img.max(), inp[idx].min(), inp[idx].max())

        self.normalization_minima_and_maxima_input = None
        if self.input_normalization is not None:
            if self.input_normalization['method'] == normalization_methods[7]:
                if 'range' in self.input_normalization:
                    normalization_range = self.input_normalization['range']
                else:
                    normalization_range = [2, 99.8]
                if 'individual_channels' in self.input_normalization and self.input_normalization[
                    'individual_channels'] is True:
                    self.normalization_minima_and_maxima_input = []
                    for c in range(input.shape[-1]):
                        lowest_percentile = np.percentile(input[..., c], normalization_range[0])
                        highest_percentile = np.percentile(input[..., c], normalization_range[1])
                        self.normalization_minima_and_maxima_input.append([lowest_percentile, highest_percentile])
                else:
                    lowest_percentile = np.percentile(input, normalization_range[0])
                    highest_percentile = np.percentile(input, normalization_range[1])
                    self.normalization_minima_and_maxima_input = [lowest_percentile, highest_percentile]
        if self.normalization_minima_and_maxima_input is not None:
            logger.debug('Normalization per percentile: ' + str(self.normalization_minima_and_maxima_input))

        # TODO maybe clip should be done here... --> would make sense in some cases, could be an option...
        # I do think so yes

        # try new normalization
        # check also where I do normalization

        if self.crop_parameters:
            input = self.crop(input, self.crop_parameters)

        # do normalization here
        if self.normalization_minima_and_maxima_input is not None:
            # --> gain of time --> à tester
            input = normalization(input, normalization_minima_and_maxima=self.normalization_minima_and_maxima_input,
                                      **self.input_normalization)
        elif self.input_normalization is not None:
            input = normalization(input, **self.input_normalization)
            # print('Classical normalization!', input.min(), input.max())

        if self.input_bg_subtraction is not None:
            if isinstance(self.input_bg_subtraction, str):
                if 'dark' in self.input_bg_subtraction:
                    input = black_top_hat(input)
                elif 'ite' in self.input_bg_subtraction:
                    input = white_top_hat(input)
            else:
                # this way I can pass white tophat or black tophat or any other algorithm we want
                input = self.input_bg_subtraction(input)

        # check where I do invert cause could be a part of it
        # ça marche bien surtout sur les images super bruitées où ça fait des miracles
        # try:
        #     # just for a test of using tophat --> quite ok in fact see if that would work also for 3D --> need be the first step of all
        #     from skimage.morphology import white_tophat, disk
        #     from skimage.morphology import square, ball, diamond, octahedron, rectangle
        #     # print(input.shape)
        #     # if input.has_c():
        #     input = input/input.max()
        #     if len(input.shape) == 3:
        #         # if input.has_c():
        #             for ch in range(input.shape[-1]):
        #                 input[..., ch] = white_tophat(input[..., ch], square(50))
        #     elif len(input.shape) == 2:
        #         input = white_tophat(input, square(50))
        # except:
        #     traceback.print_exc()

        # method = last_method

        # force invert/negative to be every other image if added to augmentation
        if method is not None:
            # hack to support chained/piped augmentations
            if isinstance(method, list):
                input =execute_chained_augmentations(method, orig=input, is_mask=False,
                                              **self.augmentation_types_and_values)
            else:
                # if not a chain assume it is an executable so run it directly
                input = method(input, False, **self.augmentation_types_and_values)

        input = np.reshape(input, (1, *input.shape))  # need add batch and need add one dim if not enough

        if self.mask_lines_and_cols_in_input_and_mask_GT_with_nans is not None:
            logger.debug(
                'final pixel masking before passing to the model ' + str(input.shape) + ' ' + str(input.dtype))
            input = self.mask_pixels_and_compute_for_those(input, False,
                                                           False if 'no' in self.mask_lines_and_cols_in_input_and_mask_GT_with_nans else True)

        if self.rotate_n_flip_independently_of_augmentation:
            logger.debug('data augmenter output before tiling ' + str(input.shape) + ' ' + str(input.dtype))
            # print('rotate n flip output')
            input = self.angular_yielder(input)
            # Img(input, dimensions='hwc').save('/home/aigouy/Bureau/trash_soon/test_input.tif')

        # if method is None:
        #     return input
        # else:
        return input

    def augment_output(self, msk, output_shape, method,

                       first_run):  # add as a parameter whether should do dilation or not and whether should change things --> then need add alos a parameter at the beginning of the class to handle that as well

        random.seed(self.random_seed)  # random seed is set for input and the same random is used for output

        logger.debug('method output ' + str(method))
        # print('method used', method)

        skip = False
        # filename = None

        if isinstance(msk, str):
            # print(msk)
            # filename = msk

            # msk = Img(msk)
            # print('self.create_epyseg_style_output', self.create_epyseg_style_output)

            if self.create_epyseg_style_output:
                # there will be a bug if file does not exist
                skip, msk = self.generate_or_load_pretrained_epyseg_style_mask(msk, output_shape, first_run)
            else:
                msk = Img(msk)

        # print('skip', skip)

        if not skip:
            # print('in')
            if self.remove_n_border_mask_pixels and self.remove_n_border_mask_pixels > 0:
                # msk.setBorder(distance_from_border_in_px=self.remove_n_border_mask_pixels)
                msk._pad_border_xy(
                    size=self.remove_n_border_mask_pixels)  # smarter than before --> I now pad symmetric the stuff

            # print('0',msk.shape)

            msk = self.increase_or_reduce_nb_of_channels(msk, output_shape, self.output_channel_of_interest,
                                                         self.output_channel_augmentation_rule,
                                                         self.output_channel_reduction_rule)

            # do a safety zone
            # copy the channel of interest and perform dialtion on each
            # ça devrait marcher
            # pas mal en fait
            # TODO

            # shall I do it here or after channel reduction ??? --> both can be good ideas --> think about it
            # added code for general preprocessing of input and or output
            if self.pre_processing_input__or_output is not None:
                if isinstance(self.pre_processing_input__or_output, list) and self.pre_processing_input__or_output:
                    if len(self.pre_processing_input__or_output) == 2:
                        pre_proc_fn = self.pre_processing_input__or_output[1]
                    else:
                        pre_proc_fn = self.pre_processing_input__or_output[0]
                else:
                    pre_proc_fn = self.pre_processing_input__or_output
                if pre_proc_fn is not None:
                    msk = pre_proc_fn(msk)

            # maybe put a smart dilation mode here to genearte a safe zone
            # NB WOULD BE SMARTER TO DO BEFORE INCREASING THE NB OF CHANNELS...
            if self.mask_dilations and not self.mask_dilations == 0:
                s = ndimage.generate_binary_structure(2, 1)
                # Apply dilation to every channel then reinject
                for c in range(output_shape[-1]):
                    dilated = msk[..., c]
                    for dilation in range(self.mask_dilations):
                        dilated = ndimage.grey_dilation(dilated, footprint=s)
                    msk[..., c] = dilated

        self.normalization_minima_and_maxima_output = None
        if self.output_normalization is not None:
            # print(self.output_normalization, type(self.output_normalization))
            if self.output_normalization['method'] == normalization_methods[7]:
                if 'range' in self.output_normalization:
                    normalization_range = self.output_normalization['range']
                else:
                    normalization_range = [2, 99.8]
                if 'individual_channels' in self.output_normalization and self.output_normalization[
                    'individual_channels'] is True:
                    self.normalization_minima_and_maxima_output = []
                    for c in range(msk.shape[-1]):
                        lowest_percentile = np.percentile(msk[..., c], normalization_range[0])
                        highest_percentile = np.percentile(msk[..., c], normalization_range[1])
                        self.normalization_minima_and_maxima_output.append([lowest_percentile, highest_percentile])
                else:
                    lowest_percentile = np.percentile(msk, normalization_range[0])
                    highest_percentile = np.percentile(msk, normalization_range[1])
                    self.normalization_minima_and_maxima_output = [lowest_percentile, highest_percentile]
        if self.normalization_minima_and_maxima_output is not None:
            logger.debug('Normalization per percentile: ' + str(self.normalization_minima_and_maxima_output))

        # pre crop images if asked
        if self.crop_parameters:
            msk = self.crop(msk, self.crop_parameters)
        # method = last_method

        # do normalization here
        if self.normalization_minima_and_maxima_output is not None:
            # --> gain of time --> à tester
            msk = normalization(msk, normalization_minima_and_maxima=self.normalization_minima_and_maxima_output,
                                    **self.input_normalization)
        elif self.output_normalization is not None:
            msk = normalization(msk, **self.input_normalization)
            # print('Classical normalization msk!', msk.min(), msk.max())

        if method is not None:
            # hack to support chained methods
            if isinstance(method, list):
                msk =execute_chained_augmentations(method, orig=msk, is_mask=True,
                                              **self.augmentation_types_and_values)
            else:
                # if not a chain assume it is an executable so run it directly
                msk = method(msk, True, **self.augmentation_types_and_values)
        msk = np.reshape(msk, (1, *msk.shape))

        if self.mask_lines_and_cols_in_input_and_mask_GT_with_nans is not None:
            logger.debug(
                'final pixel masking before passing to the model ' + str(msk.shape) + ' ' + str(msk.dtype))
            msk = self.mask_pixels_and_compute_for_those(msk, True,
                                                         False if 'no' in self.mask_lines_and_cols_in_input_and_mask_GT_with_nans else True)

        if self.rotate_n_flip_independently_of_augmentation:
            logger.debug('data augmenter output before tiling ' + ' ' + str(msk.shape) + ' ' + str(msk.dtype))
            msk = self.angular_yielder(msk)
            # Img(msk, dimensions='hwc').save('/home/aigouy/Bureau/trash_soon/test_msk.tif')
        # if method is None:
        #     return method, msk
        # else:
        #     return method, msk
        return msk

        # else:
        #
        #     #
        #     # print('tada', type(msk)) # sometimes maks
        #     # print(msk)
        #     # print(msk.shape)
        #     msk = np.reshape(msk, (1, *msk.shape))
        #     logger.debug('data augmenter output before tiling ' + ' ' + str(msk.shape) + ' ' + str(msk.dtype))
        #     if self.rotate_n_flip_independently_of_augmentation:
        #         msk = self.angular_yielder(msk)
        #         # Img(msk, dimensions='hwc').save('/home/aigouy/Bureau/trash_soon/test_msk.tif')
        #     return method, parameters, msk
        # if method is None:
        #     msk = np.reshape(msk, (1, *msk.shape))
        #     logger.debug('data augmenter output before tiling ' + ' ' + str(msk.shape) + ' ' + str(msk.dtype))
        #     if self.rotate_n_flip_independently_of_augmentation:
        #         msk = self.angular_yielder(msk)
        #         # Img(msk, dimensions='hwc').save('/home/aigouy/Bureau/trash_soon/test_msk.tif')
        #     return method, None, msk
        # else:
        #     parameters, msk = method(msk, parameters, True)
        #     #
        #     # print('tada', type(msk)) # sometimes maks
        #     # print(msk)
        #     # print(msk.shape)
        #     msk = np.reshape(msk, (1, *msk.shape))
        #     logger.debug('data augmenter output before tiling ' + ' ' + str(msk.shape) + ' ' + str(msk.dtype))
        #     if self.rotate_n_flip_independently_of_augmentation:
        #         msk = self.angular_yielder(msk)
        #         # Img(msk, dimensions='hwc').save('/home/aigouy/Bureau/trash_soon/test_msk.tif')
        #     return method, parameters, msk

    # nb could do random crops too
    # nb does crop work for images with channels ??? not so sure --> need check
    # this method id duplicated in data augmentations but maybe keep it like that !!!
    def crop(self, img, coords_dict):
        startx = coords_dict['x1']
        starty = coords_dict['y1']

        # we now allow random crop (when startx and starty are None, i.e. not defined...)
        if startx is None:
            try:
                width = coords_dict['w']
            except:
                width = coords_dict['width']
            min = img.shape[-2] - width
            if min < 0:
                min = 0
            startx = random.randrange(start=0, stop=min, step=1)
        if starty is None:
            try:
                height = coords_dict['h']
            except:
                height = coords_dict['height']
            min = img.shape[-3] - height
            if min < 0:
                min = 0
            starty = random.randrange(start=0, stop=min, step=1)

        if 'w' in coords_dict or 'width' in coords_dict:
            if 'w' in coords_dict:
                endx = startx + coords_dict['w']
            else:
                endx = startx + coords_dict['width']
        else:
            endx = coords_dict['x2']
        if 'h' in coords_dict or 'height' in coords_dict:
            if 'h' in coords_dict:
                endy = starty + coords_dict['h']
            else:
                endy = starty + coords_dict['height']
        else:
            endy = coords_dict['y2']

        if starty > endy:
            tmp = starty
            starty = endy
            endy = tmp
        if startx > endx:
            tmp = startx
            startx = endx
            endx = tmp

        # hack to support negative coords
        if endx < 0:
            endx = orig.shape[-2] + endx
        if endy < 0:
            endy = orig.shape[-3] + endy

        # print('coords', startx, endx, starty, endy) # seems ok now
        # TODO maybe implement that if crop is outside then add 0 or min to the extra region --> very important for the random crops --> TODO
        if len(img.shape) == 3:
            return img[starty:endy, startx:endx]
        else:
            return img[::, starty:endy, startx:endx]

    # only allow to load if not first pass
    def generate_or_load_pretrained_epyseg_style_mask(self, filename, output_shape, first_pass):

        if output_shape[-1] != 7:
            logger.error('Current model is incompatible with EPySeg, it will generate ' + str(output_shape[
                                                                                                  -1]) + ' outputs instead of the 7 required. Please uncheck "Produce EPySeg-style output" in output pre-processing')  # only allow it to be ticked if model is compatible
            return

        filepath = filename  # os.path.splitext(filename)[0]
        try:
            if not first_pass:
                # print('loading stored _epyseg.npy file to speed up training')
                # print(os.path.join(filepath, 'epyseg.npy'))
                msk = Img(filepath + '_epyseg.npy')
                if msk.shape[-1] != output_shape[-1]:  # check that it is correct too
                    msk = Img(filename)
                    print('dimension mismatch, assuming model architecture changed --> recreating mask')
                else:
                    # print('successfully loaded! --> speeding up')
                    return True, msk
            else:
                raise Exception("Image not found --> continuing")
        except:
            # traceback.print_exc()
            print('npy file does not exist or first pass')
            if os.path.isfile(filename):
                msk = Img(filename)
            else:
                # wrong file --> no need to continue...
                logger.error('invalid file name: ' + str(filename))
                return

        # print(msk.shape)
        # print(msk.has_c())

        channel_to_get = self.output_channel_of_interest
        if channel_to_get is None:
            channel_to_get = 0

        if msk.has_c():
            # if msk.shape[-1] != 1:
            # print(channel_to_get)
            # reduce number of channels --> take COI
            tmp = np.zeros((*msk.shape[:-1], 7), dtype=msk.dtype)
            msk = msk[..., channel_to_get]
            # print('here', msk.shape)
            tmp[..., 0] = msk
        else:
            # tmp = np.zeros((*msk.shape, 7), dtype=msk.dtype)
            # tmp[..., 0] = msk
            tmp = np.zeros((*msk.shape, 7), dtype=msk.dtype)
            tmp[..., 0] = msk
        msk = tmp

        # dilate the first 3 channels
        s = ndimage.generate_binary_structure(2, 1)
        for c in range(1, 3):
            # print('c', c)
            dilated = msk[..., c - 1]
            dilated = ndimage.grey_dilation(dilated, footprint=s)
            msk[..., c] = dilated

        # so far so good...
        #
        # plt.imshow(msk[..., 2])
        # plt.show()
        #
        # plt.imshow(msk[..., 3])
        # plt.show()

        # seeds = msk[..., 3]

        msk[..., 3][msk[..., 1] == 0] = 255
        msk[..., 3][msk[..., 1] == 255] = 0

        # msk[..., 3] = seeds

        # seeds = np.zeros_like(msk[..., 2])
        msk[..., 4][msk[..., 2] == 0] = 255
        msk[..., 4][msk[..., 2] == 255] = 0

        # msk[..., 4] = seeds

        # now we get the watershed seeds
        cells = msk[..., 0].copy()
        # from epyseg.img import invert
        cells = label(epyseg.img.invert(cells), connectivity=1, background=0)
        # if several in one cell --> just keep the biggest
        tmp, wshed_seeds = get_seeds(cells)
        # plt.imshow(tmp)
        # plt.show()
        msk[..., 5] = wshed_seeds
        # plt.imshow(cells)
        # plt.show()

        # inverted_seeds = np.zeros_like(wshed_seeds)
        # invert the seeds
        msk[..., 6][msk[..., 5] == 0] = 255
        msk[..., 6][msk[..., 5] == 255] = 0
        # msk[..., 6] = inverted_seeds

        # plt.imshow(msk[..., 6])
        # plt.show()

        # print(msk.shape)

        # filepath = os.path.dirname(filepath)
        try:
            Img(msk, dimensions='hwc').save(filepath + '_epyseg.npy')
            print('saving npy file to speed up further training', filepath + '_epyseg.npy')
        except:
            traceback.print_exc()
            print('could not save npy file --> skipping')

        return True, msk

    def check_integrity(self):
        if self.train_inputs is None:
            logger.error('Sorry, no specified input so DataGenerator has nothing to do')
            return False
        if self.train_outputs is None:
            logger.error('Sorry, no ground truth image specified so DataGenerator cannot be trained')
            return False
        if len(self.train_inputs) != len(self.train_outputs):
            logger.error('Sorry, some images have their corresponding ground truth image missing')
            return False
        return True


if __name__ == '__main__':

    # print(range(7))
    # for i in range(100):
    #     print(random.choice(range(7)))

    # ALL_AUGMENTATIONS = [{'type': None}, {'type': None}, {'type': 'zoom'}, {'type': 'blur'}, {'type': 'translate'},
    #                      {'type': 'shear'}, {'type': 'flip'}, {'type': 'rotate'}, {'type': 'invert'}]

    # TODO for blur decide to allow 3D blur or not ??? I blocked it for now but is it really smart
    # supported {'type': 'salt_n_pepper_noise'} {'type':'gaussian_noise'}{'type': 'zoom'}{'type': 'blur'}
    # {'type': 'translate'}{'type': 'flip'}, {'type': 'rotate'} {'type': 'invert'} {'type': 'shear'}
    # not finalize all noises {'type': 'poisson_noise'}
    # SELECTED_AUG = [{'type': 'random_intensity_gamma_contrast'}]  # [{'type': 'None'}]#[{'type': 'stretch'}] #[{'type': 'rotate'}] #[{'type': 'zoom'}]#[{'type': 'shear'}] #"[{'type': 'rotate'}] #[{'type': 'low noise'}] # en effet c'est destructeur... voir comment le restaurer avec un fesh wshed sur l'image originelle ou un wshed sur
    # SELECTED_AUG = [{'type': 'rotate (interpolation free)'}]
    # SELECTED_AUG = [{'type': 'intensity'}, {'type': 'random_intensity_gamma_contrast'}]
    # SELECTED_AUG = [{'type': 'roll along Z (2D + GT ignored)'}, {'type': 'shuffle images along Z (2D + GT ignored)'}]
    # SELECTED_AUG = [{'type': 'mask_pixels'}] # not existing anymore will rather be an option
    # SELECTED_AUG = [{'type': 'graded_intensity_modification'}]
    # chained_augmentation =  [pad_border_xy, rotate_interpolation_free, flip]
    # chained_augmentation2 = [None]
    # chained_augmentation3 = ['invert']
    # chained_augmentation4 = ['rotate (interpolation free)']
    # chained_augmentation5 = ['flip']
    # chained_augmentation6 = [pad_border_xy, 'elastic']
    # SELECTED_AUG = [{'type': chained_augmentation}, {'type':chained_augmentation2}, {'type':chained_augmentation3}, {'type':chained_augmentation4}, {'type':chained_augmentation5}, {'type':chained_augmentation6}]
    # SELECTED_AUG = [{'type': 'blur'}]
    SELECTED_AUG = [{'type': 'elastic'}] #strong_elastic strong elastic is  too big for cells but perfect for wings
    # SELECTED_AUG = [{'type': 'graded_intensity_modification'}] #strong_elastic strong elastic is  too big for cells but perfect for wings

    crop_with_parameters = partial(crop_augmentation, coords_dict={'x1': 10, 'x2': -10, 'y1': 10,
                                                                   'y2': -10})  # does that work with negative values by the way
    chained_augmentation = [crop_with_parameters, strong_elastic]
    chained_augmentation = [blur, invert]
    chained_augmentation = [blur, invert, rotate]

    # tt marche et ça marche vraiment super -−> I coul maybe push it soon

    # normal aug
    SELECTED_AUG = [{'type': 'elastic'}]
    # this is the test of a piped augmentation and as you can see that works
    SELECTED_AUG = [{'type': chained_augmentation}]



    normalization_meth = {'method': normalization_methods[7], 'range': [2, 99.8],'individual_channels': True, 'clip': False}
    normalization_meth = {'method': 'Rescaling (min-max normalization)', 'range': [0, 1],'individual_channels': True}



    mask_lines_and_cols_in_input_and_mask_GT_with_nans = None

    # TODO --> do a version that would take directly a numpy array as an input and then loop over the first dimension -−> should be easy

    if False:
        # 3D
        # seems to work --> but check
        augmenter = DataGenerator(
            # 'D:/dataset1/tests_focus_projection', 'D:/dataset1/tests_focus_projection',
            # 'D:/dataset1/tests_focus_projection/proj', 'D:/dataset1/tests_focus_projection/proj/*/hand*.tif',
            # inputs='/E/Sample_images/sample_images_denoise_manue/210219/raw', # 3D input
            # outputs='/E/Sample_images/sample_images_denoise_manue/210219/raw/predict', # 2D ouput otherwise change model shape... below !!!!
            # input_channel_of_interest=0,
            # comment the 2 next lines and restore the 3 above to get back to 3D
            inputs='/E/Sample_images/sample_images_PA/mini/*.png',  # 3D input
            input_channel_of_interest=1,
            # outputs='/E/Sample_images/sample_images_denoise_manue/210219/raw/predict',
            # 2D ouput otherwise change model shape... below !!!!
            # '/D/datasets_deep_learning/keras_segmentation_dataset/TA_test_set/tests_focus_projection/proj', '/D/datasets_deep_learning/keras_segmentation_dataset/TA_test_set/tests_focus_projection/proj/',
            # '/D/datasets_deep_learning/keras_segmentation_dataset/TA_test_set/tests_focus_projection', '/D/datasets_deep_learning/keras_segmentation_dataset/TA_test_set/tests_focus_projection',
            # '/home/aigouy/Bureau/last_model_not_sure_that_works/tmp', '/home/aigouy/Bureau/last_model_not_sure_that_works/tmp',
            # is_predict_generator=True,
            # crop_parameters={'x1':256, 'y1':256, 'x2':512, 'y2':512},
            # crop_parameters={'x1':512, 'y1':512, 'x2':796, 'y2':796},
            input_normalization=normalization_meth,
            # shuffle=False, input_shape=[(None, None, None, None, 1)], output_shape=[(None, None, None, None, 1)],
            # shuffle=False, input_shape=[(None, None, None, None, 1)], output_shape=[(None, None, None, 1)], # 3D input and 2D output --> if not the case there will be errors !!!
            shuffle=False, input_shape=[(None, None, None, 1)], output_shape=[(None, None, None, 1)], # 3D input and 2D output --> if not the case there will be errors !!!
            augmentations=SELECTED_AUG,
            output_channel_of_interest=0,
            # mask_dilations=7,
            # default_input_tile_width=2048, default_input_tile_height=1128,
            # default_output_tile_width=2048, default_output_tile_height=1128,
            # default_input_tile_width=512, default_input_tile_height=512,
            # default_output_tile_width=512, default_output_tile_height=512,
            default_input_tile_width=512, default_input_tile_height=512,
            default_output_tile_width=512, default_output_tile_height=512,
            # default_input_tile_width=256, default_input_tile_height=256,
            # default_output_tile_width=256, default_output_tile_height=256,
            # is_output_1px_wide=True,
            # rebinarize_augmented_output=True
            create_epyseg_style_output=False,
            rotate_n_flip_independently_of_augmentation=True,
            mask_lines_and_cols_in_input_and_mask_GT_with_nans=mask_lines_and_cols_in_input_and_mask_GT_with_nans,
            # force rotation and flip of images independently of everything
        )
    else:
        # KEEP THIS IS CODE TO RUN THE DATAGEN WITH IMAGES AS INPUT INSTEAD OF TEXT --> PLEASE ALWAYS MAKE SURE THEY HAVE THE RIGHT NB OF DIMS BECAUSE DIMS CAN'T BE ADDED TO THEM BY DEFAULT!!!
        images = [img for img in loadlist('/E/Sample_images/sample_images_PA/mini/*.png')]
        # we make a stack of these images and since the input single images have three channels then the last will have three channels
        images = to_stack(images)

        masks = [smart_name_parser(img,'TA')+'/handCorrection.tif' for img in loadlist('/E/Sample_images/sample_images_PA/mini/*.png')]
        masks = to_stack(masks)
        # here the stack misses channels because orig masks also miss channel --> so channel need be added

        print('starting shape',images.shape)
        print('starting shape masks',masks.shape)

        # # add a channel dim to masks because it uis missing and therefore I need it
        masks = masks[..., np.newaxis]

        # mask = [mask[..., np.newaxis] for mask in masks]  # really need add a channel dim to the image


        # test an augmenter already taking images as input --> probably need some small modifs
        # 3D
        # seems to work --> but check
        augmenter = DataGenerator(
            # 'D:/dataset1/tests_focus_projection', 'D:/dataset1/tests_focus_projection',
            # 'D:/dataset1/tests_focus_projection/proj', 'D:/dataset1/tests_focus_projection/proj/*/hand*.tif',
            # inputs='/E/Sample_images/sample_images_denoise_manue/210219/raw', # 3D input
            # outputs='/E/Sample_images/sample_images_denoise_manue/210219/raw/predict', # 2D ouput otherwise change model shape... below !!!!
            # input_channel_of_interest=0,
            # comment the 2 next lines and restore the 3 above to get back to 3D
            inputs=images,  # 3D input
            outputs=masks, # images,
            input_channel_of_interest=1,
            output_channel_of_interest=0, #1,
            # outputs='/E/Sample_images/sample_images_denoise_manue/210219/raw/predict',
            # 2D ouput otherwise change model shape... below !!!!
            # '/D/datasets_deep_learning/keras_segmentation_dataset/TA_test_set/tests_focus_projection/proj', '/D/datasets_deep_learning/keras_segmentation_dataset/TA_test_set/tests_focus_projection/proj/',
            # '/D/datasets_deep_learning/keras_segmentation_dataset/TA_test_set/tests_focus_projection', '/D/datasets_deep_learning/keras_segmentation_dataset/TA_test_set/tests_focus_projection',
            # '/home/aigouy/Bureau/last_model_not_sure_that_works/tmp', '/home/aigouy/Bureau/last_model_not_sure_that_works/tmp',
            # is_predict_generator=True,
            # crop_parameters={'x1':256, 'y1':256, 'x2':512, 'y2':512},
            # crop_parameters={'x1':512, 'y1':512, 'x2':796, 'y2':796},
            input_normalization=normalization_meth,
            # shuffle=False, input_shape=[(None, None, None, None, 1)], output_shape=[(None, None, None, None, 1)],
            # shuffle=False, input_shape=[(None, None, None, None, 1)], output_shape=[(None, None, None, 1)], # 3D input and 2D output --> if not the case there will be errors !!!
            shuffle=False, input_shape=[(None, None, None, 1)], output_shape=[(None, None, None, 1)],
            # 3D input and 2D output --> if not the case there will be errors !!!
            augmentations=SELECTED_AUG,
            # mask_dilations=7,
            # default_input_tile_width=2048, default_input_tile_height=1128,
            # default_output_tile_width=2048, default_output_tile_height=1128,
            # default_input_tile_width=512, default_input_tile_height=512,
            # default_output_tile_width=512, default_output_tile_height=512,
            default_input_tile_width=512, default_input_tile_height=512,
            default_output_tile_width=512, default_output_tile_height=512,
            # default_input_tile_width=256, default_input_tile_height=256,
            # default_output_tile_width=256, default_output_tile_height=256,
            # is_output_1px_wide=True,
            # rebinarize_augmented_output=True
            create_epyseg_style_output=False,
            rotate_n_flip_independently_of_augmentation=True,
            mask_lines_and_cols_in_input_and_mask_GT_with_nans=mask_lines_and_cols_in_input_and_mask_GT_with_nans,
            # force rotation and flip of images independently of everything
        )

        if False:
            # 3D
            # seems to work --> but check
            augmenter = DataGenerator(
                # 'D:/dataset1/tests_focus_projection', 'D:/dataset1/tests_focus_projection',
                # 'D:/dataset1/tests_focus_projection/proj', 'D:/dataset1/tests_focus_projection/proj/*/hand*.tif',
                # inputs='/E/Sample_images/sample_images_denoise_manue/210219/raw', # 3D input
                # outputs='/E/Sample_images/sample_images_denoise_manue/210219/raw/predict', # 2D ouput otherwise change model shape... below !!!!
                # input_channel_of_interest=0,
                # comment the 2 next lines and restore the 3 above to get back to 3D
                inputs='/E/Sample_images/sample_images_PA/mini/*.png',  # 3D input
                input_channel_of_interest=1,
                # outputs='/E/Sample_images/sample_images_denoise_manue/210219/raw/predict',
                # 2D ouput otherwise change model shape... below !!!!
                # '/D/datasets_deep_learning/keras_segmentation_dataset/TA_test_set/tests_focus_projection/proj', '/D/datasets_deep_learning/keras_segmentation_dataset/TA_test_set/tests_focus_projection/proj/',
                # '/D/datasets_deep_learning/keras_segmentation_dataset/TA_test_set/tests_focus_projection', '/D/datasets_deep_learning/keras_segmentation_dataset/TA_test_set/tests_focus_projection',
                # '/home/aigouy/Bureau/last_model_not_sure_that_works/tmp', '/home/aigouy/Bureau/last_model_not_sure_that_works/tmp',
                # is_predict_generator=True,
                # crop_parameters={'x1':256, 'y1':256, 'x2':512, 'y2':512},
                # crop_parameters={'x1':512, 'y1':512, 'x2':796, 'y2':796},
                input_normalization=normalization_meth,
                # shuffle=False, input_shape=[(None, None, None, None, 1)], output_shape=[(None, None, None, None, 1)],
                # shuffle=False, input_shape=[(None, None, None, None, 1)], output_shape=[(None, None, None, 1)], # 3D input and 2D output --> if not the case there will be errors !!!
                shuffle=False, input_shape=[(None, None, None, 1)], output_shape=[(None, None, None, 1)],
                # 3D input and 2D output --> if not the case there will be errors !!!
                augmentations=SELECTED_AUG,
                output_channel_of_interest=0,
                # mask_dilations=7,
                # default_input_tile_width=2048, default_input_tile_height=1128,
                # default_output_tile_width=2048, default_output_tile_height=1128,
                # default_input_tile_width=512, default_input_tile_height=512,
                # default_output_tile_width=512, default_output_tile_height=512,
                default_input_tile_width=512, default_input_tile_height=512,
                default_output_tile_width=512, default_output_tile_height=512,
                # default_input_tile_width=256, default_input_tile_height=256,
                # default_output_tile_width=256, default_output_tile_height=256,
                # is_output_1px_wide=True,
                # rebinarize_augmented_output=True
                create_epyseg_style_output=False,
                rotate_n_flip_independently_of_augmentation=True,
                mask_lines_and_cols_in_input_and_mask_GT_with_nans=mask_lines_and_cols_in_input_and_mask_GT_with_nans,
                # force rotation and flip of images independently of everything
            )

    # augmenter = DataGenerator(
    #     # 'D:/dataset1/tests_focus_projection', 'D:/dataset1/tests_focus_projection',
    #     # 'D:/dataset1/tests_focus_projection/proj', 'D:/dataset1/tests_focus_projection/proj/*/hand*.tif',
    #     inputs='/D/datasets_deep_learning/keras_segmentation_dataset/TA_test_set/tests_focus_projection/proj',
    #     outputs='/D/datasets_deep_learning/keras_segmentation_dataset/TA_test_set/tests_focus_projection/proj/*/hand*.tif',
    #     # '/D/datasets_deep_learning/keras_segmentation_dataset/TA_test_set/tests_focus_projection/proj', '/D/datasets_deep_learning/keras_segmentation_dataset/TA_test_set/tests_focus_projection/proj/',
    #     # '/D/datasets_deep_learning/keras_segmentation_dataset/TA_test_set/tests_focus_projection', '/D/datasets_deep_learning/keras_segmentation_dataset/TA_test_set/tests_focus_projection',
    #     # '/home/aigouy/Bureau/last_model_not_sure_that_works/tmp', '/home/aigouy/Bureau/last_model_not_sure_that_works/tmp',
    #     # is_predict_generator=True,
    #     # crop_parameters={'x1':256, 'y1':256, 'x2':512, 'y2':512},
    #     # crop_parameters={'x1':512, 'y1':512, 'x2':796, 'y2':796},
    #     input_normalization=normalization,
    #     # shuffle=False, input_shape=[(None, None, None, None, 1)], output_shape=[(None, None, None, None, 1)],
    #     shuffle=False, input_shape=[(None, None, None, 1)], output_shape=[(None, None, None, 1)],
    #     augmentations=SELECTED_AUG,
    #     input_channel_of_interest=0,
    #     output_channel_of_interest=0,
    #     # mask_dilations=7,
    #     # default_input_tile_width=2048, default_input_tile_height=1128,
    #     # default_output_tile_width=2048, default_output_tile_height=1128,
    #     # default_input_tile_width=512, default_input_tile_height=512,
    #     # default_output_tile_width=512, default_output_tile_height=512,
    #     default_input_tile_width=512, default_input_tile_height=256,
    #     default_output_tile_width=512, default_output_tile_height=256,
    #     # default_input_tile_width=256, default_input_tile_height=256,
    #     # default_output_tile_width=256, default_output_tile_height=256,
    #     # is_output_1px_wide=True,
    #     # rebinarize_augmented_output=True
    #     create_epyseg_style_output=False,
    #     rotate_n_flip_independently_of_augmentation=True
    #     # force rotation and flip of images independently of everything
    #
    # )

    if False:
        # seems fine but I really need the first pass or not now
        # set a variable to false if not first pass --> TODO

        # just for a test
        _, test = augmenter.generate_or_load_pretrained_epyseg_style_mask(
            # '/E/Sample_images/sample_images_epiguy_pyta/images_with_different_bits/predict/100708_png06.tif',
            # '/E/Sample_images/sample_images_epiguy_pyta/images_with_different_bits/predict/single_8bits.tif',
            '/E/Sample_images/sample_images_epiguy_pyta/images_with_different_bits/predict/100708_png06_rgb.tif',
            (None, None, 7), False)
        Img(test, dimensions='hwc').save(
            '/E/Sample_images/sample_images_epiguy_pyta/images_with_different_bits/predict/test_7_masks.tif')
        import sys

        sys.exit(0)

    # check all augs are ok and check degradative ones

    # import matplotlib.pyplot as plt

    # TODO try run wshed on mask and or on orig --> TODO
    # would also be more consitent to model and better comparison with other available algos

    # print(augmenter)
    # maybe need store a white mask to reapply it to prevent segmentation of dark area ?
    pause = 2

    plt.ion()
    # ZOOM
    # plt.margins(2, 2)
    # call data augmenter from the other

    # from deprecated_demos.ta.wshed import Wshed

    # mask = Wshed.run_fast(self.img, first_blur=values[0], second_blur=values[1]) # or check same width and height

    # why is this shit called twice

    LIMIT = 22
    full_count = 0
    counter = 0
    for orig, mask in augmenter.train_generator(False, True):
        print('inside loop')
        # print('out', len(orig), len(mask))
        # print(orig[0].shape, mask[0].shape)
        if False:
            # the generator ignore exit and runs one more time
            print('in')
            # just save two images for a test

            # why is that called another time ????
            print(type(orig[0]))
            print(orig[0].shape)
            # Img(orig[0], dimensions='dhwc').save('/E/Sample_images/trash_tests/orig.tif')
            # Img(mask[0], dimensions='hwc').save('/E/Sample_images/trash_tests/mask.tif')

            Img(orig[0]).save('/E/Sample_images/trash_tests/orig.tif')
            Img(mask[0]).save('/E/Sample_images/trash_tests/mask.tif')

            # somehow the generator gets called another time no big deal but I must have done smthg wrong somewhere
            print('quitting')
            import sys

            sys.exit(0)
            print('hello')


        full_count += 1
        for i in range(len(orig)):
            # print('in here', orig[i].shape)
            if counter < LIMIT:
                try:
                    center = int(len(orig[i]) / 2)
                    # center = 0

                    plt.imshow(np.squeeze(orig[i][center]))  # , cmap='gray')
                    print(orig[i].shape, orig[i].max(), orig[i].min())
                    print(mask[
                              i].dtype)  # float 32 --> marche aussi avec le wshed --> cool in fact # peut etre essayer comme ça
                    # first_blur=None, second_blur=None,
                    # orig[i][center]
                    # mask2 = Wshed.run(np.squeeze(mask[i][center]),  channel=None, seeds=np.squeeze(mask[i][center]), min_size=3,  is_white_bg=False)
                    # hack to maintain mask size constant
                    # handCorrection = Wshed.run(np.squeeze(mask[i]), seeds='mask', min_size=30) # ça marche bien mais perd les bords des images ... # --> pas si simple en fait mais maintient le truc à 1 pixel mais perd les bords --> comment va t'il se comporter si je met ça ??? faudrait idealement recropper le truc ou alors blachken image around the cells ??? maybe doable --> faut faire un flood autour de l'image et le reappliquer à l'image parente
                    # ça marche aussi sur des stacks --> cool
                    # donc je peux vraiment essayer
                    # puis-je donc tester le truc ???? oui peut etre en reel et voir ce que ça donne
                    # peut etre que ça marchera quand meme ???? pas sûr
                    # need fill outer layer and blacken it outside mask set it to min in original image
                    # TODO test it ????

                    # print(handCorrection.shape)
                    # plt.imshow((handCorrection))

                    # ZOOM
                    # plt.xlim(500, 1000)
                    # plt.ylim(500, 1000)
                    plt.pause(pause)
                except:
                    # for img in np.squeeze(orig[i][0]):
                    #     plt.imshow(np.squeeze(img), cmap='gray')
                    #     plt.pause(0.3)
                    center = int(len(np.squeeze(orig[i][0])) / 2)
                    plt.imshow(np.squeeze(orig[i][0][center]))  # , cmap='gray')
                    # ZOOM
                    # plt.xlim(500, 1000)
                    # plt.ylim(500, 1000)
                    plt.pause(pause)

                # plt.show()
        for i in range(len(mask)):
            print('in here2', mask[i].shape, mask[i].max(), mask[i].min())  # --> seems ok
            # print(mask[i]) # marche car contient des nans
            if counter < LIMIT:
                try:
                    center = int(len(mask[i]) / 2)
                    # center = 0
                    plt.imshow(np.squeeze(mask[i][center]))
                    plt.pause(0.3)

                except:
                    # for img in np.squeeze(mask[i][0]):
                    #     plt.imshow(np.squeeze(img))
                    #     plt.pause(0.3)
                    center = int(len(np.squeeze(mask[i][0])) / 2)
                    # print(mask[i][0][...,6].shape)
                    plt.imshow(np.squeeze(mask[i][0][center]))
                    # plt.imshow(mask[i][0][...,0])
                    # print(mask[0][center].shape)
                    plt.pause(0.3)
                    # print('toto')
                    Img(mask[i], dimensions='dhwc').save('/home/aigouy/Bureau/trashme.tif')
                # plt.show()
        counter += 1
        if counter >= LIMIT:
            plt.close(fig='all')
        # do stuff with that

    print('end')
    print(counter)
    print(full_count)
