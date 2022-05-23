# TODO rewrite code and make use of random seeding in order to have the same things done for augmentation of raw_input and GT --> but check code beforehand --> that works and it does not require any other code change --> much better

from builtins import enumerate
from natsort import natsorted
import traceback
from numpy import dtype
from skimage import filters
from skimage.util import random_noise
from epyseg.binarytools.cell_center_detector import get_seeds
from skimage.measure import label
from epyseg.img import Img, white_top_hat, black_top_hat, mask_rows_or_columns, elastic_deform
import random
from scipy import ndimage
from skimage import transform
import numpy as np
import glob
import os
from pathlib import Path
from skimage import exposure
from datetime import datetime
import matplotlib.pyplot as plt
from epyseg.utils.loadlist import loadlist
from epyseg.tools.logger import TA_logger # logging

# TODO add augmentation randomly swap channels --> can learn by itself which channel is containing epithelia

logger = TA_logger()


# could add non damaging rotations to augment --> would for sure be better...
# TODO try numpy.rot90(m, k=1, axes=(0, 1))

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
                                     'elastic':None}

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
                                     'elastic': None
                                     }

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
                 invert_image=False, input_bg_subtraction=None, create_epyseg_style_output=None,
                 remove_n_border_mask_pixels=None, is_output_1px_wide=False,
                 rebinarize_augmented_output=False, rotate_n_flip_independently_of_augmentation=False,
                 mask_lines_and_cols_in_input_and_mask_GT_with_nans=None,
                 # if none does nothing if noid --> cannot learn from masked pixels at all reducedid --> can learn from masked pixels --> requires specific losses to work --> find an easy way to do that ...
                 z_frames_to_add=None,
                 **kwargs):

        logger.debug('clip by freq' + str(clip_by_frequency))

        self.random_seed = datetime.now()
        self.augmentation_types_and_methods = {'None': None, 'shear': self.shear, 'zoom': self.zoom,
                                               'rotate': self.rotate,
                                               'rotate (interpolation free)': self.rotate_interpolation_free,
                                               'flip': self.flip, 'blur': self.blur,
                                               'intensity': self.change_image_intensity_and_shift_range,
                                               'translate': self.translate, 'invert': self.invert,
                                               'roll along Z (2D + GT ignored)': self.rollZ,
                                               'shuffle images along Z (2D + GT ignored)': self.shuffleZ,
                                               'low noise': self.low_noise, 'high noise': self.high_noise,
                                               'stretch': self.stretch,
                                               'random_intensity_gamma_contrast': self.random_intensity_gamma_contrast_changer,
                                               'elastic': self.elastic}# added elastic deformation because can be very useful

        self.EXTRAPOLATION_MASKS = 1  # bicubic 0 # nearest # TODO keep like that because 3 makes really weird results for binary images
        self.is_predict_generator = is_predict_generator

        # convert single input to list
        if isinstance(inputs, str):
            inputs = [inputs]
        if isinstance(outputs, str):
            outputs = [outputs]

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
                    # get path without ext
                    mask_folder = os.path.splitext(file)[0]
                    file = Path(mask_folder + '/handCorrection.tif')  # TODO do path join instead...
                    if file.exists():
                        mask = mask_folder + '/handCorrection.tif'
                    else:
                        mask = mask_folder + '/handCorrection.png'
                    lst.append(mask)
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

        # print('self.train_inputs',len(self.train_inputs[0])) --> not same size --> will crash --> use lists maybe
        # print('self.train_outputs', len(self.train_outputs[0]))

    def has_train_set(self):
        return self.train_inputs and len(self.train_inputs) != 0

    def has_validation_set(self):
        return self.validation_inputs and len(self.validation_inputs) != 0

    def has_test_set(self):
        return self.test_inputs and len(self.test_inputs) != 0

    # TODO do that more wisely
    def is_train_set_size_coherent(self):
        return len(self.train_inputs) == len(self.train_outputs)

    # TODO do that more wisely
    def is_test_set_size_coherent(self):
        return len(self.test_inputs) == len(self.test_outputs)

    # TODO do that more wisely
    def is_validation_set_size_coherent(self):
        return len(self.validation_inputs) == len(self.validation_outputs)

    def get_validation_set_length(self):
        if not self.validation_inputs:
            return 0
        return len(self.validation_inputs[0])

    def get_test_set_length(self):
        if not self.test_inputs:
            return 0
        return len(self.test_inputs[0])

    def get_train_set_length(self):
        if not self.train_inputs:
            return 0
        return len(self.train_inputs[0])

    def train_generator(self, skip_augment, first_run, __DEBUG__=False):
        if self.shuffle:
            indices = random.sample(range(len(self.train_inputs[0])), len(self.train_inputs[0]))
        else:
            indices = list(range(len(self.train_inputs[0])))

        # print('indices',indices)
        # print('vs', random.sample(range(len(self.train_inputs[0])), len(self.train_inputs[0])))

        for idx in indices:
            try:
                # print(self.train_inputs[0], self.train_inputs[1]) # seems ok --> the two lists are the same so where is the bug ???
                # can put whsed in there too

                if __DEBUG__:
                    print('self.train_inputs',self.train_inputs)
                    print('self.train_outputs',self.train_outputs)

                orig, mask = self.generate(self.train_inputs, self.train_outputs, idx,
                                           skip_augment, first_run)  # need augment in there

                #  that works check that all are there and all are possible otherwise skip
                # --> need ensure that width = height
                # need set a parameter to be sure to use it or not and need remove rotation and flip from augmentation list (or not in fact)
                # augmentations = 7
                # if orig[0].shape[-2] != orig[0].shape[-3]:
                #     augmentations = 3
                # for aug in range(augmentations):
                #     yield self.angular_yielder(orig, mask, aug)
                yield orig, mask
            except GeneratorExit:
                # except GeneratorExit
                # print("Exception!")
                # except:
                break
            except Exception:
                # erroneous/corrupt image detected --> continuing
                traceback.print_exc()
                continue

    def angular_yielder(self, orig, count=None):
        # mask = self.extra_watershed_mask(mask) # shrink mask to 1 px wide irrespective of transfo
        # NB could do here the generations of the nine stacks --> TODO --> would increase size by 9 but it is a good idea I think
        # can also copy the code of the other stuff

        # print('in angular yielder', orig.shape)

        # can create bugs if not size contant --> force constant size --> same is true for the interpolationless rotation I added recently --> fix that

        random.seed(self.random_seed)

        if count is None:
            augmentations = 8
            # quick n dirty bug fix to keep batch size fixed!!!
            if orig[0].shape[-2] != orig[0].shape[-3]:
                augmentations = 4
            count = random.choice(range(augmentations))

        if count == 0:
            return orig

        if count == 1:
            # rot 180
            return np.rot90(orig, 2, axes=(-3, -2))

        if count == 2:
            # flip hor
            return np.flip(orig, -2)

        if count == 3:
            # flip ver
            return np.flip(orig, -3)

        # make it yield the original and the nine versions of it
        # --> TODO
        # ça marche ça me genere les 9 versions du truc dans tous les sens --> probablement ce que je veux --> tt mettre ici
        if count == 4:
            # yield np.rot90(orig, axes=(-3, -2)), np.rot90(mask, axes=(-3, -2))

            # rot 90
            return np.rot90(orig, axes=(-3, -2))

        if count == 5:
            # rot 90_flipped_hor or ver
            return np.flip(np.rot90(orig, axes=(-3, -2)), -2)

        if count == 6:
            # rot 90_flipped_hor or ver
            return np.flip(np.rot90(orig, axes=(-3, -2)), -3)

        if count == 7:
            # rot 270
            return np.rot90(orig, 3, axes=(-3, -2))

    # def angular_yielder(self, orig, mask, count):
    #     # mask = self.extra_watershed_mask(mask) # shrink mask to 1 px wide irrespective of transfo
    #     # NB could do here the generations of the nine stacks --> TODO --> would increase size by 9 but it is a good idea I think
    #     # can also copy the code of the other stuff
    #
    #     if count == 0:
    #         # rot 180
    #         return np.rot90(orig, 2, axes=(-3, -2)), np.rot90(mask, 2, axes=(-3, -2))
    #
    #     if count == 1:
    #         # flip hor
    #         return np.flip(orig, -2), np.flip(mask, -2)
    #
    #     if count == 2:
    #         # flip ver
    #         return np.flip(orig, -3), np.flip(mask, -3)
    #
    #     # make it yield the original and the nine versions of it
    #     # --> TODO
    #     # ça marche ça me genere les 9 versions du truc dans tous les sens --> probablement ce que je veux --> tt mettre ici
    #     if count == 3:
    #         # yield np.rot90(orig, axes=(-3, -2)), np.rot90(mask, axes=(-3, -2))
    #
    #         # rot 90
    #         return np.rot90(orig, axes=(-3, -2)), np.rot90(mask, axes=(-3, -2))
    #
    #     if count == 4:
    #         # rot 90_flipped_hor or ver
    #         return np.flip(np.rot90(orig, axes=(-3, -2)), -2), np.flip(np.rot90(mask, axes=(-3, -2)), -2)
    #
    #     if count == 5:
    #         # rot 90_flipped_hor or ver
    #         return np.flip(np.rot90(orig, axes=(-3, -2)), -3), np.flip(np.rot90(mask, axes=(-3, -2)), -3)
    #
    #     if count == 6:
    #         # rot 270
    #         return np.rot90(orig, 3, axes=(-3, -2)), np.rot90(mask, 3, axes=(-3, -2))

    # def extra_watershed_mask(self, mask):
    #     # TODO probably need flood the borders to remove cells at the edges --> peut il y a voir une astuce pr garder 1px wide ???? sans perte sinon pas faire de nearest mais une bicubic interpolation # peut etre avec threshold --> deuxieme piste peut etre meme mieux
    #     for idx in range(len(mask)):
    #         # print(mask[idx].shape)
    #         for idx2 in range(len(mask[idx])):
    #             # np.squeeze
    #             handcorr = Wshed.run((mask[idx][idx2]), seeds='mask', min_size=30)
    #             mask[idx][idx2] = np.reshape(handcorr, (*handcorr.shape, 1))  # regenerate after stuff
    #         # print('2', mask[idx].shape)
    #     return mask

    def test_generator(self, skip_augment, first_run):
        for idx in range(len(self.test_inputs[0])):
            try:
                #  that works check that all are there and all are possible otherwise skip
                # --> need ensure that width = height
                # need set a parameter to be sure to use it or not and need remove rotation and flip from augmentation list (or not in fact)
                # DO I Need that for test gen too ??? probably not in fact --> think a bit about it
                # augmentations = 7
                # if orig[0].shape[-2] != orig[0].shape[-3]:
                #     augmentations = 3
                # for aug in range(augmentations):
                #     yield self.angular_yielder(orig, mask, aug)
                yield self.generate(self.test_inputs, self.test_outputs, idx, skip_augment, first_run)
            except GeneratorExit:  # https://stackoverflow.com/questions/46542147/elegant-way-for-breaking-a-generator-loop-generatorexit-error
                # except GeneratorExit
                # print("Exception!")
                # except:
                break
            except Exception:
                # erroneous/corrupt image detected --> continuing
                traceback.print_exc()
                continue

    def validation_generator(self, skip_augment, first_run):
        for idx in range(len(self.validation_inputs[0])):
            try:
                #  that works check that all are there and all are possible otherwise skip
                # --> need ensure that width = height
                # need set a parameter to be sure to use it or not and need remove rotation and flip from augmentation list (or not in fact)
                # augmentations = 7
                # if orig[0].shape[-2] != orig[0].shape[-3]:
                #     augmentations = 3
                # for aug in range(augmentations):
                #     yield self.angular_yielder(orig, mask, aug)
                yield self.generate(self.validation_inputs, self.validation_outputs, idx, skip_augment, first_run)
            except GeneratorExit:  # https://stackoverflow.com/questions/46542147/elegant-way-for-breaking-a-generator-loop-generatorexit-error
                # except GeneratorExit
                # print("Exception!")
                # except:
                break
            except Exception:
                # erroneous/corrupt image detected --> continuing
                continue

    def predict_generator(self, skip_augment=True):
        for idx in range(len(self.predict_inputs[0])):
            try:
                yield self.generate(self.predict_inputs, None, idx, skip_augment)
            except GeneratorExit:  # https://stackoverflow.com/questions/46542147/elegant-way-for-breaking-a-generator-loop-generatorexit-error
                # except GeneratorExit
                # print("Exception!")
                # except:
                break
            except Exception:
                # erroneous/corrupt image detected --> continuing
                traceback.print_exc()
                continue

    def _get_from(self, input_list, idx):
        if input_list is None:
            return None
        data = []
        for lst in input_list:
            # if an input list is None then return always None for this lists
            # print(lst)
            if lst and lst is not None:
                data.append(lst[idx])
            else:
                data.append(None)
        # unpack data...
        return data

    def generate(self, inputs, outputs, cur_idx, skip_augment, first_run=False):
        # print('inp',cur_idx, len(inputs))
        # print('output',cur_idx, len(outputs)) # bug here should be len =1 not two...--> I do have a bug
        # print('inp',inputs)
        # print('output',outputs)
        inp, out = self.augment(self._get_from(inputs, cur_idx),
                                self._get_from(outputs, cur_idx), skip_augment, first_run)

        # print('inp, out',inp[0].shape, out[0].shape) # no bug until there --> I do a crappy mistake after

        # import sys
        # sys.exit(0)

        if self.rebinarize_augmented_output:
            # out[out > 0] = 1
            # super_threshold_indices = a > thresh
            # a[super_threshold_indices] = 0
            for p, o in enumerate(out):
                o[o > o.min()] = o.max()
                out[p] = o

        # moved clip bu frequency to before

        # negative should be done here I think
        if self.invert_image:
            for idx, img in enumerate(inp):
                inp[idx] = Img.invert(img)

        # toss a coin to invert image if needed
        # if self.invert_in_augs:
        #     if random.uniform(0, 1) < 0.5:
        #         # print('inverting...')
        #         for idx, img in enumerate(inp):
        #             inp[idx] = Img.invert(img)
        # else:
        #     print('not inevrting...')

        inputs = []
        outputs = []
        if self.keep_original_sizes:
            for img in inp:
                # normalization moved before...
                # inputs.append(Img.normalization(img, **self.input_normalization))
                inputs.append(img)
                if self.is_predict_generator:
                    outputs.append(None)
                    return inputs, outputs  # outputs being crop parameters
            if not self.is_predict_generator:
                for img in out:
                    # normalization moved before...
                    # outputs.append(Img.normalization(img, **self.output_normalization))
                    outputs.append(img)
            return inputs, outputs
        else:
            for idx, img in enumerate(inp):
                input_shape = self.input_shape[idx]
                # assume by default shape is 4
                dimension_h = 1
                dimension_w = 2
                if len(input_shape) == 5:
                    dimension_h = 2
                    dimension_w = 3

                # print('dimension_h, dimension_w',dimension_h, dimension_w) # --> bug is here
                width = self.default_input_tile_width
                height = self.default_input_tile_height
                if input_shape[-2] is not None:
                    width = input_shape[-2]
                if input_shape[-3] is not None:
                    height = input_shape[-3]
                # nothing specified --> just take original width and height
                if width is None:
                    width = img.shape[-2]
                if height is None:
                    height = img.shape[-3]
                # normalization moved before
                # crop_parameters, tiles2D_inp = Img.get_2D_tiles_with_overlap(
                #     Img.normalization(img, **self.input_normalization), width=width - self.overlap_x,
                #     height=height - self.overlap_y, overlap_x=self.overlap_x,
                #     overlap_y=self.overlap_y, overlap=0, dimension_h=dimension_h, dimension_w=dimension_w,
                #     force_to_size=True)
                crop_parameters, tiles2D_inp = Img.get_2D_tiles_with_overlap(
                    img, width=width - self.overlap_x,
                    height=height - self.overlap_y, overlap_x=self.overlap_x,
                    overlap_y=self.overlap_y, overlap=0, dimension_h=dimension_h, dimension_w=dimension_w,
                    force_to_size=True)
                inputs.append(tiles2D_inp)
                if self.is_predict_generator:
                    outputs.append(crop_parameters)
            if not self.is_predict_generator:
                for idx, img in enumerate(out):
                    output_shape = self.output_shape[idx]
                    # assume by default shape is 4
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
                    # nothing specified --> just take original width and height
                    if width is None:
                        width = img.shape[-2]
                    if height is None:
                        height = img.shape[-3]
                    # normalization moved before
                    # _, tiles2D_out = Img.get_2D_tiles_with_overlap(img, width=width - self.overlap_x, # there should have been normalization here
                    #                                                height=height - self.overlap_y,
                    #                                                overlap_x=self.overlap_x,
                    #                                                overlap_y=self.overlap_y, overlap=0,
                    #                                                dimension_h=dimension_h, dimension_w=dimension_w,
                    #                                                force_to_size=True)
                    _, tiles2D_out = Img.get_2D_tiles_with_overlap(img, width=width - self.overlap_x,
                                                                   height=height - self.overlap_y,
                                                                   overlap_x=self.overlap_x,
                                                                   overlap_y=self.overlap_y, overlap=0,
                                                                   dimension_h=dimension_h, dimension_w=dimension_w,
                                                                   force_to_size=True)
                    outputs.append(tiles2D_out)

            bckup_input_incr = self.input_incr
            bckup_output_incr = self.output_incr
            if self.output_folder is not None and not not self.is_predict_generator:

                for idx, tiles2D_inp in enumerate(inputs):
                    self.input_incr = bckup_input_incr
                    tiles2D_inp = Img.tiles_to_linear(tiles2D_inp)
                    for idx2, inp in enumerate(tiles2D_inp):
                        if len(np.squeeze(inp).shape) != 1:
                            # normalization moved before
                            # Img(Img.normalization(inp, **self.input_normalization), dimensions='hw').save(
                            #     self.output_folder + '/input_' + str(idx) + '_' + str(self.output_incr) + '.npz')
                            Img(inp, dimensions='hw').save(
                                self.output_folder + '/input_' + str(idx) + '_' + str(self.output_incr) + '.npz')
                            self.input_incr += 1
                        else:
                            print('error size')

                # TODO allow save lists and load lists and allow allow reload based on image patterns eg input_1_*.npz, input_2_*.npz, ....
                for idx1, tiles2D_out in enumerate(outputs):
                    self.output_incr = bckup_output_incr
                    tiles2D_out = Img.tiles_to_linear(tiles2D_out)
                    for idx2, inp in enumerate(tiles2D_out):
                        if len(np.squeeze(inp).shape) != 1:
                            # Img(np.squeeze(tiles2D_out[idx]), dimensions='hw').save(self.output_folder+'/output_'+ str(self.output_incr)+'.png')
                            # normalization moved before
                            # Img(Img.normalization(inp, **self.output_normalization), dimensions='hw').save(
                            #     self.output_folder + '/output_' + str(idx) + '_' + str(self.output_incr) + '.npz')
                            Img(inp, dimensions='hw').save(
                                self.output_folder + '/output_' + str(idx) + '_' + str(self.output_incr) + '.npz')
                            self.output_incr += 1
                        else:
                            print('error size')
                # CODER SELF REMINDER: return is needed otherwise it stops at first iteration...
                return
            else:
                final_inputs = []
                for tiles2D_inp in inputs:
                    # normalisation moved before
                    # final_inputs.append(
                    #     Img.normalization(Img.tiles_to_batch(tiles2D_inp), **self.input_normalization))
                    final_inputs.append(Img.tiles_to_batch(tiles2D_inp))

                if self.is_predict_generator:
                    return final_inputs, outputs

                final_outputs = []
                for tiles2D_out in outputs:
                    # normalisation moved before
                    # final_outputs.append(
                    #     Img.normalization(Img.tiles_to_batch(tiles2D_out), **self.output_normalization))
                    final_outputs.append(Img.tiles_to_batch(tiles2D_out))
                return final_inputs, final_outputs

    def increase_or_reduce_nb_of_channels(self, img, desired_shape, channel_of_ineterest, increase_rule=None, decrease_rule=None):

        input_has_channels = img.has_c()
        if input_has_channels:
            input_channels = img.get_dimension('c')
        else:
            input_channels = None
        if input_channels is None:
            input_channels = 1

        if not input_has_channels:
            img = np.reshape(img, (*img.shape, 1))

        if desired_shape[-1] != input_channels:
            if input_channels < desired_shape[-1]:
                # too few channels in image compared to expected input of the model --> need add channels
                multi_channel_img = np.zeros((*img.shape[:-1], desired_shape[-1]),
                                             dtype=img.dtype)
                channel_to_copy = 0
                if channel_of_ineterest is not None:
                    channel_to_copy = channel_of_ineterest

                if increase_rule and 'copy' in increase_rule:
                    if 'all' in increase_rule:
                        logger.debug('Increasing nb of channels by copying COI to all available channels')
                        # case where we copy the channel of interest to all other channels
                        # can for example be used to create an RGB image out of a single channel image
                        for c in range(desired_shape[-1]):
                            multi_channel_img[..., c] = img[..., channel_to_copy]
                        img = multi_channel_img
                    elif 'missing':
                        logger.debug('Increasing nb of channels by copying COI to extra channels only')
                        # we copy the channel of interest to missing channels, other channels are kept unchanged
                        for c in range(img.shape[-1]):
                            multi_channel_img[..., c] = img[..., c]
                        for c in range(img.shape[-1], desired_shape[-1]):
                            multi_channel_img[..., c] = img[..., channel_to_copy]
                        img = multi_channel_img
                    else:
                        logger.error('unknown channel nb increase rule ' + str(increase_rule))
                elif increase_rule and 'add' in increase_rule:
                    logger.debug('Increasing nb of channels by adding empty (black) channels')
                    # copy just the existing channel and keep the rest black
                    for c in range(img.shape[-1]):
                        multi_channel_img[..., c] = img[..., c]
                    img = multi_channel_img
            elif input_channels > desired_shape[-1]:
                reduced_channel_img = np.zeros((*img.shape[:-1], desired_shape[-1]),
                                               dtype=img.dtype)
                channel_to_copy = 0
                if channel_of_ineterest is not None:
                    channel_to_copy = channel_of_ineterest

                if decrease_rule and 'copy' in decrease_rule:
                    logger.debug('Decreasing nb of channels by copying COI to available channels')
                    for c in range(desired_shape[-1]):
                        reduced_channel_img[..., c] = img[..., channel_to_copy]
                    img = reduced_channel_img
                elif decrease_rule and 'remove' in decrease_rule:
                    logger.debug('Decreasing nb of channels by removing extra channels')
                    for c in range(desired_shape[-1]):
                        reduced_channel_img[..., c] = img[..., c]
                    img = reduced_channel_img
                else:
                    logger.error('unknown channel nb decrease rule ' + str(decrease_rule))
        else:  # input_channels == desired_shape[-1]:
            if 'force' in increase_rule or 'force' in decrease_rule:
                logger.debug('Force copy COI to all channels')
                channel_to_copy = 0
                if channel_of_ineterest is not None:
                    channel_to_copy = channel_of_ineterest
                for c in range(desired_shape[-1]):
                    img[..., c] = img[..., channel_to_copy]

        return img

    @staticmethod
    def get_list_of_images(path):
        # TODO Add more file formats or make it more customizable ???

        if path is None:
            return []

        if not path:
            return []

        # if the user passes a list of images, then there is nothing to do but simply returning it...
        if isinstance(path, list):
            return path

        folderpath = path
        # TODO use python function to do that
        if not folderpath.endswith('/') and not folderpath.endswith(
                '\\') and not '*' in folderpath and not os.path.isfile(folderpath):
            folderpath += '/'

        list_of_files = []

        if folderpath.lower().endswith('.lst') or folderpath.lower().endswith('.txt'):
            # loads list of files
            list_of_files = loadlist(folderpath)
        elif '*' in folderpath:
            # loads files with pattern using glob
            list_of_files = natsorted(glob.glob(folderpath))
        elif os.path.isdir(folderpath):
            list_of_files = glob.glob(folderpath + "*.png") + glob.glob(folderpath + "*.jpg") + glob.glob(
                folderpath + "*.jpeg") + glob.glob(
                folderpath + "*.tif") + glob.glob(folderpath + "*.tiff") + glob.glob(folderpath + "*.lsm") + glob.glob(
                folderpath + "*.czi") + glob.glob(folderpath + "*.lif")
            list_of_files = natsorted(list_of_files)
        elif os.path.isfile(folderpath):
            # single image --> convert it to a list
            list_of_files.append(folderpath)
        return list_of_files

    # TODO remove
    def get_augment_method(self):
        if self.augmentations is not None and len(self.augmentations) > 0:
            method = random.choice(self.augmentations)
            if 'value' in method:
                # change range of augmentation
                self.augmentation_types_and_values[method['type']] = method['value']
            # define augmentation method
            if method['type'] is None:
                return None
            method = self.augmentation_types_and_methods[method['type']]
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
        augmentation_parameters = None

        if not skip_augment:
            method = self.get_augment_method()
        else:
            method = None

        inputs_outputs_to_skip = []
        for idx, input in enumerate(inputs):
            if idx == 0:
                # we set the new seed for the first inpout and keep it for all others
                # set seed here so that random is the same for all inputs
                self.random_seed = datetime.now()
                random.seed(self.random_seed)
            if input is None:
                input = self._rescue_missing_items(inputs, idx)
            # en fait c'est ici qu'il faut que je permette de convert des inputs en outpus --> TODO
            if not self.treat_some_inputs_as_outputs or not self.treat_some_inputs_as_outputs[idx]:
                method, augmentation_parameters, augmented = self.augment_input(input, self.input_shape[idx],
                                                                                method, augmentation_parameters,
                                                                                skip_augment, first_run)
            else:
                method, augmentation_parameters, augmented = self.augment_output(input, self.input_shape[idx],
                                                                                 method, augmentation_parameters,
                                                                                 first_run)
            if augmented is None:
                inputs_outputs_to_skip.append(idx)
            augmented_inputs.append(augmented)
        if outputs is not None:
            for idx, output in enumerate(outputs):
                method, augmentation_parameters, augmented = self.augment_output(output, self.output_shape[idx],
                                                                                 method, augmentation_parameters,
                                                                                 first_run)

                if augmented is None:
                    inputs_outputs_to_skip.append(idx)
                augmented_outputs.append(augmented)
        else:
            augmented_outputs = None

        for el in sorted(inputs_outputs_to_skip, reverse=True):
            del augmented_inputs[el]
            if augmented_outputs is not None:
                del augmented_outputs[el]
        return augmented_inputs, augmented_outputs

    # NB if there are several inputs they need to have the same random seed --> see how I can do that need to and also same is true for
    def augment_input(self, input, input_shape, last_method, parameters, skip_augment, first_run):
        logger.debug('method input ' + str(last_method) + ' ' + str(parameters) + ' ' + str(input))

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
                return None, None, None
                # so it will fail --> need continue

        input = self.increase_or_reduce_nb_of_channels(input, input_shape, self.input_channel_of_interest,
                                                       self.input_channel_augmentation_rule,
                                                       self.input_channel_reduction_rule)

        # TODO in fact that would make more sense to clip by freq before --> indeed clipping should be done before any data augmentation
        if self.clip_by_frequency is not None:
            logger.debug('Clipping image prior to any processing')
            if isinstance(self.clip_by_frequency, float):
                # for idx, img in enumerate(inp):
                input = Img.clip_by_frequency(input, upper_cutoff=self.clip_by_frequency, channel_mode=True)
                # TODO should I add clipping here for GT too ???
            elif isinstance(self.clip_by_frequency, tuple):
                if len(self.clip_by_frequency) == 2:
                    # for idx, img in enumerate(inp):
                    # print('clip by freq here', self.clip_by_frequency[0], self.clip_by_frequency[1])
                    input = Img.clip_by_frequency(input, lower_cutoff=self.clip_by_frequency[0],
                                                  upper_cutoff=self.clip_by_frequency[1], channel_mode=True)
                # TODO should I add clipping here for GT too ???
                # if not self.is_predict_generator:
                #     # clipping output...
                #     print('clipping output...')
                #     for idx, img in enumerate(out):
                #         out[idx] = Img.clip_by_frequency(img, lower_cutoff=self.clip_by_frequency[0],
                #                                          upper_cutoff=self.clip_by_frequency[1], channel_mode=True)
                else:
                    # for idx, img in enumerate(inp):
                    input = Img.clip_by_frequency(input, upper_cutoff=self.clip_by_frequency[0], channel_mode=True)
                    # TODO should I add clipping here for GT too ???
            elif isinstance(self.clip_by_frequency, dict):
                # for idx, img in enumerate(inp):
                # print('clip by freq here', self.clip_by_frequency['lower_cutoff'], self.clip_by_frequency['upper_cutoff'])
                # print('bef', img.min(), img.max())

                input = Img.clip_by_frequency(input, lower_cutoff=self.clip_by_frequency['lower_cutoff'],
                                              upper_cutoff=self.clip_by_frequency['upper_cutoff'],
                                              channel_mode=self.clip_by_frequency['channel_mode'])
                # TODO should I add clipping here for GT too ???
                # if not self.is_predict_generator:
                #     # clipping output...
                #     print('clipping output...')
                #     for idx, img in enumerate(out):
                #         out[idx] = Img.clip_by_frequency(img, lower_cutoff=self.clip_by_frequency['lower_cutoff'],
                #                                          upper_cutoff=self.clip_by_frequency['upper_cutoff'],
                #                                          channel_mode=self.clip_by_frequency['channel_mode'])
                # print('aft', img.min(), img.max(), inp[idx].min(), inp[idx].max())

        self.normalization_minima_and_maxima_input = None
        if self.input_normalization is not None:
            if self.input_normalization['method'] == Img.normalization_methods[7]:
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
            input = Img.normalization(input, normalization_minima_and_maxima=self.normalization_minima_and_maxima_input,
                                      **self.input_normalization)
        elif self.input_normalization is not None:
            input = Img.normalization(input, **self.input_normalization)
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

        method = last_method

        # force invert/negative to be every other image if added to augmentation
        if method is not None:
            parameters, input = method(input, parameters, False)

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

        if method is None:
            return method, None, input
        else:
            return method, parameters, input

    def augment_output(self, msk, output_shape, last_method,
                       parameters,
                       first_run):  # add as a parameter whether should do dilation or not and whether should change things --> then need add alos a parameter at the beginning of the class to handle that as well

        random.seed(self.random_seed)  # random seed is set for input and the same random is used for output

        logger.debug('method output ' + str(last_method) + str(parameters))

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
            if self.output_normalization['method'] == Img.normalization_methods[7]:
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
        method = last_method

        # do normalization here
        if self.normalization_minima_and_maxima_output is not None:
            # --> gain of time --> à tester
            msk = Img.normalization(msk, normalization_minima_and_maxima=self.normalization_minima_and_maxima_output,
                                    **self.input_normalization)
        elif self.output_normalization is not None:
            msk = Img.normalization(msk, **self.input_normalization)
            # print('Classical normalization msk!', msk.min(), msk.max())

        if method is not None:
            parameters, msk = method(msk, parameters, True)
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
        if method is None:
            return method, None, msk
        else:
            return method, parameters, msk

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

    def change_image_intensity_and_shift_range(self, orig, parameters, is_mask):
        if parameters is None:
            scaling_factor = random.uniform(self.augmentation_types_and_values['intensity'], 1)
        else:
            scaling_factor = parameters[0]

        # print(scaling_factor)
        if not is_mask:
            # loop over channels
            for c in range(orig.shape[-1]):
                cur = orig[..., c]
                min_before = cur.min()
                initial_range = cur.max() - min_before
                # print('bef',min_before, cur.max())
                try:
                    import numexpr
                    cur = numexpr.evaluate("cur * scaling_factor")
                except:
                    cur = cur * scaling_factor
                new_min = cur.min()
                shift_min = min_before - new_min
                try:
                    import numexpr
                    cur = numexpr.evaluate("cur + shift_min")
                except:
                    cur += shift_min
                shift_range = bool(random.getrandbits(1))
                if shift_range:
                    # shift rescaled image up
                    new_range = cur.max() - cur.min()
                    possible_range_increase = initial_range - new_range
                    random_range_shift = random.uniform(0., possible_range_increase)
                    # /2. # we divide by 2 to make it not too extreme
                    try:
                        import numexpr
                        cur = numexpr.evaluate("cur + random_range_shift")
                    except:
                        cur += random_range_shift
                    # print('range shift')
                orig[..., c] = cur
                # print('aft', orig[..., c].min(), orig[..., c].max())
                del cur
            out = orig
        else:
            out = orig
        return [scaling_factor], out

    def blur(self, orig, parameters, is_mask):
        # we just blur input and keep masks unchanged
        if parameters is None:
            gaussian_blur = random.uniform(0, self.augmentation_types_and_values['blur'])
        else:
            gaussian_blur = parameters[0]

        if not is_mask:
            # aletrantively could do a 3D blur
            if len(orig.shape) == 4:
                for n, slc in enumerate(orig):
                    orig[n] = filters.gaussian(slc, gaussian_blur, preserve_range=True, mode='wrap')
                out = orig
            else:
                out = filters.gaussian(orig, gaussian_blur, preserve_range=True, mode='wrap')
        else:
            out = orig

        return [gaussian_blur], out

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
        cells = label(Img.invert(cells), connectivity=1, background=0)
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

    # nb does crop work for images with channels ??? not so sure --> need check
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

        # print('coords', startx, endx, starty, endy) # seems ok now
        # TODO maybe implement that if crop is outside then add 0 or min to the extra region --> very important for the random crops --> TODO
        if len(img.shape) == 3:
            return img[starty:endy, startx:endx]
        else:
            return img[::, starty:endy, startx:endx]
            # return img[..., starty:endy, startx:endx] # smarter

    def invert(self, orig, parameters, is_mask):
        if not is_mask:
            max = orig.max()
            inverted_image = np.negative(orig) + max
        else:
            inverted_image = orig
        return [], inverted_image

    def rotate(self, orig, parameters, is_mask):
        # rotate by random angle between 0 and 360
        if parameters is None:
            angle = random.randint(0, self.augmentation_types_and_values['rotate'] * 360)
            angle *= 1 if random.random() < 0.5 else -1
            order = random.choice([0, 1, 3])
        else:
            angle = parameters[0]
            order = parameters[1]

        final_order = order
        if is_mask:
            final_order = self.EXTRAPOLATION_MASKS

        # bug fix for 3D images --> seems to work and should be portable even with n dims
        rot_orig = ndimage.rotate(orig, angle, axes=(-3, -2), reshape=False, order=final_order)
        # rot_mask = ndimage.rotate(mask, angle, reshape=False, order=0) # order 0 means nearest neighbor --> really required to avoid bugs here
        return [angle, order], rot_orig

    def elastic(self, orig, parameters, is_mask):
        # masks must be modified in the same way as other images
        if parameters is None:
            # create a random deformation matrix
            displacement = np.random.randn(2, 3, 3) * 25
            order = random.choice([0, 1])
        else:
            displacement = parameters[0]
            order = parameters[1]

        # for 3D images I need to change things !!!
        if len(orig.shape) == 4:
            # assume 'dhwc'
            rot_orig = elastic_deform(orig, displacement, axis=(1,2), order=order)
        else:
            # assume 'hwc'
            rot_orig = elastic_deform(orig, displacement, axis=(0, 1), order=order)

        # rot_mask = ndimage.rotate(mask, angle, reshape=False, order=0) # order 0 means nearest neighbor --> really required to avoid bugs here
        return [displacement, order], rot_orig

    # nb will that create a bug when images don't have same width and height --> most likely yes but test it --> TODO
    # in fact I should not allow that except if image has same width and height or I should use a crop of it --> can I do that ???
    def rotate_interpolation_free(self, orig, parameters, is_mask):
        # rotate by random angle between [90 ,180, 270]

        # print(orig.shape)
        if parameters is None:
            if orig.shape[-2] != orig.shape[-3]:
                angle = 180
            else:
                angle = random.choice([90, 180, 270])
        else:
            angle = parameters[0]

        rot_orig = Img.interpolation_free_rotation(orig, angle=angle)
        return [angle], rot_orig

    def translate(self, orig, parameters, is_mask):
        import numpy as np
        if parameters is None:
            trans_x = np.random.randint(-int(orig.shape[-2] * self.augmentation_types_and_values['translate']),
                                        int(orig.shape[-2] * self.augmentation_types_and_values['translate']))
            trans_y = np.random.randint(-int(orig.shape[-3] * self.augmentation_types_and_values['translate']),
                                        int(orig.shape[-3] * self.augmentation_types_and_values['translate']))
            order = random.choice([0, 1, 3])
        else:
            trans_x = parameters[0]
            trans_y = parameters[1]
            order = parameters[2]
        afine_tf = transform.AffineTransform(translation=(trans_y, trans_x))

        final_order = order
        if is_mask:
            final_order = self.EXTRAPOLATION_MASKS

        if len(orig.shape) <= 3:
            zoomed_orig = transform.warp(orig, inverse_map=afine_tf, order=final_order, preserve_range=True,
                                         mode='reflect')
        else:
            zoomed_orig = np.zeros_like(orig, dtype=orig.dtype)
            for i, slice in enumerate(orig):
                zoomed_orig[i] = transform.warp(slice, inverse_map=afine_tf, order=final_order, preserve_range=True,
                                                mode='reflect')

        return [trans_x, trans_y, order], zoomed_orig

    def stretch(self, orig, parameters, is_mask):
        if parameters is None:
            scale = random.uniform(self.stretch_range[0],
                                   self.augmentation_types_and_values['stretch'])
            order = random.choice([0, 1, 3])
            orientation = random.choice([0, 1])
        else:

            scale = parameters[0]
            order = parameters[1]
            orientation = parameters[2]

        # print('scale', scale)

        # scale = 3

        # scale = 2 # shrink 2x
        # scale = 0.5 # zoom 2x

        # plt.imshow(np.squeeze(orig))
        # plt.show()
        # print('test', orig.max(), orig.min())

        # nb the dilation is mandatory in case of stretch and 1px wide masks --> check if 1 px wide and take action
        # ou alors prendre des trucs

        # for scale = 3 need dilation 1 for scale 4 need dilation 2

        # nb if scale = 5 then I need a dilation of 2
        # if scale =3 need dilation 1

        # nb if < 2 then ok but weaker, if >=2 then need a dilation of 1 if >=5 need a dilation of 2 --> do limit its use and probably does not make sense to stretch less than 2
        # there are bugs in the interpolation of scipy

        # si 4 faut une dilation de 2 # --> voir comment faire
        # scale = 3 # make very stretched cells # 4 c'est parfait c'est comme dans le hinge --> top
        # orientation = 0

        # pb si trop de stretch --> blaste le max --> faudrait dilat avant
        if orientation == 0:
            afine_tf = transform.AffineTransform(scale=(scale, 1))
        else:
            afine_tf = transform.AffineTransform(scale=(1, scale))

        final_order = order
        if is_mask:
            final_order = self.EXTRAPOLATION_MASKS
            # TODO check that I'm not doing unnecessary stuff here but seems to work... maybe recheck dtype too
            if self.is_output_1px_wide:
                dilation = 0
                if scale >= 2 and scale <= 3:
                    # do 1 dilat
                    dilation = 1
                elif scale >= 3:
                    # do 2 dilat
                    dilation = 2
                if dilation != 0:
                    s = ndimage.generate_binary_structure(2, 1)
                    # Apply dilation to every channel then reinject
                    for c in range(orig.shape[-1]):
                        dilated = orig[..., c]
                        if len(orig.shape) == 4:
                            for n, slc in enumerate(dilated):
                                for dil in range(dilation):
                                    dilated[n] = ndimage.grey_dilation(slc, footprint=s)
                            orig[..., c] = dilated
                        else:
                            for dil in range(dilation):
                                dilated = ndimage.grey_dilation(dilated, footprint=s)
                            orig[..., c] = dilated
            #     print('here')
            #     # need dilate the mask
        # print('test2', orig.max(), orig.min())

        if len(orig.shape) == 4:
            # handle 3D images
            for n, slc in enumerate(orig):
                orig[n] = transform.warp(slc, inverse_map=afine_tf, order=final_order, preserve_range=True,
                                         mode='reflect')
            stretched_orig = orig
        else:
            stretched_orig = transform.warp(orig, inverse_map=afine_tf, order=final_order, preserve_range=True,
                                            mode='reflect')  # nb should I do this per channel to avoid issues --> MAYBE

        # print('test3', stretched_orig.max(), stretched_orig.min())
        return [scale, final_order, orientation], stretched_orig

    # CODER KEEP TIP: SCALE IS INVERTED (COMPARED TO MY OWN LOGIC SCALE 2 MEANS DIVIDE SIZE/DEZOOM BY 2 AND 0.5 MEANS INCREASE SIZE/ZOOM BY 2 --> # shall I use 1/scale
    def zoom(self, orig, parameters, is_mask):
        if parameters is None:
            scale = random.uniform(1. - self.augmentation_types_and_values['zoom'],
                                   1. + self.augmentation_types_and_values['zoom'])
            order = random.choice([0, 1, 3])
        else:
            scale = parameters[0]
            order = parameters[1]

        # scale = 2 # shrink 2x
        # scale = 0.5 # zoom 2x
        afine_tf = transform.AffineTransform(scale=(scale, scale))

        final_order = order
        if is_mask:
            final_order = self.EXTRAPOLATION_MASKS

        if len(orig.shape) == 4:
            # handle 3D images
            for n, slc in enumerate(orig):
                orig[n] = transform.warp(slc, inverse_map=afine_tf, order=final_order, preserve_range=True,
                                         mode='reflect')
            zoomed_orig = orig
        else:
            zoomed_orig = transform.warp(orig, inverse_map=afine_tf, order=final_order, preserve_range=True,
                                         mode='reflect')
        return [scale, final_order], zoomed_orig

    def shear(self, orig, parameters, is_mask):

        if parameters is None:
            shear = random.uniform(-self.augmentation_types_and_values['shear'],
                                   self.augmentation_types_and_values['shear'])
            order = random.choice([0, 1, 3])
        else:
            shear = parameters[0]
            order = parameters[1]

        afine_tf = transform.AffineTransform(shear=shear)

        final_order = order
        if is_mask:
            final_order = self.EXTRAPOLATION_MASKS

        if len(orig.shape) == 4:
            # handle 3D images
            for n, slc in enumerate(orig):
                orig[n] = transform.warp(slc, inverse_map=afine_tf, order=final_order, preserve_range=True,
                                         mode='reflect')  # is that a good idea --> maybe not...
            sheared_orig = orig
        else:
            sheared_orig = transform.warp(orig, inverse_map=afine_tf, order=final_order, preserve_range=True,
                                          mode='reflect')

        return [shear, order], sheared_orig

    # allow for shift mask but only for the mask
    def mask_pixels_and_compute_for_those(self, orig, is_mask, shift_mask=False):
        if not is_mask:
            rolled_orig = mask_rows_or_columns(orig, spacing_X=2, spacing_Y=5)
        else:
            # THIS ALLOWS MODEL TO PARTIALLY LEARN IDENTITY OR AT LEAST TO ALSO LEARN FROM THE PIXEL OF INTEREST WHICH DOES MAKE A LOT OF SENSE!!!
            initial_shift_X = 0
            initial_shift_Y = 0
            if shift_mask:
                if random.uniform(0, 1) < 0.5:
                    if random.uniform(0, 1) < 0.5:
                        if random.uniform(0, 1) < 0.5:
                            initial_shift_X = 1
                        else:
                            initial_shift_Y = 1
                    else:
                        initial_shift_X = 1
                        initial_shift_Y = 1
            rolled_orig = mask_rows_or_columns(orig, spacing_X=2, spacing_Y=5, initial_shiftX=initial_shift_X,
                                               initial_shiftY=initial_shift_Y, return_boolean_mask=True)
            orig[rolled_orig == False] = np.nan
            rolled_orig = orig  # pb non nan anymore
        return rolled_orig

    # rolls along the Z axis of a 3D image ignores for 2D images and ignores for masks
    def rollZ(self, orig, parameters, is_mask):
        # print('in rollz', orig.shape)
        if parameters is None:
            if len(orig.shape) == 4:
                random_roll = np.random.randint(0, orig.shape[0])
                # do random signed roll
                if random_roll != 0:
                    random_roll = random_roll if random.random() < 0.5 else -random_roll
            else:
                random_roll = 0
        else:
            random_roll = parameters[0]

        if not is_mask and random_roll != 0 and len(orig.shape) == 4 and \
                random_roll != orig.shape[0] and random_roll != -orig.shape[0]:
            # print('really rolling', orig.shape)
            # if image is a 3D stack roll it, except if rolling generates identity images
            rolled_orig = np.roll(orig, random_roll, axis=0)
        else:
            # print('not rolling')
            # if image is 2D then do not roll it...
            rolled_orig = orig
        return [random_roll], rolled_orig

    # shuffle images along the Z axis, may be useful for best focus algorithms
    def shuffleZ(self, orig, parameters, is_mask):
        # print('in shuffle', orig.shape)
        if not is_mask and len(orig.shape) == 4:
            # print('really shuffling', orig.shape)
            # if image is a 3D stack shuffle it otherwise ignore
            np.random.shuffle(orig)
        # else:
        #     print('not shuffling')
        #     pass
        return [], orig

    def flip(self, orig, parameters, is_mask):
        if parameters is None:
            if len(orig.shape) == 4:
                axis = random.choice([-2, -3, -4])
            else:
                axis = random.choice([-2, -3])
        else:
            axis = parameters[0]

        # THAT WAS CREATING A BIG BUG FOR MASKS --> REALLY BAD BUG converting image to int
        if axis == -4 and len(orig.shape) == 3:
            flipped_orig = orig  # there was a bug here
        else:
            flipped_orig = np.flip(orig, axis)
        return [axis], flipped_orig

    def high_noise(self, orig, parameters, is_mask):
        return self._add_noise(orig, parameters, is_mask, 'high')

    def low_noise(self, orig, parameters, is_mask):
        return self._add_noise(orig, parameters, is_mask, 'low')

    # (img, Zpos=-4, z_frames_to_add_above=0, z_frames_to_add_below=0):
    def add_z_frames(self, orig, parameters, is_mask):
        # print(orig.shape)
        # TODO do expand along the Z with black
        # de meme si l'image est 2D --> ignore I guess -->

        # ignore that for masks --> is that wise or not to ignore it for masks ???
        if is_mask:
            return orig

        if self.z_frames_to_add_below == 0 and self.z_frames_to_add_above == 0:
            # nothing to do...
            return orig

        # smarter way --> get Z pos
        zpos = -4
        try:
            zpos = orig.get_dim_idx('d')
        except:
            logger.error('dimension d/z not found, assuming Z pos = -4, i.e. ...dhwc image')

        # print('in1')

        # if orig.ndim < 4:
        #     # orig has no Z channel --> nothing to do ...
        #     return orig

        # print('in2')

        # print('zpos', zpos)
        if orig.shape[zpos] == 1:
            # image is 2D --> nothing to do ...
            return orig

        # print('in3')

        if self.z_frames_to_add_above != 0:
            # add Z frames before
            smallest_shape = list(orig.shape)
            smallest_shape[zpos] = self.z_frames_to_add_above
            missing_frames_above = np.zeros((smallest_shape), dtype=orig.dtype)

            # print(missing_frames_above.shape)

            orig = np.append(missing_frames_above, orig, axis=zpos)  # nb should do that per channel in fact... -->
            # print(orig.shape)

        if self.z_frames_to_add_below != 0:
            # add Z frames after
            smallest_shape[zpos] = self.z_frames_to_add_below
            missing_frames_below = np.zeros((smallest_shape), dtype=orig.dtype)
            # print(missing_frames_below.shape)
            orig = np.append(orig, missing_frames_below, axis=zpos)  # nb should do that per channel in fact... -->
            # print(orig.shape)

        # print(orig.shape)

        # print('final', orig.shape)
        # TODO maybe just do this as a classical function that does not return the parameters...
        return [], orig

    # use skimage to add noise
    def _add_noise(self, orig, parameters, is_mask, strength='low'):
        if parameters is None:
            # fraction of image that should contain salt black or pepper white pixels
            mode = random.choice(['gaussian', 'localvar', 'poisson', 's&p',
                                  'speckle'])
        else:
            mode = parameters[0]

        # debug force mode
        # mode = 'salt'  #'speckle' # 's&p' # 'pepper' # 'salt' # 'gaussian' # 'localvar' # localvar not that strong

        if not is_mask:
            extra_params = {}
            if mode in ['salt', 'pepper', 's&p']:
                if strength == 'low':
                    extra_params['amount'] = 0.1  # weak
                else:
                    extra_params['amount'] = 0.25  # strong
            if mode in ['gaussian', 'speckle']:
                if strength == 'low':
                    extra_params['var'] = 0.025  # weak gaussian
                else:
                    extra_params['var'] = 0.1  # strong gaussian
                if mode == 'speckle':
                    if strength == 'low':
                        extra_params['var'] = 0.1
                    else:
                        extra_params['var'] = 0.40

            min = orig.min()
            max = orig.max()

            # print('noise', mode, extra_params, is_mask)

            noisy_image = random_noise(orig, mode=mode, clip=True, **extra_params)
            if min == 0 and max == 1 or max == min:
                pass
            else:
                # we preserve original image range (very important to keep training consistent)
                noisy_image = (noisy_image * (max - min)) + min
        else:
            noisy_image = orig
        return [mode], noisy_image

    def random_intensity_gamma_contrast_changer(self, orig, parameters, is_mask):
        return self._random_intensity_gamma_contrast_changer(orig, parameters, is_mask)

    # use skimage to change gamma/contrast, ... pixel intensity and/or scale
    def _random_intensity_gamma_contrast_changer(self, orig, parameters, is_mask):
        # seems ok but try with 3D images
        if parameters is None:
            # fraction of image that should contain salt black or pepper white pixels
            mode = random.choice([exposure.rescale_intensity, exposure.adjust_gamma, exposure.adjust_log,
                                  exposure.adjust_sigmoid])
            strength = random.choice(['medium', 'strong'])
        else:
            mode = parameters[0]
            strength = parameters[1]

        # debug force mode
        # mode = exposure.rescale_intensity # exposure.adjust_gamma # exposure.adjust_log # exposure.adjust_sigmoid

        if not is_mask:
            if orig.min() >= 0:
                extra_params = {}
                if mode == exposure.rescale_intensity:
                    if strength == 'strong':
                        v_min, v_max = np.percentile(orig, (5, 95))
                    else:
                        v_min, v_max = np.percentile(orig, (0.9, 98))
                    extra_params['in_range'] = (v_min, v_max)
                elif mode == exposure.adjust_gamma:
                    if strength == 'strong':
                        extra_params['gamma'] = 0.4
                        extra_params['gain'] = 0.9
                    else:
                        extra_params['gamma'] = 0.8
                        extra_params['gain'] = 0.9
                elif mode == exposure.adjust_sigmoid:
                    if strength == 'strong':
                        extra_params['gain'] = 5
                    else:
                        extra_params['gain'] = 2
                # print(mode, extra_params, is_mask)
                contrast_changed_image = mode(orig, **extra_params)
            else:
                logger.warning(
                    'Negative intensities detected --> ignoring random_intensity_gamma_contrast_changer augmentation.')
                contrast_changed_image = orig
        else:
            contrast_changed_image = orig

        return [mode, strength], contrast_changed_image


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
    SELECTED_AUG = None

    normalization = {'method': Img.normalization_methods[7], 'range': [2, 99.8],
                     'individual_channels': True, 'clip': False}

    mask_lines_and_cols_in_input_and_mask_GT_with_nans = None

    # 3D
    # seems to work --> but check
    augmenter = DataGenerator(
        # 'D:/dataset1/tests_focus_projection', 'D:/dataset1/tests_focus_projection',
        # 'D:/dataset1/tests_focus_projection/proj', 'D:/dataset1/tests_focus_projection/proj/*/hand*.tif',
        inputs='/E/Sample_images/sample_images_denoise_manue/210219/raw', # 3D input
        outputs='/E/Sample_images/sample_images_denoise_manue/210219/raw/predict', # 2D ouput otherwise change model shape... below !!!!
        # '/D/datasets_deep_learning/keras_segmentation_dataset/TA_test_set/tests_focus_projection/proj', '/D/datasets_deep_learning/keras_segmentation_dataset/TA_test_set/tests_focus_projection/proj/',
        # '/D/datasets_deep_learning/keras_segmentation_dataset/TA_test_set/tests_focus_projection', '/D/datasets_deep_learning/keras_segmentation_dataset/TA_test_set/tests_focus_projection',
        # '/home/aigouy/Bureau/last_model_not_sure_that_works/tmp', '/home/aigouy/Bureau/last_model_not_sure_that_works/tmp',
        # is_predict_generator=True,
        # crop_parameters={'x1':256, 'y1':256, 'x2':512, 'y2':512},
        # crop_parameters={'x1':512, 'y1':512, 'x2':796, 'y2':796},
        input_normalization=normalization,
        # shuffle=False, input_shape=[(None, None, None, None, 1)], output_shape=[(None, None, None, None, 1)],
        shuffle=False, input_shape=[(None, None, None, None, 1)], output_shape=[(None, None, None, 1)], # 3D input and 2D output --> if not the case there will be errors !!!
        augmentations=SELECTED_AUG,
        input_channel_of_interest=0,
        output_channel_of_interest=0,
        # mask_dilations=7,
        # default_input_tile_width=2048, default_input_tile_height=1128,
        # default_output_tile_width=2048, default_output_tile_height=1128,
        # default_input_tile_width=512, default_input_tile_height=512,
        # default_output_tile_width=512, default_output_tile_height=512,
        default_input_tile_width=512, default_input_tile_height=256,
        default_output_tile_width=512, default_output_tile_height=256,
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
            if counter < 5:
                try:
                    center = int(len(orig[i]) / 2)
                    # center = 0

                    plt.imshow(np.squeeze(orig[i][center]))  # , cmap='gray')

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
            print('in here2', mask[i].shape)  # --> seems ok
            # print(mask[i]) # marche car contient des nans
            if counter < 5:
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
        if counter >= 6:
            plt.close(fig='all')
        # do stuff with that
    print('end')
    print(counter)
    print(full_count)
