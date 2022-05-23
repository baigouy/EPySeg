import os

from epyseg.deeplearning import zoo # epyseg model zoo
from epyseg.postprocess.filtermask import simpleFilter
from epyseg.postprocess.refine_v2 import RefineMaskUsingSeeds
from epyseg.tools.early_stopper_class import early_stop

os.environ['SM_FRAMEWORK'] = 'tf.keras'  # set env var for changing the segmentation_model framework
import traceback
import matplotlib.pyplot as plt
from epyseg.img import Img
from epyseg.deeplearning.augmentation.generators.data import DataGenerator
import numpy as np
import tensorflow as tf
import urllib.request
import hashlib
import re
from epyseg.postprocess.refine import EPySegPostProcess
import segmentation_models as sm
# sm.set_framework('tf.keras') # alternative fix = changing framework on the fly
from epyseg.deeplearning.callbacks.saver import My_saver_callback
from epyseg.deeplearning.callbacks.stop import myStopCallback
from segmentation_models.metrics import *
from segmentation_models.losses import *
from skimage import exposure
from epyseg.tools.logger import TA_logger # logging
from tensorflow.keras.callbacks import ReduceLROnPlateau
import gc

logger = TA_logger()

class EZDeepLearning:
    '''A class to handle deep learning models

    '''
    available_model_architectures = ['Unet', 'PSPNet', 'FPN', 'Linknet']

    optimizers = ['adam', 'sgd', 'rmsprop', 'adagrad', 'adadelta', 'adamax', 'nadam']

    available_sm_backbones = sm.get_available_backbone_names()

    # TODO below are the pretrained models for 2D epithelia segmentation if None --> no pretrained model exist # maybe sort them by efficiency ???
    # for each model do provide all the necessary parameters: 'model' 'model_weights' 'architecture' 'backbone' 'activation' 'classes' 'input_width' 'input_height' 'input_channels'


    pretrained_models = {
        # favourite model first
        'Linknet-vgg16-sigmoid-v2': {
        # 'EPySeg v2': {
            'url': 'https://gitlab.com/baigouy/models/raw/master/model_linknet-vgg16_shells_v2.h5',
            'input_dims': '2D',
            'md5': '98c8a51f3365e77c07a4f9e95669c259',
            # 'model': None,
            # 'model_weights': None,
            'architecture': 'Linknet',
            'backbone': 'vgg16',
            'activation': 'sigmoid',
            'classes': 7,
            'input_width': None,
            'input_height': None,
            'input_channels': 1,
            'version': 2
        },
        'Linknet-vgg16-sigmoid': {
        # 'EPySeg v1': {
            'url': 'https://gitlab.com/baigouy/models/raw/master/model_linknet-vgg16_shells.h5',
            'md5': '266ca9acd9d7a4fe74a473e17952fb6c',
            'input_dims':'2D',
            # 'model': None,
            # 'model_weights': None,
            'architecture': 'Linknet',
            'backbone': 'vgg16',
            'activation': 'sigmoid',
            'classes': 7,
            'input_width': None,
            'input_height': None,
            'input_channels': 1,
            'version': 1
        },
        # not as good as the two others...
        # 'https://github.com/baigouy/models/raw/master/model_Linknet-seresnext101.h5'
        # 'Linknet-seresnext101-sigmoid': {
        #     'url': 'https://gitlab.com/baigouy/models/raw/master/model_Linknet-seresnext101.h5',
        #     'input_dims': '2D',
        #     'md5': '209f3bf53f3e2f5aaeef62d517e8b8d8',
        #     # 'model': None,
        #     # 'model_weights': None,
        #     'architecture': 'Linknet',
        #     'backbone': 'seresnext101',
        #     'activation': 'sigmoid',
        #     'classes': 1,
        #     'input_width': None,
        #     'input_height': None,
        #     'input_channels': 1},
        # TODO finalize that some day
        # 'CARE': {
        #     'url': 'TODO',
        #     'input_dims': '3D',
        #     'md5': 'TODO',
        #     'model': 'CARE.json', # here we put the model zoo name (if it is present in the model zoo then ignore the rest cause it would be useless)
        # },
        # 'CARE_SEG': {
        #     'url': 'TODO',
        #     'input_dims': '3D',
        #     'md5': 'TODO',
        #     'model': 'CARE_SEG.json',
        #     # here we put the model zoo name (if it is present in the model zoo then ignore the rest cause it would be useless)
        # },
        # 'CARE_GUIDED': {
        #     'url': 'TODO',
        #     'md5': 'TODO',
        #     'model': 'CARE_GUIDED.json',
        #     # here we put the model zoo name (if it is present in the model zoo then ignore the rest cause it would be useless)
        # },
        # 'CARE_PLUS_HEIGHT_MAP': {
        #     'url': 'TODO',
        #     'input_dims': '3D',
        #     'md5': 'TODO',
        #     'model': '.json', #TODO
        #     # here we put the model zoo name (if it is present in the model zoo then ignore the rest cause it would be useless)
        # },
        # 'SURFACE_PROJECTION_NO_RESTORATION': {
        #     'url': 'TODO',
        #     'input_dims': '3D',
        #     'md5': 'TODO',
        #     'model': '.json',  # TODO
        #     # here we put the model zoo name (if it is present in the model zoo then ignore the rest cause it would be useless)
        # },
        # 'SURFACE_PROJECTION': { # SURFACE PROJECTION MODULE OF THE CARE MODEL
        #     'url': 'file:///home/aigouy/mon_prog/Python/epyseg_pkg/personal/pyTA/GUI/tsts/surface_projection_model_weights_1.h5', # keep to test can put a local path 'file:///home/aigouy/mon_prog/Python/epyseg_pkg/personal/pyTA/GUI/tsts/surface_projection_model_weights_1.h5'
        #     'input_dims': '3D',
        #     'md5': 'ef8b904da9d44e7196fd3efb832ec74a',
        #     'model': 'surface_projection_model.json',  # TODO
        #     # here we put the model zoo name (if it is present in the model zoo then ignore the rest cause it would be useless)
        # },
        # this seems to be the best compromise so far --> maybe set this by default and see which denoiser I will couple it to...
        'SURFACE_PROJECTION_2': { # SURFACE PROJECTION MODULE OF THE CARE MODEL
            'url': 'https://gitlab.com/baigouy/models/raw/master/surface_projection_model_weights_0.h5', # keep to test can put a local path 'file:///home/aigouy/mon_prog/Python/epyseg_pkg/personal/pyTA/GUI/tsts/surface_projection_model_weights_0.h5'
            'input_dims': '3D',
            'md5': '6dd8692c2158390e492ed752ee363695',
            'model': 'surface_projection_model.json',  # TODO
            # here we put the model zoo name (if it is present in the model zoo then ignore the rest cause it would be useless)
        },
        'SURFACE_PROJECTION_3': { # SURFACE PROJECTION MODULE OF THE CARESEG MODEL --> yes it's only at the denoiser level that I made changes
            'url': 'https://gitlab.com/baigouy/models/raw/master/surface_projection_model_weights_2.h5', # keep to test can put a local path 'file:///home/aigouy/mon_prog/Python/epyseg_pkg/personal/pyTA/GUI/tsts/surface_projection_model_weights_2.h5'
            'input_dims': '3D',
            'md5': '61f1ce15005a7138b5c2a11bae9cc5ed',
            'model': 'surface_projection_model.json',  # TODO# is that correct for CARESEG ???
            # here we put the model zoo name (if it is present in the model zoo then ignore the rest cause it would be useless)
        },
        'SURFACE_PROJECTION_4': { # SURFACE PROJECTION MODULE OF THE CARESEG MODEL --> yes it's only at the denoiser level that I made changes
            'url': 'https://gitlab.com/baigouy/models/raw/master/surface_projection_model_weights_3.h5', # keep to test can put a local path 'file:///home/aigouy/mon_prog/Python/epyseg_pkg/personal/pyTA/GUI/tsts/surface_projection_model_weights_3.h5'
            'input_dims': '3D',
            'md5': '5f778acb48e2895a490716a42fae4996',
            'model': 'surface_projection_model.json',  # TODO# is that correct for CARESEG ???
            # here we put the model zoo name (if it is present in the model zoo then ignore the rest cause it would be useless)
        },
        'SURFACE_PROJECTION_5': {
            # SURFACE PROJECTION MODULE OF THE CARESEG MODEL --> yes it's only at the denoiser level that I made changes
            'url': 'https://gitlab.com/baigouy/models/raw/master/surface_projection_model_weights_5.h5',
            # 'url': 'file:///home/aigouy/mon_prog/Python/epyseg_pkg/personal/pyTA/GUI/tsts/surface_projection_model_weights_5.h5',
            # keep to test can put a local path 'file:///home/aigouy/mon_prog/Python/epyseg_pkg/personal/pyTA/GUI/tsts/surface_projection_model_weights_3.h5'
            'input_dims': '3D',
            'md5': '3000266e665aa46f6c8eac83fe3e5649',
            'model': 'surface_projection_model.json',  # TODO# is that correct for CARESEG ???
            # here we put the model zoo name (if it is present in the model zoo then ignore the rest cause it would be useless)
        },
        'SURFACE_PROJECTION_6': {
            # SURFACE PROJECTION MODULE OF THE CARESEG MODEL --> yes it's only at the denoiser level that I made changes
            'url': 'https://gitlab.com/baigouy/models/raw/master/surface_projection_model_weights_6.h5',
            # 'url': 'file:///home/aigouy/mon_prog/Python/epyseg_pkg/personal/pyTA/GUI/tsts/surface_projection_model_weights_6.h5',
            # keep to test can put a local path 'file:///home/aigouy/mon_prog/Python/epyseg_pkg/personal/pyTA/GUI/tsts/surface_projection_model_weights_3.h5'
            'input_dims': '3D',
            'md5': '7d86782c52bf65221b6041c8b7e5ef21',
            'model': 'surface_projection_model.json',  # TODO# is that correct for CARESEG ???
            # here we put the model zoo name (if it is present in the model zoo then ignore the rest cause it would be useless)
        },
        # '2D_DENOISER': { # ORIGINAL DENOISER MODULE OF THE CARE MODEL
        #     'url': 'file:///home/aigouy/mon_prog/Python/epyseg_pkg/personal/pyTA/GUI/tsts/2D_denoiser_model_weights_1.h5',
        #     'input_dims': '2D',
        #     'md5': 'TODO',
        #     'model': '2D_denoiser_model.json',  # TODO
        #     # here we put the model zoo name (if it is present in the model zoo then ignore the rest cause it would be useless)
        # },
        '2D_DENOISEG': { # HACKED DENOISER MODULE BASED ON THE CARE DENOISER
            'url': 'https://gitlab.com/baigouy/models/raw/master/2D_denoiseg_model_weights_0.h5',# keep to test can put a local path 'file:///home/aigouy/mon_prog/Python/epyseg_pkg/personal/pyTA/GUI/tsts/2D_denoiseg_model_weights_0.h5'
            'input_dims': '2D',
            'md5': '1f39a8e9e450d1b9747fdd28c2f5a5eb',
            'model': '2D_denoiseg_model.json',  # TODO
            # here we put the model zoo name (if it is present in the model zoo then ignore the rest cause it would be useless)
        },


        # TODO should I add others and maybe retrain
    }

    # https://www.tensorflow.org/api_docs/python/tf/keras/metrics
    # TODO add smlosses iou... also add shortcuts
    metrics = {'accuracy': 'accuracy', 'f1_score': f1_score, 'f2_score': f2_score,
               'precision': precision, 'iou_score': iou_score,
               'recall': recall, 'kullback_leibler_divergence': 'kullback_leibler_divergence',
               'mean_absolute_error': 'mean_absolute_error',
               'mean_absolute_percentage_error': 'mean_absolute_percentage_error',
               'mean_squared_error': 'mean_squared_error', 'msle': 'msle',
               'binary_accuracy': 'binary_accuracy', 'binary_crossentropy': 'binary_crossentropy',
               'categorical_accuracy': 'categorical_accuracy', 'categorical_crossentropy': 'categorical_crossentropy',
               'hinge': 'hinge', 'poisson': 'poisson', 'sparse_categorical_accuracy': 'sparse_categorical_accuracy',
               'sparse_categorical_crossentropy': 'sparse_categorical_crossentropy',
               'sparse_top_k_categorical_accuracy': 'sparse_top_k_categorical_accuracy',
               'top_k_categorical_accuracy': 'top_k_categorical_accuracy', 'squared_hinge': 'squared_hinge',
               'cosine_proximity': 'cosine_proximity'}

    # https://keras.io/losses/
    loss = {'mean_squared_error': 'mean_squared_error',
            'mean_absolute_error': 'mean_absolute_error',
            'jaccard_loss': jaccard_loss, 'binary_crossentropy': 'binary_crossentropy', 'dice_loss': dice_loss,
            'binary_focal_loss': binary_focal_loss, 'categorical_focal_loss': categorical_focal_loss,
            'binary_crossentropy': binary_crossentropy, 'categorical_crossentropy': categorical_crossentropy,
            'bce_dice_loss': bce_dice_loss, 'bce_jaccard_loss': bce_jaccard_loss, 'cce_dice_loss': cce_dice_loss,
            'cce_jaccard_loss': cce_jaccard_loss, 'binary_focal_dice_loss': binary_focal_dice_loss,
            'binary_focal_jaccard_loss': binary_focal_jaccard_loss,
            'categorical_focal_dice_loss': categorical_focal_dice_loss,
            'categorical_focal_jaccard_loss': categorical_focal_jaccard_loss,
            'mean_absolute_percentage_error': 'mean_absolute_percentage_error',
            'mean_squared_logarithmic_error': 'mean_squared_logarithmic_error', 'squared_hinge': 'squared_hinge',
            'hinge': 'hinge', 'categorical_hinge': 'categorical_hinge', 'logcosh': 'logcosh',
            'huber_loss': 'huber_loss', 'categorical_crossentropy': 'categorical_crossentropy',
            'sparse_categorical_crossentropy': 'sparse_categorical_crossentropy',
            'kullback_leibler_divergence': 'kullback_leibler_divergence', 'poisson': 'poisson',
            'cosine_proximity': 'cosine_proximity', 'is_categorical_crossentropy': 'is_categorical_crossentropy'}

    # TODO explain activation layers
    # https://keras.io/activations/
    last_layer_activation = ['sigmoid', 'softmax', 'linear', 'relu', 'elu', 'tanh', 'selu', 'softplus', 'softsign',
                             'hard_sigmoid', 'exponential', 'None']

    # stop_threads = False

    def __init__(self, use_cpu=False, tf_memory_limit=None):  # TODO handle backbone and type
        '''class init

        Parameters
        ----------
        use_cpu : boolean
            if set to True tf will use CPU (slow) instead of GPU

        '''




        # use_cpu = True # can be used to test a tf model easily using the CPU while the GPU is running.

        print('Using tensorflow version ' + str(tf.__version__))
        print('Using segmentation models version ' + sm.__version__)

        if use_cpu:
            # must be set before model is compiled
            self.force_use_cpu()
            print('Using CPU')

        # gpu_options = tf.GPUOptions(allow_growth=True)
        # session = tf.InteractiveSession(config=tf.ConfigProto(gpu_options=gpu_options))

        # physical_devices = None
        # try:
        #     physical_devices = tf.config.list_physical_devices('GPU')
        # except:
        #     # dirty hack for tf2.0 support for mac OS X anaconda
        #     try:
        #         physical_devices = tf.config.experimental.list_physical_devices('GPU')
        #     except:
        #         pass
        # # is this still needed --> yes it is still mandatory or I get a weird bug...
        # # for physical_device in physical_devices:
        # #     try:
        # #         tf.config.experimental.set_memory_growth(physical_device, True)
        # #     except:
        # #         # Invalid device or cannot modify.
        # #         pass
        #
        # # the line below allow for limiting tf allocated memory which allows me to run pytorch in parallel --> set this
        #
        # # that seems to work
        # if physical_devices:
        #     for physical_device in physical_devices:
        #         try:
        #             # unfortynately there is no way in tf to know the available GPU memory
        #             tf.config.experimental.set_virtual_device_configuration(physical_device, [
        #                 tf.config.experimental.VirtualDeviceConfiguration(memory_limit=100000)]) # 100Gigs --> no card has this now --> smarter would be to limit to the minimum EPySeg needs but not so easy TODO for training... probably 4 Gigs is enough for most of the things I am doing
        #             # tf.config.experimental.set_virtual_device_configuration(physical_devices[0], [
        #             #     tf.config.experimental.VirtualDeviceConfiguration(memory_limit=25000)]) # can be set higher than the max... and can be set dynamically --> good --> put this as an option so that I can reduce it
        #
        #         except RuntimeError as e:
        #             print(e)
        self._allow_memory_growth()
        self.set_memory_limit(tf_memory_limit)

        # nb I could also use this trick to perform several training in parallel on the same card
        self.stop_cbk = None
        self.saver_cbk = None
        self.model = None

    def get_GPUs(self):
        physical_devices = None
        try:
            physical_devices = tf.config.list_physical_devices('GPU')
        except:
            # dirty hack for tf2.0 support for mac OS X anaconda
            try:
                physical_devices = tf.config.experimental.list_physical_devices('GPU')
            except:
                pass
        return physical_devices

    def _allow_memory_growth(self):
        physical_devices = self.get_GPUs()
        if physical_devices:
            for physical_device in physical_devices:
                try:
                    tf.config.experimental.set_memory_growth(physical_device, True)
                except:
                    # Invalid device or cannot modify.
                    pass

    # nb can be done only once! need restart otherwise
    def set_memory_limit(self, limit=None):
        if limit is None:
            return
        physical_devices = self.get_GPUs()
        if physical_devices:
            logger.debug('Setting max GPU memory to '+str(limit))
            for physical_device in physical_devices:
                try:
                    # unfortunately there is no way in tf to know the available GPU memory...
                    tf.config.experimental.set_virtual_device_configuration(physical_device, [
                        tf.config.experimental.VirtualDeviceConfiguration(memory_limit=limit)]) # 100Gigs --> no card has this now --> smarter would be to limit to the minimum EPySeg needs but not so easy TODO for training... probably 4 Gigs is enough for most of the things I am doing
                    # tf.config.experimental.set_virtual_device_configuration(physical_devices[0], [
                    #     tf.config.experimental.VirtualDeviceConfiguration(memory_limit=25000)]) # can be set higher than the max... and can be set dynamically --> good --> put this as an option so that I can reduce it
                except RuntimeError as e:
                    print(e)

    def get_available_pretrained_models(self):

        available_pretrained_models = []
        for pretrained_model in self.pretrained_models.keys():
            if self.pretrained_models[pretrained_model] is not None:
                available_pretrained_models.append(pretrained_model)
        return available_pretrained_models

    # encoder_weights=None,
    def load_or_build(self, model=None, model_weights=None, architecture=None, backbone=None,
                      classes=1, activation='sigmoid', input_width=None, input_height=None, input_channels=1,
                      pretraining=None, **kwargs):
        encoder_weights = None  # TODO maybe some day connect imagenet pretraining in sm model
        '''loads an existing model or builds a new one

        Both building and loading a model may fail and None is returned then

        Parameters
        ----------
        model : string
            path to a model (model can be JSON, H5, HDF5 with or without weights)

        model_weights : string
            path to model weights (hdf5, h5)

        architecture : string
            model architecture

        backbone : string
            the encoder of the model

        # encoder_weights : TODO
        #     not defined yet but will be weights to apply to the encoder only in sm models

        classes : int
            number of output classes

        activation : string
            last layer activation (typically sigmoid for binary images)

        input_width : int
            model input width (can be None)

        input_height : int
            model input height (can be None)

        input_channels : int
            model input number of channels (e.g. 1 for single channel image and 3 for RGB, can be any number but not None)

        pretraining : string
            string or url indicating the pretrained model to load

        Returns
        -------
        model
            a model or None if model could not be created
        '''
        if activation == 'None':
            # None means 'linear' activation
            activation = None

        # TODO do that for all complex models..
        if backbone == 'senet154':
            logger.warning('the ' + str(
                backbone) + ' model uses a lot of memory and time to build and save (first epoch), please be extremely patient and use a small batch size')
        elif backbone == 'inceptionresnetv2':
            logger.warning('the ' + str(
                backbone) + ' model uses a lot of memory and time save (first epoch), please be patient')

        # convert url to pretraining (very dirty and hacky make use of proper keywords directly in my models)
        # if kwargs is not None:
            # if 'url' in kwargs and pretraining is None:
            #     pretraining = kwargs['url']

        self.model = None

        if model is None and architecture is not None:
            model_architecture = None
            if architecture.lower() == 'unet':
                model_architecture = sm.Unet
                model_name = architecture
            elif architecture.lower() == 'linknet':
                model_architecture = sm.Linknet
                model_name = architecture
            elif architecture.lower() == 'pspnet':
                model_architecture = sm.PSPNet
                model_name = architecture
            elif architecture.lower() == 'fpn':
                model_architecture = sm.FPN
                model_name = architecture
            else:
                model_name = 'unknown'
                logger.error('Unknown model type ' + str(architecture) + '... model cannot be built sorry...')

            if input_width is not None and input_width <= 0:
                input_width = None
            if input_height is not None and input_height <= 0:
                input_height = None

            if model_architecture is not None:
                model_name += '-' + backbone
                if activation is not None:
                    model_name += '-' + activation
                if pretraining is not None:
                    model_name += '-pretrained'
                    # print('pretraining', pretraining)
                    if not pretraining.startswith('http') and not pretraining.startswith(
                            'file:'):  # if not an url --> try load it or use file directly
                        # model

                        if pretraining in self.pretrained_models:
                            model_parameters = self.pretrained_models[pretraining]
                            if 'md5' in model_parameters:
                                file_hash = model_parameters['md5']
                            else:
                                file_hash = None
                            if 'url' in model_parameters:
                                url = model_parameters['url']
                            else:
                                url = None
                            try:
                                # if file doesn't exist or hash do not match then it downloads it otherwise keeps it
                                # model_weights = tf.keras.utils.get_file(pretraining + '.h5', url, file_hash=file_hash,
                                #                                         cache_subdir='epyseg', hash_algorithm='auto',
                                #                                         extract=False, archive_format='auto')
                                #                                         #extract=True, archive_format='auto') # zipping is not a good idea as there is no gain in space
                                model_weights = self.get_file(pretraining + '.h5', url, file_hash=file_hash,
                                                              cache_subdir='epyseg')
                                # extract=True, archive_format='auto') # zipping is not a good idea as there is no gain in space

                            except:
                                logger.error("could not load pretrained model for '" + str(pretraining) + "'")
                                traceback.print_exc()
                    else:
                        model_weights = pretraining  # local file not url

                self.model = model_architecture(backbone, input_shape=(input_width, input_height, input_channels),
                                                encoder_weights=encoder_weights,
                                                classes=classes, activation=activation)

                self.model._name = model_name
            else:
                self.model = None
        else:
            logger.debug('loading model')
            self.model = self.load_model(model=model)
            if self.model is not None:
                logger.info('Model loaded successfully.')
                try:
                    filename0_without_path = os.path.basename(model)
                    filename0_without_ext = os.path.splitext(filename0_without_path)[0]
                    self.model._name = filename0_without_ext
                except:
                    try:
                        import tempfile
                        filename0 = tempfile.NamedTemporaryFile().name
                        filename0_without_path = os.path.basename(filename0)
                        filename0_without_ext = os.path.splitext(filename0_without_path)[0]
                        self.model._name = filename0_without_ext
                    except:
                        self.model._name = 'loaded_model'
            else:
                logger.error('Model could not be loaded sorry...')
        self.load_weights(model_weights)

    # @staticmethod
    def get_md5_hash(self, filename):
        if not os.path.isfile(filename):
            return None
        hash_md5 = hashlib.md5()
        with open(filename, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b''):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()

    def get_file(self, output_name_without_path, url, cache_dir=None, file_hash=None, cache_subdir=None):
        # hack to replace for the line below to allow for gitlab download, hopefully keras original function will be fixed some day to include user agent and prevent forbidden dl error
        # model_weights = tf.keras.utils.get_file(pretraining + '.h5', url, file_hash=file_hash,
        #                                         cache_subdir='epyseg', hash_algorithm='auto',
        #                                         extract=False, archive_format='auto')
        #                                         #extract=True, archive_format='auto') # zipping is not a good idea as there is no gain in space
        if cache_dir is None:
            cache_dir = os.path.join(os.path.expanduser('~'), '.keras')
        if cache_subdir is not None:
            final_path = os.path.join(cache_dir, cache_subdir)
        else:
            final_path = cache_dir
        os.makedirs(final_path, exist_ok=True)

        # try read the file and check if hash is ok --> if so no action taken otherwise --> take action
        file_name = os.path.join(final_path, output_name_without_path)
        if file_hash is not None:
            if self.get_md5_hash(file_name) == file_hash:
                # file is ok, no action taken
                return file_name
            else:
                print('Model file is not up to date')

        print('Downloading file from', url)
        # need download file again
        opener = urllib.request.build_opener()
        # fake that it is a normal browser otherwise it gets rejected by github
        opener.addheaders = [('User-agent', 'Mozilla/5.0')]
        urllib.request.install_opener(opener)

        # progress bar for dl
        def show_dl_progress(cur_block, block_size, total_size):
            downloaded = cur_block * block_size
            if cur_block % 100 == 0:
                print(round((downloaded / total_size) * 100, 1), '%')
            if downloaded == total_size:
                print('download complete...')

        # download the missing file
        urllib.request.urlretrieve(url, file_name,
                                   reporthook=show_dl_progress)  # it works like that, gitlab blocks python bots, i.e. and require user-agent to be set
        print('File saved as', file_name)
        return file_name

    def stop_model_training_now(self):
        '''Early stop for model training

        '''
        logger.warning('user stop received, model will stop training at epoch end, please wait...')
        if self.stop_cbk is not None:
            self.stop_cbk.stop_me = True
        if self.saver_cbk is not None:
            self.saver_cbk.stop_me = True

    def load_pre_trained_model(self, model, skip_comile=True):
        try:
            import importlib.resources as pkg_resources
        except ImportError:
            # Try backported to PY<37 `importlib_resources`.
            import importlib_resources as pkg_resources


        try:
            with pkg_resources.path(zoo, model) as p:
                package_path = str(p)
                # print('oubsi',package_path)

                if os.path.exists(package_path):
                    logger.debug('loading model from zoo: '+package_path)
                    # print('loading model from zoo: '+package_path)
                    model = self.load_model(model=package_path, skip_comile=skip_comile)
                    return model
                else:
                    logger.error('model not found '+str(model))
        except:
            traceback.print_exc()


    def load_model(self, model=None, skip_comile=False):
        '''loads a model

        Parameters
        ----------
        model : string
            path to the model

        Returns
        -------
        model
            a tf model or None if fails to load the model

        '''

        # bunch of custom objects to allow easy reload especially for the sm models
        # TODO force it to be in sync with metrics and losses
        # TODO should I add more ???
        custom_objects = {"softmax": tf.nn.softmax, "iou_score": sm.metrics.iou_score,
                          'f1_score': f1_score, 'f2_score': f2_score, 'precision': precision,
                          'recall': recall, 'jaccard_loss': jaccard_loss, 'dice_loss': dice_loss,
                          'binary_focal_loss': binary_focal_loss, 'categorical_focal_loss': categorical_focal_loss,
                          'binary_crossentropy': binary_crossentropy,
                          'categorical_crossentropy': categorical_crossentropy,
                          'bce_dice_loss': bce_dice_loss, 'bce_jaccard_loss': bce_jaccard_loss,
                          'cce_dice_loss': cce_dice_loss, 'cce_jaccard_loss': cce_jaccard_loss,
                          'binary_focal_dice_loss': binary_focal_dice_loss,
                          'binary_focal_loss_plus_dice_loss': binary_focal_dice_loss,
                          'binary_focal_jaccard_loss': binary_focal_jaccard_loss,
                          'binary_crossentropy_plus_dice_loss': bce_dice_loss,
                          'binary_focal_plus_jaccard_loss': binary_focal_jaccard_loss,
                          'categorical_focal_dice_loss': categorical_focal_dice_loss,
                          'categorical_focal_jaccard_loss': categorical_focal_jaccard_loss,
                          'binary_crossentropy_plus_jaccard_loss': bce_jaccard_loss,
                          'tf': tf, # smart especially if model uses tf in lambdas
                          }

        if model is not None:
            # TODO maybe first check wether model is in the db because if it is then try to load it
            # will need download and alike
            if model in self.pretrained_models:
                # new way to load the models --> maybe some day do the same with other pretrained models because it is cleaner
                model_parameters = self.pretrained_models[model]
                model_path = model_parameters['model']
                url = None
                try:
                    url = model_parameters['url']
                except:
                    pass
                hash = None
                try:
                    hash = model_parameters['md5']
                except:
                    pass

                # then try to load the weights too --> just duplicate the code before
                # sqdsqdsq
                # if model does not exist --> look for it in the model zoo
                # good idea in fact
                try:
                    model = self.load_pre_trained_model(model_path)
                    # try load weights
                    # model is there --> now try to add the weights
                    if url is not None:
                        # get model name
                        name = os.path.basename(url)
                        # print(name)
                        # print(hash)
                        # if the name exists then can do things
                        try:
                            model_weights = self.get_file(name, url, file_hash=hash, cache_subdir='epyseg')
                            # print(model_weights) # path to the model weights file  # --> really cool all works it seems
                            # just need load the weights and I'm done
                            self.load_weights(model_weights, model=model)
                            logger.info('successfully loaded pretrained model weights')
                        except:
                            traceback.print_exc()
                            logger.error('failed loading model weights')

                    return model
                except:
                    # failed to load the model
                    traceback.print_exc()
                    return None
            elif not model.lower().endswith('.json'):
                # load non JSON models
                if skip_comile:
                    try:
                        model_binary = tf.keras.models.load_model(model, custom_objects=custom_objects,
                                                                  compile=False)
                        return model_binary
                    except:
                        # failed to load the model
                        traceback.print_exc()
                        logger.error('Failed loading model')
                        return None
                else:
                    try:
                        model_binary = tf.keras.models.load_model(model,
                                                                  custom_objects=custom_objects)
                        return model_binary
                    except:
                        traceback.print_exc()
                        logger.error('failed loading model, retrying with compile=False')
                        try:
                            model_binary = tf.keras.models.load_model(model, custom_objects=custom_objects,
                                                                      compile=False)
                            return model_binary
                        except:
                            # failed to load the model
                            traceback.print_exc()
                            logger.error('Failed loading model')
                            return None
            else:
                # if not os.path.exists(model):


                # print('in here', model, model in self.pretrained_models)


                    # load model from a JSON file
                    with open(model, 'r') as f:
                        jsonString = f.read()
                    try:
                        model_binary = tf.keras.models.model_from_json(jsonString,
                                                                       custom_objects=custom_objects)
                        return model_binary
                    except:
                        # failed to load the model
                        traceback.print_exc()
                        logger.error('failed loading model')
                        return None
        # failed to load the model
        return None

    # TODO implement decay rate and allow to set learning rate from GUI
    # from https://gist.github.com/jeremyjordan/86398d7c05c02396c24661baa4c88165
    def step_decay_schedule(self, initial_lr=1e-3, decay_factor=0.1, step_size=50, decay_type=None):
        def schedule(epoch):
            if decay_type is not None and decay_type.startswith('exp'):
                return initial_lr * tf.math.exp(0.1 * (step_size - epoch))
            else:
                # DEFAULT DECAY
                # probably my favourite set up for reducing lr
                return initial_lr * (decay_factor ** np.floor(epoch / step_size))

        return tf.keras.callabcks.LearningRateScheduler(schedule)

    @staticmethod
    def combine(model1, model2):
        '''combine too models

        Combines two sequential models into one, the output of the first model must be compatible with the input of the second model

        Parameters
        ----------
        model1 : model


        model2 : model


        Returns
        -------
        model
            the combined model

        '''
        try:
            return tf.keras.Model(model1.inputs, model2(model1(model1.inputs)))
        except:
            traceback.print_exc()
            logger.error('models could not be combined sorry')  # TODO add more info why that did not work

    def compile(self, optimizer='Adam', loss='binary_crossentropy', metrics=['accuracy'],
                **kwargs):  # added kwargs to ignore extra args without a crash
        '''Compile the model

        Parameters
        ----------
        optimizer : string or optimizer
            the optimizer is the gradient descent algorithm

        loss : string or loss
            the loss function (to be minimized during training)

        metrics : string, list and metrics
            the human readable version of the loss

        '''
        if metrics:
            if not isinstance(metrics, list):
                metrics = [metrics]
        if isinstance(loss, str):
            if loss in self.loss:
                loss = self.loss[loss]
        for idx, metric in enumerate(metrics):
            if isinstance(metric, str):
                if metric in self.metrics:
                    metrics[idx] = self.metrics[metric]

        self.model.compile(optimizer=optimizer, loss=loss, metrics=metrics)


    # --> pb this is irreversible this way --> needs a fix --> would need backup values otherwise --> need think about it
    def force_use_cpu(self):
        '''Force tensorflow to use the CPU instead of GPU even if available

        '''
        # https://stackoverflow.com/questions/40690598/can-keras-with-tensorflow-backend-be-forced-to-use-cpu-or-gpu-at-will
        import os
        #
        # try:
        #     print('os.environ["CUDA_DEVICE_ORDER"]',os.environ["CUDA_DEVICE_ORDER"])
        # except:
        #     print('no entry')
        #     pass
        # try:
        #     print('os.environ["CUDA_VISIBLE_DEVICES"]',os.environ["CUDA_VISIBLE_DEVICES"])
        # except:
        #     print('no entry')
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

        # if hasattr(os, 'unsetenv'):
        # os.unsetenv("CUDA_DEVICE_ORDER")
        # os.unsetenv("CUDA_VISIBLE_DEVICES")


    def load_weights(self, weights, model=None):
        '''Loads model weights

        Parameters
        ----------
        weights : model
            path to model weights

        '''

        if model is None:
            model = self.model

        if weights is not None:
            try:
                logger.info("Loading weights ' " + str(weights) + "'")
                model.load_weights(weights)
            except:
                try:
                    logger.error("Error --> try loading weights by name, i.e. model and weights don't match ???")
                    model.load_weights(weights, by_name=True)
                except:
                    logger.error("Definitely failed loading weights, there is no match between weights and model!!!")

    # TODO could even hack my summary to see the content of nested models, maybe add an option
    def summary(self, line_length=150):
        '''prints a summary of the model

        Parameters
        ----------
        line_length : int
            increase the value if model columns appear truncated

        '''

        if self.model is not None:
            logger.info(self.model.summary(line_length=line_length))
        else:
            logger.error('Please load a model first')

    def plot_graph(self, save_name):
        '''Draws the model as an image

        requires pydot --> need install it
        pip install pydot
        and
        sudo apt install python-pydot python-pydot-ng graphviz # python-pydot-ng isn't in the 14.04 repos

        Parameters
        ----------
        save_name : string
            save the graph as save_name

        '''

        logger.info("Plotting model as .png")
        try:
            tf.keras.utils.plot_model(self.model, to_file=save_name, show_shapes=True, show_layer_names=True)
        except:
            logger.error('failed to save model layout as png')

    def _get_inputs(self):
        # returns model input layers
        return self.model.inputs

    def _get_outputs(self):
        # returns model output layers
        return self.model.outputs

    def get_inputs_shape(self, remove_batch_size_from_shape=False):
        '''Get model input shapes as a list of tuples

        Parameters
        ----------
        remove_batch_size_from_shape : boolean
            if True removes the first/batch size value of the shape tuple

        Returns
        -------
        list of tuples/shapes
            model inputs shapes

        '''

        shapes = []
        inputs = self._get_inputs()
        for input in inputs:
            shape = input.shape.as_list()
            if remove_batch_size_from_shape:
                shape = shape[1:]
            shapes.append(tuple(shape))
        return shapes

    def get_outputs_shape(self, remove_batch_size_from_shape=False):
        '''Get model output shapes as a list of tuples

        Parameters
        ----------
        remove_batch_size_from_shape : boolean
            if True removes the first/batch size value of the shape tuple

        Returns
        -------
        list of tuples/shapes
            model inputs shapes

        '''
        shapes = []
        outputs = self._get_outputs()

        for output in outputs:
            shape = output.shape.as_list()
            if remove_batch_size_from_shape:
                shape = shape[1:]
            shapes.append(tuple(shape))
        return shapes

    # TODO allow recursive and allow index support maybe --> yes do so!!!
    @staticmethod
    def freeze(model, layer_names=None):
        '''Allows to freeze (prevent training) of layers in the model

        Parameters
        ----------
        layer_names : string or regex pattern
            layer name to freeze

        Returns
        -------
        model
            model with the frozen layers

        '''
        if layer_names is None:
            for layer in model.layers:
                layer.trainable = False
        else:
            for layer in model.layers:
                try:
                    model.get_layer(layer_name).trainable = False
                    continue
                except:
                    pass
                for layer_name in layer_names:
                    # try match layer name using regex
                    # TODO maybe also check if layer is a model and then do things recursively in it
                    try:
                        p = re.compile(layer_name)
                        if p.match(layer.name):
                            layer.trainable = False
                    except:
                        logger.error('\'' + str(layer_name) + '\' could not be found in model')
                        pass

    @staticmethod
    def set_trainable(model, layer_names=None):
        '''Set specified layers trainable

        Parameters
        ----------
        layer_names : string or regex pattern
            layer name to freeze

        Returns
        -------
        model
            model with the trainable layers

        '''
        if layer_names is None:
            for layer in model.layers:
                layer.trainable = True
        else:
            for layer in model.layers:
                try:
                    model.get_layer(layer_name).trainable = True
                    continue
                except:
                    pass
                for layer_name in layer_names:
                    # try match layer name using regex
                    # TODO maybe also check if layer is a model and then do things recursively in it
                    try:
                        p = re.compile(layer_name)
                        if p.match(layer.name):
                            layer.trainable = True
                    except:
                        logger.error('\'' + str(layer_name) + '\' could not be found in model')
                        pass

    def get_predict_generator(self, input_shape=(512, 512, 1), output_shape=(512, 512, 1), inputs=None,
                              default_input_tile_width=256, default_input_tile_height=256, tile_width_overlap=32,
                              tile_height_overlap=32,
                              input_normalization={'method': 'Rescaling (min-max normalization)', 'range': [0, 1],
                                                   'individual_channels': True},
                              clip_by_frequency=0.05, **kwargs):

        '''retruns a predict generator used by models for their predictions

        Parameters
        ----------
        input_shape : tuple or list of tuples
            desired image shapes for model input

        output_shape : tuple or list of tuples
            desired output shapes

        inputs : list of strings
            path to input images/folder

        default_input_tile_width : int
            default tile width when None in input shape

        default_input_tile_height : int
            default tile height when None in input shape

        tile_width_overlap : int
            tile overlap along the x axis

        tile_height_overlap : int
            tile overlap along the y axis

        input_normalization : dict
            type of normalisation/standarization to apply to the input image

        clip_by_frequency : float, list of floats or None
            remove hot/cold pixels by intensity frequency

        Returns
        -------
        generator
            an image generator to be used for model predictions

        '''

        logger.debug('inputs for datagen ' + str(input_shape) + ' ' + str(output_shape) + ' ' + str(inputs) + ' ' + str(
            default_input_tile_width) + ' ' + str(default_input_tile_height) + ' ' + str(
            tile_width_overlap) + ' ' + str(tile_height_overlap) + ' ' + str(input_normalization) + ' ' + str(
            clip_by_frequency) + ' ' + str(kwargs))

        if inputs is None:
            logger.error('Please specify a valid input folder to build a predict_generator')
            return

        predict_generator = DataGenerator(inputs=inputs, input_shape=input_shape,
                                          output_shape=output_shape, input_normalization=input_normalization,
                                          clip_by_frequency=clip_by_frequency, is_predict_generator=True,
                                          default_input_tile_width=default_input_tile_width,
                                          default_input_tile_height=default_input_tile_height,
                                          overlap_x=tile_width_overlap,
                                          overlap_y=tile_height_overlap,
                                          **kwargs)
        return predict_generator



    # TODO add a loop with train_on_batch mode
    # TODO check how I can save the settings in a smart way ????
    def train(self, metagenerator, progress_callback=None, output_folder_for_models=None, keep_n_best=5,
              steps_per_epoch=-1, epochs=100,
              batch_size_auto_adjust=False, upon_train_completion_load='last', lr=None, reduce_lr_on_plateau=None,
              patience=10, use_train_on_batch=False,**kwargs):
        '''train the model

        Parameters
        ----------
        metagenerator : datagenerator
            a generator yielding input images and ground/truth output to the loss

        progress_callback : None or a progress displaying object


        output_folder_for_models : string
            path to a folder where model need be saved

        keep_n_best : int
            number of 'best' models to be saved (best = models with lower loss)

        steps_per_epoch : int
            nb of steps per epoch, if < 0 then run training on fullset

        epochs : int
            nb of train epochs

        batch_size_auto_adjust : boolean
            if True, batch size is divided by two every time train fails to run untill batch size reaches 0

        '''

        # try read model name and save right model name
        name = "model"
        if self.model._name is not None:
            name = self.model._name

        # try on an old untouched version
        # this works --> where is the fucking bug ????
        # DEBUG
        # gener = metagenerator.train_generator(infinite=True)
        # for inp, out in  gener:
        #     print('saving')
        #     # en tout cas ça ne marche pas
        #     print(inp[0].shape, out[0].shape)
        #
        #     print(isinstance(inp, tuple), type(inp))
        #     Img(inp[0], dimensions='dhwc').save('/home/aigouy/Bureau/trashme_inp.tif')
        #     Img(out[0], dimensions='dhwc').save('/home/aigouy/Bureau/trashme.tif')
        #     import sys
        #     sys.exit(0)


        # MEGA TODO implement smthg like that NB I GUESS I NEED NOT USE INFINITE IN THIS CASE (In fact I still need it...) --> SEE HOW TO DO THAT THEN
        # if not use_train_on_batch:
        # else:
        #     epoch_num = 0
        #     while epoch_num < epochs:
        #         while iter_num < step_epoch:
        #             x, y = generator.next()
        #             loss_history += self.model..train_on_batch(x, y)
        #             iter_num += 1
        #         epoch_num += 1


        if lr is not None:
            self.set_learning_rate(lr)

        try:
            validation_data = metagenerator.validation_generator(infinite=True)
            validation_steps = metagenerator.get_validation_length(first_run=True)  # use this to generate data
            if reduce_lr_on_plateau is None: # ISN T THIS THE OPPOSITE OF WHAT I WANT --> TODO CHECK SOON
                validation_freq = 5  # checks on validation data every 5 steps # TODO set this as a parameter
            else:
                validation_freq = 1

            # TODO IMPORTANT link on how to set the parameters https://segmentation-models.readthedocs.io/en/latest/api.html#unet
            if validation_steps is None:
                validation_steps = 0

            # TODO VERY IMPORTANT need shuffle if not steps_per_epoch == -1 (fullset) --> TODO
            if steps_per_epoch == -1:
                run_steps_per_epoch = metagenerator.get_train_length(first_run=True)
                logger.info('train dataset batches: ' + str(
                    run_steps_per_epoch) + '\nvalidation dataset batches: ' + str(validation_steps))
            else:
                # TODO VERY IMPORTANT need shuffle if not steps_per_epoch == -1 (fullset) --> TODO
                run_steps_per_epoch = steps_per_epoch
            train_data = metagenerator.train_generator(infinite=True)
        except:
            traceback.print_exc()
            logger.error(
                'Failed to create datagenerators (see log above), training is therefore impossible, sorry...')
            return

        # fake_crash = True

        result = None
        while result is None and metagenerator.batch_size > 0:
            try:
                # if fake_crash:
                #     fake_crash = False
                #     raise Exception('test crash')
                self.stop_cbk = myStopCallback()
                self.saver_cbk = My_saver_callback(name, self, epochs=epochs,
                                                   output_folder_for_models=output_folder_for_models,
                                                   keep_n_best=keep_n_best, progress_callback=progress_callback)
                callbacks = [self.saver_cbk, self.stop_cbk]
                if reduce_lr_on_plateau is not None and reduce_lr_on_plateau < 1:
                    logger.info('Reduce learning rate on plateau is enabled.')
                    monitor = "val_loss"
                    if validation_steps == 0:
                        monitor = 'loss'
                    logger.info('Reduce learning rate is monitoring "' + monitor + '"')

                    self.reduce_lr = ReduceLROnPlateau(monitor=monitor, factor=reduce_lr_on_plateau, patience=patience,
                                                       verbose=1, cooldown=1)
                    # https://stackoverflow.com/questions/51889378/how-to-use-keras-reducelronplateau
                    callbacks.append(self.reduce_lr)
                else:
                    logger.info('Reduce learning rate on plateau is disabled.')

                # if 'reduce_learning_rate' in kwargs and kwargs['reduce_learning_rate']:
                #     # URGENT TODO add parameters such as decay and epoch
                #     reduce_learning_rate = self.step_decay_schedule(
                #         initial_lr=tf.keras.backend.eval(self.model.optimizer.lr))
                #     callbacks.append(reduce_learning_rate) # TODO not great --> change that soon




                if validation_steps != 0:
                    # nb in fact that would probably make sense to use fit_generator in all cases --> can I restore the old code ??
                    if tf.__version__ <= "2.0.0":
                        # hack for tf 2.0.0 support for mac osX (weird bug in tf.keras somewhere)
                        # https://github.com/tensorflow/tensorflow/issues/31231#issuecomment-586630019
                        result = self.model.fit_generator(train_data,
                                                          validation_data=validation_data,
                                                          validation_steps=validation_steps,
                                                          validation_freq=validation_freq,
                                                          steps_per_epoch=run_steps_per_epoch, epochs=epochs,
                                                          callbacks=callbacks,
                                                          verbose=1)
                    else:
                        result = self.model.fit(train_data,
                                                validation_data=validation_data,
                                                validation_steps=validation_steps,
                                                validation_freq=validation_freq,
                                                steps_per_epoch=run_steps_per_epoch, epochs=epochs,
                                                callbacks=callbacks,
                                                verbose=1)
                else:
                    # same as above without validation
                    if tf.__version__ <= "2.0.0":
                        # hack for tf 2.0.0 support for mac osX (weird bug in tf.keras somewhere)
                        # https://github.com/tensorflow/tensorflow/issues/31231#issuecomment-586630019
                        result = self.model.fit_generator(train_data,
                                                          steps_per_epoch=run_steps_per_epoch, epochs=epochs,
                                                          callbacks=callbacks,
                                                          verbose=1)
                    else:
                        result = self.model.fit(train_data,
                                                steps_per_epoch=run_steps_per_epoch, epochs=epochs,
                                                callbacks=callbacks,
                                                verbose=1)

            except:
                traceback.print_exc()
                if batch_size_auto_adjust:
                    metagenerator.batch_size = int(metagenerator.batch_size / 2)
                    if validation_steps != 0:
                        validation_steps = metagenerator.get_validation_length()
                    if steps_per_epoch == -1:
                        run_steps_per_epoch = metagenerator.get_train_length()  # need recompute how many steps there will be because of the batch size reduction by 2
                else:
                    traceback.print_exc()
                    # if user does not want batch size to be adjusted --> quit loop
                    break
                logger.error(
                    'An error occurred but soft did not crash, most likely batch size is too big, giving rise to oom, reducing bacth size to ' + str(
                        metagenerator.batch_size))
                self.clear_mem()

        if result is None:
            logger.error(
                'Something went wrong during the training, if you get oom, you could try to reduce \'tile input width\' and \'tile input height\'')
        else:
            # load best or last model (by default last model is loaded...)
            if upon_train_completion_load == 'best':
                try:
                    path_to_best_model = self.saver_cbk.get_best_model()
                    if path_to_best_model is not None:
                        logger.info("Loading best model '" + str(path_to_best_model) + "'")
                        self.load_or_build(model=path_to_best_model)
                    else:
                        logger.error('No best model found, nothing to load')
                    if self.model is None:
                        logger.critical(
                            'Could not load best model, something wrong happened, please load or build a new model')
                except:
                    traceback.print_exc()
                    logger.error('Failed to load best model upon training completion')
        self.clear_mem()

    # it obviously doesn't work I should just delete the whole object
    def clear_mem(self):
        '''attempt to clear mem on oom TODO test that it really works

        '''
        try:
            tf.keras.backend.clear_session()
            import gc
            gc.collect()
            # print(len(gc.get_objects())) # to see that it really works
        except:
            traceback.print_exc()

    # hq_pred_options --> if all use px deteriorating augs otherwise just use the other --> TODO add this to the code
    def predict(self, datagenerator, output_shapes, progress_callback=None, batch_size=1, predict_output_folder=None,
                hq_predictions='mean', post_process_algorithm=None, append_this_to_save_name='', hq_pred_options='all',
                **kwargs):
        '''run the model

        Parameters
        ----------
        datagenerator : datagenerator
            a generator yielding model input images

        progress_callback : None or a progress displaying object

        batch_size : int
            by setting it to one you do not really affect speed much but you really ensure that no oom occurs

        predict_output_folder : string
            path to a folder where model predictions should be saved

        '''

        # print('passed _output_shape for control', output_shapes)

        logger.debug('hq_predictions mode' + str(hq_predictions))
        logger.debug('hq_pred_options mode' + str(hq_pred_options))
        predict_generator = datagenerator.predict_generator()

        # bckup_predict_output_folder = predict_output_folder
        TA_mode = False
        if predict_output_folder == 'TA_mode':
            TA_mode = True

        if predict_output_folder is None:
            predict_output_folder = ''

        multiple_output = False
        if isinstance(output_shapes, list):
            if len(output_shapes) > 1:
                multiple_output = True
                predict_output_folder = os.path.join(predict_output_folder, 'output_#$#$#')

        self.stop_cbk = myStopCallback()

        for i, (files, crop_parameters) in enumerate(predict_generator):
            try:
                if early_stop.stop == True:
                    return
                if progress_callback is not None:
                    progress_callback.emit((i / len(datagenerator.predict_inputs[0])) * 100)
                else:
                    print(str((i / len(datagenerator.predict_inputs[0])) * 100) + '%')
            except:
                pass

            # allow early stop
            # do I need to do that ??? probably not...
            if self.stop_cbk.stop_me:
                return

            # try:
            # global stop_threads
            # if stop_threads:
            #     print('stop_threads', stop_threads)
            #     return
            # except:
            #     pass

            # we will use this file name to generate the outputname if needed
            filename0 = datagenerator._get_from(datagenerator.predict_inputs, i)[0]
            filename0_without_path = os.path.basename(filename0)
            filename0_without_ext = os.path.splitext(filename0_without_path)[0]
            parent_dir_of_filename0 = os.path.dirname(filename0)

            TA_output_filename = os.path.join(parent_dir_of_filename0, filename0_without_ext,
                                              'epyseg_raw_predict.tif')  # TODO allow custom names here to allow ensemble methods
            non_TA_final_output_name = os.path.join(predict_output_folder, filename0_without_ext + '.tif')

            filename_to_use_to_save = non_TA_final_output_name
            if TA_mode:
                filename_to_use_to_save = TA_output_filename

            # print(files[0].shape)
            # there is a huge bug in the shape of the image... --> why ?

            try:
                if self.stop_cbk.stop_me:
                    return

                    # print('early stop',early_stop.stop)
                if early_stop.stop == True:
                    # print('early stop')
                    return
                results = self.model.predict(files, verbose=1, batch_size=batch_size)
                # tf.keras.backend.clear_session() # quick test to see if it can free up memory # marche pas du tout...

                # print(type(results), len(results)) #<class 'list'> 3

                if hq_predictions is not None:
                    # TODO fix the line below to handle several outputs support --> TODO (do that in a smart way)
                    results = self.get_HQ_predictions(files, results, batch_size=batch_size,
                                                      projection_method=hq_predictions, hq_pred_options=hq_pred_options)# quick test to see if it can free up memory
                    # tf.keras.backend.clear_session()
            except:
                traceback.print_exc()
                logger.error('Could not predict output for image \'' + str(
                    filename0_without_path) + '\', please check it manually. Prediction continues with the next image.')
                continue

            if results is None:
                logger.warning('Prediction interrupted or failed. Stopping...')
                if progress_callback is not None:
                    progress_callback.emit(100)
                return

            if isinstance(results, np.ndarray):  # convert to list if single output
                results = [results]

            # TODO if format is unsupported then try to get a better thing done

            # nb I do miss here a loop over the results to be able to save them all

            # ALMOST THERE BUT STILL NEEDS A BIT OF LOVE THOUGH!!!

            # the two first outputs are actually 2D images and the 3rd is 3D --> almost there in fact
            for cur_count, result in enumerate(results):
                # append_this_to_save_name=str(cur_count)+append_this_to_save_name
                # print('crop parameters', len(crop_parameters))
                for j in range(len(
                        crop_parameters)):  # DO I REALLY NEED THIS LOOP BTW I DOUBT SO --> THINK ABOUT IT AND CHECK HOW MANY TIMES IT RUNS I BET IT ONLY RUNS ONCE BUT MAYBE KEEP IT STILL BECAUSE OK IF RUNS ONLY ONCE!!!
                    # if j == 0:
                    #     print('processing ', cur_count, len(result))
                    #     print(result[j].shape)
                    #     print(crop_parameters[j])
                    #     # cur_output_shape = output_shapes[j] # the bug was here it was not getting the shape properly
                    cur_output_shape = output_shapes[cur_count]
                    #     print('cur output_shape', cur_output_shape)  # make sure it does read it properly there too
                    #     '''
                    #         processing  0 81
                    #         (256, 256, 1)
                    #         {'overlap_y': 32, 'overlap_x': 32, 'final_height': 2048, 'final_width': 2048, 'n_cols': 9, 'n_rows': 9, 'nb_tiles': 81}
                    #         cur output_shape (None, None, None, 1)
                    #         INFO - 2021-03-26 15:32:37,518 - deepl.py - predict - line 1332 - saving file as /E/Sample_images/sample_images_denoise_manue/210324_ON_suz_22h45_armGFP_line2/predict/210324_ON_suz_22h45_armGFP_line2.lif - Series096.tif
                    #         processing  1 81
                    #         (256, 256, 1)
                    #         {'overlap_y': 32, 'overlap_x': 32, 'final_height': 2048, 'final_width': 2048, 'n_cols': 9, 'n_rows': 9, 'nb_tiles': 81}
                    #         cur output_shape (None, None, None, 1)
                    #         INFO - 2021-03-26 15:32:37,737 - deepl.py - predict - line 1332 - saving file as /E/Sample_images/sample_images_denoise_manue/210324_ON_suz_22h45_armGFP_line2/predict/210324_ON_suz_22h45_armGFP_line2.lif - Series096.tif
                    #         processing  2 81
                    #         (35, 256, 256, 1)
                    #         {'overlap_y': 32, 'overlap_x': 32, 'final_height': 2048, 'final_width': 2048, 'n_cols': 9, 'n_rows': 9, 'nb_tiles': 81}
                    #         cur output_shape (None, None, None, 1) # bug is here!!!
                    #         Traceback (most recent call last):
                    #         File "/home/aigouy/mon_prog/Python/epyseg_pkg/epyseg/worker/threaded.py", line 67, in run
                    #         result = self.fn(*self.args, **self.kwargs)
                    #         File "/home/aigouy/mon_prog/Python/epyseg_pkg/epyseg/epygui.py", line 1850, in _predict_using_model
                    #         **predict_parameters)
                    #         File "/home/aigouy/mon_prog/Python/epyseg_pkg/epyseg/deeplearning/deepl.py", line 1298, in predict
                    #         Img(reconstructed_tile, dimensions='hwc').save(filename_to_use_to_save+append_this_to_save_name)
                    #         File "/home/aigouy/mon_prog/Python/epyseg_pkg/epyseg/img.py", line 957, in save
                    #         'dimensions'] is not None and self.has_c() else {})  # small hack to keep only non RGB images as composite and self.get_dimension('c')!=3
                    #         File "/home/aigouy/.local/lib/python3.7/site-packages/tifffile/tifffile.py", line 849, in imwrite
                    #         result = tif.write(data, shape, dtype, **kwargs)
                    #         File "/home/aigouy/.local/lib/python3.7/site-packages/tifffile/tifffile.py", line 1618, in write
                    #         datashape, ijrgb, metadata.get('axes', None)
                    #         File "/home/aigouy/.local/lib/python3.7/site-packages/tifffile/tifffile.py", line 13508, in imagej_shape
                    #         raise ValueError('ImageJ hyperstack is not a grayscale image')
                    #         ValueError: ImageJ hyperstack is not a grayscale image
                    #
                    #
                    #     '''

                    # bug preventing the stuff is there
                    # ordered_tiles = Img.linear_to_2D_tiles(result[j], crop_parameters[j])
                    ordered_tiles = Img.linear_to_2D_tiles(result, crop_parameters[j])

                    # 2D image
                    if len(cur_output_shape) == 4:
                        reconstructed_tile = Img.reassemble_tiles(ordered_tiles, crop_parameters[j])

                        # print('post_process_algorithm', post_process_algorithm)
                        # print(reconstructed_tile.dtype)
                        # print(reconstructed_tile.min())
                        # print(reconstructed_tile.max())
                        # print(reconstructed_tile[50,50])

                        # run post process directly on the image if available
                        if cur_output_shape[-1] != 7 and (post_process_algorithm is not None and (
                                isinstance(post_process_algorithm, str) and not (
                                'imply' in post_process_algorithm or 'first' in post_process_algorithm))):
                            logger.error(
                                'Model is not compatible with epyseg and cannot be optimized, so the desired post processing cannot be applied, sorry...')

                        if isinstance(post_process_algorithm,
                                      str) and 'imply' in post_process_algorithm:  # or cur_output_shape[-1]!=7 # bug why did I put that ??? # if model is incompatible
                            # simply binarise all
                            reconstructed_tile = simpleFilter(Img(reconstructed_tile, dimensions='hwc'), **kwargs)
                            # print('oubsi 1')
                            Img(reconstructed_tile, dimensions='hwc').save(self.hack_name_for_multiple_outputs_models(
                                filename_to_use_to_save + append_this_to_save_name, multiple_output, TA_mode,
                                cur_count))
                            del reconstructed_tile
                        elif post_process_algorithm is not None:
                            try:
                                logger.info('post processing/refining mask, please wait...')
                                # print('post_process_algorithm', post_process_algorithm)
                                reconstructed_tile = self.run_post_process(Img(reconstructed_tile, dimensions='hwc'),
                                                                           post_process_algorithm,
                                                                           progress_callback=progress_callback,
                                                                           **kwargs)
                                if 'epyseg_raw_predict.tif' in filename_to_use_to_save:
                                    filename_to_use_to_save = filename_to_use_to_save.replace('epyseg_raw_predict.tif',
                                                                                              'handCorrection.tif')
                                # print('oubsi 2')

                                # print('bug her"',reconstructed_tile.shape)  # most likely not 2D

                                # Img(reconstructed_tile, dimensions='hw').save(filename_to_use_to_save)
                                Img(reconstructed_tile).save(self.hack_name_for_multiple_outputs_models(
                                    filename_to_use_to_save + append_this_to_save_name, multiple_output, TA_mode,
                                    cur_count))  # TODO check if that fixes bugs
                                del reconstructed_tile
                            except:
                                logger.error('running post processing/refine mask failed')
                                traceback.print_exc()
                        else:
                            # import tifffile
                            # tifffile.imwrite('/home/aigouy/Bureau/201104_armGFP_different_lines_tila/predict/test_direct_save.tif', reconstructed_tile, imagej=True)
                            # print('oubsi 3')
                            # print(filename_to_use_to_save+append_this_to_save_name+str(cur_count)+".tif") # the two first are correct
                            Img(reconstructed_tile, dimensions='hwc').save(self.hack_name_for_multiple_outputs_models(
                                filename_to_use_to_save + append_this_to_save_name, multiple_output, TA_mode,
                                cur_count))  # +str(cur_count)+".tif"
                            del reconstructed_tile
                    else:
                        # 3D image
                        reconstructed_tile = Img.reassemble_tiles(ordered_tiles, crop_parameters[j], three_d=True)
                        # run post process directly on the image if available
                        if cur_output_shape[-1] != 7 and (post_process_algorithm is not None or (
                                isinstance(post_process_algorithm, str) and 'imply' in post_process_algorithm)):
                            logger.error(
                                'Model is not compatible with epyseg and cannot be optimized, so it will simply be thresholded according to selected options, sorry...')
                        if isinstance(post_process_algorithm,
                                      str) and 'imply' in post_process_algorithm:  # or cur_output_shape[-1]!=7 --> there was a bug here ...
                            # simply binarise all
                            # nb that will NOT WORK TODO FIX BUT OK FOR NOW
                            # reconstructed_tile = simpleFilter(Img(reconstructed_tile, dimensions='dhwc'), **kwargs)
                            logger.error('not supported yet please threshold outside the software')
                            Img(reconstructed_tile, dimensions='dhwc').save(self.hack_name_for_multiple_outputs_models(
                                filename_to_use_to_save + append_this_to_save_name, multiple_output, TA_mode,
                                cur_count))
                            del reconstructed_tile
                        elif post_process_algorithm is not None:
                            try:
                                logger.info('post processing/refining mask, please wait...')
                                reconstructed_tile = self.run_post_process(Img(reconstructed_tile, dimensions='dhwc'),
                                                                           post_process_algorithm,
                                                                           progress_callback=progress_callback,
                                                                           **kwargs)
                                if 'epyseg_raw_predict.tif' in filename_to_use_to_save:
                                    filename_to_use_to_save = filename_to_use_to_save.replace('epyseg_raw_predict.tif',
                                                                                              'handCorrection.tif')  # nb java TA does not support 3D masks yet --> maybe do that specifically for the python version
                                Img(reconstructed_tile, dimensions='dhw').save(
                                    self.hack_name_for_multiple_outputs_models(
                                        filename_to_use_to_save + append_this_to_save_name, multiple_output, TA_mode,
                                        cur_count))
                                del reconstructed_tile
                            except:
                                logger.error('running post processing/refine mask failed')
                                traceback.print_exc()
                        else:
                            Img(reconstructed_tile, dimensions='dhwc').save(self.hack_name_for_multiple_outputs_models(
                                filename_to_use_to_save + append_this_to_save_name, multiple_output, TA_mode,
                                cur_count))
                            del reconstructed_tile
                    logger.info('saving file as ' + str(
                        self.hack_name_for_multiple_outputs_models(filename_to_use_to_save + append_this_to_save_name,
                                                                   multiple_output, TA_mode,
                                                                   cur_count)))  # du coup c'est faux
            del results
            gc.collect() # try to force free memory to avoid oom errors
        try:
            if progress_callback is not None:
                progress_callback.emit(100)
            else:
                print(str(100) + '%')
        except:
            pass

    # not sure I gain anything from that, maybe a bit of control
    # in fact I really need a predict gen even with a single image because the image needs be tiled
    # nb this assumes the predict gen contains only one file
    def predict_single(self, single_image_predict_generator, output_shapes, progress_callback=None, cur_progress=None, batch_size=1,
                       hq_predictions='mean', post_process_algorithm=None, hq_pred_options='all',
                       **kwargs):

        results=None
        # multiple_output = False
        # if isinstance(output_shapes, list):
        #     if len(output_shapes) > 1:
        #         multiple_output = True

        self.stop_cbk = myStopCallback()


        if progress_callback is not None and cur_progress is not None:
            progress_callback.emit(cur_progress)
        # for i, (files, crop_parameters) in enumerate(predict_generator):
        #
        #     try:
        #         if progress_callback is not None:
        #             progress_callback.emit((i / len(datagenerator.predict_inputs[0])) * 100)
        #         else:
        #             print(str((i / len(datagenerator.predict_inputs[0])) * 100) + '%')
        #     except:
        #         pass

        # allow early stop
        # do I need to do that ??? probably not...
        if self.stop_cbk.stop_me:
            return

        # global stop_threads
        # if stop_threads:
        #
        #     print('stop_threads', stop_threads)
        #     return

        # we will use this file name to generate the outputname if needed


        # print(files[0].shape)
        # there is a huge bug in the shape of the image... --> why ?

        ########### hack to make it work on a single image --> TODO --> do that in a smarter way some day
        # if isinstance(single_image_predict_generator, np.ndarray):
        #     image = single_image_predict_generator
        #     crop_parameters = None
        # else:
        ############ end hack
        image,crop_parameters = next(single_image_predict_generator)

        try:
            results = self.model.predict(image, verbose=1, batch_size=batch_size)
            # tf.keras.backend.clear_session() # quick test to see if it can free up memory # marche pas du tout...

            # print(type(results), len(results)) #<class 'list'> 3

            if hq_predictions is not None:
                # TODO fix the line below to handle several outputs support --> TODO (do that in a smart way)
                results = self.get_HQ_predictions(image, results, batch_size=batch_size,
                                                  projection_method=hq_predictions, hq_pred_options=hq_pred_options)# quick test to see if it can free up memory
            ######## code for complex support of single images
            # if crop_parameters is None:
            #     return results
            ########## end code for single images
            # tf.keras.backend.clear_session()
        except:
            traceback.print_exc()
            logger.error('Could not predict output for cur image')

        if results is None:
            logger.warning('Prediction interrupted or failed. Stopping...')
            # if progress_callback is not None:
            #     progress_callback.emit(100)
            return

        if isinstance(results, np.ndarray):  # convert to list if single output
            results = [results]

        # TODO if format is unsupported then try to get a better thing done

        # nb I do miss here a loop over the results to be able to save them all

        # ALMOST THERE BUT STILL NEEDS A BIT OF LOVE THOUGH!!!

        # the two first outputs are actually 2D images and the 3rd is 3D --> almost there in fact
        final_output=[]
        for cur_count, result in enumerate(results):
            # append_this_to_save_name=str(cur_count)+append_this_to_save_name
            # print('crop parameters', len(crop_parameters))
            for j in range(len(
                    crop_parameters)):  # DO I REALLY NEED THIS LOOP BTW I DOUBT SO --> THINK ABOUT IT AND CHECK HOW MANY TIMES IT RUNS I BET IT ONLY RUNS ONCE BUT MAYBE KEEP IT STILL BECAUSE OK IF RUNS ONLY ONCE!!!
                # if j == 0:
                #     print('processing ', cur_count, len(result))
                #     print(result[j].shape)
                #     print(crop_parameters[j])
                #     # cur_output_shape = output_shapes[j] # the bug was here it was not getting the shape properly
                cur_output_shape = output_shapes[cur_count]
                #     print('cur output_shape', cur_output_shape)  # make sure it does read it properly there too
                #     '''
                #         processing  0 81
                #         (256, 256, 1)
                #         {'overlap_y': 32, 'overlap_x': 32, 'final_height': 2048, 'final_width': 2048, 'n_cols': 9, 'n_rows': 9, 'nb_tiles': 81}
                #         cur output_shape (None, None, None, 1)
                #         INFO - 2021-03-26 15:32:37,518 - deepl.py - predict - line 1332 - saving file as /E/Sample_images/sample_images_denoise_manue/210324_ON_suz_22h45_armGFP_line2/predict/210324_ON_suz_22h45_armGFP_line2.lif - Series096.tif
                #         processing  1 81
                #         (256, 256, 1)
                #         {'overlap_y': 32, 'overlap_x': 32, 'final_height': 2048, 'final_width': 2048, 'n_cols': 9, 'n_rows': 9, 'nb_tiles': 81}
                #         cur output_shape (None, None, None, 1)
                #         INFO - 2021-03-26 15:32:37,737 - deepl.py - predict - line 1332 - saving file as /E/Sample_images/sample_images_denoise_manue/210324_ON_suz_22h45_armGFP_line2/predict/210324_ON_suz_22h45_armGFP_line2.lif - Series096.tif
                #         processing  2 81
                #         (35, 256, 256, 1)
                #         {'overlap_y': 32, 'overlap_x': 32, 'final_height': 2048, 'final_width': 2048, 'n_cols': 9, 'n_rows': 9, 'nb_tiles': 81}
                #         cur output_shape (None, None, None, 1) # bug is here!!!
                #         Traceback (most recent call last):
                #         File "/home/aigouy/mon_prog/Python/epyseg_pkg/epyseg/worker/threaded.py", line 67, in run
                #         result = self.fn(*self.args, **self.kwargs)
                #         File "/home/aigouy/mon_prog/Python/epyseg_pkg/epyseg/epygui.py", line 1850, in _predict_using_model
                #         **predict_parameters)
                #         File "/home/aigouy/mon_prog/Python/epyseg_pkg/epyseg/deeplearning/deepl.py", line 1298, in predict
                #         Img(reconstructed_tile, dimensions='hwc').save(filename_to_use_to_save+append_this_to_save_name)
                #         File "/home/aigouy/mon_prog/Python/epyseg_pkg/epyseg/img.py", line 957, in save
                #         'dimensions'] is not None and self.has_c() else {})  # small hack to keep only non RGB images as composite and self.get_dimension('c')!=3
                #         File "/home/aigouy/.local/lib/python3.7/site-packages/tifffile/tifffile.py", line 849, in imwrite
                #         result = tif.write(data, shape, dtype, **kwargs)
                #         File "/home/aigouy/.local/lib/python3.7/site-packages/tifffile/tifffile.py", line 1618, in write
                #         datashape, ijrgb, metadata.get('axes', None)
                #         File "/home/aigouy/.local/lib/python3.7/site-packages/tifffile/tifffile.py", line 13508, in imagej_shape
                #         raise ValueError('ImageJ hyperstack is not a grayscale image')
                #         ValueError: ImageJ hyperstack is not a grayscale image
                #
                #
                #     '''

                # bug preventing the stuff is there
                # ordered_tiles = Img.linear_to_2D_tiles(result[j], crop_parameters[j])
                ordered_tiles = Img.linear_to_2D_tiles(result, crop_parameters[j])

                # 2D image
                if len(cur_output_shape) == 4:
                    reconstructed_tile = Img.reassemble_tiles(ordered_tiles, crop_parameters[j])

                    # print('post_process_algorithm', post_process_algorithm)
                    # print(reconstructed_tile.dtype)
                    # print(reconstructed_tile.min())
                    # print(reconstructed_tile.max())
                    # print(reconstructed_tile[50,50])

                    # run post process directly on the image if available
                    if cur_output_shape[-1] != 7 and (post_process_algorithm is not None and (
                            isinstance(post_process_algorithm, str) and not (
                            'imply' in post_process_algorithm or 'first' in post_process_algorithm))):
                        logger.error(
                            'Model is not compatible with epyseg and cannot be optimized, so the desired post processing cannot be applied, sorry...')

                    if isinstance(post_process_algorithm,
                                  str) and 'imply' in post_process_algorithm:  # or cur_output_shape[-1]!=7 # bug why did I put that ??? # if model is incompatible
                        # simply binarise all
                        reconstructed_tile = simpleFilter(Img(reconstructed_tile, dimensions='hwc'), **kwargs)
                        # print('oubsi 1')
                        # Img(reconstructed_tile, dimensions='hwc').save(self.hack_name_for_multiple_outputs_models(
                        #     filename_to_use_to_save + append_this_to_save_name, multiple_output, TA_mode,
                        #     cur_count))
                        # del reconstructed_tile
                        # return reconstructed_tile
                        final_output.append(reconstructed_tile)
                    elif post_process_algorithm is not None:
                        try:
                            logger.info('post processing/refining mask, please wait...')
                            # print('post_process_algorithm', post_process_algorithm)
                            reconstructed_tile = self.run_post_process(Img(reconstructed_tile, dimensions='hwc'),
                                                                       post_process_algorithm,
                                                                       progress_callback=progress_callback,
                                                                       **kwargs)
                            # if 'epyseg_raw_predict.tif' in filename_to_use_to_save:
                            #     filename_to_use_to_save = filename_to_use_to_save.replace('epyseg_raw_predict.tif',
                            #                                                               'handCorrection.tif')
                            # print('oubsi 2')

                            # print('bug her"',reconstructed_tile.shape)  # most likely not 2D

                            # Img(reconstructed_tile, dimensions='hw').save(filename_to_use_to_save)
                            # Img(reconstructed_tile).save(self.hack_name_for_multiple_outputs_models(
                            #     filename_to_use_to_save + append_this_to_save_name, multiple_output, TA_mode,
                            #     cur_count))  # TODO check if that fixes bugs
                            # del reconstructed_tile
                            # return reconstructed_tile
                            final_output.append(reconstructed_tile)
                        except:
                            logger.error('running post processing/refine mask failed')
                            traceback.print_exc()
                    else:
                        # import tifffile
                        # tifffile.imwrite('/home/aigouy/Bureau/201104_armGFP_different_lines_tila/predict/test_direct_save.tif', reconstructed_tile, imagej=True)
                        # print('oubsi 3')
                        # print(filename_to_use_to_save+append_this_to_save_name+str(cur_count)+".tif") # the two first are correct
                        # Img(reconstructed_tile, dimensions='hwc').save(self.hack_name_for_multiple_outputs_models(
                        #     filename_to_use_to_save + append_this_to_save_name, multiple_output, TA_mode,
                        #     cur_count))  # +str(cur_count)+".tif"
                        # del reconstructed_tile
                        # return reconstructed_tile
                        final_output.append(reconstructed_tile)
                else:
                    # 3D image
                    reconstructed_tile = Img.reassemble_tiles(ordered_tiles, crop_parameters[j], three_d=True)
                    # run post process directly on the image if available
                    if cur_output_shape[-1] != 7 and (post_process_algorithm is not None or (
                            isinstance(post_process_algorithm, str) and 'imply' in post_process_algorithm)):
                        logger.error(
                            'Model is not compatible with epyseg and cannot be optimized, so it will simply be thresholded according to selected options, sorry...')
                    if isinstance(post_process_algorithm,
                                  str) and 'imply' in post_process_algorithm:  # or cur_output_shape[-1]!=7 --> there was a bug here ...
                        # simply binarise all
                        # nb that will NOT WORK TODO FIX BUT OK FOR NOW
                        # reconstructed_tile = simpleFilter(Img(reconstructed_tile, dimensions='dhwc'), **kwargs)
                        logger.error('not supported yet please threshold outside the software')
                        # Img(reconstructed_tile, dimensions='dhwc').save(self.hack_name_for_multiple_outputs_models(
                        #     filename_to_use_to_save + append_this_to_save_name, multiple_output, TA_mode,
                        #     cur_count))
                        # del reconstructed_tile
                        # return reconstructed_tile
                        final_output.append(reconstructed_tile)
                    elif post_process_algorithm is not None:
                        try:
                            logger.info('post processing/refining mask, please wait...')
                            reconstructed_tile = self.run_post_process(Img(reconstructed_tile, dimensions='dhwc'),
                                                                       post_process_algorithm,
                                                                       progress_callback=progress_callback,
                                                                       **kwargs)
                            # if 'epyseg_raw_predict.tif' in filename_to_use_to_save:
                            #     filename_to_use_to_save = filename_to_use_to_save.replace('epyseg_raw_predict.tif',
                            #                                                               'handCorrection.tif')  # nb java TA does not support 3D masks yet --> maybe do that specifically for the python version
                            # Img(reconstructed_tile, dimensions='dhw').save(
                            #     self.hack_name_for_multiple_outputs_models(
                            #         filename_to_use_to_save + append_this_to_save_name, multiple_output, TA_mode,
                            #         cur_count))
                            # return reconstructed_tile
                            final_output.append(reconstructed_tile)
                        except:
                            logger.error('running post processing/refine mask failed')
                            traceback.print_exc()
                    else:
                        # Img(reconstructed_tile, dimensions='dhwc').save(self.hack_name_for_multiple_outputs_models(
                        #     filename_to_use_to_save + append_this_to_save_name, multiple_output, TA_mode,
                        #     cur_count))
                        # del reconstructed_tile
                        # return reconstructed_tile
                        final_output.append(reconstructed_tile)
                # logger.info('saving file as ' + str(
                #     self.hack_name_for_multiple_outputs_models(filename_to_use_to_save + append_this_to_save_name,
                #                                                multiple_output, TA_mode,
                #                                                cur_count)))  # du coup c'est faux
        return final_output


    # TODO add tqdm prog bar
    # that may work --> should I try with a small sample
    def step_by_step_training(self, datagenerator, output_shapes, progress_callback=None, batch_size=1, **kwargs):

        # maybe compare the predict gen vs the datagen to see where I do the fusion between consecutive images and do prevent that but keep the rest of the stuff --> should be easy TODO
        # --> maybe simplest is to rely on datagen instead but keep the same random logic as for the train gen --> TODO --> probably not that hard...
        # try models with different inputs too


        # TODO do check this /home/aigouy/mon_prog/Python/epyseg_pkg/personal/train_deepl.py --> ok

        # This is the code based on datagen --> it is also very easy to use --> I should allow models to generate things with different batch size and no batch mix (I could still split the patch --> probably need a few lines of code)
        # try to do that --> in a way that is the simplest model I can do


        # for orig, mask in augmenter.train_generator(False, True):
        #     print('inside loop')
        #     # print('out', len(orig), len(mask))
        #     # print(orig[0].shape, mask[0].shape)
        #     if True:
        #         # the generator ignore exit and runs one more time
        #         print('in')
        #         # just save two images for a test
        #
        #         # why is that called another time ????
        #         print(type(orig[0]))

        # def get_train_length(self, first_run=False):
        #     # need run the train algo once with real tiled data to get the counts
        #     train_generator = self._train_generator(skip_augment=True, first_run=first_run)
        #     nb_batches = 0
        #     for _, _ in train_generator:
        #         nb_batches += 1
        #     return nb_batches


        # also training is easy here because it is just trained one image after the other --> should I just change the code of the train gen so that it does that ??? --> maybe requires a code change
        # In Fact I could also simply rely on the data gen --> loop on them one by one --> pb is if I do that I would not have shuffling which most likely helps in learning


        train_generator = datagenerator.train_generator(infinite=True) # do not need it to be infinite but just need a reset at the end of the stuff ???
        validation_data = datagenerator.validation_generator(infinite=True) # could do validation after n steps --> also very easy TODO --> and could easily save data during training...



        # since images are passed one by one --> I could easily crop them manually --> simpler TODO


        self.stop_cbk = myStopCallback()

        epochs = 3
        step_per_epoch = 5

        # should I rely on datagen instead --> in a way that maybe smarter but then I just need to get the stuff and also assume
        # think about it but I'm almost there
        # I just need to get the optimizer and the loss and so on --> TODO
        # could loop over datagenerators randomly --> make sure not to mix samples from different images --> TODO --> but should not be too hard

        # init loss
        loss_history = 0

        # check what the training generator generates
        # can I also do the tiling in tf directly ??? --> maybe
        for i in range(epochs):
            # for j, x, y in enumerate(train_generator):
            for j in range(step_per_epoch):
                x,y = train_generator.next() # will that work ??? --> maybe
                loss_history += self.model.train_on_batch(x,y) # NB THERE IS SAMPLE WEIGHT --> MAYBE I COULD MAKE SENSE OF THIS TOO TO HANDLE THE DATASETS THAT ARE POORLY REPRESENTED!!!
            print("EPOCH {} FINISHED".format(i + 1))

        # while epoch_num < epochs2:
        #   while iter_num < step_epoch:
        #     x,y = next_batch_train(iter_num)
        #     loss_history += model2.train_on_batch(x,y)
        #
        #     iter_num += 1
        #
        #   print("EPOCH {} FINISHED".format(epoch_num + 1))
        #   epoch_num += 1
        #   iter_num = 0 # reset counter


    # en fait faudrait le faire dès le départ...
    # sauver le nom au depart avec un truc magique à remplacer ici
    def hack_name_for_multiple_outputs_models(self, input_name, multi_output_model, TA_mode, current_output):
        if not multi_output_model:
            return input_name
        else:
            # replace the predict folder or the extension
            # filename_to_use_to_save = input_name.replace('epyseg_raw_predict.tif',
            #                                                               'handCorrection.tif')
            if 'output_#$#$#' in input_name:
                filename_to_use_to_save = input_name.replace('#$#$#', str(current_output))
            else:
                # append text just before the extension
                # --> get filename without the ext
                filename0_without_ext, file_extension = os.path.splitext(input_name)
                filename_to_use_to_save = filename0_without_ext + '_output_' + str(current_output) + file_extension
            return filename_to_use_to_save

    def run_post_process(self, image_to_process, post_process_algorithm, progress_callback=None, **kwargs):
        # do I really need that ???
        # now refine masks if the user wants it, even though it can be done as a post process

        # Default(Moderately fast but robust)')
        # self.post_process_method_selection.addItem('Fast (More errors)')
        # # self.post_process_method_selection.addItem('Slow') # check if I add this ???
        # self.post_process_method_selection.addItem('Old method (Overall less constant, sometimes better)')
        # self.post_process_method_selection.addItem('None (Raw model output)')

        # if does not have 7 ouputs --> deactivate my own post proc and only allow none or simply threshold

        if isinstance(post_process_algorithm, str):

            # print('chosen', post_process_algorithm)
            if 'ld' in post_process_algorithm:
                method = EPySegPostProcess
            # elif 'imply' in post_process_algorithm:
            #     method = SimplyThresholdMask
            else:  # MEGA TODO add parameters with partial according to input
                method = RefineMaskUsingSeeds
        else:
            method = post_process_algorithm

        return method().process(input=image_to_process, mode=post_process_algorithm, **kwargs,
                                progress_callback=progress_callback)

        # if method is EPySegPostProcess:
        #     # if 'filter' in kwargs:
        #     # if TA_mode:
        #     #     kwargs['input'] = kwargs['inputs'][0]
        #     # else:
        #     #     kwargs['input'] = bckup_predict_output_folder
        #     # if TA_mode:
        #     #     kwargs['output_folder'] = bckup_predict_output_folder
        #     # else:
        #     #     kwargs['output_folder'] = os.path.join(bckup_predict_output_folder,'refined_predictions')
        #     return method.process(input=image_to_process, **kwargs, progress_callback=progress_callback)
        # else:
        #     return method(input=image_to_process, **kwargs, progress_callback=progress_callback)

    # nb if it is a list then I need handle it as such...
    # TODO add median as avg_method
    def get_HQ_predictions(self, files, results, batch_size=1,
                           projection_method='mean',
                           hq_pred_options='all'):  # 'max' #'mean' # max_mean # do max for flips and mean for increase contrast
        DEBUG = False  # True
        path = '/D/datasets_deep_learning/keras_segmentation_dataset/TA_test_set/output_models/test_spliiting_augs'
        counter = 1

        logger.info('HQ predictions')
        # interpolation free transformations list
        # none
        # rot90
        # rot 180
        # rot 270
        # rot 90_flipped_hor
        # rot 90_flipped_ver
        # flip hor
        # flip ver --> a combination of 8 images to get perfect segmentation

        # loop over files

        # rotate all in the folder
        # TODO allow for multiple inputs and outputs --> not so easy cause need several loops

        if self.stop_cbk is not None and self.stop_cbk.stop_me:
            return

        if early_stop.stop == True:
            # print('early stop')
            return

        # check if we can rotate by 90 degrees (i.e. if image has same w and height)
        # for odd number of angle rotation need check width and height
        # in fact should check all

        width_and_height_are_always_the_same = True
        for file in files:
            if file.shape[-2] != file.shape[-3]:
                width_and_height_are_always_the_same = False
                break

        # print('width_and_height_are_always_the_same', width_and_height_are_always_the_same)

        if width_and_height_are_always_the_same:
            files2 = []
            for idx, file in enumerate(files):
                if DEBUG:
                    Img(file, dimensions='dhwc').save(
                        os.path.join(os.path.splitext(path)[0], 'orig-' + str(idx) + '.tif'))
                files2.append(np.rot90(file, axes=(-3, -2)))
            results2 = self.model.predict(files2, verbose=1, batch_size=batch_size)
            if self.stop_cbk.stop_me:
                return
            for idx, result in enumerate(results2):
                result = np.rot90(result, 3, axes=(-3, -2))
                if DEBUG:
                    Img(result, dimensions='dhwc').save(
                        os.path.join(os.path.splitext(path)[0], '0-' + str(idx) + '.tif'))
                if projection_method == 'max':
                    results[idx] = np.maximum(results[idx],
                                              result)  # restore original rotation angle # est/ce que ça marche en fait je sais pas --> je peux tenter
                elif projection_method == 'mean':
                    results[idx] += result
            if projection_method == 'mean':
                counter += 1
            del results2

            files3 = []
            for file in files2:
                files3.append(np.flip(file, -2))
            results2 = self.model.predict(files3, verbose=1, batch_size=batch_size)
            if self.stop_cbk.stop_me:
                return
            for idx, result in enumerate(results2):
                result = np.flip(result, -2)
                result = np.rot90(result, 3, axes=(-3, -2))  # restore original rotation angle
                if DEBUG:
                    Img(result, dimensions='dhwc').save(
                        os.path.join(os.path.splitext(path)[0], '1-' + str(idx) + '.tif'))
                if projection_method == 'max':
                    results[idx] = np.maximum(results[idx], result)
                elif projection_method == 'mean':
                    results[idx] += result
            if projection_method == 'mean':
                counter += 1
            del results2

            files3 = []
            for file in files2:
                files3.append(np.flip(file, -3))
            results2 = self.model.predict(files3, verbose=1, batch_size=batch_size)
            if self.stop_cbk.stop_me:
                return
            for idx, result in enumerate(results2):
                result = np.flip(result, -3)
                result = np.rot90(result, 3, axes=(-3, -2))  # restore original rotation angle
                if DEBUG:
                    Img(result, dimensions='dhwc').save(
                        os.path.join(os.path.splitext(path)[0], '2-' + str(idx) + '.tif'))
                if projection_method == 'max':
                    results[idx] = np.maximum(results[idx], result)
                elif projection_method == 'mean':
                    results[idx] += result
            if projection_method == 'mean':
                counter += 1
            del results2
            del files3
            del files2

            files2 = []
            for file in files:
                files2.append(np.rot90(file, 3, axes=(-3, -2)))
            results2 = self.model.predict(files2, verbose=1, batch_size=batch_size)
            if self.stop_cbk.stop_me:
                return
            del files2
            for idx, result in enumerate(results2):
                result = np.rot90(result, axes=(-3, -2))  # restore original rotation angle
                if DEBUG:
                    Img(result, dimensions='dhwc').save(
                        os.path.join(os.path.splitext(path)[0], '3-' + str(idx) + '.tif'))
                if projection_method == 'max':
                    results[idx] = np.maximum(results[idx], result)
                elif projection_method == 'mean':
                    results[idx] += result
            if projection_method == 'mean':
                counter += 1
            del results2
        else:
            logger.warning(
                "Suboptimal HQ predictions. You see this warning because image width and height aren't the same. If the model allows it, please use the same width and height or the same tile width and height to enable HQ mode!")

        # below width and height are kept unchanged so this can always be used
        # 180 degrees rotation is always ok cause does not interchange width and height
        files2 = []
        for file in files:
            files2.append(np.rot90(file, 2, axes=(-3, -2)))
        results2 = self.model.predict(files2, verbose=1, batch_size=batch_size)
        if self.stop_cbk.stop_me:
            return
        del files2
        for idx, result in enumerate(results2):
            result = np.rot90(result, 2, axes=(-3, -2))  # restore original rotation angle
            if DEBUG:
                Img(result, dimensions='dhwc').save(os.path.join(os.path.splitext(path)[0], '4-' + str(idx) + '.tif'))
            if projection_method == 'max':
                results[idx] = np.maximum(results[idx], result)
            elif projection_method == 'mean':
                results[idx] += result
        if projection_method == 'mean':
            counter += 1
        del results2

        # flip hor (check)
        files2 = []
        for file in files:
            files2.append(np.flip(file, -2))
        results2 = self.model.predict(files2, verbose=1, batch_size=batch_size)
        if self.stop_cbk.stop_me:
            return
        del files2
        for idx, result in enumerate(results2):
            result = np.flip(result, -2)  # restore original orientation
            if DEBUG:
                Img(result, dimensions='dhwc').save(os.path.join(os.path.splitext(path)[0], '5-' + str(idx) + '.tif'))
            if projection_method == 'max':
                results[idx] = np.maximum(results[idx], result)
            elif projection_method == 'mean':
                results[idx] += result
        if projection_method == 'mean':
            counter += 1
        del results2

        # flip ver (check)
        files2 = []
        for file in files:
            files2.append(np.flip(file, -3))
        results2 = self.model.predict(files2, verbose=1, batch_size=batch_size)
        if self.stop_cbk.stop_me:
            return
        del files2
        for idx, result in enumerate(results2):
            result = np.flip(result, -3)  # restore original orientation
            if DEBUG:
                Img(result, dimensions='dhwc').save(os.path.join(os.path.splitext(path)[0], '6-' + str(idx) + '.tif'))
            if projection_method == 'max':
                results[idx] = np.maximum(results[idx], result)
            elif projection_method == 'mean':
                results[idx] += result
        if projection_method == 'mean':
            counter += 1
        del results2

        # made px deteriorating augs optional
        skip_deteriorting_augs = False
        if hq_pred_options is not None:
            if isinstance(hq_pred_options, str):
                if not 'all' in hq_pred_options.lower():
                    skip_deteriorting_augs = True

        if not skip_deteriorting_augs:
            logger.debug('Applying pixel deteriorating augs')
            # TODO add now some contrast/intensity augmentations
            try:
                no_negative_values = True
                for file in files:
                    if file.min() < 0:
                        no_negative_values = False
                        break

                # TODO check if works with 3D and with multi channels
                if no_negative_values:
                    # increase contrast 1
                    files2 = []
                    for file in files:
                        v_min, v_max = np.percentile(file, (0.9, 98))
                        files2.append(exposure.rescale_intensity(file, in_range=(v_min, v_max)))
                    results2 = self.model.predict(files2, verbose=1, batch_size=batch_size)
                    if self.stop_cbk.stop_me:
                        return
                    del files2
                    for idx, result in enumerate(results2):
                        if DEBUG:
                            Img(result, dimensions='dhwc').save(
                                os.path.join(os.path.splitext(path)[0], '7-' + str(idx) + '.tif'))
                        if projection_method == 'max':
                            results[idx] = np.maximum(results[idx], result)
                        elif projection_method == 'mean':
                            results[idx] += result
                    if projection_method == 'mean':
                        counter += 1
                    del results2

                    # increase contrast 2
                    files2 = []
                    for file in files:
                        v_min, v_max = np.percentile(file, (5, 95))
                        files2.append(exposure.rescale_intensity(file, in_range=(v_min, v_max)))
                    results2 = self.model.predict(files2, verbose=1, batch_size=batch_size)
                    if self.stop_cbk.stop_me:
                        return
                    del files2
                    for idx, result in enumerate(results2):
                        if DEBUG:
                            Img(result, dimensions='dhwc').save(
                                os.path.join(os.path.splitext(path)[0], '8-' + str(idx) + '.tif'))
                        if projection_method == 'max':
                            results[idx] = np.maximum(results[idx], result)
                        elif projection_method == 'mean':
                            results[idx] += result
                    if projection_method == 'mean':
                        counter += 1
                    del results2

                    # change gamma 2
                    files2 = []
                    for file in files:
                        files2.append(exposure.adjust_gamma(file, gamma=0.8, gain=0.9))
                    results2 = self.model.predict(files2, verbose=1, batch_size=batch_size)
                    if self.stop_cbk.stop_me:
                        return
                    del files2
                    for idx, result in enumerate(results2):
                        if DEBUG:
                            Img(result, dimensions='dhwc').save(
                                os.path.join(os.path.splitext(path)[0], '10-' + str(idx) + '.tif'))
                        if projection_method == 'max':
                            results[idx] = np.maximum(results[idx], result)
                        elif projection_method == 'mean':
                            results[idx] += result
                    if projection_method == 'mean':
                        counter += 1
                    del results2

                    # adjust_log
                    files2 = []
                    for file in files:
                        files2.append(exposure.adjust_log(file))
                    results2 = self.model.predict(files2, verbose=1, batch_size=batch_size)
                    if self.stop_cbk.stop_me:
                        return
                    del files2
                    for idx, result in enumerate(results2):
                        if DEBUG:
                            Img(result, dimensions='dhwc').save(
                                os.path.join(os.path.splitext(path)[0], '11-' + str(idx) + '.tif'))
                        if projection_method == 'max':
                            results[idx] = np.maximum(results[idx], result)
                        elif projection_method == 'mean':
                            results[idx] += result
                    if projection_method == 'mean':
                        counter += 1
                    del results2

                    # adjust_sigmoid 1
                    files2 = []
                    for file in files:
                        files2.append(exposure.adjust_sigmoid(file, gain=5))
                    results2 = self.model.predict(files2, verbose=1, batch_size=batch_size)
                    if self.stop_cbk.stop_me:
                        return
                    del files2
                    for idx, result in enumerate(results2):
                        if DEBUG:
                            Img(result, dimensions='dhwc').save(
                                os.path.join(os.path.splitext(path)[0], '12-' + str(idx) + '.tif'))
                        if projection_method == 'max':
                            results[idx] = np.maximum(results[idx], result)
                        elif projection_method == 'mean':
                            results[idx] += result
                    if projection_method == 'mean':
                        counter += 1
                    del results2
                else:
                    logger.warning(
                        'Suboptimal HQ predictions. You see this warning because your input image contains negative values, therefore some of the data augmentation cannot be performed.')

            except:
                traceback.print_exc()
        else:
            logger.debug('Skipping pixel deteriorating augs')

        # TODO test if ok...
        # QUICK FIX FOR SUPPORT OF MULTIPLE OUTPUTS MODELS/that seems to work but need check it more thoroughly
        if projection_method == 'mean':
            if not isinstance(results, list):
                results /= counter
            else:
                for iii, r in enumerate(results):
                    results[iii] = results[iii] / counter

        return results

    # TODO put this outside of the class
    # TODO ask for a save path
    def saveAsJsonWithWeights(self, model=None):
        # save model as a json file and save weights independently
        if model is None:
            model = self.model
        # serialize model to JSON
        name = "model"
        if model._name is not None:
            name = model._name
        model_json = model.to_json(indent=4)
        with open(name + ".json", "w") as json_file:

            # print(name + ".json")

            json_file.write(model_json)
        # serialize weights to HDF5
        # self.model.save_weights("unet_membrane.hdf5")
        self.saveWeights(model)

    def saveWeights(self, model, name=None):
        # save model weights
        if name is None:
            name = "model"
            if model._name is not None:
                name = model._name
            name += "_weights.h5"
        # print('saving ',name)
        model.save_weights(name)

    # TODO ask for a save path
    def saveModel(self):
        # save model
        name = "model"
        if self.model._name is not None:
            name = self.model._name
        self.model.save(name + '.model')

    # TODO ask for a save path
    def saveModel2(self):
        name = "model"
        if self.model._name is not None:
            name = self.model._name
        # see https://www.tensorflow.org/api_docs/python/tf/keras/models/save_model --> maybe better and can save the optimizer...
        tf.keras.models.save_model(self.model, name, include_optimizer=True,
                                   overwrite=True, save_format='h5')

    # TODO put outside of the class and add a model parameter
    # TODO make it generic so that it can loop on any generator even when they have mutiple inputs and outputs, could be really useful though
    def preview_data_generator(self, data_generator, max_iter=10):
        # show a preview of the image with the given parameter
        count = 0
        for data in data_generator:
            if isinstance(data, tuple):
                # create an array that shows all in that array as a matplotlib stuff
                fig, images = plt.subplots(1, len(data))
                fig.suptitle('Horizontally stacked subplots')
                # TODO handle any dimensions
                for i, img in enumerate(data):
                    img_to_show = np.squeeze(img[0])
                    if len(img_to_show.shape) == 3 or len(img_to_show.shape) == 4:
                        img_to_show = img_to_show[0]
                    images[i].imshow(
                        img_to_show)  # cause if images have batch then they have yet another first dimension...
                plt.show()
            if count >= max_iter:
                break

            count += 1

    # TODO add nested model support and store outside of the class
    def loop_over_layers_and_tell_if_trainable_or_not(self):
        for i, layer in enumerate(self.model.layers):
            print("layer#", i, layer.trainable, layer.name)

    def is_model_compiled(self):
        '''returns True if model is compiled, False otherwise

        '''
        if self.model is None:
            logger.error("Model not loaded, can't check its compilation status...")
            return False
        return self.model.optimizer is not None

    def get_loaded_model_params(self):
        '''prints model optimizer and its parameters

        '''
        try:
            print(self.model.optimizer)
            if self.model.optimizer is None:
                print(
                    'No training configuration found in save file: the model was *not* compiled. Compile it manually.')
                return

            print(tf.keras.backend.eval((self.model.optimizer.lr)))
            print('name', self.model.optimizer._name)

            try:
                # print learning rates, decay and total model iterations
                print('lr', tf.keras.backend.eval(self.model.optimizer.lr))
                print('lr2', tf.keras.backend.eval(self.model.optimizer.decay))
                print('lr3', tf.keras.backend.eval(self.model.optimizer.iterations))
            except:
                pass

            try:
                print('lr4', tf.keras.backend.eval(self.model.optimizer.beta_1))
                print('lr5', tf.keras.backend.eval(self.model.optimizer.beta_2))
            except:
                pass

            print(self.model.optimizer)  # prints the optimizer
            print(
                self.model.optimizer.__dict__)  # this contains a lot of the model infos
            print(self.model.optimizer._hyper)
            print(self.model.optimizer._hyper['learning_rate'])
            print('_iterations', tf.keras.backend.eval(
                self.model.optimizer.__dict__['_iterations']))  # probably the total nb of iterations
            print('learning_rate', tf.keras.backend.eval(self.model.optimizer._hyper['learning_rate']))
            print('decay', tf.keras.backend.eval(self.model.optimizer._hyper['decay']))
            print('beta_1', tf.keras.backend.eval(self.model.optimizer._hyper['beta_1']))
            print('beta_2', tf.keras.backend.eval(self.model.optimizer._hyper['beta_2']))
        except:
            pass

    def set_learning_rate(self, learning_rate):
        if self.model is None:
            logger.error('Please load/build a model first')
            return
        try:
            import tensorflow.keras.backend as K
            K.set_value(self.model.optimizer.learning_rate, learning_rate)
        except:
            traceback.print_exc()
            logger.error('Could not change learning rate, sorry...')


if __name__ == '__main__':
    deepTA = EZDeepLearning()

    if True:
        import sys

        deepTA.load_or_build(model='CARE_GUIDED.json')

        sys.exit(0)



    deepTA.load_or_build(architecture='Unet', backbone='vgg19', activation='sigmoid', classes=1)
    # deepTA.load_or_build(model='/path/to/model.h5')
    deepTA.get_loaded_model_params()
    deepTA.summary()

    print(deepTA._get_inputs())
    print(deepTA._get_outputs())

    print('input shapes', deepTA.get_inputs_shape())
    print('output shapes', deepTA.get_outputs_shape())

    input_shape = deepTA.get_inputs_shape()
    output_shape = deepTA.get_outputs_shape()

    input_normalization = {'method': 'Rescaling (min-max normalization)', 'range': [0, 1],
                           'individual_channels': True}

    # metaAugmenter = MetaAugmenter.get_epithelia_data_augmenter()
    #
    # optimizer = 'adam'  # 'adadelta' # 'adam' #Adam() #keras.optimizers.Adam() #Adam(lr=1e-4) #optimizer='rmsprop' #'sgd' #keras.optimizers.SGD(learning_rate=learning_rate_fn)
    # loss = sm.losses.jaccard_loss #'binary_crossentropy'  # 'binary_crossentropy' #'categorical_crossentropy' #'mean_squared_error'#'mean_squared_error' #sm.losses.bce_jaccard_loss #'binary_crossentropy' #'mean_squared_error'
    # metrics = [sm.metrics.iou_score] # 'accuracy' # ['binary_accuracy'] #[sm.metrics.iou_score] #['accuracy'] ['binary_accuracy'] ['mae']
    #
    # # TRAIN SETTINGS
    # if not deepTA.is_model_compiled():
    #     print('compiling model')
    #     deepTA.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    #
    # NB_EPOCHS = 100  # 80 # 100 # 10
    #
    # deepTA.get_loaded_model_params()
    # deepTA.train(metaAugmenter, epochs=NB_EPOCHS, batch_size_auto_adjust=True)
    #
    # deepTA.saveModel()
    # deepTA.saveAsJsonWithWeights()
    # deepTA.plot_graph(deepTA.model._name + '_graph.png')

    default_input_width = 256  # 576  # 128 # 64
    default_input_height = 256  # 576 # 128 # 64

    predict_generator = deepTA.get_predict_generator(
        inputs=['/home/aigouy/Bureau/last_model_not_sure_that_works/tmp/'], input_shape=input_shape,
        output_shape=output_shape, default_input_tile_width=default_input_width,
        default_input_tile_height=default_input_height,
        tile_width_overlap=32,
        tile_height_overlap=32, input_normalization=input_normalization, clip_by_frequency=0.05)

    predict_output_folder = os.path.join('/D/datasets_deep_learning/keras_segmentation_dataset/TA_test_set/trash',
                                         deepTA.model._name if deepTA.model._name is not None else 'model')  # 'TA_mode'

    deepTA.predict(predict_generator, output_shape, predict_output_folder=predict_output_folder,
                   batch_size=1)
