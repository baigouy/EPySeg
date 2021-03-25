import os

from tensorflow.python.keras.callbacks import ReduceLROnPlateau

from epyseg.postprocess.filtermask import simpleFilter
from epyseg.postprocess.refine_v2 import RefineMaskUsingSeeds

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
# logging
from epyseg.tools.logger import TA_logger

logger = TA_logger()

class EZDeepLearning:
    '''A class to handle deep learning models

    '''
    available_model_architectures = ['Unet', 'PSPNet', 'FPN', 'Linknet']

    optimizers = ['adam', 'sgd', 'rmsprop', 'adagrad', 'adadelta', 'adamax', 'nadam']

    available_sm_backbones = sm.get_available_backbone_names()

    # TODO below are the pretrained models for 2D epithelia segmentation if None --> no pretrained model exist # maybe sort them by efficiency ???
    # for each model do provide all the necessary parameters: 'model' 'model_weights' 'architecture' 'backbone' 'activation' 'classes' 'input_width' 'input_height' 'input_channels'

    pretrained_models_2D_epithelia = {
        'Unet-vgg19-sigmoid': None,
        'Unet-vgg16-sigmoid': None,
        'Unet-seresnext50-sigmoid': None,
        'Unet-seresnext101-sigmoid': None,
        'Unet-seresnet50-sigmoid': None,
        'Unet-seresnet34-sigmoid': None,
        'Unet-seresnet18-sigmoid': None,
        'Unet-seresnet152-sigmoid': None,
        'Unet-seresnet101-sigmoid': None,
        'Unet-senet154-sigmoid': None,
        'Unet-resnext50-sigmoid': None,
        'Unet-resnext101-sigmoid': None,
        'Unet-resnet50-sigmoid': None,
        'Unet-resnet34-sigmoid': None,
        'Unet-resnet18-sigmoid': None,
        'Unet-resnet152-sigmoid': None,
        'Unet-resnet101-sigmoid': None,
        'Unet-mobilenetv2-sigmoid': None,
        'Unet-mobilenet-sigmoid': None,
        'Unet-inceptionv3-sigmoid': None,
        'Unet-inceptionresnetv2-sigmoid': None,
        'Unet-efficientnetb7-sigmoid': None,
        'Unet-efficientnetb6-sigmoid': None,
        'Unet-efficientnetb5-sigmoid': None,
        'Unet-efficientnetb4-sigmoid': None,
        'Unet-efficientnetb3-sigmoid': None,
        'Unet-efficientnetb2-sigmoid': None,
        'Unet-efficientnetb1-sigmoid': None,
        'Unet-efficientnetb0-sigmoid': None,
        'Unet-densenet201-sigmoid': None,
        'Unet-densenet169-sigmoid': None,
        'Unet-densenet121-sigmoid': None,
        'PSPNet-vgg19-sigmoid': None,
        'PSPNet-vgg16-sigmoid': None,
        'PSPNet-seresnext50-sigmoid': None,
        'PSPNet-seresnext101-sigmoid': None,
        'PSPNet-seresnet50-sigmoid': None,
        'PSPNet-seresnet34-sigmoid': None,
        'PSPNet-seresnet18-sigmoid': None,
        'PSPNet-seresnet152-sigmoid': None,
        'PSPNet-seresnet101-sigmoid': None,
        'PSPNet-senet154-sigmoid': None,
        'PSPNet-resnext50-sigmoid': None,
        'PSPNet-resnext101-sigmoid': None,
        'PSPNet-resnet50-sigmoid': None,
        'PSPNet-resnet34-sigmoid': None,
        'PSPNet-resnet18-sigmoid': None,
        'PSPNet-resnet152-sigmoid': None,
        'PSPNet-resnet101-sigmoid': None,
        'PSPNet-mobilenetv2-sigmoid': None,
        'PSPNet-mobilenet-sigmoid': None,
        'PSPNet-inceptionv3-sigmoid': None,
        'PSPNet-inceptionresnetv2-sigmoid': None,
        'PSPNet-efficientnetb7-sigmoid': None,
        'PSPNet-efficientnetb6-sigmoid': None,
        'PSPNet-efficientnetb5-sigmoid': None,
        'PSPNet-efficientnetb4-sigmoid': None,
        'PSPNet-efficientnetb3-sigmoid': None,
        'PSPNet-efficientnetb2-sigmoid': None,
        'PSPNet-efficientnetb1-sigmoid': None,
        'PSPNet-efficientnetb0-sigmoid': None,
        'PSPNet-densenet201-sigmoid': None,
        'PSPNet-densenet169-sigmoid': None,
        'PSPNet-densenet121-sigmoid': None,
        'FPN-vgg19-sigmoid': None,
        'FPN-vgg16-sigmoid': None,
        'FPN-seresnext50-sigmoid': None,
        'FPN-seresnext101-sigmoid': None,
        'FPN-seresnet50-sigmoid': None,
        'FPN-seresnet34-sigmoid': None,
        'FPN-seresnet18-sigmoid': None,
        'FPN-seresnet152-sigmoid': None,
        'FPN-seresnet101-sigmoid': None,
        'FPN-senet154-sigmoid': None,
        'FPN-resnext50-sigmoid': None,
        'FPN-resnext101-sigmoid': None,
        'FPN-resnet50-sigmoid': None,
        'FPN-resnet34-sigmoid': None,
        'FPN-resnet18-sigmoid': None,
        'FPN-resnet152-sigmoid': None,
        'FPN-resnet101-sigmoid': None,
        'FPN-mobilenetv2-sigmoid': None,
        'FPN-mobilenet-sigmoid': None,
        'FPN-inceptionv3-sigmoid': None,
        'FPN-inceptionresnetv2-sigmoid': None,
        'FPN-efficientnetb7-sigmoid': None,
        'FPN-efficientnetb6-sigmoid': None,
        'FPN-efficientnetb5-sigmoid': None,
        'FPN-efficientnetb4-sigmoid': None,
        'FPN-efficientnetb3-sigmoid': None,
        'FPN-efficientnetb2-sigmoid': None,
        'FPN-efficientnetb1-sigmoid': None,
        'FPN-efficientnetb0-sigmoid': None,
        'FPN-densenet201-sigmoid': None,
        'FPN-densenet169-sigmoid': None,
        'FPN-densenet121-sigmoid': None,
        'Linknet-vgg19-sigmoid': None,
        'Linknet-vgg16-sigmoid': {'url': 'https://gitlab.com/baigouy/models/raw/master/model_linknet-vgg16_shells.h5',
                                  # TODO change this
                                  'md5': '266ca9acd9d7a4fe74a473e17952fb6c',
                                  'model': None,
                                  'model_weights': None,
                                  'architecture': 'Linknet',
                                  'backbone': 'vgg16',
                                  'activation': 'sigmoid',
                                  'classes': 7,
                                  'input_width': None,
                                  'input_height': None,
                                  'input_channels': 1,
                                  'version':1},
        'Linknet-vgg16-sigmoid-v2': {'url': 'https://gitlab.com/baigouy/models/raw/master/model_linknet-vgg16_shells_v2.h5',
                                  'md5': '98c8a51f3365e77c07a4f9e95669c259',
                                  'model': None,
                                  'model_weights': None,
                                  'architecture': 'Linknet',
                                  'backbone': 'vgg16',
                                  'activation': 'sigmoid',
                                  'classes': 7,
                                  'input_width': None,
                                  'input_height': None,
                                  'input_channels': 1,
                                  'version': 1},
        'Linknet-seresnext50-sigmoid': None,
        # 'https://github.com/baigouy/models/raw/master/model_Linknet-seresnext101.h5'
        'Linknet-seresnext101-sigmoid': {
            'url': 'https://gitlab.com/baigouy/models/raw/master/model_Linknet-seresnext101.h5',
            'md5': '209f3bf53f3e2f5aaeef62d517e8b8d8',
            'model': None,
            'model_weights': None,
            'architecture': 'Linknet',
            'backbone': 'seresnext101',
            'activation': 'sigmoid',
            'classes': 1,
            'input_width': None,
            'input_height': None,
            'input_channels': 1},
        'Linknet-seresnet50-sigmoid': None,
        'Linknet-seresnet34-sigmoid': None,
        'Linknet-seresnet18-sigmoid': None,
        'Linknet-seresnet152-sigmoid': None,
        'Linknet-seresnet101-sigmoid': None,
        'Linknet-senet154-sigmoid': None,
        'Linknet-resnext50-sigmoid': None,
        'Linknet-resnext101-sigmoid': None,
        'Linknet-resnet50-sigmoid': None,
        'Linknet-resnet34-sigmoid': None,
        'Linknet-resnet18-sigmoid': None,
        'Linknet-resnet152-sigmoid': None,
        'Linknet-resnet101-sigmoid': None,
        'Linknet-mobilenetv2-sigmoid': None,
        'Linknet-mobilenet-sigmoid': None,
        'Linknet-inceptionv3-sigmoid': None,
        'Linknet-inceptionresnetv2-sigmoid': None,
        'Linknet-efficientnetb7-sigmoid': None,
        'Linknet-efficientnetb6-sigmoid': None,
        'Linknet-efficientnetb5-sigmoid': None,
        'Linknet-efficientnetb4-sigmoid': None,
        'Linknet-efficientnetb3-sigmoid': None,
        'Linknet-efficientnetb2-sigmoid': None,
        'Linknet-efficientnetb1-sigmoid': None,
        'Linknet-efficientnetb0-sigmoid': None,
        'Linknet-densenet201-sigmoid': None,
        'Linknet-densenet169-sigmoid': None,
        'Linknet-densenet121-sigmoid': None}

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

    def __init__(self, use_cpu=False):  # TODO handle backbone and type
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

        try:
            physical_devices = tf.config.list_physical_devices('GPU')
        except:
            # dirty hack for tf2.0 support for mac OS X anaconda
            physical_devices = tf.config.experimental.list_physical_devices('GPU')
        for physical_device in physical_devices:
            try:
                tf.config.experimental.set_memory_growth(physical_device, True)
            except:
                # Invalid device or cannot modify.
                pass

        self.stop_cbk = None
        self.saver_cbk = None
        self.model = None

    def get_available_pretrained_models(self):

        available_pretrained_models = []
        for pretrained_model in self.pretrained_models_2D_epithelia.keys():
            if self.pretrained_models_2D_epithelia[pretrained_model] is not None:
                available_pretrained_models.append(pretrained_model)
        return available_pretrained_models

    # encoder_weights=None,
    def load_or_build(self, model=None, model_weights=None, architecture=None, backbone=None,
                      classes=1, activation='sigmoid', input_width=None, input_height=None, input_channels=1,
                      pretraining=None, **kawrgs):
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

                        if pretraining in self.pretrained_models_2D_epithelia:
                            model_parameters = self.pretrained_models_2D_epithelia[pretraining]
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
                          'binary_crossentropy': binary_crossentropy, 'categorical_crossentropy': categorical_crossentropy,
                          'bce_dice_loss': bce_dice_loss, 'bce_jaccard_loss': bce_jaccard_loss,
                          'cce_dice_loss': cce_dice_loss, 'cce_jaccard_loss': cce_jaccard_loss,
                          'binary_focal_dice_loss': binary_focal_dice_loss,
                          'binary_focal_loss_plus_dice_loss': binary_focal_dice_loss,
                          'binary_focal_jaccard_loss': binary_focal_jaccard_loss,
                          'binary_crossentropy_plus_dice_loss': bce_dice_loss,
                          'binary_focal_plus_jaccard_loss': binary_focal_jaccard_loss,
                          'categorical_focal_dice_loss': categorical_focal_dice_loss,
                          'categorical_focal_jaccard_loss': categorical_focal_jaccard_loss,
                          'binary_crossentropy_plus_jaccard_loss': bce_jaccard_loss
                          }

        if model is not None:
            if not model.lower().endswith('.json'):
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
                            model_binary = tf.keras.models.load_model(model, custom_objects=custom_objects, compile=False)
                            return model_binary
                        except:
                            # failed to load the model
                            traceback.print_exc()
                            logger.error('Failed loading model')
                            return None
            else:
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

    def force_use_cpu(self):
        '''Force tensorflow to use the CPU instead of GPU even if available

        '''
        # https://stackoverflow.com/questions/40690598/can-keras-with-tensorflow-backend-be-forced-to-use-cpu-or-gpu-at-will
        import os
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

    def load_weights(self, weights):
        '''Loads model weights

        Parameters
        ----------
        weights : model
            path to model weights

        '''
        if weights is not None:
            try:
                logger.info("Loading weights ' " + str(weights) + "'")
                self.model.load_weights(weights)
            except:
                try:
                    logger.error("Error --> try loading weights by name, i.e. model and weights don't match ???")
                    self.model.load_weights(weights, by_name=True)
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

    # TODO check how I can save the settings in a smart way ????
    def train(self, metagenerator, progress_callback=None, output_folder_for_models=None, keep_n_best=5,
              steps_per_epoch=-1, epochs=100,
              batch_size_auto_adjust=False, upon_train_completion_load='last', lr=None, reduce_lr_on_plateau=None, patience=10,  **kwargs):
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

        if lr is not None:
            self.set_learning_rate(lr)

        try:
            validation_data = metagenerator.validation_generator(infinite=True)
            validation_steps = metagenerator.get_validation_length(first_run=True)  # use this to generate data
            if reduce_lr_on_plateau is None:
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
                        monitor='loss'
                    logger.info('Reduce learning rate is monitoring "'+monitor+'"')
                    self.reduce_lr = ReduceLROnPlateau(monitor=monitor, factor=reduce_lr_on_plateau, patience=patience, verbose=1, cooldown=1)
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
                        run_steps_per_epoch = metagenerator.get_train_length() # need recompute how many steps there will be because of the batch size reduction by 2
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

    def predict(self, datagenerator, output_shapes, progress_callback=None, batch_size=1, predict_output_folder=None,
                hq_predictions='mean', post_process_algorithm=None, append_this_to_save_name='', **kwargs):
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

        logger.debug('hq_predictions mode' + str(hq_predictions))
        predict_generator = datagenerator.predict_generator()

        # bckup_predict_output_folder = predict_output_folder
        TA_mode = False
        if predict_output_folder == 'TA_mode':
            TA_mode = True

        if predict_output_folder is None:
            predict_output_folder = ''

        self.stop_cbk = myStopCallback()

        for i, (files, crop_parameters) in enumerate(predict_generator):

            try:
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

            try:
                results = self.model.predict(files, verbose=1, batch_size=batch_size)
                if hq_predictions is not None:
                    results = self.get_HQ_predictions(files, results, batch_size=batch_size,
                                                      projection_method=hq_predictions)
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

            if isinstance(results, np.ndarray):
                results = [results]

            for j in range(len(crop_parameters)):
                ordered_tiles = Img.linear_to_2D_tiles(results[j], crop_parameters[j])
                output_shape = output_shapes[j]

                if len(output_shape) == 4:
                    reconstructed_tile = Img.reassemble_tiles(ordered_tiles, crop_parameters[j])

                    # print('post_process_algorithm', post_process_algorithm)
                    # print(reconstructed_tile.dtype)
                    # print(reconstructed_tile.min())
                    # print(reconstructed_tile.max())
                    # print(reconstructed_tile[50,50])

                    # run post process directly on the image if available
                    if output_shape[-1]!=7 and (post_process_algorithm is not None and (isinstance(post_process_algorithm, str) and not ('imply' in post_process_algorithm or 'first' in post_process_algorithm))):
                        logger.error('Model is not compatible with epyseg and cannot be optimized, so the desired post processing cannot be applied, sorry...')

                    if isinstance(post_process_algorithm, str) and 'imply' in post_process_algorithm: #  or output_shape[-1]!=7 # bug why did I put that ??? # if model is incompatible
                        # simply binarise all
                        reconstructed_tile = simpleFilter(Img(reconstructed_tile, dimensions='hwc'), **kwargs)
                        # print('oubsi 1')
                        Img(reconstructed_tile, dimensions='hwc').save(filename_to_use_to_save+append_this_to_save_name)
                        del reconstructed_tile
                    elif post_process_algorithm is not None:
                        try:
                            logger.info('post processing/refining mask, please wait...')
                            # print('post_process_algorithm', post_process_algorithm)
                            reconstructed_tile = self.run_post_process(Img(reconstructed_tile, dimensions='hwc'),
                                                                       post_process_algorithm,
                                                                       progress_callback=progress_callback, **kwargs)
                            if 'epyseg_raw_predict.tif' in filename_to_use_to_save:
                                filename_to_use_to_save = filename_to_use_to_save.replace('epyseg_raw_predict.tif',
                                                                                          'handCorrection.tif')
                            # print('oubsi 2')

                            # print('bug her"',reconstructed_tile.shape)  # most likely not 2D

                            # Img(reconstructed_tile, dimensions='hw').save(filename_to_use_to_save)
                            Img(reconstructed_tile).save(filename_to_use_to_save+append_this_to_save_name)  # TODO check if that fixes bugs
                            del reconstructed_tile
                        except:
                            logger.error('running post processing/refine mask failed')
                            traceback.print_exc()
                    else:
                        # import tifffile
                        # tifffile.imwrite('/home/aigouy/Bureau/201104_armGFP_different_lines_tila/predict/test_direct_save.tif', reconstructed_tile, imagej=True)
                        # print('oubsi 3')
                        Img(reconstructed_tile, dimensions='hwc').save(filename_to_use_to_save+append_this_to_save_name)
                        del reconstructed_tile
                else:
                    reconstructed_tile = Img.reassemble_tiles(ordered_tiles, crop_parameters[j], three_d=True)
                    # run post process directly on the image if available
                    if output_shape[-1] != 7 and (post_process_algorithm is not None or (
                            isinstance(post_process_algorithm, str) and 'imply' in post_process_algorithm)):
                        logger.error(
                            'Model is not compatible with epyseg and cannot be optimized, so it will simply be thresholded according to selected options, sorry...')
                    if isinstance(post_process_algorithm, str) and 'imply' in post_process_algorithm: #or output_shape[-1]!=7 --> there was a bug here ...
                        # simply binarise all
                        # nb that will NOT WORK TODO FIX BUT OK FOR NOW
                        # reconstructed_tile = simpleFilter(Img(reconstructed_tile, dimensions='dhwc'), **kwargs)
                        logger.error('not supported yet please threshold outside the software')
                        Img(reconstructed_tile, dimensions='dhwc').save(filename_to_use_to_save+append_this_to_save_name)
                        del reconstructed_tile
                    elif post_process_algorithm is not None:
                        try:
                            logger.info('post processing/refining mask, please wait...')
                            reconstructed_tile = self.run_post_process(Img(reconstructed_tile, dimensions='dhwc'),
                                                                       post_process_algorithm,
                                                                       progress_callback=progress_callback, **kwargs)
                            if 'epyseg_raw_predict.tif' in filename_to_use_to_save:
                                filename_to_use_to_save = filename_to_use_to_save.replace('epyseg_raw_predict.tif',
                                                                                          'handCorrection.tif')  # nb java TA does not support 3D masks yet --> maybe do that specifically for the python version
                            Img(reconstructed_tile, dimensions='dhw').save(filename_to_use_to_save+append_this_to_save_name)
                            del reconstructed_tile
                        except:
                            logger.error('running post processing/refine mask failed')
                            traceback.print_exc()
                    else:
                        Img(reconstructed_tile, dimensions='dhwc').save(filename_to_use_to_save+append_this_to_save_name)
                        del reconstructed_tile
                logger.info('saving file as ' + str(filename_to_use_to_save))
            del results
        try:
            if progress_callback is not None:
                progress_callback.emit(100)
            else:
                print(str(100) + '%')
        except:
            pass

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

    # TODO add median as avg_method
    def get_HQ_predictions(self, files, results, batch_size=1,
                           projection_method='mean'):  # 'max' #'mean' # max_mean # do max for flips and mean for increase contrast
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

        if self.stop_cbk.stop_me:
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
                    results[idx] = np.maximum(results[idx], result)  # restore original rotation angle
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

        if projection_method == 'mean':
            results /= counter

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