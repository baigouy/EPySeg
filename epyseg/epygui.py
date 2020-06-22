import sys

from epyseg.postprocess.gui import PostProcessGUI
from epyseg.postprocess.refine import EPySegPostProcess
from epyseg.worker.fake import FakeWorker
from epyseg.uitools.blinker import Blinker
import os
import logging
import traceback
from epyseg.deeplearning.deepl import EZDeepLearning
import json
from epyseg.deeplearning.augmentation.meta import MetaAugmenter
from epyseg.gui.augmenter import DataAugmentationGUI
from PyQt5.QtWidgets import QListWidgetItem, QAbstractItemView, QSpinBox, QComboBox, QProgressBar, \
    QVBoxLayout, QHBoxLayout, QLabel, QCheckBox, QRadioButton, QButtonGroup, QGroupBox, \
    QTextBrowser
from PyQt5.QtCore import Qt, QThreadPool
from PyQt5.QtGui import QColor, QTextCharFormat, QTextCursor
from PyQt5.QtWidgets import QGridLayout, QListWidget, QFrame, QTabWidget
from epyseg.gui.open import OpenFileOrFolderWidget
from PyQt5.QtWidgets import QApplication
from PyQt5 import QtWidgets, QtCore
from epyseg.worker.threaded import Worker
from epyseg.gui.img import image_input_settings
from epyseg.tools.qthandler import XStream, QtHandler
from epyseg.gui.markdown import PyQT_markdown
from epyseg.img import Img
from PyQt5.QtWidgets import QPushButton, QWidget
# logging
from epyseg.tools.logger import TA_logger

logger = TA_logger()

QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling, True)  # high DPI fix

DEBUG = False  # set to True if GUI crashes
__MAJOR__ = 0
__MINOR__ = 1
__MICRO__ = 0
__RELEASE__ = ''  # a #b  # https://www.python.org/dev/peps/pep-0440/#public-version-identifiers --> alpha beta, ...
__VERSION__ = ''.join([str(__MAJOR__), '.', str(__MINOR__),
                       '.'.join([str(__MICRO__)]) if __MICRO__ != 0 else '', __RELEASE__])
__AUTHOR__ = 'Benoit Aigouy'
__NAME__ = 'EPySeg'


class EPySeg(QWidget):
    '''a deep learning GUI

    '''

    def __init__(self, parent=None):
        '''init for gui

        Parameters
        ----------
        parent : QWidget
            parent window if any (for future dvpt such as tyssue analyzer)

        '''
        self.currently_selected_metrics = []
        super().__init__(parent)
        self.blinker = Blinker()
        self.initUI()
        self.to_blink_after_worker_execution = None
        self.deepTA = EZDeepLearning()  # init empty model

    def initUI(self):
        '''init ui

        '''
        self.threading_enabled = True
        # logging
        self.logger_console = QTextBrowser(self)
        self.logger_console.setReadOnly(True)
        self.logger_console.textCursor().movePosition(QTextCursor.Left, QTextCursor.KeepAnchor, 1)
        self.logger_console.setHtml('<html>')
        self.logger_console.ensureCursorVisible()
        if not DEBUG:
            XStream.stdout().messageWritten.connect(self.set_html_black)
            XStream.stderr().messageWritten.connect(self.set_html_red)
        self.handler = QtHandler()
        self.handler.setFormatter(logging.Formatter(TA_logger.default_format))
        # change default handler for logging
        TA_logger.setHandler(self.handler)

        self.threadpool = QThreadPool()
        self.threadpool.setMaxThreadCount(self.threadpool.maxThreadCount() - 1)  # spare one core for system
        self.setWindowTitle(__NAME__ + ' v' + str(__VERSION__))

        # Initialize tab screen
        self.tabs = QTabWidget(self)
        self.model_tab = QWidget()
        self.train_tab = QWidget()
        self.ensemble_tab = QWidget()
        self.predict_tab = QWidget()
        self.advanced_tab = QWidget()
        self.post_process = QWidget()

        # Add tabs
        self.tabs.addTab(self.model_tab, 'Model')
        self.tabs.addTab(self.train_tab, 'Train')
        self.tabs.setTabEnabled(1, False)  # only activate when model is compiled
        self.tabs.addTab(self.predict_tab, 'Predict')
        self.tabs.setTabEnabled(2, False)  # only activate when model is compiled
        # self.tabs.addTab(self.ensemble_tab, 'Ensemble')  # To combine several outputs/models to improve seg quality
        self.tabs.setTabEnabled(3, False)
        self.ensemble_tab.setVisible(False)
        # self.tabs.addTab(self.advanced_tab, 'Advanced')  # mutate model, feeze, 3D, ... TODO
        self.tabs.setTabEnabled(4, False)
        self.advanced_tab.setVisible(False)
        # self.tabs.setVisible(False)
        self.tabs.addTab(self.post_process, 'Post Process')

        self.tabs.currentChanged.connect(self._onTabChange)

        # creating model tab
        self.model_tab.layout = QGridLayout()
        self.model_tab.layout.setAlignment(Qt.AlignTop)
        self.model_tab.layout.setColumnStretch(0, 25)
        self.model_tab.layout.setColumnStretch(1, 75)
        self.model_tab.layout.setHorizontalSpacing(3)
        self.model_tab.layout.setVerticalSpacing(3)

        # choice between opening an existing model, building a new model or using a pretrained model
        self.build_model_radio = QRadioButton('Build a new model')
        self.load_model_radio = QRadioButton('Load an existing model')
        self.model_pretrain_on_epithelia = QRadioButton('Use a pre-trained model')
        model_build_load_radio_group = QButtonGroup()
        model_build_load_radio_group.addButton(self.load_model_radio)
        model_build_load_radio_group.addButton(self.build_model_radio)
        model_build_load_radio_group.addButton(self.model_pretrain_on_epithelia)
        # self.build_model_radio.setChecked(True)
        self.model_pretrain_on_epithelia.setChecked(True)

        # if 'open an existing model' is selected then provide path to the model
        self.input_model = OpenFileOrFolderWidget(parent_window=self, is_file=True,
                                                  extensions="Supported Files (*.h5 *.H5 *.hdf5 *.HDF5 *.json *.JSON);;All Files (*)")  # TODO check and add .pb (*.pb *.PB)

        # parameters for the pretrained models
        self.groupBox_pretrain = QGroupBox('Pretrained models')
        self.groupBox_pretrain.setEnabled(True)
        # groupBox layout
        self.groupBox_pretrain_layout = QGridLayout()
        self.groupBox_pretrain_layout.setAlignment(Qt.AlignTop)
        self.groupBox_pretrain_layout.setColumnStretch(0, 25)
        self.groupBox_pretrain_layout.setColumnStretch(1, 75)
        self.groupBox_pretrain_layout.setHorizontalSpacing(3)
        self.groupBox_pretrain_layout.setVerticalSpacing(3)

        # populate the list of available pretrained model
        pretrained_label = QLabel('Models trained on 2D epithelia')
        self.pretrained_models = QComboBox()
        for pretrained_model in EZDeepLearning.pretrained_models_2D_epithelia:
            if EZDeepLearning.pretrained_models_2D_epithelia[pretrained_model] is not None:
                self.pretrained_models.addItem(pretrained_model)
        pretrained_label_infos = QLabel(
            'NB: Pretrained models expect a dark backrgound (bg=low intensity), please invert your images if they have a white backrgound.')

        self.groupBox_pretrain_layout.addWidget(pretrained_label, 0, 0)
        self.groupBox_pretrain_layout.addWidget(self.pretrained_models, 0, 1)
        self.groupBox_pretrain_layout.addWidget(pretrained_label_infos, 1, 0, 1, 2)
        self.groupBox_pretrain.setLayout(self.groupBox_pretrain_layout)

        # parameters for the model
        self.groupBox = QGroupBox('Model parameters')
        self.groupBox.setEnabled(False)
        self.input_model.setEnabled(False)

        # enable the right input according to mode
        self.build_model_radio.toggled.connect(self._load_or_build_model_settings)
        self.load_model_radio.toggled.connect(self._load_or_build_model_settings)
        self.model_pretrain_on_epithelia.toggled.connect(self._load_or_build_model_settings)

        # groupBox layout
        self.model_builder_layout = QGridLayout()
        self.model_builder_layout.setAlignment(Qt.AlignTop)
        self.model_builder_layout.setColumnStretch(0, 25)
        self.model_builder_layout.setColumnStretch(1, 75)
        self.model_builder_layout.setHorizontalSpacing(3)
        self.model_builder_layout.setVerticalSpacing(3)

        # Architecture
        model_architecture_label = QLabel('Architecture')
        self.model_architecture = QComboBox()
        for method in EZDeepLearning.available_model_architectures:
            self.model_architecture.addItem(method)
        # add a listener to model Architecture
        self.model_architecture.currentTextChanged.connect(self._architecture_change)

        # backbone/encoder
        model_backbone_label = QLabel('Backbone')
        self.model_backbones = QComboBox()
        for backbone in EZDeepLearning.available_sm_backbones:
            self.model_backbones.addItem(backbone)

        # last layer activation
        model_activation_label = QLabel('Activation (last layer)')
        self.model_last_layer_activation = QComboBox()
        for activation in EZDeepLearning.last_layer_activation:
            self.model_last_layer_activation.addItem(activation)

        # model input width
        model_width_label = QLabel('Input width (0 = None = Any size) (optional)')
        self.model_width = QSpinBox()
        self.model_width.setSingleStep(1)
        self.model_width.setRange(0, 100_000)  # 100_000 makes no sense (oom) but anyway
        self.model_width.setValue(0)

        # TODO could be useful to find closest multiple of a value
        # model input height
        model_height_label = QLabel('Input height (0 = None = Any size) (optional)')
        self.model_height = QSpinBox()
        self.model_height.setSingleStep(1)
        self.model_height.setRange(0, 100_000)  # 100_000 makes no sense (oom) but anyway
        self.model_height.setValue(0)

        # model input nb of channels
        model_channels_label = QLabel('Input channels (1 for gray, 3 for RGB images)')
        self.model_channels = QSpinBox()
        self.model_channels.setSingleStep(1)
        self.model_channels.setRange(1, 100_000)  # 100_000 makes no sense (oom) but anyway
        self.model_channels.setValue(1)

        # nb of classes/indepent semantic segmentations
        model_nb_classes_label = QLabel('Number of classes to predict (output nb of channels)')
        self.nb_classes = QSpinBox()
        self.nb_classes.setSingleStep(1)
        self.nb_classes.setRange(1, 1_000_000)  # nb 1000 would already be a lot but anyway...
        self.nb_classes.setValue(1)

        # parameters for the model
        # model weights optional
        groupBox_weights = QGroupBox('Model weights (can be optional)')
        groupBox_weights.setEnabled(True)

        # groupBox layout
        groupBox_weights_layout = QGridLayout()
        groupBox_weights_layout.setAlignment(Qt.AlignTop)
        groupBox_weights_layout.setColumnStretch(0, 25)
        groupBox_weights_layout.setColumnStretch(1, 75)
        groupBox_weights_layout.setHorizontalSpacing(3)
        groupBox_weights_layout.setVerticalSpacing(3)

        self.input_weights = OpenFileOrFolderWidget(parent_window=self, label_text='Load weights',
                                                    is_file=True,
                                                    extensions="Supported Files (*.h5 *.H5 *.hdf5 *.HDF5);;All Files (*)")

        # arrange groupbox
        self.model_builder_layout.addWidget(model_architecture_label)
        self.model_builder_layout.addWidget(self.model_architecture)
        self.model_builder_layout.addWidget(model_backbone_label)
        self.model_builder_layout.addWidget(self.model_backbones)
        self.model_builder_layout.addWidget(model_width_label)
        self.model_builder_layout.addWidget(self.model_width)
        self.model_builder_layout.addWidget(model_height_label)
        self.model_builder_layout.addWidget(self.model_height)
        self.model_builder_layout.addWidget(model_channels_label)
        self.model_builder_layout.addWidget(self.model_channels)
        self.model_builder_layout.addWidget(model_activation_label)
        self.model_builder_layout.addWidget(self.model_last_layer_activation)
        self.model_builder_layout.addWidget(model_nb_classes_label)
        self.model_builder_layout.addWidget(self.nb_classes)
        self.groupBox.setLayout(self.model_builder_layout)

        # line separator
        line_sep_model = QFrame()
        line_sep_model.setFrameShape(QFrame.HLine)
        line_sep_model.setFrameShadow(QFrame.Sunken)

        # do load/build the model
        self.pushButton2 = QPushButton('Go')
        self.pushButton2.clicked.connect(self.load_or_build_model)

        # arrange tab layout
        combo_hlayout = QHBoxLayout()
        combo_hlayout.addWidget(self.build_model_radio)
        combo_hlayout.addWidget(self.load_model_radio)
        combo_hlayout.addWidget(self.model_pretrain_on_epithelia)
        self.model_tab.layout.addLayout(combo_hlayout, 0, 0, 1, 2)
        self.model_tab.layout.addWidget(self.groupBox_pretrain, 1, 0, 1, 2)
        self.model_tab.layout.addWidget(self.input_model, 2, 0, 1, 2)
        self.model_tab.layout.addWidget(self.groupBox, 3, 0, 1, 2)

        groupBox_weights_layout.addWidget(self.input_weights, 0, 0, 1, 2)
        groupBox_weights.setLayout(groupBox_weights_layout)
        self.model_tab.layout.addWidget(groupBox_weights, 4, 0, 1, 2)

        self.model_tab.layout.addWidget(line_sep_model, 5, 0, 1, 2)
        self.model_tab.layout.addWidget(self.pushButton2, 6, 0, 1, 2)
        self.model_tab.setLayout(self.model_tab.layout)

        # Train tab
        self.train_tab.layout = QGridLayout()
        self.train_tab.layout.setAlignment(Qt.AlignTop)
        self.train_tab.layout.setColumnStretch(0, 25)
        self.train_tab.layout.setColumnStretch(1, 75)
        self.train_tab.layout.setHorizontalSpacing(3)
        self.train_tab.layout.setVerticalSpacing(3)

        # model compilation parameters
        self.groupBox_compile = QGroupBox('Compile model')
        self.groupBox_compile.setEnabled(True)

        # sometimes can be useful to recompile a pretrained model to change learning rate or loss
        self.force_recompile = QCheckBox('Force recompile')
        self.force_recompile.setEnabled(False)
        self.force_recompile.setChecked(False)
        self.force_recompile.stateChanged.connect(
            lambda: self.groupBox_compile.setEnabled(self.force_recompile.isChecked()))

        # model compilation groupBox layout
        groupBox_compile_layout = QGridLayout()
        groupBox_compile_layout.setAlignment(Qt.AlignTop)
        groupBox_compile_layout.setColumnStretch(0, 25)
        groupBox_compile_layout.setColumnStretch(1, 65)
        groupBox_compile_layout.setColumnStretch(2, 5)
        groupBox_compile_layout.setColumnStretch(3, 5)
        groupBox_compile_layout.setHorizontalSpacing(3)
        groupBox_compile_layout.setVerticalSpacing(3)

        # optimizer (gradient descent algortithm)
        optimizer_label = QLabel('Optimizer')
        # put this in a group and deactivate it if model is already compiled ... or offer recompile...
        self.model_optimizers = QComboBox()
        for optimizer in EZDeepLearning.optimizers:
            self.model_optimizers.addItem(optimizer)

        # loss used to update weights (determines how well the model fits the data)
        loss_label = QLabel('Loss')
        self.model_loss = QComboBox()
        for l in EZDeepLearning.loss:
            self.model_loss.addItem(l)

        # metrics: measures how well the model fits the data (not used for backprop)
        metrics_label = QLabel('Metrics')
        self.model_metrics = QComboBox()
        for metric in EZDeepLearning.metrics:
            self.model_metrics.addItem(metric)

        self.add_metric = QPushButton('+')
        self.add_metric.clicked.connect(self._add_selected_metric)
        # remove dataset
        self.remove_metric = QPushButton('-')
        self.remove_metric.clicked.connect(self._remove_selected_metric)

        selected_metrics_label = QLabel('Selected metrics')
        self.selected_metrics = QLabel()

        # ask user where models should be saved
        self.output_models_to = OpenFileOrFolderWidget(parent_window=self, label_text='Output models to')
        from os.path import expanduser
        home = expanduser('~')
        home = os.path.join(home, 'trained_models/')
        self.output_models_to.path.setText(home)

        # freeze encoder or not
        # self.encoder_freeze = QCheckBox('Freeze encoder')  # /pretrain
        # self.encoder_freeze.setChecked(False)
        # self.encoder_freeze.setEnabled(False)  # coming soon

        self.groupBox_training = QGroupBox('Training parameters')
        self.groupBox_training.setEnabled(True)

        # model compilation groupBox layout
        groupBox_training_layout = QGridLayout()
        groupBox_training_layout.setAlignment(Qt.AlignTop)
        groupBox_training_layout.setColumnStretch(0, 25)
        groupBox_training_layout.setColumnStretch(1, 75)
        groupBox_training_layout.setHorizontalSpacing(3)
        groupBox_training_layout.setVerticalSpacing(3)

        # nb of epochs
        nb_epochs_label = QLabel('Epochs')
        self.nb_epochs = QSpinBox()
        self.nb_epochs.setSingleStep(1)
        self.nb_epochs.setRange(0, 1_000_000)  # 1_000_000 makes no sense but anyway
        self.nb_epochs.setValue(100)

        # steps per epoch
        steps_per_epoch_label = QLabel('Steps per epoch (-1 = fullset)')
        self.steps_per_epoch = QSpinBox()
        self.steps_per_epoch.setSingleStep(1)
        self.steps_per_epoch.setRange(-1, 1_000_000)  # 1_000_000 makes no sense but anyway
        self.steps_per_epoch.setValue(-1)

        # batch size
        bs_label = QLabel('Batch size (bs)')
        self.bs = QSpinBox()
        self.bs.setSingleStep(1)
        self.bs.setRange(1, 1_000_000)  # 1_000_000 makes no sense but anyway
        self.bs.setValue(16)
        self.bs_checkbox = QCheckBox('Auto reduce bs on OOM (recommended!)')
        self.bs_checkbox.setChecked(True)

        # keep n best models
        keep_n_best_models_label = QLabel('Keep')
        self.keep_n_best_models = QSpinBox()
        self.keep_n_best_models.setSingleStep(1)
        self.keep_n_best_models.setRange(-1, 100)
        self.keep_n_best_models.setValue(5)
        keep_n_best_models_label2 = QLabel('best models (-1 = keep all)')

        # maybe rather put than in model input
        # self.shuffle = QCheckBox('Shuffle') # maybe reactivate it later but remove for now
        # self.shuffle.setChecked(True)

        # load best or last model once training is completed or when 'stop asap' is pressed
        load_model_upon_completion_of_training = QLabel('Upon completion of training, load the')
        self.load_best_model_upon_completion = QRadioButton('best model')
        self.load_last_model_upon_completion = QRadioButton('last model')
        best_or_last_radio_group = QButtonGroup()
        best_or_last_radio_group.addButton(self.load_best_model_upon_completion)
        best_or_last_radio_group.addButton(self.load_last_model_upon_completion)
        self.load_best_model_upon_completion.setChecked(True)

        train_validation_split_label = QLabel('Validation split (please keep this % low or null)')

        self.validation_split = QSpinBox()
        self.validation_split.setSingleStep(1)
        self.validation_split.setRange(0, 100)
        self.validation_split.setValue(0)

        # Tiling parameters
        self.groupBox_tiling = QGroupBox('Tiling')
        self.groupBox_tiling.setEnabled(True)

        # model compilation groupBox layout
        groupBox_tiling_layout = QGridLayout()
        groupBox_tiling_layout.setAlignment(Qt.AlignTop)
        groupBox_tiling_layout.setColumnStretch(0, 10)
        groupBox_tiling_layout.setColumnStretch(1, 90)
        groupBox_tiling_layout.setHorizontalSpacing(3)
        groupBox_tiling_layout.setVerticalSpacing(3)

        default_tile_width_label = QLabel('Default tile width')
        self.tile_width = QSpinBox()
        self.tile_width.setSingleStep(1)
        self.tile_width.setRange(8, 1_000_000)  # 1_000_000 makes no sense but anyway
        self.tile_width.setValue(256)  # 128 could also be a good default value also
        default_tile_height_label = QLabel('Default tile height')
        self.tile_height = QSpinBox()
        self.tile_height.setSingleStep(1)
        self.tile_height.setRange(8, 1_000_000)  # 1_000_000 makes no sense but anyway
        self.tile_height.setValue(256)  # 128 could also be a good default value also

        # TODO ALSO handle TA architecture of files --> can maybe add that all is ok if TA mode or put TA mode detected

        # request user for its training sets
        self.groupBox_training_dataset = QGroupBox('Training datasets')
        self.groupBox_training_dataset.setEnabled(True)

        # model compilation groupBox layout
        groupBox_training_dataset_layout = QGridLayout()
        groupBox_training_dataset_layout.setAlignment(Qt.AlignTop)
        groupBox_training_dataset_layout.setColumnStretch(0, 10)
        groupBox_training_dataset_layout.setColumnStretch(1, 90)
        groupBox_training_dataset_layout.setHorizontalSpacing(3)
        groupBox_training_dataset_layout.setVerticalSpacing(3)

        # list of training datasets with their preprocessing parameters
        self.list_datasets = QListWidget(self)
        self.list_datasets.setSelectionMode(QAbstractItemView.ExtendedSelection)
        # add dataset
        self.add_dataset = QPushButton('+')
        self.add_dataset.clicked.connect(self._add_data)
        # remove dataset
        self.remove_dataset = QPushButton('-')
        self.remove_dataset.clicked.connect(self._remove_data)

        # request user for its training sets
        self.groupBox_data_aug = QGroupBox('Data augmentation')
        self.groupBox_data_aug.setEnabled(True)

        # model compilation groupBox layout
        groupBox_data_aug_layout = QGridLayout()
        groupBox_data_aug_layout.setAlignment(Qt.AlignTop)
        groupBox_data_aug_layout.setColumnStretch(0, 10)
        groupBox_data_aug_layout.setColumnStretch(1, 90)
        groupBox_data_aug_layout.setHorizontalSpacing(3)
        groupBox_data_aug_layout.setVerticalSpacing(3)

        # set data augmentations (eg, flip, rotate, ...)
        # list of augmentations
        self.list_augmentations = QListWidget()
        self.list_augmentations.setSelectionMode(QAbstractItemView.ExtendedSelection)
        # add new augmentation
        self.add_data_aug = QPushButton('+')
        self.add_data_aug.clicked.connect(self._add_augmenter)
        # remove augmentation
        self.del_data_aug = QPushButton('-')
        self.del_data_aug.clicked.connect(self._delete_augmentations)

        # line separator
        line_sep_train = QFrame()
        line_sep_train.setFrameShape(QFrame.HLine)
        line_sep_train.setFrameShadow(QFrame.Sunken)

        # train or do stop the current training
        self.train = QPushButton('Go (Train model)')
        self.train.clicked.connect(self.compile_model)
        self.stop = QPushButton('Stop ASAP')
        self.stop.setEnabled(self.threading_enabled)
        self.stop.clicked.connect(self._stop_training)

        # arrange groupBox_compile
        groupBox_compile_layout.addWidget(optimizer_label, 0, 0, 1, 1)
        groupBox_compile_layout.addWidget(self.model_optimizers, 0, 1, 1, 1)
        groupBox_compile_layout.addWidget(loss_label, 1, 0, 1, 1)
        groupBox_compile_layout.addWidget(self.model_loss, 1, 1, 1, 1)
        groupBox_compile_layout.addWidget(metrics_label, 2, 0, 1, 1)
        groupBox_compile_layout.addWidget(self.model_metrics, 2, 1, 1, 1)
        groupBox_compile_layout.addWidget(self.add_metric, 2, 2, 1, 1)
        groupBox_compile_layout.addWidget(self.remove_metric, 2, 3, 1, 1)
        groupBox_compile_layout.addWidget(selected_metrics_label, 3, 0, 1, 1)
        groupBox_compile_layout.addWidget(self.selected_metrics, 3, 1, 1, 1)
        self.groupBox_compile.setLayout(groupBox_compile_layout)

        self.train_tab.layout.addWidget(self.force_recompile, 0, 0, 1, 3)
        self.train_tab.layout.addWidget(self.groupBox_compile, 1, 0, 1, 3)

        # arrange dataset layout
        groupBox_training_dataset_layout.addWidget(self.list_datasets, 0, 0, 3, 3)
        groupBox_training_dataset_layout.addWidget(self.add_dataset, 0, 4, 1, 1)
        groupBox_training_dataset_layout.addWidget(self.remove_dataset, 1, 4, 1, 1)
        self.groupBox_training_dataset.setLayout(groupBox_training_dataset_layout)

        # arrange data aug layout
        groupBox_data_aug_layout.addWidget(self.list_augmentations, 21, 0, 3, 3)
        groupBox_data_aug_layout.addWidget(self.add_data_aug, 21, 4, 1, 1)
        groupBox_data_aug_layout.addWidget(self.del_data_aug, 22, 4, 1, 1)
        self.groupBox_data_aug.setLayout(groupBox_data_aug_layout)

        group_dataset_n_aug = QHBoxLayout()
        group_dataset_n_aug.addWidget(self.groupBox_training_dataset)
        group_dataset_n_aug.addWidget(self.groupBox_data_aug)
        self.train_tab.layout.addLayout(group_dataset_n_aug, 3, 0, 1, 3)

        # tiling parameters
        groupBox_tiling_layout.addWidget(default_tile_width_label, 9, 0, 1, 1)
        groupBox_tiling_layout.addWidget(self.tile_width, 9, 1, 1, 2)
        groupBox_tiling_layout.addWidget(default_tile_height_label, 10, 0, 1, 1)
        groupBox_tiling_layout.addWidget(self.tile_height, 10, 1, 1, 2)
        self.groupBox_tiling.setLayout(groupBox_tiling_layout)
        self.train_tab.layout.addWidget(self.groupBox_tiling, 9, 0, 1, 3)

        # THIS IS NOT REALLY BEAUTIFUL SHOULD UNPACK STUFF BUT OK FOR NOW --> WASTE TIME MAKING IT APPEAR NICER WHEN ALL IS DONE AND I HAVE NOTHING ELSE TO DO
        self.groupBox_input_output_normalization_method = QGroupBox('Normalization')
        self.groupBox_input_output_normalization_method.setEnabled(True)
        groupBox_input_output_normalization_method_layout = QGridLayout()
        groupBox_input_output_normalization_method_layout.setContentsMargins(0, 0, 0, 0)

        self.input_output_normalization_method = image_input_settings(parent_window=self,
                                                                      show_normalization=True,
                                                                      show_preview=False)
        groupBox_input_output_normalization_method_layout.addWidget(self.input_output_normalization_method)
        self.groupBox_input_output_normalization_method.setLayout(groupBox_input_output_normalization_method_layout)
        self.train_tab.layout.addWidget(self.groupBox_input_output_normalization_method, 12, 0, 1, 3)

        # arrange train tab
        groupBox_training_layout.addWidget(self.output_models_to, 3, 0, 1, 4)
        # groupBox_training_layout.addWidget(self.encoder_freeze, 4, 0, 1, 3) # coming soon
        groupBox_training_layout.addWidget(nb_epochs_label, 5, 0)
        groupBox_training_layout.addWidget(self.nb_epochs, 5, 1)
        groupBox_training_layout.addWidget(steps_per_epoch_label, 5, 2)
        groupBox_training_layout.addWidget(self.steps_per_epoch, 5, 3)

        groupBox_training_layout.addWidget(bs_label, 7, 0, 1, 1)
        groupBox_training_layout.addWidget(self.bs, 7, 1, 1, 1)
        groupBox_training_layout.addWidget(self.bs_checkbox, 7, 2, 1, 2)

        groupBox_training_layout.addWidget(keep_n_best_models_label, 8, 0, 1, 1)
        groupBox_training_layout.addWidget(self.keep_n_best_models, 8, 1, 1, 1)
        groupBox_training_layout.addWidget(keep_n_best_models_label2, 8, 2, 1, 2)

        # groupBox_training_layout.addWidget(self.shuffle)

        groupBox_training_layout.addWidget(load_model_upon_completion_of_training, 9, 0, 1, 1)
        groupBox_training_layout.addWidget(self.load_best_model_upon_completion, 9, 1, 1, 1)
        groupBox_training_layout.addWidget(self.load_last_model_upon_completion, 9, 2, 1, 1)

        groupBox_training_layout.addWidget(train_validation_split_label, 10, 0, 1, 2)
        groupBox_training_layout.addWidget(self.validation_split, 10, 2, 1, 2)

        self.groupBox_training.setLayout(groupBox_training_layout)
        self.train_tab.layout.addWidget(self.groupBox_training, 21, 0, 1, 4)

        self.train_tab.layout.addWidget(line_sep_train, 26, 0, 1, 3)
        self.train_tab.layout.addWidget(self.train, 27, 0, 1, 2)
        self.train_tab.layout.addWidget(self.stop, 27, 2)

        self.train_tab.setLayout(self.train_tab.layout)

        # predict tab
        self.predict_tab.layout = QGridLayout()
        self.predict_tab.layout.setAlignment(Qt.AlignTop)
        self.predict_tab.layout.setColumnStretch(0, 75)
        self.predict_tab.layout.setColumnStretch(1, 25)
        self.predict_tab.layout.setHorizontalSpacing(3)
        self.predict_tab.layout.setVerticalSpacing(3)

        # ask for data to feed the model
        self.set_custom_predict_parameters = image_input_settings(show_input=True,
                                                                  show_channel_nb_change_rules=True,
                                                                  show_normalization=True,
                                                                  show_tiling=True,
                                                                  show_overlap=True,
                                                                  show_predict_output=True,
                                                                  input_mode_only=True,
                                                                  show_preview=True,
                                                                  show_HQ_settings=True,
                                                                  show_run_post_process=True)

        line_sep_predict = QFrame()
        line_sep_predict.setFrameShape(QFrame.HLine)
        line_sep_predict.setFrameShadow(QFrame.Sunken)

        # launch predictions
        self.predict = QPushButton('Go (Predict)')
        self.predict.clicked.connect(self.predict_using_model)

        self.stop2 = QPushButton('Stop ASAP')
        self.stop2.setEnabled(self.threading_enabled)
        self.stop2.clicked.connect(self._stop_training)

        self.predict_tab.layout.addWidget(self.set_custom_predict_parameters, 0, 0, 1, 2)
        # self.predict_tab.layout.addWidget(groupBox_post_process, 0, 0, 4, 1)

        self.predict_tab.layout.addWidget(line_sep_predict, 8, 0, 1, 2)
        self.predict_tab.layout.addWidget(self.predict, 9, 0)
        self.predict_tab.layout.addWidget(self.stop2, 9, 1)
        self.predict_tab.setLayout(self.predict_tab.layout)

        # predict tab
        self.post_process.layout = QGridLayout()
        self.post_process.layout.setAlignment(Qt.AlignTop)
        self.post_process.layout.setColumnStretch(0, 25)
        self.post_process.layout.setColumnStretch(1, 75)
        self.post_process.layout.setHorizontalSpacing(3)
        self.post_process.layout.setVerticalSpacing(3)

        # Post processing tab to refine the watershed masks from raw data
        self.groupBox_post_process = PostProcessGUI(parent_window=self)
        # self.set_custom_predict_parameters.setVisible(False)

        self.input_output_normalization_method = image_input_settings(parent_window=self,
                                                                      show_normalization=True,
                                                                      show_preview=False)

        line_sep_predict3 = QFrame()
        line_sep_predict3.setFrameShape(QFrame.HLine)
        line_sep_predict3.setFrameShadow(QFrame.Sunken)

        # launch predictions
        self.post_proc = QPushButton('Go (Refine masks)')
        self.post_proc.clicked.connect(self.run_post_process)

        self.stop_post_process = QPushButton('Stop ASAP')
        self.stop_post_process.setEnabled(self.threading_enabled)
        self.stop_post_process.clicked.connect(self._stop_post_process)

        # Post processing tab to refine the watershed masks from raw data
        self.set_custom_post_process = image_input_settings(show_input=True,
                                                            show_channel_nb_change_rules=False,
                                                            show_normalization=False,
                                                            show_tiling=False,
                                                            show_overlap=False,
                                                            show_predict_output=True,
                                                            input_mode_only=True,
                                                            show_preview=False, label_input='Path to EPySeg raw output')

        self.post_process.layout.addWidget(self.set_custom_post_process, 0, 0, 4, 4)
        self.post_process.layout.addWidget(self.groupBox_post_process, 5, 0, 4, 4)
        self.post_process.layout.addWidget(line_sep_predict3, 10, 0, 1, 4)
        self.post_process.layout.addWidget(self.post_proc, 11, 0, 1, 3)
        self.post_process.layout.addWidget(self.stop_post_process, 11, 3, 1, 1)
        self.post_process.setLayout(self.post_process.layout)
        # see what it will do

        # widget global layout
        table_widget_layout = QVBoxLayout()
        table_widget_layout.addWidget(self.tabs)

        log_and_main_layout = QHBoxLayout()

        # TODO put this in a group to get the stuff
        log_groupBox = QGroupBox('Log')
        log_groupBox.setEnabled(True)

        help = PyQT_markdown()
        try:
            this_dir, this_filename = os.path.split(__file__)
            help.set_markdown_from_file(os.path.join(this_dir, 'deeplearning/docs', 'getting_started2.md'),
                                        title='getting started: predict using pre-trained network')
            help.set_markdown_from_file(os.path.join(this_dir, 'deeplearning/docs', 'getting_started.md'),
                                        title='getting started: build and train a custom network')
            help.set_markdown_from_file(os.path.join(this_dir, 'deeplearning/docs', 'pretrained_model.md'),
                                        title='Load a pre-trained model')
            help.set_markdown_from_file(os.path.join(this_dir, 'deeplearning/docs', 'model.md'),
                                        title='Build a model from scratch')
            help.set_markdown_from_file(os.path.join(this_dir, 'deeplearning/docs', 'load_model.md'),
                                        title='Load a model')
            help.set_markdown_from_file(os.path.join(this_dir, 'deeplearning/docs', 'train.md'), title='Train')
            help.set_markdown_from_file(os.path.join(this_dir, 'deeplearning/docs', 'predict.md'), title='Predict')
            help.set_markdown_from_file(os.path.join(this_dir, 'deeplearning/docs', 'training_dataset.md'),
                                        title='Training dataset parameters')
            help.set_markdown_from_file(os.path.join(this_dir, 'deeplearning/docs', 'data_augmentation.md'),
                                        title='Data augmentation')
            # TODO prevent this window to be resize upon tab changes
        except:
            traceback.print_exc()

        # Initialize tab screen
        self.help_tabs = QTabWidget(self)
        self.help_tabs.setMinimumWidth(500)
        self.log_tab = QWidget()
        self.help_html_tab = QWidget()
        self.settings_GUI = QWidget()

        # Add tabs
        self.help_tabs.addTab(self.log_tab, 'Log')
        self.help_tabs.addTab(self.help_html_tab, 'Help')
        self.help_tabs.addTab(self.settings_GUI, 'GUI settings')

        # creating model tab
        self.log_tab.layout = QVBoxLayout()
        self.log_tab.layout.setAlignment(Qt.AlignTop)

        # global system progress bar
        self.pbar = QProgressBar(self)

        self.log_tab.layout.addWidget(self.logger_console)
        self.log_tab.layout.addWidget(self.pbar)
        self.log_tab.setLayout(self.log_tab.layout)

        self.help_html_tab.layout = QVBoxLayout()
        self.help_html_tab.layout.setContentsMargins(0, 0, 0, 0)
        self.help_html_tab.layout.setAlignment(Qt.AlignTop)
        self.help_html_tab.layout.addWidget(help)
        self.help_html_tab.setLayout(self.help_html_tab.layout)

        self.enable_threading_check = QCheckBox(
            'Threading enable/disable')
        self.enable_threading_check.setChecked(True)
        self.enable_threading_check.stateChanged.connect(self._set_threading)

        self.settings_GUI.layout = QVBoxLayout()
        self.settings_GUI.layout.setAlignment(Qt.AlignTop)
        self.settings_GUI.layout.addWidget(self.enable_threading_check)
        self.settings_GUI.setLayout(self.settings_GUI.layout)

        log_and_main_layout.addLayout(table_widget_layout)
        log_and_main_layout.addWidget(self.help_tabs)

        self.setLayout(log_and_main_layout)

        try:
            screen = QApplication.desktop().screenNumber(QApplication.desktop().cursor().pos())
            centerPoint = QApplication.desktop().screenGeometry(screen).center()
            self.setGeometry(QtCore.QRect(centerPoint.x() - self.width(), centerPoint.y() - self.height(), self.width(),
                                          self.height()))
        except:
            pass
        self.show()

    def set_html_red(self, text):
        # quick n dirty log coloring --> improve when I have time
        textCursor = self.logger_console.textCursor()
        textCursor.movePosition(QTextCursor.End)
        self.logger_console.setTextCursor(textCursor)
        format = QTextCharFormat()
        format.setForeground(QColor(255, 0, 0))  # red
        self.logger_console.setCurrentCharFormat(format)
        self.logger_console.insertPlainText(text)
        self.logger_console.verticalScrollBar().setValue(self.logger_console.verticalScrollBar().maximum())

    def set_html_black(self, text):
        # quick n dirty log coloring --> improve when I have time
        textCursor = self.logger_console.textCursor()
        textCursor.movePosition(QTextCursor.End)
        self.logger_console.setTextCursor(textCursor)
        format = QTextCharFormat()
        format.setForeground(QColor(0, 0, 0))  # black
        self.logger_console.setCurrentCharFormat(format)
        self.logger_console.insertPlainText(text)

    def print_output(self, s):
        print(s)

    def _set_threading(self):
        self.threading_enabled = self.enable_threading_check.isChecked()
        # disable early stop if threading not enabled otherwise do allow it
        self.stop.setEnabled(self.threading_enabled)
        self.stop2.setEnabled(self.threading_enabled)

    def thread_complete(self):
        '''Called every time a thread completed

        I use it to blink things in case there are errors

        '''
        # reset progress upon thread complete
        self.pbar.setValue(0)

        if self.to_blink_after_worker_execution is not None:
            self.blinker.blink(self.to_blink_after_worker_execution)
            self.to_blink_after_worker_execution = None

    def progress_fn(self, current_progress):
        '''basic progress function

        '''
        print("%d%% done" % current_progress)
        self.pbar.setValue(current_progress)

    def get_model_parameters(self):
        '''gets the parameters of the model

        Returns
        -------
        dict
            containing model parameters

        '''
        self.model_parameters = {}
        if self.load_model_radio.isChecked():
            self.model_parameters['model'] = self.input_model.text()  # can be None if empty
            self.model_parameters['model_weights'] = self.input_weights.text()  # can be None if empty
            self.model_parameters['architecture'] = None
            self.model_parameters['backbone'] = None
            self.model_parameters['activation'] = None
            self.model_parameters['classes'] = None
            self.model_parameters['input_width'] = None
            self.model_parameters['input_height'] = None
            self.model_parameters['input_channels'] = None
            self.model_parameters['pretraining'] = None
        elif self.build_model_radio.isChecked():
            self.model_parameters['model'] = None
            self.model_parameters['model_weights'] = self.input_weights.text()  # can be None if empty
            self.model_parameters['architecture'] = self.model_architecture.currentText()
            self.model_parameters['backbone'] = self.model_backbones.currentText()
            self.model_parameters['activation'] = self.model_last_layer_activation.currentText()
            self.model_parameters['classes'] = self.nb_classes.value()
            self.model_parameters['input_width'] = self.model_width.value()
            self.model_parameters['input_height'] = self.model_height.value()
            self.model_parameters['input_channels'] = self.model_channels.value()
            self.model_parameters['pretraining'] = None
        else:
            # load pretrained model
            pretrained_model_parameters = self.deepTA.pretrained_models_2D_epithelia[
                self.pretrained_models.currentText()]
            self.model_parameters['model'] = pretrained_model_parameters['model']
            self.model_parameters['model_weights'] = pretrained_model_parameters['model_weights']
            self.model_parameters['architecture'] = pretrained_model_parameters['architecture']
            self.model_parameters['backbone'] = pretrained_model_parameters['backbone']
            self.model_parameters['activation'] = pretrained_model_parameters['activation']
            self.model_parameters['classes'] = pretrained_model_parameters['classes']
            self.model_parameters['input_width'] = pretrained_model_parameters['input_width']
            self.model_parameters['input_height'] = pretrained_model_parameters['input_height']
            self.model_parameters['input_channels'] = pretrained_model_parameters['input_channels']
            self.model_parameters['pretraining'] = self.pretrained_models.currentText()
            # except:
            #     traceback.print_exc()
            #     logger.error('could not load url of pretrained model, please check pretraining parameters')
            #     self.model_parameters['pretraining'] = None
            #
            # print('pretraining',self.model_parameters['pretraining'])

        return self.model_parameters

    def get_post_process_parameters(self):
        '''Get the parameters for model training

        Returns
        -------
        dict
            containing training parameters

        '''

        # TODO fix it that it really gets the parameters and do not save as predict but as a different name ??? or keep indeed predict as a name
        self.post_process_parameters = self.set_custom_post_process.get_parameters_directly()
        if 'inputs' in self.post_process_parameters:
            self.post_process_parameters['input'] = self.post_process_parameters['inputs'][0]
        if 'predict_output_folder' in self.post_process_parameters:
            self.post_process_parameters['output_folder'] = self.post_process_parameters['predict_output_folder']
        self.post_process_parameters.update(self.groupBox_post_process.get_parameters_directly())

        if DEBUG:
            print('post proc params', self.post_process_parameters)

        return self.post_process_parameters

    def get_train_parameters(self):
        '''Get the parameters for model training

        Returns
        -------
        dict
            containing training parameters

        '''

        self.train_parameters = {}
        self.train_parameters['optimizer'] = self.model_optimizers.currentText()
        self.train_parameters['loss'] = self.model_loss.currentText()
        self.train_parameters['metrics'] = self.currently_selected_metrics if self.currently_selected_metrics else [
            self.model_metrics.currentText()]
        self.train_parameters['datasets'] = self.get_list_of_datasets()
        self.train_parameters['augmentations'] = self.get_list_of_augmentations()
        output = self.input_output_normalization_method.get_parameters_directly()
        self.train_parameters['input_normalization'] = output['input_normalization']
        self.train_parameters['output_normalization'] = output['output_normalization']
        self.train_parameters['epochs'] = self.nb_epochs.value()
        self.train_parameters['steps_per_epoch'] = self.steps_per_epoch.value()
        self.train_parameters['default_input_tile_width'] = self.tile_width.value()
        self.train_parameters['default_input_tile_height'] = self.tile_height.value()
        self.train_parameters['default_output_tile_width'] = self.tile_width.value()
        self.train_parameters['default_output_tile_height'] = self.tile_height.value()
        self.train_parameters['output_folder_for_models'] = self.output_models_to.text()
        # self.train_parameters['freeze_encoder'] = self.encoder_freeze.isChecked() #TODO
        self.train_parameters['keep_n_best'] = self.keep_n_best_models.value()
        self.train_parameters[
            'shuffle'] = False  # self.shuffle.isChecked() TODO maybe reactivate, but optimize code first
        self.train_parameters['batch_size'] = self.bs.value()
        self.train_parameters['batch_size_auto_adjust'] = self.bs_checkbox.isChecked()
        self.train_parameters['clip_by_frequency'] = self.input_output_normalization_method.get_clip_by_freq()
        self.train_parameters['validation_split'] = self.validation_split.value()
        # TODO add a parameter for 'upon_train_completion_load' 'best' or 'last' model
        self.train_parameters[
            'upon_train_completion_load'] = 'best' if self.load_best_model_upon_completion.isChecked() else 'last'
        return self.train_parameters

    def get_predict_parameters(self):
        '''Get the parameters for running the model (predict)

        Returns
        -------
        dict
            containing training parameters

        '''

        self.predict_parameters = self.set_custom_predict_parameters.get_parameters_directly()
        return self.predict_parameters

    def get_list_of_augmentations(self):
        '''Returns all data augmentations as a list

        Returns
        -------
        list
            containing training parameters

        '''

        items = []
        for index in range(self.list_augmentations.count()):
            items.append(json.loads(self.list_augmentations.item(index).text()))
        return items

    def get_list_of_datasets(self):
        '''Returns training datasets as a list

        Returns
        -------
        list
            containing datasets for training the model

        '''

        items = []
        for index in range(self.list_datasets.count()):
            items.append(json.loads(self.list_datasets.item(index).text()))

        return items

    def _onTabChange(self):
        # model_parameters = self.get_model_parameters()
        if self.deepTA.model is not None:
            inputs = self.deepTA.get_inputs_shape()
            # if model has no specified width
            if inputs[0][-2] is None:
                # allow user to define one
                self.tile_width.setEnabled(True)
                self.set_custom_predict_parameters.tile_width_pred.setEnabled(True)
            else:
                # otherwise set width to default model width and do not allow it to change
                self.tile_width.setEnabled(False)
                self.tile_width.setValue(inputs[0][-2])
                self.set_custom_predict_parameters.tile_width_pred.setEnabled(False)
                self.set_custom_predict_parameters.tile_width_pred.setValue(inputs[0][-2])
            # if model has no specified height
            if inputs[0][-3] is None:
                # allow user to define one
                self.tile_height.setEnabled(True)
                self.set_custom_predict_parameters.tile_height_pred.setEnabled(True)
            else:
                # otherwise set height to default model height and do not allow it to change
                self.tile_height.setEnabled(False)
                self.tile_height.setValue(inputs[0][-3])
                self.set_custom_predict_parameters.tile_height_pred.setEnabled(False)
                self.set_custom_predict_parameters.tile_height_pred.setValue(inputs[0][-3])

    # also upon model check just see if I have the corresponding model and architecture present
    def _load_or_build_model_settings(self):
        self.groupBox.setEnabled(self.build_model_radio.isChecked())
        self.input_model.setEnabled(self.load_model_radio.isChecked())
        self.groupBox_pretrain.setEnabled(self.model_pretrain_on_epithelia.isChecked())
        # enable load weights if load model or if no epithelial training is checked
        if self.load_model_radio.isChecked():
            self.input_weights.setEnabled(True)
        else:
            self.input_weights.setEnabled(not self.model_pretrain_on_epithelia.isChecked())
        # if people changed settings assume model reset (TODO improve that later)
        self._enable_training(False)
        self._enable_predict(False)

    def _get_worker(self, func, *args, **kwargs):
        # returns the worker to proceed with building, training or running the model
        if self.threading_enabled:
            # threaded worker 
            return Worker(func, *args, **kwargs)
        else:
            # non threaded worker
            return FakeWorker(func, *args, **kwargs)

    def _reset_metrics_on_model_change(self):
        # resets all metrics when called
        self.selected_metrics.setText('')
        self.currently_selected_metrics = []

    def load_or_build_model(self):
        '''Loads or builds a model, warns upon errors

        '''
        self._reset_metrics_on_model_change()
        model_parameters = self.get_model_parameters()
        if self.load_model_radio.isChecked() and self.model_parameters['model'] is None:
            logger.error('Please provide a valid path to the model to be loaded')
            self.blinker.blink(self.input_model)
            return

        worker = self._get_worker(self._load_or_build_model, model_parameters=model_parameters)
        worker.signals.result.connect(self.print_output)
        worker.signals.finished.connect(self.thread_complete)
        worker.signals.progress.connect(self.progress_fn)

        # Execute
        if isinstance(worker, FakeWorker):
            # no threading
            worker.run()
        else:
            # threading
            self.threadpool.start(worker)

    def _load_or_build_model(self, progress_callback, model_parameters={}):
        self._enable_training(False)
        self._enable_predict(False)

        progress_callback.emit(5)

        if self.load_model_radio.isChecked():
            logger.info('loading an existing model')
            try:
                progress_callback.emit(10)
                self.deepTA.load_or_build(**model_parameters)
                progress_callback.emit(90)
            except:
                logger.error('Could not load model... please try something else')
                traceback.print_exc()
                self._enable_training(False)
                self._enable_predict(False)
                return
            if self.deepTA.model is None:
                logger.error('Please load a valid model')
                self.to_blink_after_worker_execution = self.input_model
                return

            if self.deepTA.is_model_compiled():
                self.groupBox_compile.setEnabled(False)
                self.force_recompile.setEnabled(True)
            self.deepTA.summary()

            self._enable_training(True)
            self._enable_predict(True)

        else:
            # build a model
            print('building model, please wait...')
            try:
                progress_callback.emit(10)
                self.deepTA.load_or_build(**model_parameters)
                progress_callback.emit(90)
            except:
                logger.error(
                    traceback.format_exc() + '\nCould not build model... you may want to try to change/set model'
                                             ' input width or height to a different value (e.g. 256, 128, 64, 512, 576,'
                                             ' 331, 224, 299 or to a multiple of 8, 32 or 64, ...\nAlso note that for'
                                             ' PSPnet model width and height should be a multiple of 48 eg 96, '
                                             '144, ...)')  # 299 --> inceptionresnetv2, 331 --> nasnet, resnet --> 224
                self.to_blink_after_worker_execution = [self.model_width, self.model_height]
                self._enable_training(False)
                self._enable_predict(False)
                return
            if self.deepTA.model is None:
                logger.error('Model building failed')
            if self.deepTA.is_model_compiled():
                self.groupBox_compile.setEnabled(True)
                self.force_recompile.setEnabled(False)

            progress_callback.emit(100)

            self.deepTA.summary()
            self._enable_training(True)
            self._enable_predict(True)

        self._set_model_inputs_and_outputs()

    def _set_model_inputs_and_outputs(self):
        # sets model inputs/outputs in the deep learning class (used to check input/output data during training)
        try:
            inputs = self.deepTA.get_inputs_shape()
        except:
            inputs = None

        self.input_output_normalization_method.set_model_inputs(inputs)
        self.set_custom_predict_parameters.set_model_inputs(inputs)

        try:
            outputs = self.deepTA.get_outputs_shape()
        except:
            outputs = None
        self.input_output_normalization_method.set_model_outputs(outputs)
        self.set_custom_predict_parameters.set_model_outputs(outputs)

    def _enable_training(self, bool):
        self.tabs.setTabEnabled(1, bool)

    def _enable_predict(self, bool):
        self.tabs.setTabEnabled(2, bool)

    def _stop_training(self):
        '''method to stop training or predict as soon as possible

        '''
        if self.deepTA.model is None:
            logger.error('Please load or build a model first.')
            return
        self.deepTA.stop_model_training_now()

    def _add_data(self):
        '''adds a training dataset

        '''

        augment, ok = image_input_settings.getDataAndParameters(parent_window=self,
                                                                show_preprocessing=True,
                                                                show_channel_nb_change_rules=True,
                                                                show_input=True,
                                                                show_output=True,
                                                                allow_ROI=True,
                                                                allow_wild_cards_in_path=True,
                                                                show_preview=True,
                                                                model_inputs=self.deepTA.get_inputs_shape(),
                                                                model_outputs=self.deepTA.get_outputs_shape(),
                                                                show_HQ_settings=False)

        if ok:
            item = QListWidgetItem(str(augment), self.list_datasets)
            self.list_datasets.addItem(item)

    def _remove_data(self):
        '''removes a training dataset

        '''

        for sel in self.list_datasets.selectedItems():
            self.list_datasets.takeItem(self.list_datasets.row(sel))

    def _add_augmenter(self):
        '''adds a data augmentation method

        '''

        augment, ok = DataAugmentationGUI.getAugmentation(parent=self)
        if ok:
            item = QListWidgetItem(str(augment), self.list_augmentations)
            self.list_augmentations.addItem(item)

    def _delete_augmentations(self):
        '''removes a data augmentation method

        '''

        for sel in self.list_augmentations.selectedItems():
            self.list_augmentations.takeItem(self.list_augmentations.row(sel))

    def compile_model(self):
        '''compiles the model

        '''

        train_parameters = self.get_train_parameters()
        try:
            logger.debug('training parameters' + str(train_parameters))
        except:
            pass

        # print(train_parameters)

        if not train_parameters['datasets']:
            logger.error('Please provide one or more valid training input/output dataset first')
            self.blinker.blink(self.groupBox_training_dataset)
            return

        worker = self._get_worker(self._compile_model, train_parameters=train_parameters)
        worker.signals.result.connect(self.print_output)
        worker.signals.finished.connect(self.thread_complete)
        worker.signals.progress.connect(self.progress_fn)

        # Execute
        if isinstance(worker, FakeWorker):
            # no threading
            worker.run()
        else:
            # threading
            self.threadpool.start(worker)

    def _compile_model(self, progress_callback, train_parameters={}):
        if self.deepTA.model is None:
            logger.error('Please load or build a model first.')
            return

        try:
            logger.debug('training parameters ' + str(train_parameters))
        except:
            pass

        is_compiled = self.deepTA.is_model_compiled()
        if not is_compiled or self.groupBox_compile.isEnabled():
            cur_info = 'compiling the model'
            if is_compiled:
                cur_info = 're' + cur_info
            logger.info(cur_info)
            self.deepTA.compile(**self.get_train_parameters())
            is_compiled = self.deepTA.is_model_compiled()
            if not is_compiled:
                logger.error('model could not be compiled, please try another set of optimizer, loss and metrics')
            else:
                logger.info('model successfully compiled')
        else:
            logger.info('model is already compiled, nothing to do...')

        input_shape = self.deepTA.get_inputs_shape()
        output_shape = self.deepTA.get_outputs_shape()

        logger.debug('inp, out ' + str(input_shape) + ' ' + str(output_shape))
        metaAugmenter = MetaAugmenter(input_shape=input_shape, output_shape=output_shape, **train_parameters)
        metaAugmenter.appendDatasets(**train_parameters)
        self.deepTA.train(metaAugmenter, progress_callback=progress_callback, **train_parameters)

    def _output_normalization_changed(self):
        if self.output_normalization.currentText() == Img.normalization_methods[0] \
                or self.output_normalization.currentText() == Img.normalization_methods[1]:
            self.output_norm_range.setEnabled(True)
        else:
            self.output_norm_range.setEnabled(False)

    def _input_normalization_changed(self):
        if self.input_normalization.currentText() == Img.normalization_methods[0] \
                or self.input_normalization.currentText() == Img.normalization_methods[1]:
            self.input_norm_range.setEnabled(True)
        else:
            self.input_norm_range.setEnabled(False)

    def _add_selected_metric(self):
        '''adds a metric (human readable measure of segmentation/training quality printed during training)

        '''

        current_metric = self.model_metrics.currentText()
        if not current_metric in self.currently_selected_metrics:
            self.currently_selected_metrics.append(current_metric)
        self.selected_metrics.setText(str(self.currently_selected_metrics))

    def _remove_selected_metric(self):
        '''removes a metric

        '''
        current_metric = self.model_metrics.currentText()
        try:
            self.currently_selected_metrics.remove(current_metric)
        except:
            pass
        if self.currently_selected_metrics:
            self.selected_metrics.setText(str(self.currently_selected_metrics))
        else:
            self.selected_metrics.setText('')

    def _architecture_change(self):
        # checks model input width and height upon architecture change
        current_model = self.model_architecture.currentText()
        if current_model.lower().startswith('psp'):
            model_height = self.model_height.value()
            model_width = self.model_width.value()
            self.restored_height_after_PSP_net = model_height
            self.restored_width_after_PSP_net = model_width
            if model_height < 48:
                self.model_height.setValue(144)  # could also set 96...
            elif model_height % 48:
                # get next multiple of 48
                model_height = model_height + (48 - model_height % 48)
                self.model_height.setValue(model_height)
            if model_width < 48:
                self.model_width.setValue(144)
            elif model_width % 48:
                # get next multiple of 48
                model_width = model_width + (48 - model_width % 48)
                self.model_width.setValue(model_width)
        else:
            try:
                # ignore error if the variables haven't been defined
                self.model_height.setValue(self.restored_height_after_PSP_net)
                self.model_width.setValue(self.restored_width_after_PSP_net)
            except:
                pass

    def run_post_process(self):
        # run the image post process
        post_process_params = self.get_post_process_parameters()

        if 'input' not in post_process_params or post_process_params['input'] is None:
            logger.error('Please provide epyseg raw output files (highlighted in red)')
            self.blinker.blink(self.set_custom_post_process.open_input_button)
            return

        if 'output_folder' not in post_process_params or post_process_params['output_folder'] is None:
            logger.error('Please provide a valid output folder')
            self.blinker.blink(self.set_custom_post_process.output_predictions_to)
            return

        # check output and if not correct say the pb
        # self.set_custom_post_process.check_custom_dir()

        worker = self._get_worker(self._run_post_process, model_parameters=post_process_params)
        worker.signals.result.connect(self.print_output)
        worker.signals.finished.connect(self.thread_complete)
        worker.signals.progress.connect(self.progress_fn)

        # Execute+
        if isinstance(worker, FakeWorker):
            # no threading
            worker.run()
        else:
            # threading
            self.threadpool.start(worker)

    def _run_post_process(self, progress_callback, model_parameters={}):
        # run post process
        self.post_process_func = EPySegPostProcess(**model_parameters, progress_callback=progress_callback)

        return "Done."

    def _stop_post_process(self):
        # stop immediately the post process --> see how to do --> best is to use a worker and a global variable maybe ???
        try:

            logger.info('Stopping post process')
            # self.post_process_func.early_stop()
            EPySegPostProcess.stop_now = True
            self.pbar.reset()  # setValue(0)
        except:
            # traceback.print_exc()
            pass

    def predict_using_model(self):
        '''run the model (get the predictions), warns upon errors

        '''
        predict_parameters = self.get_predict_parameters()
        try:
            logger.debug('predict_parameters' + str(predict_parameters))
        except:
            pass

        # if predict parameters is None
        if predict_parameters['inputs'] == None or predict_parameters['inputs'] == [None]:
            logger.error('Please provide a valid input dataset (highlighted in red)')
            self.blinker.blink(self.set_custom_predict_parameters.open_input_button)
            return

        if 'predict_output_folder' not in predict_parameters or predict_parameters['predict_output_folder'] is None:
            logger.error('Please provide a valid output folder')
            self.blinker.blink(self.set_custom_predict_parameters.output_predictions_to)
            return

        # TODO add this in Img code too as a control...
        if 'tile_width_overlap' in predict_parameters:
            if predict_parameters['tile_width_overlap'] >= predict_parameters['default_input_tile_width']:
                logger.error('Image width overlap is bigger than image width, this does not make sense please increase'
                             ' tile width or decrease overlap width. Ideally width should be at least '
                             'twice the size of the overlap.')
                self.blinker.blink(self.set_custom_predict_parameters.tiling_group)
                return

        if 'tile_height_overlap' in predict_parameters:
            if predict_parameters['tile_height_overlap'] >= predict_parameters['default_input_tile_height']:
                logger.error('Image height overlap is bigger than image height, '
                             'this does not make sense please increase tile height or decrease overlap height. '
                             'Ideally height should be at least twice the size of the overlap.')
                self.blinker.blink(self.set_custom_predict_parameters.tiling_group)
                return

        worker = self._get_worker(self._predict_using_model, predict_parameters=predict_parameters)
        worker.signals.result.connect(self.print_output)
        worker.signals.finished.connect(self.thread_complete)
        worker.signals.progress.connect(self.progress_fn)

        # Execute
        if isinstance(worker, FakeWorker):
            # no threading
            worker.run()
        else:
            # threading
            self.threadpool.start(worker)

    def _predict_using_model(self, progress_callback, predict_parameters={}):
        if self.deepTA.model is None:
            logger.error('Please load or build a model first.')
            return

        input_shape = self.deepTA.get_inputs_shape()
        output_shape = self.deepTA.get_outputs_shape()

        logger.debug('inp, out ' + str(input_shape) + ' ' + str(output_shape))
        logger.debug('test values ' + str(input_shape) + ' ' + str(output_shape))
        logger.debug('predict params' + str(predict_parameters))

        predict_generator = self.deepTA.get_predict_generator(input_shape=input_shape, output_shape=output_shape,
                                                              **predict_parameters)

        self.deepTA.predict(predict_generator, output_shape, progress_callback=progress_callback, batch_size=1,
                            **predict_parameters)

        return "Done."


if __name__ == '__main__':
    app = QApplication(sys.argv)
    w = EPySeg()
    w.show()
    sys.exit(app.exec_())
