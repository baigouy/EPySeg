import json
from PyQt5.QtWidgets import QApplication, QGridLayout, \
    QGroupBox, QLabel, QComboBox, QDialog, QDialogButtonBox
from PyQt5 import QtCore

from epyseg.deeplearning.augmentation.generators.data import DataGenerator
from epyseg.gui.preview import crop_or_preview
from epyseg.gui.open import OpenFileOrFolderWidget
from epyseg.img import Img
import sys

class MetaAugmenterGUI(QDialog):

    def __init__(self, parent_window=None):
        super().__init__(parent=parent_window)
        self.first_image = None
        self.first_mask = None
        self.initUI()

    def initUI(self):
        layout = QGridLayout()
        layout.setColumnStretch(0, 80)
        layout.setColumnStretch(1, 20)
        layout.setHorizontalSpacing(3)
        layout.setVerticalSpacing(3)
        OpenFileOrFolderWidget.finalize_text_change = self.check_input
        self.open_input_button = OpenFileOrFolderWidget(parent_window=self, add_timer_to_changetext=True,
                                                        show_ok_or_not_icon=True, label_text='Path to the raw dataset')
        layout.addWidget(self.open_input_button, 0, 0, 1, 2)


        OpenFileOrFolderWidget.finalize_text_change = self.check_labels
        self.open_labels_button = OpenFileOrFolderWidget(parent_window=self, add_timer_to_changetext=True,
                                                         show_ok_or_not_icon=True,
                                                         label_text='Segmentation masks/labels')

        layout.addWidget(self.open_labels_button, 1, 0, 1, 2)

        self.label_nb_inputs_over_outputs = QLabel('')
        layout.addWidget(self.label_nb_inputs_over_outputs, 2, 0, 1, 2)

        self.nb_inputs = 0
        self.nb_outputs = 0

        self.setLayout(layout)

        groupBox = QGroupBox('Pre-processing settings')
        groupBox.setEnabled(True)

        self.model_builder_layout = QGridLayout()
        self.model_builder_layout.setHorizontalSpacing(3)
        self.model_builder_layout.setVerticalSpacing(3)

        input_channel_label = QLabel('input channel of interest')
        self.model_builder_layout.addWidget(input_channel_label, 5, 0)
        self.input_channel_of_interest = QComboBox()
        self.input_channel_of_interest.currentIndexChanged.connect(self._input_channel_changed)
        self.model_builder_layout.addWidget(self.input_channel_of_interest, 5, 1)

        input_channel_reduction_rule_label = QLabel('rule to reduce nb of input channels (if needed)')
        self.model_builder_layout.addWidget(input_channel_reduction_rule_label, 6, 0)
        self.channel_input_reduction_rule = QComboBox()
        self.channel_input_reduction_rule.addItem('copy the input channel of interest to all available channels')
        self.channel_input_reduction_rule.addItem('remove supernumerary input channels (keep other channels unchanged)')
        self.model_builder_layout.addWidget(self.channel_input_reduction_rule, 6, 1, 1, 2)

        input_channel_augmentation_rule_label = QLabel('rule to increase nb of input channels (if needed)')
        self.model_builder_layout.addWidget(input_channel_augmentation_rule_label, 7, 0)
        self.channel_input_augmentation_rule = QComboBox()
        self.channel_input_augmentation_rule.addItem('copy the input channel of interest to all channels')
        self.channel_input_augmentation_rule.addItem('copy the input channel of interest to missing channels only')
        self.channel_input_augmentation_rule.addItem('add empty channels (0 filled)')
        self.model_builder_layout.addWidget(self.channel_input_augmentation_rule, 7, 1, 1, 2)

        output_channel_label = QLabel('output channel of interest')
        self.model_builder_layout.addWidget(output_channel_label, 8, 0)
        self.output_channel_of_interest = QComboBox()
        self.output_channel_of_interest.currentIndexChanged.connect(self._output_channel_changed)
        self.model_builder_layout.addWidget(self.output_channel_of_interest, 8, 1)

        output_channel_reduction_rule_label = QLabel('rule to reduce nb of output channels (if needed)')
        self.model_builder_layout.addWidget(output_channel_reduction_rule_label, 9, 0)
        # maybe rather put than in model input
        self.channel_output_reduction_rule = QComboBox()
        self.channel_output_reduction_rule.addItem('copy the output channel of interest to all available channels')
        self.channel_output_reduction_rule.addItem('remove supernumerary output channels (keep other channels unchanged)')
        self.model_builder_layout.addWidget(self.channel_output_reduction_rule, 9, 1, 1, 2)

        output_channel_augmentation_rule_label = QLabel('rule to increase nb of output channels (if needed)')
        self.model_builder_layout.addWidget(output_channel_augmentation_rule_label, 10, 0)
        self.channel_output_augmentation_rule = QComboBox()
        self.channel_output_augmentation_rule.addItem('copy the output channel of interest to all channels')
        self.channel_output_augmentation_rule.addItem('copy the output channel of interest to missing channels only')
        self.channel_output_augmentation_rule.addItem('add empty channels (0 filled)')
        self.model_builder_layout.addWidget(self.channel_output_augmentation_rule, 10, 1, 1, 2)

        crop_info_label = QLabel('If ROI should be used for training, please draw a rect over below images')
        self.model_builder_layout.addWidget(crop_info_label, 12, 0, 1, 3)
        input_preview_label = QLabel('input preview (channel of interest)')
        self.model_builder_layout.addWidget(input_preview_label, 14, 0, 1, 3)
        self.image_cropper_UI = crop_or_preview()
        self.model_builder_layout.addWidget(self.image_cropper_UI, 15, 0, 1, 3)

        output_preview_label = QLabel('output preview (channel of interest)')
        self.model_builder_layout.addWidget(output_preview_label, 14, 1, 1, 3)
        self.mask_preview = crop_or_preview(preview_only=True)
        self.model_builder_layout.addWidget(self.mask_preview, 15, 1, 1, 3)

        groupBox.setLayout(self.model_builder_layout)

        layout.addWidget(groupBox, 3, 0, 1, 2)

        # OK and Cancel buttons
        self.buttons = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel,
            QtCore.Qt.Horizontal, self)
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        layout.addWidget(self.buttons)

    def _output_channel_changed(self):
        if self.first_mask is not None and self.first_mask.has_c():
            if self.output_channel_of_interest.currentIndex() != -1:
                channel_img = self.first_mask.imCopy(c=self.output_channel_of_interest.currentIndex())
                self.mask_preview.set_image(channel_img)
            else:
                self.mask_preview.set_image(self.first_mask)

    def _input_channel_changed(self):
        if self.first_image is not None and self.first_image.has_c():
            if self.input_channel_of_interest.currentIndex() != -1:
                channel_img = self.first_image.imCopy(c=self.input_channel_of_interest.currentIndex())
                self.image_cropper_UI.set_image(channel_img)
            else:
                self.image_cropper_UI.set_image(self.first_image)

    def _update_inputs_and_outputs_nb(self):
        info_text = 'everything seems fine'
        if self.nb_outputs == 0 or self.nb_inputs == 0 or self.nb_outputs != self.nb_inputs:
            info_text = 'there seems to be a pb\n with your input/label data\n please check'
        self.label_nb_inputs_over_outputs.setText(str(self.nb_inputs) + ' / ' + str(self.nb_outputs) + " " + info_text)

    def check_input(self):
        print('in 2', self.open_input_button.text())
        txt = self.open_input_button.text()
        if '*' in txt:
            # then check if files do exist and try to open them
            file_list = self.open_input_button.get_list_using_glob()
            if file_list is not None and file_list:
                import os
                self.first_image, can_read = self._can_read_file(file_list[0], self.input_channel_of_interest)
                self.open_input_button.set_icon_ok(can_read)
                if can_read:
                    self.nb_inputs = len(file_list)
                else:
                    self.nb_inputs = 0

                if self.open_labels_button.text() is not None:
                    if self._can_read_mask(file_list[0]):
                        self.nb_outputs = self.nb_inputs
                    else:
                        self.nb_outputs = 0
            else:
                self.open_input_button.set_icon_ok(False)
                self.nb_inputs = 0
                self.first_image = None
                if self.open_labels_button.text() is not None:
                    self.open_labels_button.set_icon_ok(False)
                    self.nb_outputs = 0
        else:
            import os
            file_list = DataGenerator.get_list_of_images(self.open_input_button.text())
            if file_list:
                self.first_image, can_read = self._can_read_file(file_list[0], self.input_channel_of_interest)
                self.open_input_button.set_icon_ok(can_read)
                self.nb_inputs = len(file_list)

                if self.open_labels_button.text() is not None:
                    if self._can_read_mask(file_list[0]):
                        self.nb_outputs = self.nb_inputs
                    else:
                        self.nb_outputs = 0
            else:
                self.nb_inputs = 0
                self.first_image = None
                self.open_input_button.set_icon_ok(False)
                if self.open_labels_button.text() is not None:
                    self.open_labels_button.set_icon_ok(False)
                    self.nb_outputs = 0
                print('error no files matching "', self.open_input_button.text(), '"')

        self._update_inputs_and_outputs_nb()
        self.image_cropper_UI.set_image(self.first_image)  # not smart to load it twice TODO


    def _can_read_mask(self, path):
        import os
        mask_path = os.path.join(os.path.splitext(path)[0], 'handCorrection.png')
        if os.path.isfile(mask_path):
            if self._can_read_file(path, self.output_channel_of_interest):
                print('TA structure') # TODO replace by logger
                self.open_labels_button.set_icon_ok(True)
                return True
            else:
                print('unknown structure')
                self.open_labels_button.set_icon_ok(False)
        else:
            mask_path = os.path.join(os.path.splitext(path)[0], 'handCorrection.tif')
            if os.path.isfile(mask_path):
                if self._can_read_file(path, self.output_channel_of_interest):
                    print('TA structure')
                    self.open_labels_button.set_icon_ok(True)
                    return True
                else:
                    print('unknown structure')
                    self.open_labels_button.set_icon_ok(False)
            else:
                self.open_labels_button.set_icon_ok(False)
        return False

    def _can_read_file(self, path, channels_to_update=None):
        try:
            img = Img(path)
            try:
                print(img.has_c(), channels_to_update, img.shape, img.shape[-1])
                if img.has_c() and channels_to_update is not None:
                    # empty channel combo
                    channels_to_update.clear()
                    # add new channels to combo
                    for chan in range(img.shape[-1]):
                        channels_to_update.addItem(str(chan))
                    # deselect it
                    channels_to_update.setCurrentIndex(-1)
                    channels_to_update.update()
                if not img.has_c():
                    channels_to_update.clear()
                    channels_to_update.setCurrentIndex(-1)
                    channels_to_update.update()
            except:
                import traceback
                import logging
                logging.error(traceback.format_exc())
            return img, True
        except:
            print('could not read image ' + path)
        return None, False

    def check_labels(self):
        txt = self.open_labels_button.text()
        if txt is not None and '*' in txt:
            file_list = self.open_labels_button.get_list_using_glob()
            if file_list is not None and file_list:
                import os
                self.first_mask, can_read = self._can_read_file(file_list[0], self.output_channel_of_interest)
                self.open_labels_button.set_icon_ok(can_read)
                if can_read:
                    self.nb_outputs = len(file_list)
                else:
                    self.nb_outputs = 0
            else:
                self.first_mask = None
                self.open_labels_button.set_icon_ok(False)
                self.nb_outputs = 0
        else:
            import os
            file_list = DataGenerator.get_list_of_images(self.open_labels_button.text())
            if file_list:
                self.first_mask, can_read = self._can_read_file(file_list[0], self.output_channel_of_interest)
                self.open_labels_button.set_icon_ok(can_read)
                if can_read:
                    self.nb_outputs = len(file_list)
                else:
                    self.nb_outputs = 0
            else:
                self.first_mask =None
                self.open_labels_button.set_icon_ok(False)
                self.nb_outputs = 0
                print('error no files matching "', self.open_input_button.text(), '"')
        self._update_inputs_and_outputs_nb()
        self.mask_preview.set_image(self.first_mask)

    def getParams(self):
        # will that work and do I need to show the channel
        input_channel_of_interest = self.input_channel_of_interest.currentIndex()
        if input_channel_of_interest == -1:
            input_channel_of_interest = None
        else:
            input_channel_of_interest = int(self.input_channel_of_interest.currentText())
        output_channel_of_interest = self.output_channel_of_interest.currentIndex()
        if output_channel_of_interest == -1:
            output_channel_of_interest = None
        else:
            output_channel_of_interest = int(self.output_channel_of_interest.currentText())
        return json.dumps({'inputs': [self.open_input_button.text()], 'outputs': [self.open_labels_button.text()],
                           'input_channel_reduction_rule': self.channel_input_reduction_rule.currentText(),
                           'input_channel_augmentation_rule': self.channel_input_augmentation_rule.currentText(),
                           'output_channel_reduction_rule': self.channel_output_reduction_rule.currentText(),
                           'output_channel_augmentation_rule': self.channel_output_augmentation_rule.currentText(),
                           'input_channel_of_interest': input_channel_of_interest,
                           'output_channel_of_interest': output_channel_of_interest,
                           'crop_parameters': self.image_cropper_UI.get_crop_parameters()})

    # get all the params for augmentation
    @staticmethod
    def getDataAndParameters(parent=None):
        dialog = MetaAugmenterGUI(parent)
        result = dialog.exec_()
        augment = dialog.getParams()
        return (augment, result == QDialog.Accepted)

    def activate_ok_button_if_all_si_fine(self):
        all_ok = False
        self.buttons.button(QDialogButtonBox.Ok).setEnabled(all_ok) # do that always in all GUIs where it's required to avoid issues...

if __name__ == '__main__':
    app = QApplication(sys.argv)
    augment, ok = MetaAugmenterGUI.getDataAndParameters(parent=None)
    print(augment, ok)
    sys.exit(0)
