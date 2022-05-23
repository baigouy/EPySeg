# TODO depending on selected mode activate or not stuff
# if output is not 0 --> then use min max as 0 1
# if none --> remove all parameters

from PyQt5.QtWidgets import QDialog, QDoubleSpinBox, QToolTip, QPushButton, QDialogButtonBox
from PyQt5.QtWidgets import QApplication, QGridLayout
from PyQt5.QtWidgets import QSpinBox, QComboBox, QVBoxLayout, QLabel, QCheckBox, QGroupBox
from PyQt5.QtCore import Qt, QPoint
from PyQt5 import QtWidgets, QtCore
import sys
from epyseg.deeplearning.docs.doc2html import markdown_file_to_html
from epyseg.tools.logger import TA_logger # logging

logger = TA_logger()


class PostProcessGUI(QDialog):

    def __init__(self, parent_window=None, _is_dialog=False):
        super().__init__(parent=parent_window)
        self._is_dialog = _is_dialog
        self.initUI()

    def initUI(self):
        input_v_layout = QVBoxLayout()
        input_v_layout.setAlignment(Qt.AlignTop)
        input_v_layout.setContentsMargins(0, 0, 0, 0)
        # TODO add a set of parameters there for the post process
        self.groupBox_post_process = QGroupBox(
            'Refine segmentation/Create a binary mask', objectName='groupBox_post_process')
        self.groupBox_post_process.setCheckable(True)
        self.groupBox_post_process.setChecked(True)
        # self.groupBox_post_process.setEnabled(True)

        group_box_post_process_parameters_layout = QGridLayout()
        group_box_post_process_parameters_layout.setAlignment(Qt.AlignTop)
        group_box_post_process_parameters_layout.setHorizontalSpacing(3)
        group_box_post_process_parameters_layout.setVerticalSpacing(3)

        # do a radio dialog with all the stuff needed...
        # test all

        post_process_method_selection_label = QLabel('Post process method')  # (or bond score for pretrained model)
        post_process_method_selection_label.setStyleSheet("QLabel { color : red; }")

        self.post_process_method_selection = QComboBox(objectName='post_process_method_selection')
        self.post_process_method_selection.addItem('Default (Slow/robust) (EPySeg pre-trained model only!)')
        self.post_process_method_selection.addItem('Fast (May contain more errors) (EPySeg pre-trained model only!)')
        self.post_process_method_selection.addItem('Old method (Sometimes better than default) (EPySeg pre-trained model only!)')
        self.post_process_method_selection.addItem('Simply binarize output using threshold')
        self.post_process_method_selection.addItem('Keep first channel only')
        self.post_process_method_selection.addItem('None (Raw model output)')
        self.post_process_method_selection.currentTextChanged.connect(self._post_process_method_changed)

        group_box_post_process_parameters_layout.addWidget(post_process_method_selection_label, 0, 0, 1, 1)
        group_box_post_process_parameters_layout.addWidget(self.post_process_method_selection, 0, 1, 1, 3)

        # TODO --> always make this relative
        threshold_label = QLabel(
            'Threshold: (in case of over/under segmentation, please increase/decrease, respectively)')  # (or bond score for pretrained model)
        threshold_label.setStyleSheet("QLabel { color : red; }")
        self.threshold_bond_or_binarisation = QDoubleSpinBox(objectName='threshold_bond_or_binarisation')
        self.threshold_bond_or_binarisation.setSingleStep(0.01)
        self.threshold_bond_or_binarisation.setRange(0.01, 1)  # 100_000 makes no sense (oom) but anyway
        self.threshold_bond_or_binarisation.setValue(0.42)  # probably should be 1 to 3 depending on the tissue
        self.threshold_bond_or_binarisation.setEnabled(False)
        # threshold_hint = QLabel()  # (or bond score for pretrained model)

        self.autothreshold = QCheckBox("Auto",objectName='autothreshold')
        self.autothreshold.setChecked(True)
        self.autothreshold.stateChanged.connect(self._threshold_changed)

        group_box_post_process_parameters_layout.addWidget(threshold_label, 1, 0, 1, 2)
        group_box_post_process_parameters_layout.addWidget(self.threshold_bond_or_binarisation, 1, 2)
        group_box_post_process_parameters_layout.addWidget(self.autothreshold, 1, 3)
        # groupBox_post_process_parameters_layout.addWidget(threshold_hint, 0, 3)

        filter_by_size_label = QLabel('Further filter segmentation by size:')
        self.filter_by_cell_size_combo = QComboBox(objectName='filter_by_cell_size_combo')
        self.filter_by_cell_size_combo.addItem('None (quite often the best choice)')
        self.filter_by_cell_size_combo.addItem('Local median (slow/very good) divided by')
        self.filter_by_cell_size_combo.addItem('Cells below Average area (global) divided by')
        self.filter_by_cell_size_combo.addItem('Global median divided by')
        self.filter_by_cell_size_combo.addItem('Cells below size (in px)')

        # add a listener to model Architecture
        self.filter_by_cell_size_combo.currentTextChanged.connect(self._filter_changed)

        group_box_post_process_parameters_layout.addWidget(filter_by_size_label, 2, 0)
        group_box_post_process_parameters_layout.addWidget(self.filter_by_cell_size_combo, 2, 1, 1, 2)

        self.avg_area_division_or_size_spinbox = QSpinBox(objectName='avg_area_division_or_size_spinbox')
        self.avg_area_division_or_size_spinbox.setSingleStep(1)
        self.avg_area_division_or_size_spinbox.setRange(1, 10000000)  # 100_000 makes no sense (oom) but anyway
        self.avg_area_division_or_size_spinbox.setValue(2)  # probably should be 1 to 3 depending on the tissue
        self.avg_area_division_or_size_spinbox.setEnabled(False)
        group_box_post_process_parameters_layout.addWidget(self.avg_area_division_or_size_spinbox, 2, 3)

        self.prevent_exclusion_of_too_many_cells_together = QCheckBox('Do not exclude groups bigger than', objectName='prevent_exclusion_of_too_many_cells_together')
        self.prevent_exclusion_of_too_many_cells_together.setChecked(False)
        self.prevent_exclusion_of_too_many_cells_together.setEnabled(False)

        # max_nb_of_cells_to_be_excluded_together_label = QLabel('Group size')
        self.max_nb_of_cells_to_be_excluded_together_spinbox = QSpinBox(objectName='max_nb_of_cells_to_be_excluded_together_spinbox')
        self.max_nb_of_cells_to_be_excluded_together_spinbox.setSingleStep(1)
        self.max_nb_of_cells_to_be_excluded_together_spinbox.setRange(1, 10000000)  # max makes no sense
        self.max_nb_of_cells_to_be_excluded_together_spinbox.setValue(
            3)  # default should be 2 or 3 because seg is quite good so above makes no sense
        self.max_nb_of_cells_to_be_excluded_together_spinbox.setEnabled(False)
        cells_text_labels = QLabel('cells')

        self.restore_secure_cells = QCheckBox('Restore most likely cells',objectName='restore_secure_cells')
        self.restore_secure_cells.setChecked(False)
        self.restore_secure_cells.setEnabled(False)

        # help for post process
        # help_ico = QIcon.fromTheme('help-contents')
        self.help_button_postproc = QPushButton('?', None)
        bt_width = self.help_button_postproc.fontMetrics().boundingRect(self.help_button_postproc.text()).width() + 7
        self.help_button_postproc.setMaximumWidth(bt_width * 2)
        self.help_button_postproc.clicked.connect(self.show_tip)

        group_box_post_process_parameters_layout.addWidget(self.restore_secure_cells, 3, 0)
        group_box_post_process_parameters_layout.addWidget(self.prevent_exclusion_of_too_many_cells_together, 3, 1)
        group_box_post_process_parameters_layout.addWidget(self.max_nb_of_cells_to_be_excluded_together_spinbox, 3, 2)
        group_box_post_process_parameters_layout.addWidget(cells_text_labels, 3, 3)

        # TODO --> improve layout to make help button smaller
        group_box_post_process_parameters_layout.addWidget(self.help_button_postproc, 0, 5, 3, 1)

        self.groupBox_post_process.setLayout(group_box_post_process_parameters_layout)
        input_v_layout.addWidget(self.groupBox_post_process)
        self.setLayout(input_v_layout)

        if self._is_dialog:
            # OK and Cancel buttons
            self.buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel,
                                            QtCore.Qt.Horizontal, self)
            self.buttons.accepted.connect(self.accept)
            self.buttons.rejected.connect(self.reject)
            self.layout().addWidget(self.buttons)

    def _threshold_changed(self):
        self.threshold_bond_or_binarisation.setEnabled(not self.autothreshold.isChecked())

    # self.post_process_method_selection.addItem('Default (Slow/robust)')
    # self.post_process_method_selection.addItem('Fast (May contain more errors)')
    # self.post_process_method_selection.addItem('Old method (Less constant than default but sometimes better)')
    # self.post_process_method_selection.addItem('Simply binarize output using threshold')
    # self.post_process_method_selection.addItem('None (Raw model output)')

    def _post_process_method_changed(self):
        text = self.post_process_method_selection.currentText().lower()
        if 'none' in text or 'first' in text:
            self.set_threshold_enabled(False)
            self.set_safety_parameters(False)
            self.set_filter_by_size_enabled(False)
        elif 'simply' in text:
            self.set_threshold_enabled(True)
            self.set_safety_parameters(False)
            self.set_filter_by_size_enabled(False)
        elif 'old' in text:
            self.set_threshold_enabled(False)
            self.set_safety_parameters(True)
            self.set_filter_by_size_enabled(True)
        else:
            self.set_threshold_enabled(True)
            self.set_safety_parameters(False)
            self.set_filter_by_size_enabled(True)

    def set_filter_by_size_enabled(self, bool):
        if bool is False:
            self.filter_by_cell_size_combo.setEnabled(False)
            self.avg_area_division_or_size_spinbox.setEnabled(False)
        else:
            self.filter_by_cell_size_combo.setEnabled(True)
            self.avg_area_division_or_size_spinbox.setEnabled(True)

    def set_threshold_enabled(self, bool):
        if bool is False:
            self.autothreshold.setEnabled(False)
            self.threshold_bond_or_binarisation.setEnabled(False)
        else:
            self.autothreshold.setEnabled(True)
            self._threshold_changed()

    def set_safety_parameters(self, bool):
        self._filter_changed()

    def show_tip(self):
        QToolTip.showText(self.sender().mapToGlobal(QPoint(30, 30)), markdown_file_to_html('refine_segmentation.md'))

    def isChecked(self):
        return self.groupBox_post_process.isChecked()

    def setChecked(self, bool):
        return self.groupBox_post_process.setChecked(bool)

    def _filter_changed(self):
        current_filter = self.filter_by_cell_size_combo.currentText().lower()
        current_mode = self.post_process_method_selection.currentText().lower()
        if 'one' in current_filter:
            self.avg_area_division_or_size_spinbox.setEnabled(False)
            self.max_nb_of_cells_to_be_excluded_together_spinbox.setEnabled(False)
            self.prevent_exclusion_of_too_many_cells_together.setEnabled(False)
            self.restore_secure_cells.setEnabled(False)
        else:
            self.avg_area_division_or_size_spinbox.setEnabled(True)
            self.max_nb_of_cells_to_be_excluded_together_spinbox.setEnabled(True)
            self.prevent_exclusion_of_too_many_cells_together.setEnabled(True)
            self.restore_secure_cells.setEnabled(True)
            if 'divided' in current_filter:
                self.avg_area_division_or_size_spinbox.setValue(2)
            else:
                self.avg_area_division_or_size_spinbox.setValue(300)
            if not 'old' in current_mode:
                self.max_nb_of_cells_to_be_excluded_together_spinbox.setEnabled(False)
                self.prevent_exclusion_of_too_many_cells_together.setEnabled(False)
                self.restore_secure_cells.setEnabled(False)

    def _get_post_process_filter(self):
        current_filter = self.filter_by_cell_size_combo.currentText().lower()
        if 'one' in current_filter or not self.filter_by_cell_size_combo.isEnabled():
            return None
        if 'size' in current_filter:
            return self.avg_area_division_or_size_spinbox.value()
        if 'verage' in current_filter:
            return 'avg'
        if 'local' in current_filter:
            return 'local'
        if 'global' in current_filter:
            return 'global median'

    def get_parameters_directly(self):
        '''Get the parameters for model training

            Returns
            -------
            dict
                containing post processing parameters

            '''

        self.post_process_parameters = {}
        post_proc_method = self.post_process_method_selection.currentText().lower()
        if 'none' in post_proc_method:
            self.post_process_parameters['post_process_algorithm'] = None
        else:
            self.post_process_parameters['post_process_algorithm'] = post_proc_method
        self.post_process_parameters['filter'] = self._get_post_process_filter()
        if self.threshold_bond_or_binarisation.isEnabled():
            self.post_process_parameters['threshold'] = self.threshold_bond_or_binarisation.value()
        if self.autothreshold.isEnabled() and self.autothreshold.isChecked():
            self.post_process_parameters[
                'threshold'] = None  # None means autothrehsold # maybe add more options some day
        if self.avg_area_division_or_size_spinbox.isEnabled():
            self.post_process_parameters['correction_factor'] = self.avg_area_division_or_size_spinbox.value()
        if self.restore_secure_cells.isEnabled():
            self.post_process_parameters['restore_safe_cells'] = self.restore_secure_cells.isChecked()
        if self.max_nb_of_cells_to_be_excluded_together_spinbox.isEnabled():
            self.post_process_parameters[
                'cutoff_cell_fusion'] = self.max_nb_of_cells_to_be_excluded_together_spinbox.value() if self.prevent_exclusion_of_too_many_cells_together.isChecked() else None

        if 'old' in self.post_process_method_selection.currentText().lower():
            # just for max use that --> maybe do this as an option some day
            self.post_process_parameters['hq_predictions'] = 'max'

        return self.post_process_parameters

    def get_parameters(self):
        return (self.get_parameters_directly())

    @staticmethod
    def getDataAndParameters(parent_window=None, _is_dialog=False):
        # get all the params for augmentation
        dialog = PostProcessGUI(parent_window=parent_window, _is_dialog=_is_dialog)
        result = dialog.exec_()
        parameters = dialog.get_parameters()
        return (parameters, result == QDialog.Accepted)


# now really try to get the parameters properly

if __name__ == '__main__':
    # just for a test
    app = QApplication(sys.argv)
    parameters, ok = PostProcessGUI.getDataAndParameters(parent_window=None)
    print(parameters, ok)
    sys.exit(0)

# TODO change default parameters depending on whether a pre-trained model is selected or not
# TODO allow retraining of the model --> just give it a try...
