from PyQt5.QtWidgets import QDialog
from PyQt5.QtWidgets import  QApplication, QGridLayout
from PyQt5.QtWidgets import QSpinBox, QComboBox, QVBoxLayout, QLabel, QCheckBox, QGroupBox
from PyQt5.QtCore import Qt
import sys

# logging
from epyseg.tools.logger import TA_logger
logger = TA_logger()

class PostProcessGUI(QDialog):

    def __init__(self, parent_window=None):
        super().__init__(parent=parent_window)
        self.initUI()


    def initUI(self):
        input_v_layout = QVBoxLayout()
        input_v_layout.setAlignment(Qt.AlignTop)
        input_v_layout.setContentsMargins(0,0,0,0)
        # TODO add a set of parameters there for the post process
        groupBox_post_process = QGroupBox('Refine segmentation ("None" or "local median" should be prefered)')
        groupBox_post_process.setEnabled(True)

        groupBox_post_process_parameters_layout = QGridLayout()
        groupBox_post_process_parameters_layout.setAlignment(Qt.AlignTop)
        groupBox_post_process_parameters_layout.setHorizontalSpacing(3)
        groupBox_post_process_parameters_layout.setVerticalSpacing(3)

        # do a radio dialog with all the stuff needed...
        # test all

        filter_by_size_label = QLabel('Filter segmentation/cells by size:')
        self.filter_by_cell_size_combo = QComboBox()
        self.filter_by_cell_size_combo.addItem('Local median (slow/very good) divided by')
        self.filter_by_cell_size_combo.addItem('Cells below size (in px)')
        self.filter_by_cell_size_combo.addItem('cells below Average area (global) divided by')
        self.filter_by_cell_size_combo.addItem('Global median divided by')
        self.filter_by_cell_size_combo.addItem('None (segmentation may sometimes work better without filters)')
        # add a listener to model Architecture
        self.filter_by_cell_size_combo.currentTextChanged.connect(self._filter_changed)

        groupBox_post_process_parameters_layout.addWidget(filter_by_size_label, 1, 0)
        groupBox_post_process_parameters_layout.addWidget(self.filter_by_cell_size_combo, 1, 1, 1, 2)

        self.avg_area_division_or_size_spinbox = QSpinBox()
        self.avg_area_division_or_size_spinbox.setSingleStep(1)
        self.avg_area_division_or_size_spinbox.setRange(1, 1000_0000)  # 100_000 makes no sense (oom) but anyway
        self.avg_area_division_or_size_spinbox.setValue(2)  # probably should be 1 to 3 depending on the tissue
        groupBox_post_process_parameters_layout.addWidget(self.avg_area_division_or_size_spinbox, 1, 3)

        self.prevent_exclusion_of_too_many_cells_together = QCheckBox('Do not exclude groups bigger than')
        self.prevent_exclusion_of_too_many_cells_together.setChecked(True)

        # max_nb_of_cells_to_be_excluded_together_label = QLabel('Group size')
        self.max_nb_of_cells_to_be_excluded_together_spinbox = QSpinBox()
        self.max_nb_of_cells_to_be_excluded_together_spinbox.setSingleStep(1)
        self.max_nb_of_cells_to_be_excluded_together_spinbox.setRange(1, 1000_0000)  # max makes no sense
        self.max_nb_of_cells_to_be_excluded_together_spinbox.setValue(3)  # default should be 2 or 3 because seg is quite good so above makes no sense
        cells_text_labels = QLabel('cells')

        self.restore_secure_cells = QCheckBox('Restore most likely cells')
        self.restore_secure_cells.setChecked(True)

        groupBox_post_process_parameters_layout.addWidget(self.restore_secure_cells, 2, 0)
        groupBox_post_process_parameters_layout.addWidget(self.prevent_exclusion_of_too_many_cells_together, 2, 1)
        groupBox_post_process_parameters_layout.addWidget(self.max_nb_of_cells_to_be_excluded_together_spinbox, 2, 2)
        groupBox_post_process_parameters_layout.addWidget(cells_text_labels, 2, 3)

        groupBox_post_process.setLayout(groupBox_post_process_parameters_layout)
        input_v_layout.addWidget(groupBox_post_process)
        self.setLayout(input_v_layout)

    def _filter_changed(self):
        current_filter = self.filter_by_cell_size_combo.currentText().lower()
        if 'one' in current_filter:
            self.avg_area_division_or_size_spinbox.setEnabled(False)
        else:
            self.avg_area_division_or_size_spinbox.setEnabled(True)
            if 'divided' in current_filter:
                self.avg_area_division_or_size_spinbox.setValue(2)
            else:
                self.avg_area_division_or_size_spinbox.setValue(300)

    def _get_post_process_filter(self):
        current_filter = self.filter_by_cell_size_combo.currentText().lower()
        if 'one' in current_filter:
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
                containing training parameters

            '''

        self.post_process_parameters={}
        self.post_process_parameters['filter'] = self._get_post_process_filter()
        self.post_process_parameters['correction_factor'] = self.avg_area_division_or_size_spinbox.value()
        self.post_process_parameters['restore_safe_cells'] = self.restore_secure_cells.isChecked()
        self.post_process_parameters[
            'cutoff_cell_fusion'] = self.max_nb_of_cells_to_be_excluded_together_spinbox.value() if self.prevent_exclusion_of_too_many_cells_together.isChecked() else None

        return self.post_process_parameters

    def get_parameters(self):
        return (self.get_parameters_directly())

    @staticmethod
    def getDataAndParameters(parent_window=None):
        # get all the params for augmentation
        dialog = PostProcessGUI(parent_window=parent_window)
        result = dialog.exec_()
        parameters = dialog.get_parameters()
        return (parameters, result == QDialog.Accepted)


if __name__ == '__main__':
    # just for a test
    app = QApplication(sys.argv)
    parameters, ok = PostProcessGUI.getDataAndParameters(parent_window=None)
    print(parameters, ok)
    sys.exit(0)
