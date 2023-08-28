import os
from epyseg.settings.global_settings import set_UI  # set the UI to be used by qtpy
set_UI()
from qtpy import QtGui, QtWidgets
from qtpy.QtCore import Qt, QMimeData
from qtpy.QtWidgets import QTableWidget, QApplication, QAbstractItemView


class MyTableWidget(QTableWidget):
    def __init__(self, parent=None):
        super(MyTableWidget, self).__init__(parent)
        self.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectItems)  # Set the selection behavior to select individual items
        self.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)  # Set the selection mode to select a single item at a time
        self.copy = None

    def keyPressEvent(self, event):
        """
        Handle key press events.

        Args:
            event (QKeyEvent): The key event.

        """
        key = event.key()

        if key == Qt.Key_Delete:
            # Delete key pressed, clear the text of the selected item
            item = self.get_selected_item()
            if item is not None:
                item.setText('')
        elif key == Qt.Key_C and (event.modifiers() & Qt.ControlModifier):
            # Ctrl+C pressed, copy the selected item(s) text
            self.copy = self.get_selection()
        elif key == Qt.Key_V and (event.modifiers() & Qt.ControlModifier):
            # Ctrl+V pressed, paste the copied text to the selected item
            if self.copy is not None:
                item = self.get_selected_item()
                if item is not None:
                    item.setText(self.copy)
        else:
            super(MyTableWidget, self).keyPressEvent(event)

    def get_selection(self):
        """
        Get the selected item(s) text.

        Returns:
            str or list: The selected item(s) text.

        """
        selection = []
        items = self.selectedItems()
        for item in items:
            print('selected', item.text())
            selection.append(item.text())
        if len(selection) == 0:
            return None
        if len(selection) == 1:
            return selection[0]
        return selection

    def get_selected_item(self):
        """
        Get the selected item.

        Returns:
            QTableWidgetItem or None: The selected item.

        """
        items = self.selectedItems()
        for item in items:
            return item


if __name__ == "__main__":
    pass
