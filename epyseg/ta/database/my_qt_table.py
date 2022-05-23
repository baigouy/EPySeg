from PyQt5 import QtGui, QtWidgets
from PyQt5.QtCore import Qt, QMimeData
from PyQt5.QtWidgets import QTableWidget, QApplication, QAbstractItemView


class MyTableWidget(QTableWidget):
    def __init__(self, parent=None):
        super(MyTableWidget, self).__init__(parent)
        # self.setSelectionBehavior(QtWidgets.QAbstractItemView.SingleSelection) #SelectRows
        self.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectItems) #SelectRows
        self.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection) #SelectRows
        self.copy = None

    def keyPressEvent(self, event):
         key = event.key()

         # # find the key ressed
         # if key == Qt.Key_Return or key == Qt.Key_Enter:
         #     # Process current item here
         #    print('test')
         # el
         if    key == Qt.Key_Delete:
            # print("DELETE")
            item =self.get_selected_item()
            if item is not None:
                item.setText('')
         elif  key == Qt.Key_C and (event.modifiers() & Qt.ControlModifier):
             # print('ctrlC')
             # print(self.get_selection())
             # mimeData = QMimeDvata()
             # get selected item text
             # QApplication.clipboard().setMimeData(mimeData)
             self.copy = self.get_selection()
             # if self.copy is not None:
             #    mimeData.setData("text/plain", self.copy)
         elif key == Qt.Key_V and (event.modifiers() & Qt.ControlModifier):
             # mimeData = QMimeData()
             if self.copy is not None:
                 item = self.get_selected_item()
                 if item is not None:
                     item.setText(self.copy)
         else:
             super(MyTableWidget, self).keyPressEvent(event)

        # super(TableView, self).keyPressEvent(event)


    def get_selection(self):
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
        items = self.selectedItems()
        for item in items:
            return item


if __name__ == "__main__":
    pass

