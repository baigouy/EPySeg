# faire deux modes --> un in mem
# faire une table 'properties' si elle existe


# TODO lui faire aussi charger une liste d'images et tt un tas de fichiers db --> à faire
# peut etre faire une table style image properties
# je pourrais reinjecter

# cols to have --> time and d/x ratio and  maybe also the xyz px sizes and the unit of size
# see how and where I can have that done
# maybe also set it as a parameter
# can I allow copy below if 1 or None
# maybe if not known set it to none then it is ignored
# then allow plots
# ce truc pourrait etre genere avant meme de rentrer la dedans à l'etape du preprocess --> good idea and no need to modify the file which is a bit painful

# ça marche mais faut le rendre un peu plus smart

import numpy as np
import sqlite3
# import pandas as pd
# from PyQt5 import QtSql
# from PyQt5.QtCore import Qt
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QWidget, QApplication, QTableWidgetItem, QPushButton, QVBoxLayout

from epyseg.ta.database.sql_column_name_dialog import SQL_column_name_and_type
from epyseg.utils.loadlist import loadlist, smart_TA_list
from epyseg.ta.database.my_qt_table import MyTableWidget
from epyseg.ta.database.sql import TAsql, get_properties_master_db, \
    reinject_properties_to_TA_files


class Example(QWidget):
    def __init__(self, table_name, lst=None, db_to_connect_to=None, non_editable_columns=[0,1]):
        super().__init__()
        self.db_connect = None
        self.non_editable_columns = non_editable_columns
        self.lst = lst
        self.table_name = table_name
        self.initUI()
        self.set_db(db_to_connect_to)
        # self.update_required = False
        self.rows_to_update = []


    def initUI(self):
        hbox = QVBoxLayout(self)

        self.table_widget = MyTableWidget()
        self.table_widget.itemChanged.connect(self.item_changed)

        # db_connect = QtSql.QSqlDatabase.addDatabase("QSQLITE")
        # db_connect.setDatabaseName("exemple.db")
        # db_connect.open()
        # model = QtSql.QSqlQueryModel(parent=self)
        # model.setQuery("SELECT * FROM stocks")
        # tableView.setModel(model)
        #
        # db_connect.close()

        # model = QtGui.QStandardItemModel()

        # print(len(cursor_obj.fetchall()))

        # if self.db_connect is not None:

        # faudrait creer une hbox où je sauve le reste --> TODO but ok for now

        hbox.addWidget(self.table_widget)

        save_button = QPushButton('Save')
        save_button.clicked.connect(self.reinject_modified_properties)
        hbox.addWidget(save_button)

        # self.tab1c.layout = QVBoxLayout()
        self.add_column_to_properties_table_button = QPushButton("Add column")
        self.add_column_to_properties_table_button.clicked.connect(self.add_column_to_properties_table)
        # self.tab1c.layout.addWidget(self.add_column_to_properties_table_button)
        # self.tab1c.setLayout(self.tab1c.layout)
        hbox.addWidget(self.add_column_to_properties_table_button)



        self.setLayout(hbox)

        self.show()

    def get_column_names(self):
        col_names = []
        for i in range(self.table_widget.columnCount()):
            header_item = self.table_widget.horizontalHeaderItem(i)
            col_names.append(header_item.text())
        return col_names

    def add_column_to_properties_table(self):
        # ask the user for new col name and type
        colnmame_n_type, ok = SQL_column_name_and_type.get_value(parent=self,title="New Column Name", existing_column_name=self.get_column_names())
        if ok:
            db = TAsql(filename_or_connection=self.db_connect)
            # We add a column with the given type to the current table
            db.add_column(self.table_name,colnmame_n_type[0],col_type=colnmame_n_type[1]) # maybe format string so that it is nice looking # shall i also specify type of column such as number or string or ignore ???
            self.update_db()
            # Force update all rows! because we have added a new column
            for rrr in range(self.table_widget.rowCount()):
                self.rows_to_update.append(rrr)

    def reinject_modified_properties(self):
        self.reinject_updated_props()

    def set_db(self, db):
        # if a db was already opened before then close it
        self.close()
        # if self.db_connect is not None:
        #     self.db_connect.close()

        if isinstance(db, list):
            self.lst = db
            if db is None or not db:
                db = None
            else:
                db = get_properties_master_db(db)

        if db is None:
            self.db_connect = None
            self.cursor = None
            self.table_widget.clear()
            return

        if isinstance(db, str):
            self.db_connect = sqlite3.connect(db)
            self.cursor = self.db_connect.cursor()
        else:
                self.db_connect =db.con
                self.cursor = db.cur

        self.update_db()


    def update_db(self):
        if self.db_connect is None:
            return

        # print('self.table_name', self.table_name)
        # self.db_connect = db

        # rowcount = cursor.execute('''SELECT COUNT(*) FROM '''+table_name).fetchone()[0]
        #

        # print(rowcount)
        self.cursor.execute('''SELECT * FROM ''' + self.table_name)
        content = self.cursor.fetchall()

        self.table_widget.setRowCount(len(content))
        self.table_widget.setColumnCount(len((content[0])))

        column_names = [description[0] for description in self.cursor.description]

        # print(cursor.rowcount)

        # print(cursor.c)
        # tableView.setColumnCount(cursor.) # --> ok --> needs be set too
        # tableView.setColumnCount(2) # --> ok --> needs be set too

        # for row, form in enumerate(cursor):
        #     for column, item in enumerate(form):
        #         self.tblTable.setItem(row, column, QtGui.QTableWidgetItem(str(item)))
        # first = True

        self.table_widget.itemChanged.disconnect()

        for row, form in enumerate(content):
            # tableView.insertRow(row)

            # print(len(form))
            # if first:
            #     first = False
            #     tableView.setColumnCount(len(form))
            #     tableView.setRowCount(len(cursor))

            for column, item in enumerate(form):
                # print(row, column, str(item))# this is ok
                itm = QTableWidgetItem(str(item))
                # we make unsaved TA only columns not editable to avoid issues
                if self.non_editable_columns is not None and self.non_editable_columns:
                    if column in self.non_editable_columns:
                        itm.setFlags(itm.flags() ^ Qt.ItemIsEditable)
                self.table_widget.setItem(row, column,itm)

        # db_connect.close()
        # tableView.update()

        # need set all column names and headers
        self.table_widget.setHorizontalHeaderLabels(column_names)

        self.table_widget.itemChanged.connect(self.item_changed)
        # self.update_required = False # reset update required
        self.rows_to_update=[]
        # print('oubsi', column_names)

    def item_changed(self, item):
        # self.update_required = True
        # print('something changed --> update required')
        row = item.row()
        # col = item.column()
        self.rows_to_update.append(row)
        # print(self.rows_to_update) # rather use this because it also allows me to save just the necessary stuff --> no need to update all in fact

    # TODO --> do that more properly some day
    def saveToDb(self, tablename):
        table_data = {}
        for i in range(self.table_widget.columnCount()):
            items = []
            for j in range(self.table_widget.rowCount()):
                item = self.table_widget.item(j, i)
                items.append(item.text())
            header_item = self.table_widget.horizontalHeaderItem(i)
            # n_column = str(i) if h_item is None else h_item.text()
            # print(h_item.text())
            table_data[header_item.text()] = items

        # df = pd.DataFrame(data=d)
        # engine = sqlite3.connect(db_filename)
        # df.to_sql(tablename, con=self.db_connect,if_exists='replace')# pas top car ajoute une colonne et fait une conversion de type

        # print(d)
        self.dict_to_sql(table_data, tablename)

    def dict_to_sql(self, values, tablename):
        # columns = ', '.join(values.keys())
        # placeholders = ', '.join('?' * len(values))

        # print(columns, placeholders)

        # what I want is not insert but update all
        # sql = 'INSERT INTO stocks ({}) VALUES ({})'.format(columns, placeholders)
        # sql = 'UPDATE stocks ({}) VALUES ({})'.format(columns, placeholders)
        # sql = ''' UPDATE stocks
        #           SET ({}) VALUES ({})
        #           WHERE id = ?'''.format(columns, placeholders)

        # sql = 'UPDATE table_name SET column1 = value1, column2 = value2...., columnN = valueN WHERE [condition];'


        # values = [int(x) if isinstance(x, bool) else x for x in values.values()]

        # print(values)
        # print(sql)

        # bug because of too many entries

        # for iii in range(len(values[0])):
        #     row = [x[iii] for x in values]
        #     self.cursor.execute(sql, row)

        # keys = values.keys()
        # for value in values.values():

        columns =' = ? ,\n\t'.join(values.keys())+' = ? '
        # print(columns)

        # in fact need update row by row --> see how

        # purchases = [('2006-03-28', 'BUY', 'IBM', 1000, 45.00),
        #              ('2006-04-05', 'BUY', 'MSFT', 1000, 72.00),
        #              ('2006-04-06', 'SELL', 'IBM', 500, 53.00),
        #              ]
        # c.executemany('INSERT INTO stocks VALUES (?,?,?,?,?)', purchases)

        datas = list(values.values())
        # print(datas)
        # print(datas)

        rowid = [iii+1 for iii in range(len(datas[0]))]
        # datas = np.asarray(datas)
        # print(datas.shape)
        # print(rowid)
        # datas = np.vstack([datas, rowid])
        # datas = np.append(datas, rowid)
        datas.append(rowid)
        # print(np.array(datas))
        # ok but needs be reordered into a format that is more friendly --> that takes
        datas = np.asarray(datas).T.tolist()
        # print('-->',datas)
        # update all at once maybe
        # for iii, data in enumerate(datas):
        #     data.append(iii+1)

        # for iii,val in enumerate(datas):

        sql = "UPDATE "+tablename+" \nSET " +columns +"\nWHERE ROWID = ?"
        # sql ='''UPDATE Database_Name.Table_Name
        #         SET Column1_Name = value1, Column2_Name = value2,...
        #         WHERE condition'''
        # val = #("Valley 345", "Canyon 123")

        # print(sql)
        # print(val)

        # self.cursor.execute(sql, val)
        self.cursor.executemany(sql, datas)

        self.db_connect.commit()
        # self.cursor.executemany(sql, values)

    def close(self):
        try:
            if self.db_connect is not None:
                self.db_connect.close()
                self.db_connect = None
        except:
            pass

    def reinject_updated_props(self):
        if self.rows_to_update:
                # lst = self.get_full_list(warn_on_empty_list=False)
                if self.lst is not None and self.lst:
                    self.saveToDb('properties')
                    # self.properties_table.update_required = False
                    # I need get its db and reinject it
                    reinject_properties_to_TA_files(self.lst, TAsql(filename_or_connection=self.db_connect), indices_to_update=self.rows_to_update) # TODO rinject just the necessary lines
                # else maybe ask where to save ???
        self.rows_to_update = []
        # self.properties_table.close()

#
# db_connect = sqlite3.connect("database.db")
# cur = db_connect.cursor()
# cur.execute("SELECT id, num, data FROM table")
# content = cur.fetchall()
# row_count = len(content)
# db_connect.commit()
# cur.close()
# db_connect.close()
# model = QtGui.QStandardItemModel()
#
# lst1 = []
# lst2 = []
# lst3 = []
#
# for i in content:
# lst1.append(i[0])
# lst2.append(i[1])
# lst3.append(i[2])
#
# for i in range(0, row_count):
# item1 = QtGui.QStandardItem(lst1[i])
# item2 = QtGui.QStandardItem(lst2[i])
# item3 = QtGui.QStandardItem(lst3[i])
#
# model.appendRow([item1, item2, item3])
# item1.setEditable(False)
# window.tableView.setModel(model)


def _tmp_remove_props(lst):
    database_list = smart_TA_list(lst, 'pyTA.db')
    for db_file in database_list:
        db = TAsql(db_file)
        db.drop_table('properties2')
        db.close()


# TODO connect this to the main db et creer le truc


if __name__ == "__main__":
    # if True:
    #     db_connect = sqlite3.connect('test.db')
    #     print(type(db_connect))
    #     import sys
    #     sys.exit(0)



    app = QApplication([])

    lst = loadlist('/E/Sample_images/sample_images_PA/trash_test_mem/mini_different_nb_of_channels/list.lst')
    # _tmp_remove_props(lst)



    # db = get_properties_master_db(lst)

    ex = Example(table_name = 'properties')
    ex.set_db(lst)
    ex.show()
    app.exec_()

    # TODO do some code to reinject all the lines into each file of the list --> quite easy
    # just copy row per row and get all but the two first columns

    # in fact I need upate the master db first but ok for a test
    # ex.saveToDb('properties')
    # reinject_properties_to_TA_files(lst, db)

    ex.close()