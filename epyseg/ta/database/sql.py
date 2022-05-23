import sqlite3
import traceback
import os
import numpy as np
import math
from prettytable import PrettyTable
from epyseg.img import Img, has_metadata
from epyseg.ta.selections.selection import convert_coords_to_IDs
from epyseg.tools.early_stopper_class import early_stop
from epyseg.tools.logger import TA_logger  # logging
from epyseg.utils.loadlist import smart_TA_list,loadlist
from epyseg.ta.tracking.tools import smart_name_parser

logger = TA_logger()

# TODO
# do something to list the columns of a table
# TODO implement a database reader so that I can associate the thing to plot to a local id for cells, bonds or vertices
# TODO add possibility to reuse and open connection

class TAsql:

    def __init__(self, filename_or_connection=None, add_useful_missing_SQL_commands=True):
        self.db_name = filename_or_connection
        if isinstance(filename_or_connection, sqlite3.Connection):
            self.con = filename_or_connection
            self.db_name = None
            logger.debug('Opened database from connection')
        elif filename_or_connection is None:
            # if no file name is specified --> create an in mem table
            self.con = sqlite3.connect(":memory:")
            self.db_name = ':memory:'
        else:
            if not os.path.exists(self.db_name):
                parent_dir = os.path.dirname(self.db_name)
                if not os.path.exists(parent_dir):
                    # if parent folder does not exist create it so that the db can be created too....
                    os.makedirs(parent_dir, exist_ok=True)
            self.con = sqlite3.connect(self.db_name)
            logger.debug('Opened database: ' + str(self.db_name))

        if add_useful_missing_SQL_commands:
            self.add_useful_missing_SQL_commands()
        self.cur = self.con.cursor()

    def add_useful_missing_SQL_commands(self):
        # TODO maybe add a few more such as some parsing ones maybe
        self.con.create_function("sin", 1, math.sin)
        self.con.create_function("atan2", 2, math.atan2)
        self.con.create_function("sqrt", 1, math.sqrt)

    def drop_table(self, table_name):
        self.cur.execute('DROP TABLE IF EXISTS ' + table_name)

    def create_and_append_table(self, table_name, datas, temporary=False):
        # I need recover types from dict
        self.create_table(table_name, list(datas.keys()), column_types=get_types_from_data(datas), temporary=temporary)
        # shall I convert data into a master row per row based database ???? --> think about it
        self.fill_table(table_name, datas)

    # if types are specified --> need add it
    def create_table(self, table_name, columns, column_types=None, temporary=False):
        self.drop_table(table_name)
        # print('column_types',column_types)
        # if two lists
        col_content = columns
        if isinstance(col_content, list):

            if column_types is not None:
                concat = list(zip(columns, column_types))
                # print('concat', concat)
                concat = [str(name) + ' ' + str(type) for name, type in concat]
                # print('concat', concat)
                col_content = _list_to_string(concat, add_quotes=False)
            else:
                col_content = _list_to_string(col_content, add_quotes=False)
        # print('col_content',col_content)
        # keep for debug
        # print('CREATE TABLE ' + table_name + ' (' + col_content + ')')
        self.cur.execute('CREATE' + (' TEMPORARY' if temporary else '')+ ' TABLE ' + table_name + ' (' + col_content + ')')

    def fill_table(self, table_name, datas):
        if isinstance(datas, dict):
            #
            # print('requires pre processing')
            # convert to a big list the entire dict
            datas = list(datas.values())
            # print(datas)

            # ok but needs be reordered into a format that is more friendly --> that takes
            datas = np.array(datas, dtype=object).T.tolist()
            # print('arr_t',datas)

        for data in datas:
            # print('data', data) # if data is already a string
            list_as_string = _list_to_string(data)

            # print(list_as_string, data)
            # print('list_as_sting',list_as_string) # --> returns None
            # keep for debug
            # print("INSERT INTO " + table_name + " VALUES (" + list_as_string + ")")



            self.cur.execute(
                "INSERT INTO " + table_name + " VALUES (" + list_as_string + ")")  # can I do it in a smarter way by auto parse now
        self.con.commit()

    def exists(self, table_name, attached_table=None):
        if table_name is None:
            return None
        # get the count of tables with the name

        # self.cur.execute(
        #     "SELECT COUNT(name) FROM tmp.sqlite_master WHERE type='table' AND name='" + table_name.replace('tmp.',
        #                                                                                                    '') + "';")
        #
        # print(self.cur.fetchone())

        # print("SELECT COUNT(name) FROM "+('' if attached_table is None else str(attached_table)+'.')+"sqlite_master WHERE type='table' AND name='" + table_name +"';")

        if not '.' in table_name:
            self.cur.execute("SELECT COUNT(name) FROM " + ('' if attached_table is None else str(
                attached_table) + '.') + "sqlite_master WHERE type='table' AND name='" + table_name + "';")
        else:
            # trick to detect tmp tables
            master, table = table_name.split('.')
            self.cur.execute("SELECT COUNT(name) FROM " + str(
                master) + '.' + "sqlite_master WHERE type='table' AND name='" + table + "';")

        # output = self.cur.fetchone()
        # print(output)

        # if the count is 1, then table exists
        if self.cur.fetchone()[0] == 1:
            return True

        # check also in temp/attached databases
        # shall I remove temp from name
        # self.cur.execute("SELECT COUNT(name) FROM sqlite_temp_master WHERE type='table' AND name='" + table_name.replace() + "';")
        # self.cur.execute("SELECT COUNT(name) FROM tmp.sqlite_master WHERE type='table' AND name='" + table_name.replace('tmp.', '') + "';")
        #
        #
        # print(self.cur.fetchone())
        #
        # if self.cur.fetchone()[0] == 1:
        #     return True
        # sqlite_temp_master
        return False

    def save_query_to_csv_file(self,sql_command, output_file_name):
        # final_scores = []  # image name, iou score epyseg, iou score cellpose, oversegmentation/undersegmentation epyseg, oversegmentation/undersegmentation cellpose
        # final_scores.append(
        #     ['image_name', 'nb cells', 'iou_epyseg', 'iou_cellpose', 'AP_score_epyseg', 'AP_score_cellpose',
        #      'SEG_score_epyseg', 'SEG_score_cellpose', 'segmentation_quality_epyseg',
        #      'segmentation_quality_cellpose'])

        # this will create dupes for the columns
        # data = master_db.run_SQL_command_and_get_results(sql_command)

        # file = open(os.path.join(filename_without_ext, 'nuc_to_golgi_3d.csv'), 'w', newline='')
        # with file:
        #     writer = csv.writer(file, delimiter='\t')  # make csv file tab separated --> not a real csv file
        #     writer.writerows(data)

        # with open('output.csv', 'wb') as f:
        #     writer = csv.writer(f)
        # writer.writerow(['Column 1', 'Column 2', ...])
        # writer.writerows(data)
        #
        # db_df = pd.read_sql_query("SELECT * FROM error_log", conn)
        # db_df.to_csv('database.csv', index=False)

        import pandas as pd
        db_df = pd.read_sql_query(sql_command, self.con)

        # probably can do better but ok for now
        db_df.to_csv(output_file_name,
                     index=False)

    def get_tables(self, force_lower_case=False):
        try:
            self.cur.execute("SELECT name FROM sqlite_master WHERE type='table';")
            query_result = self.cur.fetchall()
            # names = [tb_name[0] for tb_name in query_result]
            names = self._unpack(query_result)
            if force_lower_case:
                names = _to_lower(names)
            return names
        except:
            # something went wrong assume no table
            return None

    def close(self):
        # nb should I do that before closing --> maybe a good idea
        try:
            self.con.commit()
        except:
            # traceback.print_exc()
            pass

        logger.debug('Closing database: ' + str(self.db_name))
        self.con.close()

    # gets the header of the specified table --> will be useful for plots
    def get_table_header(self, tablename):
        # if not db.exists(tablename):
        #     return None
        # header, _ = self.run_SQL_command_and_get_results('SELECT * FROM '+tablename +' LIMIT 1', return_header=True)
        header = self.get_table_column_names_and_types(table_name=tablename, return_colnames_only=True)
        return header

    # /**
    #  * Attaches a new table to the current table (can be used to transfer data
    #  * from one table to another)
    #  *
    #  * @param dbName filename of the table to attach
    #  * @throws SQLException
    #  */
    def attach_table(self, dbName, nickName):
        if dbName is not None:
            # dbName = CommonClasses.change_path_separators_to_system_ones(dbName)
            dbName = dbName.replace('\\\\', '/').replace('\\', '/')
            # self.cur.e("ATTACH DATABASE '" + dbName + "' AS '" + nickName + "'").execute()
            self.execute_command("ATTACH DATABASE '" + dbName + "' AS '" + nickName + "'")

    # /**
    #  * Detaches a table from the current table
    #  *
    #  * @param dbName nickName of the table to detach
    #  * @throws SQLException
    #  */
    def detach_table(self, nickName):
        if nickName is not None:
            # conn.prepareStatement("DETACH DATABASE '" + nickName + "'").execute()
            self.execute_command("DETACH DATABASE '" + nickName + "'")

    def execute_command(self, SQL_command, warn_on_error=True):
        if SQL_command is None:
            return
        try:
            # print('SQL_command',SQL_command)
            self.cur.execute(SQL_command)
            self.cur.fetchall()
            self.con.commit()
        except:
            if warn_on_error:
                traceback.print_exc()
                logger.error(
                    'error executing the following command:\n"' + str(SQL_command) + '"' + '\ntable name:' + str(
                        self.db_name))

    def EXCEPT(self, table_name, *columns_to_exclude): #concatTableName=False,
        # col_name = self.getColumns(table_name)

        # print('columns_to_exclude',columns_to_exclude)


        columns_to_exclude = [col.lower() for col in columns_to_exclude]
        columns = self.get_table_column_names_and_types(table_name, return_colnames_only=True)
        columns = [col.lower() for col in columns]
        columns = [col for col in columns if col not in columns_to_exclude]

        # columns_to_excludelist = [0]
        #
        # if columns_to_exclude is not None and columns_to_excludelist:
        #     for columns_to_excludelist1 in columns_to_exclude:
        #         columns_to_excludelist.append(columns_to_excludelist1.toLowerCase())
        # else:
        #     return "" + table_name + ".*"
        # out = ""
        # for col_name1 in col_name:
        #     string = col_name1.strip().lower()
        #     if string not in columns_to_excludelist:
        #         if concatTableName:
        #             # /* added handling of quotes to avoid pbs with weird names */
        #             out += " \"" + table_name + "." + string + "\","
        #         else:
        #             # /* added handling of quotes to avoid pbs with weird names */
        #             out += " \"" + string + "\","
        #
        # if (out.endsWith(",")):
        #     out = out.substring(0, out.length() - 1)
        # return out

        return ', '.join(columns)

    def run_SQL_command_and_get_results(self, SQL_command, return_header=False, warn_on_error=True):
        if SQL_command is None:
            if return_header:
                return None, None
            return None

        if SQL_command.count(';') > 1:
            # need split the command and effectively return just the last output
            # à tester
            SQL_commands = SQL_command.strip().split(';')

            SQL_commands = [sql_command for sql_command in SQL_commands if sql_command.strip() != '']
            # print(SQL_commands)
            if len(SQL_commands) > 1:
                last_command = SQL_commands[-1]
                for command in SQL_commands[:-1]:
                    # print('running ', command)
                    self.run_SQL_command_and_get_results(command, return_header=False, warn_on_error=warn_on_error)
                # print('running last command and getting results from it', last_command)
                return self.run_SQL_command_and_get_results(last_command, return_header=return_header,
                                                            warn_on_error=warn_on_error)

        try:
            self.cur.execute(SQL_command)
            query_result = self.cur.fetchall()
            # print(len(query_result[0]))

            # print('inside', query_result)

            if return_header:
                # print('self.cur.description', self.cur.description)  # --> indeed contains table header
                # shall I return it ???
                # headers = [header[0] for header in self.cur.description]
                headers = self._unpack(self.cur.description)
                # print('headers', headers)
                return headers, query_result

            return query_result
            # if len(query_result[0]) < 15:
            #     # cur.execute(SINGLE_CLONE_FERET_SQL_COMMAND)
            #     try:
            #         cur.executescript(SINGLE_CLONE_FERET_SQL_COMMAND_3D)
            #     except:
            #         cur.executescript(SINGLE_CLONE_FERET_SQL_COMMAND)
            # else:
            #     # cur.execute(DOUBLE_CLONE_FERET_SQL_COMMAND)
            #     try:
            #         cur.executescript(DOUBLE_CLONE_FERET_SQL_COMMAND_3D)
            #     except:
            #         cur.executescript(DOUBLE_CLONE_FERET_SQL_COMMAND)
            # cur.fetchall()
        except:
            # command failed (e.g. table does not exist, ...)
            if warn_on_error:
                traceback.print_exc()
            # --> return None
            if return_header:
                return None, None
            return None

        # pandas can be useful to do plots --> TODO
        # df = pd.read_sql_query("SELECT ROWID AS frame_nb,* FROM feret_dynamic", db)

        # print('nb of columns', len(df.columns))

        # print(df[0])

        # df.columns[0]='frame_nb'
        # print(df)

        # if len(df.columns) <= 3:
        #     fig = df.plot(x='frame_nb', y=['area_variation', 'FERET_VARIATION'], ylim=ylim)
        # else:
        #     fig = df.plot(x='frame_nb', y=['area_variation', 'FERET_VARIATION', 'area2_variation', 'FERET_VARIATION2'],
        #                   ylim=ylim)

    #   /*
    # * this removes all unnecessary data from the db --> strong size reduction
    # */
    def clean(self):
        logger.debug('Cleaning the database: ' + str(self.db_name))
        self.run_SQL_command_and_get_results("VACUUM;")

    # returns the values of one column --> very useful
    # sort can be 'ASC' or 'DESC'
    def get_column(self, table_name, column_name, sort=None):
        if not self.exists(table_name) or column_name is None:
            return None
        try:
            # print('SELECT "'+column_name+'" FROM "'+table_name+'"'+ ('' if sort is None else 'ORDER BY "'+column_name+'" '+sort))
            results = self.run_SQL_command_and_get_results('SELECT "' + column_name + '" FROM "' + table_name + '"' + (
                '' if sort is None else 'ORDER BY "' + column_name + '" ' + sort), return_header=False,
                                                           warn_on_error=False)
            results = self._unpack(results)
            return results
        except:
            return None

    def _unpack(self, lst):
        out = [elem[0] for elem in lst]
        return out

    def get_table_column_names_and_types(self, table_name, return_colnames_only=False, attached_table=None):
        # check also if in temp tables maybe
        # print('self.exists(',attached_table, table_name, self.exists(table_name, attached_table=attached_table))
        if not self.exists(table_name, attached_table=attached_table):
            return None

        if not '.' in table_name:
            cols_and_types = self.run_SQL_command_and_get_results('PRAGMA ' + ('' if attached_table is None else str(
                attached_table) + '.') + 'table_info("' + table_name + '");')  # does not return attached ones --> need another code
        else:
            master, table = table_name.split('.')
            cols_and_types = self.run_SQL_command_and_get_results(
                'PRAGMA ' + master + '.' + 'table_info("' + table + '");')  # does not return attached ones --> need another code

        # attached_table
        if cols_and_types is None:
            return None
        # print(cols_and_types)
        cols_and_types = {col[1]: col[2] for col in cols_and_types}
        # print(cols_and_types)
        if return_colnames_only:
            return list(cols_and_types.keys())
        else:
            return cols_and_types



    # a quick and easy way to get max and min of data
    # see if I can do this in a smarter way --> for example based on a query or on the last column
    def get_min_max(self, table_name, column_name, freq=None, ignore_None_and_string=True, force_numeric=False):
        if table_name is None or column_name is None:
            return None, None
        sorted_data = self.get_column(table_name, column_name, sort='ASC')
        if sorted_data is None:
            return None, None

        return sort_col_numpy(sorted_data, freq=freq, ignore_None_and_string=ignore_None_and_string, force_numeric=force_numeric, sort=False)
        # # slower but can handle strings --> may be useful
        # if force_numeric:
        #     sorted_data = self.any_to_numeric(sorted_data)
        # # print('sorted_data', len(sorted_data))
        # if ignore_None_and_string:
        #     sorted_data = [val for val in sorted_data if val or isinstance(val, str)]
        # length = len(sorted_data)
        # # print('sorted_data', len(sorted_data))
        #
        # min = sorted_data[0]
        # max = sorted_data[-1]
        # if freq is None or freq == 0.:
        #     # print('min, max',min, max)
        #     return min, max
        # else:
        #     if isinstance(freq, list) or isinstance(freq, tuple):
        #         if len(freq) == 1:
        #             lower = freq[0]
        #             upper = freq[0]
        #         else:
        #             lower = freq[0]
        #             upper = freq[1]
        #         # print('lower, upper',lower, upper)
        #     else:
        #         lower = freq
        #         upper = freq
        #         # print('lower, upper', lower, upper)
        # if lower > 0.:
        #     # convert frequency to value
        #     # get closest element
        #     idx = round(lower * length)
        #     try:
        #         min = sorted_data[idx]
        #     except:
        #         pass
        #     # print('idx',idx, length)
        # if upper > 0.:
        #     idx = round(upper * length)
        #     # max =
        #     # print('idx',idx, length)
        #     try:
        #         max = sorted_data[-idx]
        #     except:
        #         pass
        # # print('min, max',min, max)
        # return min, max



    # /**
    #  * counts the nb of rows of a table
    #  *
    #  * @param tableName table name
    #  * @return nb of rows
    #  */
    def getNbRows(self, tableName):
        if not self.exists(tableName):
            return 0

        SQLQuery = "SELECT COUNT(*) from '" + tableName + "';"
        value = self.run_SQL_command_and_get_results(SQLQuery)

        # print(value)
        try:
            return value[0][0]
        except:
            # LogFrame2.printStackTrace(e);
            print('error')

        # /* if there was an error return -1 */
        return 0



    def add_column(self, table_name, column_name, col_type=None, default_value='NULL'):
        # print('adding column ', table_name, column_name)
        if not self.exists(table_name):
            logger.error('table ' + str(table_name) + ' does not exist.')
            return

        # KEEP NB adding quotes around the table creates pbs when using an attached table such as tmp.cells_2D --> see how to do that properly...
        # print('ALTER TABLE '+table_name+' ADD COLUMN "'+column_name+'"'+ ('' if default_value is None else '  DEFAULT ' +str(default_value))+';')
        # print('ALTER TABLE "'+table_name+'" ADD COLUMN "'+column_name+'"  DEFAULT ' +default_value+';')
        self.execute_command('ALTER TABLE ' + table_name + ' ADD COLUMN "' + column_name + '"' + (
            '' if col_type is None else (' ' + str(col_type) + ' ')) + (
                                 '' if default_value is None else '  DEFAULT ' + str(
                                     default_value)) + ';')  # VARCHAR DEFAULT 'N';

    def remove_column(self, table_name, column_name):
        if not self.exists(table_name):
            logger.error('table ' + str(table_name) + ' does not exist.')
            return
        try:
            self.execute_command('ALTER TABLE ' + table_name + ' DROP COLUMN "' + column_name + '"')
        except:
            # if the column does not exist -> nothing to drop but no big deal
            logger.warning('column does not exist and could not be dropped ' + str(table_name) + ' ' + str(column_name))
            pass

    def print_query(self, SQL_command):
        # prettytable
        # https://stackoverflow.com/questions/10865483/print-results-in-mysql-format-with-python
        # https://stackoverflow.com/questions/9516247/how-to-print-a-mysqldb-unicode-result-in-human-readable-way
        header, data = self.run_SQL_command_and_get_results(SQL_command, return_header=True)

        # import prettytable
        x = PrettyTable(header)
        for row in data:
            x.add_row(row)
        print(x)

    # /**
    #  * Checks if a table is empty or does not exist, false otherwise
    #  *
    #  * @param tableName the name of the table
    #  * @return true if the table is empty or does not exist, false otherwise
    #  */
    def isTableEmpty(self, tableName):
        if not self.exists(tableName):
            return True

        if self.getNbRows(tableName) == 0:
            return True

        return False

    # see how I can do that
    # should I run a master SQL command before that??? --> not so easy --> try be smart
    # this stuff converts a clone to a table
    # not so easy

    # maybe I can post filter but not so easy for the master db --> see how I can do that
    # maybe simpler to add the column to the clone db and to do a join on the cell id column if available rather than a natural join because otherwise it will miss cells as soon as they have changed their point inside the cell
    # en fait ce truc convertit juste des coords en local IDs et en cree une table
    # puis je faire la filtration hors sql aussi ???
    # maybe not that bad
    # sinon convertir ça en une très longue query mais sub optimal
    # maybe I would then not even need to create a table --> that would make my life easier in a way

    # very good too --> can use the SELECT * FROM cells_2D WHERE local_id IN (120, 2, 3); # --> selectionne juste 3 cellules qui peuvent etre utilisees pr refiltrer une table --> assez facile à faire

    # filtering_elements_dict must be a dict where the key is the column name and the values is a list of elements that need be filtered --> {'area':[120, 130], 'local_ID':[33]} # --> will keep only cell with a certain local ID and certain area from the main query
    def create_filtered_query(self, SQL_command, filtering_elements_dict=None):

        # for each entry
        if filtering_elements_dict is None:
            # empty filter --> nothing todo --> return default command
            return SQL_command

        if not filtering_elements_dict:
            return SQL_command


        initial_command = 'SELECT * FROM (' + SQL_command + ') WHERE '
        extra_command = ''
        for kkk, (k, v) in enumerate(filtering_elements_dict.items()):
            filters = ', '.join(str(f) for f in v )
            extra_command+=' '+(' AND ' if kkk>0 else '')+k + ' IN (%s)' % filters




        query =  initial_command +extra_command
        return query

    # create filtered query
    # this further filters a query --> quite powerful in fact and simpler than creating a temp table
    def create_filtered_query_old(self, SQL_command, filtering_elements, filter_name='local_ID'):
        filters = ', '.join(str(f) for f in filtering_elements)
        query = 'SELECT * FROM ('+SQL_command+') WHERE '+filter_name+' IN (%s)' % filters
        # print(query)
        # cursor.execute(query, l)
        # self.execute_command()
        return query

    # do I even need that ???
    def filter_by_clone(self, table_name, cell_label):
        # TODO I need to embed some of this in the clone and in the master db creation so that there is no bug...

        # this is an easy way to convert a clone to a db filter out all the necessary things of the clone --> need convert it to a single method so that everything is easy TODO and to handle
        # devrait vraiment pas etre trop dur a faire du coup
        # try to filter a db based on clones --> probably not that hard to do -->
        # load the db and try filter
        # filename = '/E/Sample_images/sample_images_PA/trash_test_mem/mini_empty/focused_Series012.png'
        # db_path = smart_name_parser(filename, 'pyTA.db')
        # table_name = 'tracked_clone'
        coords = []
        # db = None
        try:
            # db = TAsql(db_path)
            # now I need get the coords from the db and check the image for that
            # NB THIS TABLES ASSUMES THE QUERY RETURNS 2D COORDS --> ALWAYS AN X AND A Y
            # headers,
            coords = self.run_SQL_command_and_get_results('SELECT * FROM ' + table_name, return_header=False)
        except:
            traceback.print_exc()
        # finally:
        #     if db is not None:
        #         try:
        #             db.close()
        #         except:
        #             traceback.print_exc()
        # now I have the coords and I need to tranaslate them back into a local ID --> then do it
        # maybe if there is a mismatch between local ID in Db and in the label image and or the coord I could warn the user so that he updates his files --> and there would be no need for a drastic change I guess and no need for a check
        # print(coords)
        # now we convert coords to local ID using the handCorrection mask
        # --> TODO
        # see if I need the cells file or not
        # convert_coords_to_IDs(img_to_analyze_RGB24_or_label, selected_coords, forbidden_colors = [0xFFFFFF,0], return_image=False, new_color_to_give_to_cells_if_return_image=None):

        # handcorrection = handCorrection_to_label(Img(smart_name_parser(filename, 'handCorrection.tif')))
        local_ids = convert_coords_to_IDs(cell_label, coords, forbidden_colors=[0], return_image=False,
                                          new_color_to_give_to_cells_if_return_image=None)
        # print('equivalent local ids', local_ids)

        # I could create a temporary table to filter the db
        # db = None
        # try:
        #     db = TAsql(db_path)
        # try filter the cells db based on the local IDs extracted from the clones
        # --> maybe create a temp table

        # local_ID
        # self.create_and_append_table(tmp_table_name, {'local_ID': local_ids},temporary=True)  # --> perfect that seems to work!!!
        # print(len(local_ids))
        # the addition of the clone can be done left or right but will gain a lot of time if left # nut maybe simpler on the right of it --> try that!!!
        # filtered_db = db.run_SQL_command_and_get_results('SELECT * FROM cells_2D NATURAL JOIN tmp_filtering_table')
        # print(filtered_db)
        # print(len(filtered_db))

        # ça marche vraiment du tonnerre du coup ça va etre assez facile d'ajouter des clones aux plots et de faire les plots de l'ovipositor
        # --> TODO
        # db.print_query('SELECT * FROM cells_2D NATURAL JOIN tmp_filtering_table')

        # SELECT * FROM cells_2D NATURAL JOIN tmp_filtering_table   --> in fact this is faster and safer --> maybe this is what I should rather do
        # SELECT * FROM tmp_filtering_table NATURAL JOIN cells_2D
        # except:
        #     traceback.print_exc()
        # finally:
        #     if db is not None:
        #         try:
        #             db.close()
        #         except:
        #             traceback.print_exc()
        # ça a l'air de marcher --> maintenant il ne me reste plus qu'à filtrer les dbs
        # img = convert_coords_to_IDs(handcorrection, coords, forbidden_colors = [0], return_image=True, new_color_to_give_to_cells_if_return_image=None)
        # plt.imshow(img)
        # plt.show()
        return local_ids


    def filter_by_clone_old(self, table_name, cell_label, tmp_table_name='tmp_filtering_table'):
        # TODO I need to embed some of this in the clone and in the master db creation so that there is no bug...

        # this is an easy way to convert a clone to a db filter out all the necessary things of the clone --> need convert it to a single method so that everything is easy TODO and to handle
        # devrait vraiment pas etre trop dur a faire du coup
        # try to filter a db based on clones --> probably not that hard to do -->
        # load the db and try filter
        # filename = '/E/Sample_images/sample_images_PA/trash_test_mem/mini_empty/focused_Series012.png'
        # db_path = smart_name_parser(filename, 'pyTA.db')
        # table_name = 'tracked_clone'
        coords = []
        # db = None
        try:
            # db = TAsql(db_path)
            # now I need get the coords from the db and check the image for that
            # NB THIS TABLES ASSUMES THE QUERY RETURNS 2D COORDS --> ALWAYS AN X AND A Y
            # headers,
            coords = self.run_SQL_command_and_get_results('SELECT * FROM ' + table_name, return_header=False)
        except:
            traceback.print_exc()
        # finally:
        #     if db is not None:
        #         try:
        #             db.close()
        #         except:
        #             traceback.print_exc()
        # now I have the coords and I need to tranaslate them back into a local ID --> then do it
        # maybe if there is a mismatch between local ID in Db and in the label image and or the coord I could warn the user so that he updates his files --> and there would be no need for a drastic change I guess and no need for a check
        # print(coords)
        # now we convert coords to local ID using the handCorrection mask
        # --> TODO
        # see if I need the cells file or not
        # convert_coords_to_IDs(img_to_analyze_RGB24_or_label, selected_coords, forbidden_colors = [0xFFFFFF,0], return_image=False, new_color_to_give_to_cells_if_return_image=None):

        # handcorrection = handCorrection_to_label(Img(smart_name_parser(filename, 'handCorrection.tif')))
        local_ids = convert_coords_to_IDs(cell_label, coords, forbidden_colors=[0], return_image=False,
                                          new_color_to_give_to_cells_if_return_image=None)
        # print('equivalent local ids', local_ids)

        # I could create a temporary table to filter the db
        # db = None
        # try:
        #     db = TAsql(db_path)
        # try filter the cells db based on the local IDs extracted from the clones
        # --> maybe create a temp table

        # local_ID
        self.create_and_append_table(tmp_table_name, {'local_ID': local_ids},
                                   temporary=True)  # --> perfect that seems to work!!!
        # print(len(local_ids))
        # the addition of the clone can be done left or right but will gain a lot of time if left # nut maybe simpler on the right of it --> try that!!!
        # filtered_db = db.run_SQL_command_and_get_results('SELECT * FROM cells_2D NATURAL JOIN tmp_filtering_table')
        # print(filtered_db)
        # print(len(filtered_db))

        # ça marche vraiment du tonnerre du coup ça va etre assez facile d'ajouter des clones aux plots et de faire les plots de l'ovipositor
        # --> TODO
        # db.print_query('SELECT * FROM cells_2D NATURAL JOIN tmp_filtering_table')

        # SELECT * FROM cells_2D NATURAL JOIN tmp_filtering_table   --> in fact this is faster and safer --> maybe this is what I should rather do
        # SELECT * FROM tmp_filtering_table NATURAL JOIN cells_2D
        # except:
        #     traceback.print_exc()
        # finally:
        #     if db is not None:
        #         try:
        #             db.close()
        #         except:
        #             traceback.print_exc()
        # ça a l'air de marcher --> maintenant il ne me reste plus qu'à filtrer les dbs
        # img = convert_coords_to_IDs(handcorrection, coords, forbidden_colors = [0], return_image=True, new_color_to_give_to_cells_if_return_image=None)
        # plt.imshow(img)
        # plt.show()


def any_to_numeric( values):
    if values is None:
        return None
    for iii, v in enumerate(values):
        if isinstance(v, str):
            if '.' in v:
                try:
                    values[iii] = float(v)
                except ValueError:
                    values[iii] = None
            else:
                try:
                    values[iii] = int(v)
                except ValueError:
                    values[iii] = None
    return values

def get_numeric_value( v):
    if isinstance(v, str):
        if '.' in v:
            try:
                return float(v)
            except ValueError:
                return None
        else:
            try:
                return int(v)
            except ValueError:
                return None
    return v

def sort_col_numpy(unsorted_data, freq=None, ignore_None_and_string=True, force_numeric=False, sort=True):

    if unsorted_data is None:
        return None, None

    sorted_data = unsorted_data

        # slower but can handle strings --> may be useful
    if force_numeric:
        sorted_data = any_to_numeric(sorted_data)
        # shall I resort ???


        # print('sorted_data', len(sorted_data))
    if ignore_None_and_string:
        # is not none is required otherwise zeros are removed and we don't want that
        sorted_data = [val for val in sorted_data if val is not None or isinstance(val, str)]

    if sort:
        sorted_data = np.sort(unsorted_data)
        # print(sorted_data)
        # print('sorting')

        # shall I resort ???

    length = len(sorted_data)
    # print('sorted_data', len(sorted_data))

    min = sorted_data[0]
    max = sorted_data[-1]

    # print(min, max, sorted_data) # all the zeros have been removed --> why???

    # nb if nan or none need select non complex values
    if freq is None or freq == 0.:
        # print('min, max',min, max)
        return min, max
    else:
        if isinstance(freq, list) or isinstance(freq, tuple):
            if len(freq) == 1:
                lower = freq[0]
                upper = freq[0]
            else:
                lower = freq[0]
                upper = freq[1]
            # print('lower, upper',lower, upper)
        else:
            lower = freq
            upper = freq
            # print('lower, upper', lower, upper)
    if lower > 0.:
        # convert frequency to value
        # get closest element
        idx = round(lower * length)
        try:
            min = sorted_data[idx]
        except:
            pass
        # print('idx',idx, length)
    if upper > 0.:
        idx = round(upper * length)
        # max =
        # print('idx',idx, length)
        try:
            max = sorted_data[-idx]
        except:
            pass
    # print('min, max',min, max)
    return min, max


def update_db_properties_using_image_properties(input_file, ta_path):
    try:
        tmp = Img(input_file)
        if has_metadata(tmp):
            db_path = os.path.join(ta_path, 'pyTA.db')
            db = TAsql(db_path)
            try:
                # these are all the variables of the db I may want to keep...
                voxel_size_x = None
                voxel_size_y = None
                voxel_size_z = None
                voxel_z_over_x_ratio = None
                time = None
                creation_time = None

                if 'vx' in tmp.metadata:
                    voxel_size_x = tmp.metadata['vx']
                if 'vy' in tmp.metadata:
                    voxel_size_y = tmp.metadata['vy']
                if 'vz' in tmp.metadata:
                    voxel_size_z = tmp.metadata['vz']
                if 'AR' in tmp.metadata:
                    voxel_z_over_x_ratio = tmp.metadata['AR']
                if 'time' in tmp.metadata:
                    time = tmp.metadata['time']
                if 'creation_time' in tmp.metadata:
                    creation_time = tmp.metadata['creation_time']

                neo_data = {'voxel_size_x': voxel_size_x,
                            'voxel_size_y': voxel_size_y,
                            'voxel_size_z': voxel_size_z,
                            'voxel_z_over_x_ratio': voxel_z_over_x_ratio,
                            'time': time,
                            'creation_time': creation_time}

                if db.exists('properties'):
                    # db exists --> update it rather than recreating everything
                    header, cols = db.run_SQL_command_and_get_results('SELECT * FROM properties', return_header=True)
                    cols = cols[0]
                    data = _to_dict(header, cols)
                    for key in list(neo_data.keys()):
                        if neo_data[key] is None:
                            if not key in data:
                                data[key] = neo_data[key]
                        else:
                            data[str(key)] = neo_data[key]
                else:
                    data = neo_data

                data = {k: [v] for k, v in data.items()}
                db.create_and_append_table('properties', data)
            except:
                traceback.print_exc()
                print('An error occurred while reading properties from the image or writing them to the database')
            finally:
                try:
                    db.close()
                except:
                    pass
        # else:
        #     # image has no metadata --> nothing TODO --> I can just skip things
        #     pass
        del tmp
    except:
        traceback.print_exc()
        print('Error could not save image properties to the TA database')

def _to_dict(header, col):
    dct = {}
    for iii, he in enumerate(header):
        dct[he] = col[iii]
    # print(dct)
    return dct


def _list_to_string(lst, add_quotes=True):
    if isinstance(lst, list):
        # if not add_quotes:
        #     return ', '.join(map(str, lst))
        # else:
        #     return "'" + '\', \''.join(map(str, lst)) + "'"
        # small change to allow for conversion of None values to Null SQL which is I think what I want
        return ', '.join(map(lambda x: "'" + str(x) + "'" if x is not None and add_quotes else str(x) if x is not None and not add_quotes else 'Null', lst))
    elif isinstance(lst, dict):
        # return ', '.join(str(x) + ' ' + str(y) for x, y in lst.items())
        return ', '.join((str(x) + ' ' + str(y)) if y is not None else str(x) + ' ' + 'Null' for x, y in lst.items()) # TODO maybe add support for quotes too...
    # elif lst is not None:
    #     # assume just a single value is passed --> convert it to a list and continue
    #     return _list_to_string([lst], add_quotes=add_quotes)

# TODO get master db --> TODO --> required for global plots --> easy way of doing things in fact, I love it

# maybe allow unknown
def get_type(value):
    if isinstance(value, bool): # or isinstance(value, bool): # deprecated apparently
        return 'BOOLEAN'
    if isinstance(value, str):
        return 'TEXT'
    if isinstance(value, int):
        return 'INTEGER'
    if isinstance(value, float):
        # types.append('REAL')
        return 'FLOAT'

    if isinstance(value, list):
        return 'TEXT'

    # add numpy types --> maybe need more.....
    if isinstance(value, np.int64) or isinstance(value, np.uint64) \
            or isinstance(value, np.int32) or isinstance(value, np.uint32) \
            or isinstance(value, np.int8) or isinstance(value, np.uint8):
        return 'INTEGER'

    # somehow some values are saved like that error type not supported --> report this to baigouy@gmail.com <class 'epyseg.img.Img'> 1034 --> dirty hack --> best would be to compute value wand not return a piece of image but just convert it
    # what I do here is not great easpecially if I wanna save images some day into the db, but I would then rather save them as BLOBS --> maybe ok
    if isinstance(value, Img):
        return 'INTEGER'

    # if type is unknow put None or TEXT ??? or BLOB
    # if not isinstance(value, Img):
    print('error type not supported --> report this to baigouy@gmail.com', type(value), value)
    return 'TEXT'


def get_types_from_data(one_row_of_data):
    types = []
    if isinstance(one_row_of_data, dict):
        for k, v in one_row_of_data.items():
            if v is None:
                types.append('TEXT')
                continue

            if not isinstance(v, list):
                types.append(get_type(v))
            else:
                # loop over the list until a not None value is found
                success = False
                for vv in v:
                    if vv is not None:
                        types.append(get_type(vv))
                        success = True
                        break
                if not success:
                    types.append('TEXT')
    else:
        # loop over the one row of data
        for data in one_row_of_data:
            types.append(get_type(data))
            # continue
            # TODO should I support blob .??? --> maybe
            # if isinstance(data, double):
            #     # types.append('REAL')
            #     types.append('FLOAT')
            #     continue
    return types


def tst_frequency_from_list():
    a = [1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 4, 5, 5, 5, 5, 5]
    from itertools import groupby
    print([len(list(group)) for key, group in groupby(a)])

    print({key: len(list(group)) for key, group in
           groupby(a)})  # --> that is what I want --> I have the frequence then I need compute the histo

    mn = min(a)
    mx = max(a)
    total = len(a)
    print(mn, mx, total)

    # or even better
    import collections
    # using Counter to find frequency of elements
    frequency = collections.Counter(a)

    # printing the frequency
    print(dict(frequency))
    # {'A': 3, 'B': 3, 'C': 1, 'D': 2}

    # this is one way of doing, cool also do it very quickly in sql directly --> easy to get max and min by percentage --> sort values ascending then get total count of values then get x percent from top and x percent from bottom
    # qdsqdsqdqsdq

    # print(dict(tuple(groupby(a))))


def _to_lower(lst):
    return [str(elm).lower() for elm in lst]


# def create_or_update_empty_track_cells_db_if_missing(db_name):
#
#     # if the table does not exist or then create it
#
#     # make sure the table is in the query otherwise forget about it
#
#     # CREATE TABLE cell_tracks AS SELECT local_ID, NULL as track_id FROM cells_2D
#     # SELECT track_id FROM cell_tracks LIMIT 1
#
#     # maybe if track_id is null then update it on the fly or if table is missing
#     # --> TODO
#
#     # maybe update those in order to be sure I don't lose anything can I do this with a natural JOIN
#
#     pass


#     # how can I see if all is ok and need check before hand that all is ok
#     # force ensure it --> shall I delete it again ?? or simply create it as a querry -> fairly easy in fact and it will not damage or change anything
#     pass
# if table does not exist and starts with cell add it a cell local_id
# if it is a bond one then add it


# TODO do a get min and a get max for the table or some of its columns --> terribly useful in fact!!!
# TODO if table track cells does not exist --> need add it --> need copy the localID of cells and add NUll to the other column
# this is really required


# TODO KEEP MEGA URGENT do force_track_cells_db_update for bonds and vertices when the tracking will be implemented.

# TODO --> find a way to handle clones here
# sqdsqdsqd
def createMasterDB(lst, outputName=None, progress_callback=None, database_name=None, force_track_cells_db_update=False):
    masterDB = None
    try:
        frame_nb = "frame_nb"
        fileName = "filename"
        # /**
        #  * The master db containing data from all other frames --> I need to
        #  * copy and append tables from small dbs into a bigger one
        #  */
        if outputName is not None:
            try:
                os.remove(outputName)
            except OSError:
                pass
        database_list = smart_TA_list(lst, 'pyTA.db')

        masterDB = TAsql(filename_or_connection=outputName, add_useful_missing_SQL_commands=False)
        for l, db_l in enumerate(database_list):
            try:
                if early_stop.stop:
                    return
                if progress_callback is not None:
                    progress_callback.emit((l / len(database_list)) * 100)
                else:
                    logger.info(str((l / len(database_list)) * 100) + '%')
            except:
                pass
            # need get the path
            # output_folder = CommonClasses.getName(list.get(l))

            # if some columns are missing in some case --> add them too --> TODO

            dbHandler = None
            try:
                dbHandler = TAsql(filename_or_connection=db_l)
                # /**
                #  * Get all tables and create missing ones. If table already
                #  * exists, then just append it otherwise create it, then
                #  * append it --> how can I do that ???
                #  */
                tables = dbHandler.get_tables()
                if force_track_cells_db_update:
                    # could make this a method --> more genric to handle everyDB create_or_update_empty_track_cells_db_if_missing
                    force_update = False
                    if 'cell_tracks' not in tables or dbHandler.isTableEmpty('cell_tracks'):
                        force_update == True
                    if not force_update:
                        try:
                            data = dbHandler.run_SQL_command_and_get_results('SELECT track_id FROM cell_tracks LIMIT 1')
                            if data == None or data[0][0]==None:
                                force_update = True
                        except:
                            force_update = True
                        if force_update:
                            dbHandler.drop_table('cell_tracks')
                            dbHandler.execute_command('CREATE TABLE cell_tracks AS SELECT local_id as local_id, CAST(NULL AS INTEGER) AS track_id FROM cells_2D')
                    tables = dbHandler.get_tables()


                masterDBTables = masterDB.get_tables()

                # print('masterDBTables',masterDBTables) # --> empty

                # need get table and header type
                for string in tables:
                    if database_name is not None:
                        if database_name.lower() != string.lower():
                            continue
                    if string not in masterDBTables:  # (!masterDBTables.contains(string)) {
                        tableHeaderAndType = dbHandler.get_table_column_names_and_types(string)
                        extra = {}
                        extra[frame_nb] = 'INTEGER'
                        extra[fileName] = 'TEXT'


                        # probably need that too for the master db creation
                        # # if we append the extra properties then do that
                        # if append_TA_properties:
                        #     if string.lower != 'properties':
                        #         # add all the tbales from the extra db
                        #         table_properties_header_n_type = dbHandler.get_table_column_names_and_types('properties')
                        #         for string1, typ in table_properties_header_n_type.items():
                        #             extra[string1]=typ

                        for string1 in tableHeaderAndType.keys():
                            extra[string1] = tableHeaderAndType[string1]
                        masterDB.create_table(string, list(extra.keys()), list(extra.values()))

                masterDB.attach_table(db_l, 'tmp')  # --> so it is needed then # do I use that ????
                dbHandler.close()
                # in case there is a mismatch in the columns then do stuff to fix it

                for string in tables:
                    if database_name is not None:
                        if database_name.lower() != string.lower():
                            continue
                    DB_to_read = 'tmp.' + str(string)
                    cols = masterDB.get_table_column_names_and_types(DB_to_read)
                    cols_master = masterDB.get_table_column_names_and_types(string)

                    #



                    # https://stackoverflow.com/questions/2082152/case-insensitive-dictionary --> maybe could be useful to have a case insensitive dict or https://stackoverflow.com/questions/3296499/case-insensitive-dictionary-search


                    # print('tmp.'+string, cols)
                    # print(string, cols_master)

                    fixed_cols = set(_to_lower(cols.keys()))
                    fixed_cols.add(frame_nb)
                    fixed_cols.add(fileName)

                    # error mismatch: --> see how to fix that
                    #  {'local_ID': 'INTEGER', 'length': 'FLOAT', 'pixel_count': 'INTEGER', 'orientation': 'FLOAT', 'vx_1': 'INTEGER', 'vx_2': 'INTEGER', 'first_pixel_x': 'INTEGER', 'first_pixel_y': 'INTEGER', 'cell_id1_around_bond': 'INTEGER', 'cell_id2_around_bond': 'INTEGER', 'is_border_bond': 'INTEGER', 'sum_px_int_vertices_included_ch0': 'TEXT', 'avg_px_int_vertices_included_ch0': 'TEXT'} vs:
                    #  {'frame_nb': 'INTEGER', 'filename': 'TEXT', 'local_ID': 'INTEGER', 'length': 'FLOAT', 'pixel_count': 'INTEGER', 'orientation': 'FLOAT', 'vx_1': 'INTEGER', 'vx_2': 'INTEGER', 'first_pixel_x': 'INTEGER', 'first_pixel_y': 'INTEGER', 'cell_id1_around_bond': 'INTEGER', 'cell_id2_around_bond': 'INTEGER', 'is_border': 'INTEGER', 'sum_px_int_vertices_included_ch0': 'INTEGER', 'avg_px_int_vertices_included_ch0': 'FLOAT', 'sum_px_int_vertices_included_ch1': 'INTEGER', 'avg_px_int_vertices_included_ch1': 'FLOAT', 'sum_px_int_vertices_included_ch2': 'INTEGER', 'avg_px_int_vertices_included_ch2': 'FLOAT'}
                    # 13 19

                    present_in_master_but_missing_in_cur = None
                    fixed_master = set(_to_lower(cols_master.keys()))


                    # make it case insensitive
                    cols_master = {k.lower():v for k,v in cols_master.items()} # TODO maybe replace by a case insensitive dict
                    cols= {k.lower():v for k,v in cols.items()}

                    if not fixed_cols == fixed_master:
                        # print('error mismatch:')
                        # print(fixed_cols, 'vs:\n', fixed_master)
                        # print(len(cols), len(cols_master))

                        present_in_master_but_missing_in_cur = fixed_master - fixed_cols
                        present_in_cur_but_missing_in_master = fixed_cols - fixed_master
                        # print('present in master but missing in cur', present_in_master_but_missing_in_cur)
                        # print('present in cur but missing in master', present_in_cur_but_missing_in_master)
                        # do convert none to null

                        # probably need reattcah the table --> think about it and try again
                        # maybe add column of given type --> would be better and fix if not set

                        if present_in_master_but_missing_in_cur:
                            # create a temp db that we gonna remove later
                            # masterDB.execute_command('DROP TABLE pytaTMP IF EXISTS ')
                            masterDB.drop_table('pytaTMP')

                            # this command is ok...

                            masterDB.execute_command('CREATE TABLE pytaTMP AS SELECT * from ' + str(
                                DB_to_read))

                            #+(" CROSS JOIN (SELECT * FROM tmp.properties LIMIT 1)" if append_TA_properties else ''))  # does a copy of the table rather than editing the attached table


                            # masterDB.print_query('SELECT * FROM pytaTMP LIMIT 1') # contains all with time

                            #  CROSS JOIN (SELECT * FROM properties LIMIT 1);
                            for col in present_in_master_but_missing_in_cur:
                                # add missing cols to cur
                                # pb it does add it to the tmp db which is not what I want --> would need create a tmp table
                                # KEEP MEGA PB --> THIS DOES MODIFY THE ORIGINAL ATTACHED DB --> I DONT WANT THAT !!!! TODO RATHER CREATE A TMP DB THAT IS A COPY OF THIS ONE AND ADD COLS TO IT AND THEN READ FROM IT THEN DELETE IT!!! --> create a temp name that is different and read from it
                                masterDB.add_column('pytaTMP', col, col_type=cols_master[
                                    col])  # not what I want I want to change the tmp table not the original one !!!!
                            DB_to_read = 'pytaTMP'
                            # in fact all the added cols need be removed after then

                        for col in present_in_cur_but_missing_in_master:
                            # add missing column in master
                            masterDB.add_column(string, col, col_type=cols[col])

                        # add missing columns with default values

                    name = smart_name_parser(lst[l], ordered_output=['short'])[0]


                    # masterDB.print_query('SELECT * FROM ' + string +' LIMIT 1') # contains all with time

                    # masterDB.execute_command("INSERT INTO '" + str(string) + "' SELECT " + str(l) + " AS '" + str(frame_nb) + "', '" +  str(name) + "' AS '" + str(fileName) + "', * FROM tmp." + str(string) + "")
                    masterDB.execute_command(
                        "INSERT INTO '" + str(string) + "' SELECT " + str(l) + " AS '" + str(frame_nb) + "', '" + str(
                            name) + "' AS '" + str(fileName) + "', * FROM " + str(DB_to_read))

                    # "', * FROM (SELECT " + str(DB_to_read) + " CROSS JOIN (SELECT * FROM tmp.properties LIMIT 1))")
                    # +(" CROSS JOIN (SELECT * FROM tmp.properties LIMIT 1)" if append_TA_properties else '')

                    masterDB.drop_table('pytaTMP')

                    # if present_in_master_but_missing_in_cur is not None:
                    #
                    #     # KEEP MEGA TODO super mega dirty hack --> remove added columns --> not super smart --> best would be to create a tmp table and load from it, remove that and use a temp table instead --> dirty but ok for now
                    #     for col in present_in_master_but_missing_in_cur:
                    #         # add missing cols to cur
                    #         # pb it does add it to the tmp db which is not what I want --> would need create a tmp table
                    #         masterDB.remove_column('tmp.' + string, col)

            except:
                traceback.print_exc()
            finally:
                # if (dbHandler != null) {
                #     try {
                #         masterDB.detachTable("tmp")
                #     } catch (Exception e) {
                #         StringWriter sw = new StringWriter();
                #         PrintWriter pw = new PrintWriter(sw);
                #         e.printStackTrace(pw);
                #         String stacktrace = sw.toString();
                #         pw.close();
                #         System.err.println(stacktrace);
                #     }
                #     try:
                #         dbHandler.closeDb()
                #     catch (Exception e):
                #         pass
                if dbHandler is not None:
                    masterDB.detach_table('tmp')
                    dbHandler.close()
    except:
        traceback.print_exc()
    finally:
        print(outputName)
        # assume the file should be returned if in mem
        if masterDB is not None and outputName is not None:
            masterDB.clean()
            masterDB.close()
        else:
            return masterDB


'''
things that need be implemented



    public final void speedUpDB() throws SQLException {
        /**
         * it implies that there is no concurrent access to the DB but it's
         * worse it because it it several orders of magnitude faster
         */
        //NB this causes an error on slow disks (bad network connection)
        setSynchronous(false);
    }

    public void setSynchronous(boolean sync) throws SQLException {
        /**
         * allows to speed up access to the table and should not be a problem in
         * my case ??? NB: it is only useful at creation or during db edit can
         * easily be up to 20 times faster. NB2: the drawback is the db will get
         * corrupted if the computer crashes (but how often does this happen ?).
         */
        stat.executeUpdate("PRAGMA synchronous = " + (sync ? "NORMAL" : "OFF") + ";");//OFF NORMAL or FULL //OFF 3.5s NORMAL 32s FULL 40s 
//        stat.executeUpdate("PRAGMA journal_mode = " + (sync ? "MEMORY" : "OFF") + ";");
//        stat.executeUpdate("PRAGMA temp_store = " + (sync ? "ON" : "OFF") + ";");
    }

    public void setMemory(boolean sync) throws SQLException {
        /**
         * allows to speed access to the table and should not be a problem in my
         * case ???
         */
        stat.executeUpdate("PRAGMA journal_mode = " + (sync ? "MEMORY" : "OFF") + ";");
    }


    public boolean updateTableByHiddenIndex(String tableName, int rowIndex, HashMap<String, Object> data) {
        //ex: UPDATE bloomington SET "Stk #" ="toto", "Genotype" = "tutu" WHERE ROWID = 1;
        //--> tres facile a faire en fait
        try {
            String SQL_command = "UPDATE \"" + tableName + "\" SET ";
            if (!containsTable(tableName)) {
                System.err.println("table " + tableName + " does not exist");
                return false;
            }
            ArrayList<String> cols = getColumns(tableName);
            if (rowIndex < 1) {
                System.err.println("HIDDEN_TABLE_INDEX < 1 --> error");
                return false;
            }
//        System.out.println(cols);
            boolean first = true;
            for (Map.Entry<String, Object> entry : data.entrySet()) {
                String colName = entry.getKey();
                if (!cols.contains(colName.toLowerCase())) {
                    System.err.println("Column " + colName + " does not exist --> skipping");
                    continue;
                }
                Object val = "NULL";
                if (entry.getValue() != null) {
                    val = entry.getValue().toString();
                }
                if (!first) {
                    SQL_command += ", \"" + colName + "\" = " + "\"" + val + "\"";
                } else {
                    SQL_command += "\"" + colName + "\" = " + "\"" + val + "\"";
                    first = false;
                }
                //System.out.println("Key : " + entry.getKey() +  " Value : " + entry.getValue());
            }
            SQL_command += " WHERE HIDDEN_TABLE_INDEX = " + rowIndex + ";";
            executeCommand(SQL_command);
        } catch (Exception e) {
            LogFrame2.printStackTrace(e);
            return false;
        }
        return true;
    }
    



    /*
     * allows for update of a row by its index (very useful to synchronize with a jTable for example
     */
    public boolean updateTableByRowIndex(String tableName, int rowIndex, HashMap<String, Object> data) {
        //ex: UPDATE bloomington SET "Stk #" ="toto", "Genotype" = "tutu" WHERE ROWID = 1;
        //--> tres facile a faire en fait
        try {
            String SQL_command = "UPDATE \"" + tableName + "\" SET ";
            if (!containsTable(tableName)) {
                System.err.println("table " + tableName + " does not exist");
                return false;
            }
            ArrayList<String> cols = getColumns(tableName);
            if (rowIndex < 1) {
                System.err.println("row idx < 1 --> error");
                return false;
            }
//        System.out.println(cols);
            boolean first = true;
            for (Map.Entry<String, Object> entry : data.entrySet()) {
                String colName = entry.getKey();
                if (!cols.contains(colName.toLowerCase())) {
                    System.err.println("Column " + colName + " does not exist --> skipping");
                    continue;
                }
                Object val = "NULL";
                if (entry.getValue() != null) {
                    val = entry.getValue().toString();
                }
                if (!first) {
                    SQL_command += ", \"" + colName + "\" = " + "\"" + val + "\"";
                } else {
                    SQL_command += "\"" + colName + "\" = " + "\"" + val + "\"";
                    first = false;
                }
                //System.out.println("Key : " + entry.getKey() +  " Value : " + entry.getValue());
            }
            SQL_command += " WHERE ROWID = " + rowIndex + ";";
            executeCommand(SQL_command);
        } catch (Exception e) {
            LogFrame2.printStackTrace(e);
            return false;
        }
        return true;
    }

    public void updateColumn(String tableName, String columnName, ArrayList<Object> colData) {
        if (colData == null || colData.isEmpty()) {
            return;
        }
        int count = 1;
        for (Object object : colData) {
            try {
                String SQL_command = "update '" + tableName + "' SET '" + columnName + "'" + "='" + object + "' WHERE ROWID =" + count++ + ";";
                executeCommand(SQL_command);
            } catch (Exception e) {
                LogFrame2.printStackTrace(e);
            }
        }
    }

    //derniere chose a faire est de d'updater la db
    //cela dit je pourrais creer l'image dans une nouvelle table et croiser les deux avec une commande SQL c'est probablement mieux comme ca
    //en fait un linked hashmap serait tres facile a gerer
    public void appendToExistingTableOrCreateItIfItDoesNotExist(String table_name, HashMap<String, Object> data) throws SQLException {
        //d'abord verif si la taille existe et si oui on la recree pas si elle existe evrif que les colonnes existent
        //si oui il faut trier les truc dans le bon ordre et rentrer les donnees dedans
        //TODO
//editer la ligne 
        //on verif si la table existe car si elle existe on l'appende sinon on la cree
        if (!containsTable(table_name)) {
            createTable(table_name, generateHeadersWithTypeFromKeySet(data, ","));
        } else {
            //on verif que ttes les colonnes existent sinon on les cree
            //faire un colName exist
            for (String key : data.keySet()) {
                if (!colExists(table_name, key)) {
//                    System.out.println("--"+key + " "+getColumns(table_name));
                    /*
                     * the colName does not exist --> we create it
                     */
                    Object data_val = data.get(key);
                    if (data_val instanceof Integer) {
                        addColumn(table_name, key, "INTEGER");
                    } else if (data_val instanceof Double) {
                        addColumn(table_name, key, "DOUBLE");
                    } else if (data_val instanceof Double) {
                        addColumn(table_name, key, "FLOAT");
                    } else if (data_val instanceof byte[]) {
                        addColumn(table_name, key, "BLOB");
                    } else if (data_val instanceof String) {
                        addColumn(table_name, key, "TEXT");
                    } else if (data_val instanceof Boolean) {
                        addColumn(table_name, key, "BOOLEAN");
                    } else {
                        addColumn(table_name, key);
                    }
                }
            }
            //maintenant que les colonnes ont ete crees on va ajouter les donnees a la table
        }
        addToTableFastObject(table_name, data);
    }


    //remove table
    //http://stackoverflow.com/questions/8442147/how-to-delete-or-add-column-in-sqlite
    /**
     * create new table as the one you are trying to change, copy all data, drop
     * old table, rename the new one.
     */
    public void renameTable(String oldTableName, String newTableName) throws SQLException {
        if (!containsTable(newTableName)) {
            //ALTER TABLE foo  bar
            String SQLQuery = "ALTER TABLE '" + oldTableName + "' RENAME TO '" + newTableName + "'";
            executeCommand(SQLQuery);
        }
    }


    /**
     * adds an empty row to the current table
     *
     * @param table
     */
    public void addRow(String table) throws SQLException {
        String SQLQuery = "INSERT INTO '" + table + "' DEFAULT VALUES;";
        executeCommand(SQLQuery);
    }

    public void addColumn(String table, String new_col_name, String type) throws SQLException {
        if (!colExists(table, new_col_name)) {
            String SQLQuery = "ALTER TABLE '" + table + "' ADD COLUMN '" + new_col_name + "' " + type;
            executeCommand(SQLQuery);
        }
    }    

    public void addColumn(String table, String new_col_name) throws SQLException {
        if (!colExists(table, new_col_name)) {
            String SQLQuery = "ALTER TABLE '" + table + "' ADD COLUMN '" + new_col_name + "'";
            executeCommand(SQLQuery);
        }
    }

    public boolean colExists(String table_name, String column_name) {
        try {
            ArrayList<String> cols = getColumns(table_name);
//            System.out.println(cols);
//            System.out.println(column_name);
            if (cols != null && !cols.isEmpty()) {
                if (cols.contains(column_name.toLowerCase())) {
                    return true;
                }
            }
        } catch (Exception e) {
        }
        return false;
    }


    public void swap2Rows(String table, int row1, int row2) throws SQLException {
        String SQLCommand = "update '" + table + "' set HIDDEN_TABLE_INDEX = \"tmp\" WHERE HIDDEN_TABLE_INDEX = " + row1 + ";";
        SQLCommand += "update '" + table + "' set HIDDEN_TABLE_INDEX = " + row1 + " WHERE HIDDEN_TABLE_INDEX = " + row2 + ";";
        SQLCommand += "update '" + table + "' set HIDDEN_TABLE_INDEX = " + row2 + " WHERE HIDDEN_TABLE_INDEX = \"tmp\";";
//        System.out.println(SQLCommand);
        executeManyCommands(SQLCommand);
    }


    public void appendToExistingTableOrCreateItIfItDoesNotExist(String table_name, ArrayList<String> data) throws SQLException {
        if (data != null && !data.isEmpty()) {
            if (containsTable(table_name)) {
                /*
                 * table already exists --> we do not recreate it and we just
                 * add the new data to it at some point add controls to be sure
                 * that they match but ok for now
                 */
                data.remove(0);
            } else {
                String header = data.get(0);
                if (header != null && !header.equals("")) {
                    //           System.out.println(header);
                    String[] headers = header.split("\t");
                    createTable(table_name, headers);
                }
            }
            if (!data.isEmpty()) {
                addToTableFast(table_name, data);
            }
        }

    }

    public void dropTempTable(String table_name) throws SQLException {
        executeCommand("DROP TABLE IF EXISTS '" + table_name + "';");
    }
    
    public void replacejavaNullBySQLNULL() throws SQLException {
        replaceAnyBy("'null'", "NULL", false);
    }

    public void replaceTrueByTRUE() throws SQLException {
        replaceAnyBy("true", "TRUE", true);
    }

    public void replaceNaNByNULL() throws SQLException {
        replaceAnyBy("'NaN'", "NULL", false);
    }
    public void replaceAnyBy(String stuff_to_replace, String replacement_val, boolean addquotes) throws SQLException {
        ArrayList<String> tables = getTables();
        for (String string : tables) {
            ArrayList<String> columns = getColumns(string);
            for (String string1 : columns) {
                if (addquotes) {
                    executeCommand("UPDATE " + string + " SET " + string1 + " = REPLACE(" + string1 + ", '" + stuff_to_replace + "', '" + replacement_val + "') WHERE " + string1 + " = '" + stuff_to_replace + "';");
                } else {
                    executeCommand("UPDATE " + string + " SET " + string1 + " = REPLACE(" + string1 + ", " + stuff_to_replace + ", " + replacement_val + ") WHERE " + string1 + " = " + stuff_to_replace + ";");
                }
            }
        }
    }

    public void replaceAnyStringInColumnBy(String col_name, String stuff_to_replace, String replacement_val, boolean addquotes) throws SQLException {
        ArrayList<String> tables = getTables();
        for (String string : tables) {
            ArrayList<String> columns = getColumns(string);
            for (String string1 : columns) {
                if (!string1.toLowerCase().equals(col_name.toLowerCase())) {
                    continue;
                }
                if (addquotes) {
                    executeCommand("UPDATE " + string + " SET " + string1 + " = REPLACE(" + string1 + ", '" + stuff_to_replace + "', '" + replacement_val + "') WHERE " + string1 + " = '" + stuff_to_replace + "';");
                } else {
                    executeCommand("UPDATE " + string + " SET " + string1 + " = REPLACE(" + string1 + ", " + stuff_to_replace + ", " + replacement_val + ") WHERE " + string1 + " = " + stuff_to_replace + ";");
                }
            }
        }
    }


    //ca marche vraiment dickav
    //on va ajouter ttes ces databases a la db en cours
    //voir si ca marche encore qd des tables differents --> il faudrait faire un map entre les deux dbs des tables //--> map 
    //TODO ajouter les tables mergees dans une nouvelle db comme ca on sait ce qui est contenu et du coup on n'a qu'a ajouter celles qui manquent
    public void mergeSeveralDatabases(ArrayList<String> list_of_databases, boolean force_matching_of_partially_overlapping_databases, String... tables2merge) throws SQLException {
        for (int i = 0; i < list_of_databases.size(); i++) {
            String string = list_of_databases.get(i);
            String attach_db = "ATTACH DATABASE " + string + " AS tmp";
            executeCommand(attach_db);
            for (String string1 : tables2merge) {
                if (containsTable(string1)) {
                    String insert_all = "INSERT INTO " + string1 + "SELECT * FROM tmp." + string1 + ";";
                    executeCommand(insert_all);
                } else {
                    /*
                     * if the table doe not exist --> just create it
                     */
                    String insert_all = "CREATE TABLE " + string1 + "AS SELECT * FROM tmp." + string1 + ";";
                    executeCommand(insert_all);
                }
            }
            String detach_db = "DETACH DATABASE tmp";
            executeCommand(detach_db);
        }
    }


    public String SELECT_ALL(String table_name) {
        return "SELECT * FROM TABLE " + table_name;
    }

    public String SELECT_COLUMNS(String table_name, String... columns_to_select) {
        String columns = createSelection(columns_to_select);
        return "SELECT " + columns + " FROM TABLE " + table_name;
    }

    public String SELECT_COLUMN(String table_name, String columns_to_select) {
        return "SELECT " + columns_to_select + " FROM TABLE " + table_name;
    }

    public String CREATE_TABLE_AS_SELECT(String newTableName, String SQLCommandUsedForCreation) {
        String command = "CREATE TABLE '" + newTableName + "' AS " + SQLCommandUsedForCreation;
        if (!SQLCommandUsedForCreation.trim().endsWith(";")) {
            command += ";";
        }
        return command;
    }


    /**
     * returns SQLite code to rename a table
     *
     * @param orig_table_name
     * @param new_table_name
     * @return
     */
    public String RENAME_TABLE(String orig_table_name, String new_table_name) {
        return "ALTER TABLE '" + orig_table_name + "' RENAME TO '" + new_table_name + "';";
    }

    public String createSelection(String... cols) {
        String out = "";
        for (String string : cols) {
            out += string + ",";
        }
        if (out.endsWith(",")) {
            out = out.substring(0, out.length() - 1);
        }
        return out;
    }

    /*
     * cree la selection avec le nom complet de la table --> peut etre tres
     * utile dans le cas ou on a plusieurs tables
     */
    public String createSelection(String table_name, String... cols) {
        String out = "";
        for (String string : cols) {
            out += table_name + "." + string + ",";
        }
        if (out.endsWith(",")) {
            out = out.substring(0, out.length() - 1);
        }
        return out;
    }

    public String SELECT_ALL_EXCEPT(String table_name, Object... columns_to_exclude) throws SQLException {
        String[] names = new String[columns_to_exclude.length];
        int i = 0;
        for (Object string : columns_to_exclude) {
            names[i] = string.toString().trim();
            i++;
        }
        String cols = EXCEPT(table_name, names);
        return SELECT_COLUMN(table_name, cols);
    }

    /*
     * creates a string that contains all columns except the ones we don't want
     * to select --> bonne idee en fait
     */
    public String EXCEPT(String table_name, String... columns_to_exclude) throws SQLException {
        return EXCEPT(table_name, true, columns_to_exclude);
    }


    public String getAllColumnsAsASingleString(String table_name) throws SQLException {
        ArrayList<String> col_name = getColumns(table_name);
        String out = "";
        for (String col_name1 : col_name) {
            String string = col_name1.trim();
            out += " " + table_name + "." + string + ",";
        }
        if (out.endsWith(",")) {
            out = out.substring(0, out.length() - 1);
        }
        return out;
    }

    public String SELECT_ALL_EXCEPT(String table_name, ArrayList<String> columns_to_exclude) throws SQLException {
        return SELECT_ALL_EXCEPT(table_name, ((List) columns_to_exclude).toArray());
    }

    /*
     * enleve ttes les colonnes identiques des deux tables --> sql output that
     * looks better
     */
    public String NODUPES(String table_name1, String table_name2) throws SQLException {
        ArrayList<String> col_name1 = getColumns(table_name1);
        ArrayList<String> col_name2 = getColumns(table_name2);
        col_name2.removeAll(col_name1);
        String columns = "";
        for (String cur_name : col_name1) {
            columns += " " + table_name1 + "." + cur_name + ",";
        }
        for (String cur_name : col_name2) {
            columns += " " + table_name2 + "." + cur_name + ",";
        }
        if (columns.endsWith(",")) {
            columns = columns.substring(0, columns.length() - 1);
        }
        columns = "SELECT " + columns + " FROM TABLE " + table_name1 + "," + table_name2;
        return columns;
    }


    public static void mkDirsFromFileName(String name) {
        new File(new File(name).getParent()).mkdirs();
    }
    
    public ArrayList<String> getTables() throws SQLException {
        return listTablesInCurrentDb();
    }

    public ArrayList<String> getColumns(String table_name) throws SQLException {
        return listTableHeaders(table_name);
    }
    
    public ArrayList<String> listTablesInCurrentDb() throws SQLException {
        ArrayList<String> tables = new ArrayList<String>();
        DatabaseMetaData dmd = conn.getMetaData();
        String[] data = {"TABLE"};
        ResultSet rs = dmd.getTables(null, null, "%", data);
        while (rs.next()) {
            tables.add(rs.getString(3).toLowerCase());
        }
        try {
            rs.close();
        } catch (Exception e) {
            LogFrame2.printStackTrace(e);
        }
        return tables;
    }
    
    
    # TODO --> add this maybe
            if (obj instanceof byte[]) {
            type = "BLOB";
        }        
/tt mettre en double ?
    public void csv2sql(String path2csv, String table_name) throws SQLException {
        if (!new File(path2csv).exists()) {
            return;
        }
        ArrayList<String> txt = new TextParser().OnlyGetTextNoEscape(path2csv);
        csv2sql(txt, table_name);
        //on balance chaque ligne dans la db
    }

    public void realCsv2sql(String path2csv, String table_name) throws SQLException {
        if (!new File(path2csv).exists()) {
            return;
        }
        ArrayList<String> txt = new TextParser().OnlyGetTextNoEscape(path2csv);
        csv2sql(txt, table_name, ",");
        //on balance chaque ligne dans la db
    }
    //TODO faire un truc pr importer divers arrays/arraylist d'objets
    //--> il faut connaitre le separateur texte et pas splitter a l'interieur --> faire ca un autre jour 

    public void csv2sql(ArrayList<String> txt, String table_name, String splitter) throws SQLException {
        String[] col_names = txt.get(0).replace("#", " ").split(splitter);
        txt.remove(0); //on supprime le header
//        
//        for (String string : col_names) {
//            System.out.println(string);
//        }

        //il faut ignorer les commas qui sont dans les comments //--> splitter par "" en fait
//        for (String string : col_names) {
//            System.out.println(string);
//            
//        }
        createTable(table_name, col_names);
        if (!txt.isEmpty()) {
            addToTableFast(table_name, txt, splitter);
        }
        //on balance chaque ligne dans la db
    }

    public void csv2sql(ArrayList<String> txt, String table_name) throws SQLException {
        csv2sql(txt, table_name, "\t");
    }

    public void csv2sql(LinkedHashMap<String, Object> header_n_type, ArrayList<String> data, String table_name) throws SQLException {
        createTable(table_name, header_n_type);
        if (!data.isEmpty()) {
            addToTableFast(table_name, data);
        }
        //on balance chaque ligne dans la db
    }


    public void palette2DB(String table_name, int[] palette) throws SQLException {
        int[] palette_fused;
        if (palette.length > 260) {
            palette_fused = new PaletteCreator().fused_RGB_palette(palette);
        } else {
            palette_fused = palette;
        }
        //rows and cols
        ArrayList<ArrayList<Object>> values = new ArrayList<ArrayList<Object>>();
        ArrayList<Object> header = new ArrayList<Object>();
        header.add("gray_value");
        header.add("corresponding_RGB_color");
        header.add("R_only");
        header.add("G_only");
        header.add("B_only");
        values.add(header);
        for (int i = 0; i < palette_fused.length; i++) {
            ArrayList<Object> row = new ArrayList<Object>();
            int RGB = palette_fused[i];
            int red = RGB >> 16 & 0xFF;
            int green = RGB >> 8 & 0xFF;
            int blue = RGB & 0xFF;
            row.add(i);
            row.add(RGB);
            row.add(red);
            row.add(green);
            row.add(blue);
            values.add(row);
        }
        arrayListObjects2SQL(table_name, values);
    }

    public void csv2sql(String path2csv) throws SQLException {
        csv2sql(path2csv, CommonClasses.getName(new File(path2csv).getName()));
    }

    /*
     * permet de convertir des arraylist string en db sqlite
     */
    public void csv2sql(String table_name, ArrayList<String> data) throws SQLException {
        String[] col_names = data.get(0).split("\t");
        data.remove(0); //on supprime le header
        //    System.out.println(col_names);
        createTable(table_name, col_names);
        addToTableFast(table_name, data);
    }

    public void csv2sql(String table_name, String column_header, ArrayList<String> data) throws SQLException {
        createTable(table_name, column_header);
        if (data != null && !data.isEmpty()) {
            addToTableFast(table_name, data);
        }
    }


    public ArrayList<String> listTableHeaders(String table_name) throws SQLException {
        return getHeaders(table_name);
    }

    public int getNbOfCols(String table_name) throws SQLException {
        return getHeaders(table_name).size();
    }

    public ArrayList<String> getHeaders(String table_name) throws SQLException {
        if (!containsTable(table_name)) {
            return null;
        }
        return new ArrayList<String>(getColumnNamesAndTypes(table_name).keySet());
    }

    public String getHeader(String table_name) throws SQLException {
        ArrayList<String> header = new ArrayList<String>(getColumnNamesAndTypes(table_name).keySet());
        String head = "";
        for (int i = 0; i < header.size(); i++) {
            String string = header.get(i);
            if (i >= 0 && header.size() != 1 && i < header.size() - 1) {
                head += string + "\t";
            } else {
                head += string;
            }
        }
        return head;
    }


    public String createTempTable(ArrayList<String> columns) throws SQLException {
        String[] headers = new String[columns.size()];
        columns.toArray(headers);
        return createTempTable(headers);
    }

    public String createTempTable(String... columns) throws SQLException {
        String table_name = "";
        do {
            table_name = CommonClasses.getName(new CommonClasses().CreateTempFile(".db").getName());
        } while (containsTable(table_name));
        //stat.executeUpdate("DROP TABLE IF EXISTS " + table_name + ";"); //--> si la table existe deja ca l'ecrase en entier --> pas mal comme truc
        String create_table = "CREATE TEMPORARY TABLE " + table_name + " ";
        String cols = "(";
        for (int i = 0; i < columns.length; i++) {
            String string = columns[i];
            if (i > 0) {
                cols += "," + string;
            } else {
                cols += string;
            }
        }
        cols += ");";
        create_table += cols;
//        System.out.println(create_table);
        stat.executeUpdate(create_table);
        return table_name;
    }


    public void popTable(ResultSet rs) {
        TableModel jtab = resultSetToTableModel(rs, 1000, false);
        int result = JOptionPane.showOptionDialog(CommonClasses.getGUIComponent(), new Object[]{new JTable(jtab)}, "Edit Palette", JOptionPane.CANCEL_OPTION, JOptionPane.PLAIN_MESSAGE, null, null, null);
    }

    public String toString(String SQLCommand, int limit) throws SQLException {
        ResultSet rs = executeQueryAndGetResultset(SQLCommand);
        String result = toString(rs, limit);
        rs.close();
        return result;
    }


  /**
     * prints a resultset as a formatted string --> useful for debug purposes
     *
     * @param rs
     * @param limit
     * @return
     */
    public String toString(ResultSet rs, int limit) {
        String output = "";
        //need to get string size
        //need to set a font where all chars have the same length --> check
        //TODO populate the table then get max size then we're set
        try {
            ResultSetMetaData metaData = rs.getMetaData();
            if (metaData == null) {
                return output;
            }
            int numberOfColumns = metaData.getColumnCount();
            ArrayList<String> colNames = new ArrayList<String>();
            for (int column = 0; column < numberOfColumns; column++) {
                colNames.add(metaData.getColumnLabel(column + 1));
            }
            int counter = 0;
            ArrayList<ArrayList<String>> rows = new ArrayList<ArrayList<String>>();
            rows.add(colNames);
            boolean breakBeforeEnd = false;
            while (rs.next()) {
                if (limit > 0 && counter > limit) {
                    breakBeforeEnd = true;
                    break;
                }
                ArrayList<String> curRow = new ArrayList<String>();
                for (int i = 1; i <= numberOfColumns; i++) {
                    curRow.add(rs.getObject(i).toString());
                }
                rows.add(curRow);
                counter++;
            }

            /**
             * we now get the max size of each string in each col to make it
             * nice looking
             */
            ArrayList<Integer> maxColWidth = new ArrayList<Integer>();
            ArrayList<String> empty = new ArrayList<String>();
            for (String colName : colNames) {
                maxColWidth.add(0);
                empty.add("");
            }
            for (ArrayList<String> row : rows) {
                for (int i = 0; i < row.size(); i++) {
                    String get = row.get(i);
                    maxColWidth.set(i, Math.max(maxColWidth.get(i), get.toString().length()));
                }
            }
            for (int i = 0; i < maxColWidth.size(); i++) {
                Integer length = maxColWidth.get(i);
                String get = "";
                while (get.length() < length) {
                    get += "-";
                }
                empty.set(i, get);
            }
            /**
             * add borders to the table
             */
            rows.add(0, empty);
            rows.add(2, empty);
            /**
             * now we pad all cols
             */
            for (ArrayList<String> row : rows) {
                for (int i = 0; i < row.size(); i++) {

                    int curColWidth = maxColWidth.get(i);
                    String get = row.get(i);
                    while (get.length() < curColWidth) {
                        get += " ";
                    }
                    //get=get;
                    row.set(i, get);
                    maxColWidth.set(i, Math.max(maxColWidth.get(i), get.toString().length()));
                }
                output += row.toString().replace(",", "|").replace("[", "|").replace("]", "|") + "\n";
            }
            if (limit > 0 && !breakBeforeEnd) {
                output += "...";
            }
        } catch (Exception e) {
            LogFrame2.printStackTrace(e);
            return null;
        }
        return output;
    }

    //pas mal mais manque le header 
    //TODO en faire une version qui soit capable de splitter la table pr pouvoir faire plusieurs pages web connectees par des liens
    public String resultSetToHTMLTable(ResultSet rs, int max_cols, boolean split_header, boolean number_cols) throws SQLException {
        /*
         * TODO en fait je pourrais aussi breaker des colonnes --> si besoin est --> plus simple en fait que d'utiliser le truc du systeme
         * sinon proposer des taille auto par le systeme ???
         * sinon demander aux gens les colonnes qu'ils veulent j'aurais bien jette un coup d'oeil au truc de tim cross pr voir comment il avait fait
         * //certaines choses peuvent etre hidden et on peut les afficher que si on clicke sur un bouton
         * voir comment mettre des checkbox dans le truc --> faire ca apres commencer une presentation
         */
        ResultSetMetaData metaData = rs.getMetaData();
        int numberOfColumns = metaData.getColumnCount();
        String header = "<tr>\n";
        for (int column = 0; column < numberOfColumns; column++) {
            if (split_header) {
                String col_header = metaData.getColumnLabel(column + 1);
                String[] data;
                if (col_header.contains(" ")) {
                    data = col_header.split(" ");
                } else if (col_header.contains("/")) {
                    data = col_header.split("/");
                } else {
                    data = new String[]{col_header};
                }
                header += "<th>";
                for (String string : data) {
                    header += string + "<BR>";
                }
                header += "</th>\n";
            } else {
                header += "<th>\n" + metaData.getColumnLabel(column + 1) + "\n</th>\n";
            }
        }
        header += "</tr>";
        int counter = 1;
        if (max_cols == -1) {
            max_cols = Integer.MAX_VALUE;//peu de chances d'y arriver
        }
        //tester pr voir si ca marche
        String html_table = "<table style=\"width:100%;\" id=\"SQLtable\" cellpadding=\"5\" cellspacing=\"0\"\">\n";
        html_table += header;
        String final_table = "";
        while (rs.next() && counter < max_cols) {
            String numeratorString;
            if (counter % 2 == 0) {
                numeratorString = "<tr row_nb=\"" + counter + "\">\n";
            } else {
                //ca sert juste a rendre plus joli a l'aide des styles css --> une ligne sur deux est coloree par exemple
                numeratorString = "<tr class=\"odd\" row_nb=\"" + counter + "\">\n";
            }
            for (int j = 1, col_counter = 0; j <= numberOfColumns; j++, col_counter++) {
//                if (counter % 2 == 0) {
                if (!number_cols) {
                    numeratorString += "<td  align=\"center\">\n" // style="border-bottom: 1px solid #000;"   style="border-bottom: 1px solid #000;" // nowrap=\"nowrap\"
                            + "           " + rs.getObject(j) + "\n"
                            + "        </td>\n";
                } else {
                    numeratorString += "<td id=\"col_nb_" + col_counter + "\" align=\"center\">\n" // style="border-bottom: 1px solid #000;"   style="border-bottom: 1px solid #000;" // nowrap=\"nowrap\"
                            + "           " + rs.getObject(j) + "\n"
                            + "        </td>\n";
                }
            }
            numeratorString += "</tr>";
            final_table += numeratorString;
            counter++;
        }
        html_table += final_table + "\n";
        html_table += "</table>";
        return html_table;
    }


    private static TableModel resultSetToTableModel(ResultSet rs, int max_cols, boolean isTableEditable) {
        try {
            ResultSetMetaData metaData = rs.getMetaData();
            int numberOfColumns = metaData.getColumnCount();
            Vector colNames = new Vector<String>();
            for (int column = 0; column < numberOfColumns; column++) {
                colNames.add(metaData.getColumnLabel(column + 1));
            }
            int counter = 0;
            Vector rows = new Vector<Object>();
            while (rs.next()) {
                if (max_cols > 0 && counter > max_cols) {
                    break;
                }
                Vector curRow = new Vector<Object>();
                for (int i = 1; i <= numberOfColumns; i++) {
                    curRow.add(rs.getObject(i));
                }
                rows.add(curRow);
                counter++;
            }
            /*
             * we make the table cells non editable
             */
            DefaultTableModel tableModel;

            if (!isTableEditable) {
                tableModel = new DefaultTableModel(rows, colNames) {
                    @Override
                    public boolean isCellEditable(int row, int column) {
                        return false;
                    }
                };
            } else {
                tableModel = new DefaultTableModel(rows, colNames);
            }
            return tableModel;
        } catch (Exception e) {
            LogFrame2.printStackTrace(e);
            return null;
        }
    }

    public int[] getColumnSQLType(String table_name) throws SQLException {
        ResultSet rs = stat.executeQuery("SELECT * FROM " + table_name + " LIMIT 1");
        int[] columnSQLType = getColumnSQLType(rs);
        try {
            rs.close();
        } catch (Exception e) {
            LogFrame2.printStackTrace(e);
        }
        return columnSQLType;
    }
    
    //voila le meilleur moyen de catser les donnees en fait --> permet a l'aide d'un switch de bien sauver les donnees dans le bon type de format --> double, blob, integer, float, ...
    public int[] getColumnSQLType(ResultSet rs) throws SQLException {
        ResultSetMetaData rsMetaData = rs.getMetaData();
        int numberOfColumns = rsMetaData.getColumnCount();
        int[] types = new int[numberOfColumns];

        for (int i = 1; i < numberOfColumns + 1; i++) {
            int tableTypeInt = rsMetaData.getColumnType(i); //--> ca marche tres bien en fait et c'est bcp plus classe que l'autre
            types[i - 1] = tableTypeInt;
        }
        return types;
    }
    

    
    public void executeManyCommands(String SQLcommand) throws SQLException {
        String[] commands = SQLcommand.split(";");
        for (String cur : commands) {
            executeCommand(cur);
        }
    }

    public void executeCommand(String SQLcommand) throws SQLException {
//        System.out.println(SQLcommand);
        stat.execute(SQLcommand);
    }

    //--> faire une vraie db pr mes projections, etc ...
    //--> bonne idee
    public String findCellWithID(String id) {
        return "SELECT * FROM cells WHERE cell_id = " + id;
    }

    public String findBondWithID(String id) {
        return "SELECT * FROM bonds WHERE bond_id = " + id;
    }

    public void exportSQLCommand2csv(String SQLCommand, String output_file_name, boolean append, boolean writeHeader) throws SQLException {
        /**
         * debug only
         */
//                System.out.println(SQLCommand);
        ResultSet rs = null;
        DataOutputStream fos = null;
        try {
            rs = stat.executeQuery(SQLCommand);
            fos = new DataOutputStream(new FileOutputStream(output_file_name, append));
            ArrayList<String> getColumnHeaders = new ArrayList<String>(getColumnNamesAndTypesCurrentSearch(rs).keySet());
            String cur_line = "";
            if (writeHeader) {
                for (String columnHeader : getColumnHeaders) {
                    cur_line += columnHeader + "\t";
                }
                cur_line += "\n";
                fos.writeBytes(cur_line);
            }
            while (rs.next()) {
                cur_line = "";
                for (int i = 0; i < getColumnHeaders.size(); i++) {
                    cur_line += rs.getObject(i + 1) /*getColumnHeaders.get(i))*/ + "\t";
                }
                cur_line = cur_line.trim();
                fos.writeBytes(cur_line + "\n");
            }
            fos.flush();
        } catch (Exception e) {
            //TODO maybe display this in a popup window ?
            CommonClasses.Warning("the following SQL command could not be executed (table may not exist):\n" + SQLCommand);
            LogFrame2.printStackTrace(e);
        } finally {
            try {
                if (rs != null) {
                    rs.close();
                }
            } catch (Exception e) {
                LogFrame2.printStackTrace(e);
            }
            try {
                if (fos != null) {
                    fos.close();
                }
            } catch (Exception e) {
              LogFrame2.printStackTrace(e);
            }
        }
    }

    /*
     * sqlite> .mode list sqlite> .separator , sqlite> .output test_file_1.txt
     * sqlite> select * from yourtable; sqlite> .exit
     */
    public void exportTable2csv(String table_name, String output_file_name) throws SQLException {
        exportSQLCommand2csv("SELECT * FROM " + table_name, output_file_name, false, true);
    }
    
    
    private String formatSearch(String searchLike) {
        if (searchLike.trim().startsWith("\"") && searchLike.trim().endsWith("\"")) {
            return searchLike.replace("\"", "");
        } else if (searchLike.contains("*")) {
            searchLike = searchLike.replace("*", "%");
        } else {
            searchLike = "%" + searchLike + "%";
        }
        return searchLike;
    }
    
    
    public void removeRowsByHiddenIndices(String table, ArrayList<Integer> rowNb) throws SQLException {
        if (rowNb == null || rowNb.isEmpty()) {
            System.out.println("no rows to delete");
            return;
        }
        String SQLcommand = "DELETE FROM '" + table + "' WHERE";
        for (int s : rowNb) {
            SQLcommand += " HIDDEN_TABLE_INDEX = " + s + " OR";
        }
        if (SQLcommand.endsWith(" OR")) {
            SQLcommand = SQLcommand.substring(0, SQLcommand.length() - " OR".length());
        }
        SQLcommand += ";";
        executeCommand(SQLcommand);
    }
    

    public void removeRows(String table, int... rowNb) throws SQLException {
        if (rowNb == null || rowNb.length == 0) {
            System.out.println("no rows to delete");
            return;
        }
        String SQLcommand = "DELETE FROM '" + table + "' WHERE";
        for (int s : rowNb) {
            SQLcommand += " ROWID = " + s + " OR";
        }
        if (SQLcommand.endsWith(" OR")) {
            SQLcommand = SQLcommand.substring(0, SQLcommand.length() - " OR".length());
        }
        SQLcommand += ";";
        executeCommand(SQLcommand);
    }
    
   /**
     * This removes columns from an SQLite DB
     *
     * @param table
     * @param ColumnsToDelete
     * @throws SQLException
     */
    public void removeColumns(String table, ArrayList<String> ColumnsToDelete) throws SQLException {
        /**
         * http://stackoverflow.com/questions/8442147/how-to-delete-or-add-column-in-sqlite
         * SQLite supports a limited subset of ALTER TABLE. The ALTER TABLE
         * command in SQLite allows the user to rename a table or to add a new
         * column to an existing table. It is not possible to rename a column,
         * remove a column, or add or remove constraints from a table.
         */
        if (ColumnsToDelete == null || ColumnsToDelete.isEmpty()) {
            System.out.println("no columns to delete");
            return;
        }
        String SQLcommand = "";
        /* we select everything from the table except the columns to be deleted */
        String SELECT_ALL_COLS_EXCEPT = EXCEPT(table, false, ColumnsToDelete.toArray(new String[0]));
        SQLcommand += "\n" + CREATE_TABLE_AS_SELECT("tmp_table_for_column_deletion", "SELECT " + SELECT_ALL_COLS_EXCEPT + " FROM '" + table + "';");
        /* we drop the old table */
        SQLcommand += "\n" + DROP_TABLE_IF_EXISTS(table);
        /* we rename the tmp table to restore the original file */
        SQLcommand += "\n" + RENAME_TABLE("tmp_table_for_column_deletion", table);
        /* for debug */
        //System.out.println(SQLcommand);
        executeManyCommands(SQLcommand);
    }
    
   //new
    public LinkedHashMap<Integer, Integer> getColumnPosAndSQLTypes(String table_name) throws SQLException {
        ResultSet rs = stat.executeQuery("SELECT * FROM " + table_name + " LIMIT 1");
        LinkedHashMap<Integer, Integer> posAndType = getColumnPosAndSQLTypesCurrentSearch(rs);
        try {
            rs.close();
        } catch (Exception e) {
            LogFrame2.printStackTrace(e);
        }
        return posAndType;
    }

    //tjrs fermer la research
    //new
    public LinkedHashMap<String, Integer> getColumnNamesAndSQLTypes(String table_name) throws SQLException {
        ResultSet rs = stat.executeQuery("SELECT * FROM " + table_name + " LIMIT 1");
        LinkedHashMap<String, Integer> ColumnNameAndType = getColumnNamesAndSQLTypesCurrentSearch(rs);
        try {
            rs.close();
        } catch (Exception e) {
            LogFrame2.printStackTrace(e);
        }
        return ColumnNameAndType;
    }
    

    public ArrayList<Object>[] getColumns(ResultSet rs) throws SQLException {
        int nbOfCols = getColumnCount(rs);
        if (nbOfCols == 0) {
            return null;
        }

        ArrayList<Object>[] columns = new ArrayList[nbOfCols];
        for (int i = 0; i < nbOfCols; i++) {
            columns[i] = new ArrayList<Object>();
        }
        try {
            ResultSetMetaData metaData = rs.getMetaData();
            int numberOfColumns = metaData.getColumnCount();
            if (numberOfColumns == 0) {
                return null;
            }
            while (rs.next()) {
                for (int i = 0; i < numberOfColumns; i++) {

                    columns[i].add(rs.getObject(i + 1));
                }
            }
            return columns;
        } catch (Exception e) {
           LogFrame2.printStackTrace(e);
            return null;
        }
    }

    public int getColumnCount(ResultSet rs) throws SQLException {
        if (rs == null) {
            return 0;
        }
        ResultSetMetaData rsMetaData = rs.getMetaData();
        int numberOfColumns = rsMetaData.getColumnCount();
        return numberOfColumns;
    }

    public ArrayList<String> getColumnNames(ResultSet rs) throws SQLException {
        ArrayList<String> columnNames = new ArrayList<String>();
        if (rs == null) {
            return columnNames;
        }
        ResultSetMetaData rsMetaData = rs.getMetaData();
        int numberOfColumns = rsMetaData.getColumnCount();

        for (int i = 1; i <= numberOfColumns; i++) {
            String name = rsMetaData.getColumnName(i);
            /**
             * force names to be lower case by convention
             */
            columnNames.add(name != null ? name.toLowerCase() : name);
        }

        return columnNames;
    }

    //new
    public ArrayList<Object>[] getColumns(String tableName, int... col_nbs) throws SQLException {
        return getColumnsWhere(tableName, null, col_nbs);
    }
    
    
    public static String parseBonds(String path_to_db) {
        String script = "var dbName = \"/Path/To/Your/Db.db\";" + "\n"
                + "var mydb_handler = new Packages.DB.MySQLDatabseHandler(dbName);" + "\n"
                + "mydb_handler.executeCommand(\"DROP TABLE IF EXISTS TMP_SPLIT;\");" + "\n"
                + "mydb_handler.executeCommand(\"CREATE TABLE TMP_SPLIT AS SELECT frame_nb_cells, local_id_cells,local_id_of_bonds FROM Cells;\");" + "\n"
                + "mydb_handler.executeCommand(\"DROP TABLE IF EXISTS splitted_bonds;\");" + "\n"
                + "mydb_handler.executeCommand(\"CREATE TABLE splitted_bonds (frame_nb_cells, local_id_cells, bd_id);\");" + "\n"
                + "var i=0;\n"
                + "for (i=0; i<20; i++)" + "\n"
                + "{" + "\n"
                + "  mydb_handler.executeCommand(\"INSERT INTO splitted_bonds (frame_nb_cells, local_id_cells, bd_id) SELECT frame_nb_cells, local_id_cells, (SUBSTR(local_id_of_bonds, 1, LENGTH(local_id_of_bonds)-LENGTH(LTRIM(local_id_of_bonds, '0123456789')))) FROM TMP_SPLIT WHERE local_id_of_bonds <> '';\");" + "\n"
                + "  mydb_handler.executeCommand(\"UPDATE TMP_SPLIT SET local_id_of_bonds = LTRIM(SUBSTR(local_id_of_bonds,  LENGTH(SUBSTR(local_id_of_bonds, 1, LENGTH(local_id_of_bonds)-LENGTH(LTRIM(local_id_of_bonds, '0123456789'))))+1,  LENGTH(local_id_of_bonds)), '#');\");" + "\n"
                + "}" + "\n"
                + "mydb_handler.executeCommand(\"DROP TABLE IF EXISTS TMP_SPLIT;\");";
        if (path_to_db != null && !path_to_db.equals("")) {
            script = script.replace("/Path/To/Your/Db.db", CommonClasses.change_path_separators_to_system_ones(path_to_db));
        }
        return script;
    }

    public static String parseVertices(String path_to_db) {
        String script = "var dbName = \"/Path/To/Your/Db.db\";" + "\n"
                + "var mydb_handler = new Packages.DB.MySQLDatabseHandler(dbName);" + "\n"
                + "mydb_handler.executeCommand(\"DROP TABLE IF EXISTS splitted_vertices;\");" + "\n"
                + "mydb_handler.executeCommand(\"CREATE TABLE splitted_vertices AS SELECT frame_nb_cells, local_id_cells, vx_coords_cells FROM Cells;\");" + "\n"
                + "var parsers = new Array();" + "\n"
                + "parsers[0] =\"#\";" + "\n"
                + "parsers[1] =\":\";" + "\n"
                + "mydb_handler.parse_column(\"splitted_vertices\", \"vx_coords_cells\", \"splitted_vertices\",parsers);";
        if (path_to_db != null && !path_to_db.equals("")) {
            script = script.replace("/Path/To/Your/Db.db", CommonClasses.change_path_separators_to_system_ones(path_to_db));
        }
        return script;
    }

    public static String getDefaultScript(String path_to_db) {
        String script = "importPackage(java.sql);" + "\n"
                + "var p = new java.util.Properties();" + "\n"
                + "var dbName = \"/Path/To/Your/Db.db\";" + "\n"
                + "Packages.DB.MySQLDatabseHandler.mkDirsFromFileName(dbName);" + "\n"
                + "var conn = new org.sqlite.JDBC().connect(\"jdbc:sqlite:\"+dbName,p);" + "\n"
                + "var stat = conn.createStatement();" + "\n"
                + "stat.executeUpdate(\"drop table if exists people ;\");" + "\n"
                + "stat.executeUpdate(\"create table people (name, occupation);\");" + "\n"
                + "var prep = conn.prepareStatement(\"insert into people values (?, ?);\");" + "\n"
                + "prep.setString(1, \"Gandhi\"); prep.setString(2, \"politics\");" + "\n"
                + "prep.addBatch(); prep.setString(1, \"Turing\");" + "\n"
                + "prep.setString(2, \"Computers\");" + "\n"
                + "prep.addBatch();" + "\n"
                + "conn.setAutoCommit(false); " + "\n"
                + "prep.executeBatch();" + "\n"
                + "conn.setAutoCommit(true); " + "\n"
                + "var resultSet = stat.executeQuery(\"select * from people;\");" + "\n"
                + "while (resultSet.next()){" + "\n"
                + "println(resultSet.getString(\"name\") + \" – \" + resultSet.getString(\"occupation\"));" + "\n"
                + "}" + "\n"
                + "resultSet.close();" + "\n"
                + "stat.close();" + "\n"
                + "conn.close();";
        if (path_to_db != null && !path_to_db.equals("")) {
            script = script.replace("/Path/To/Your/Db.db", CommonClasses.change_path_separators_to_system_ones(path_to_db));
        }
        return script;
    }


    public byte[] getArrayByteFromFile(String filename) {
        File file = new File(filename);
        int length = (int) file.length();
        byte[] data = new byte[length];
        try {
            BufferedInputStream in;
            in = new BufferedInputStream(new FileInputStream(file));
            int result = in.read(data, 0, length);
        } catch (Exception e) {
           LogFrame2.printStackTrace(e);
        }
        return data;
    }
    
    public TableModel querryCell(int cell_id) throws SQLException {
        //--> on va chercher dans la DB tt ce dont on a besoin pour la cellule
        TableModel objs = executeSQLQuery("SELECT * FROM cells WHERE local_id_cells='" + cell_id + "' ;", false, false);
        return objs;
    }

    public TableModel querryBond(int bond_id) throws SQLException {
        TableModel objs = executeSQLQuery("SELECT * FROM Bonds WHERE local_id_bonds='" + bond_id + "' ;", false, false);
        return objs;
    }
    
    public String getVersion() {
        String val = "sqlite initiation error";
        ResultSet rs = null;
        try {
            rs = executeQueryAndGetResultset("SELECT sqlite_version()");
            while (rs.next()) {
                val = rs.getString(1);
                break;
            }
        } catch (Exception e) {
            LogFrame2.printStackTrace(e);
        } finally {
            try {
                rs.close();
            } catch (Exception e) {
                LogFrame2.printStackTrace(e);
            }
            return val;
        }
    }
    
    public void restoreSavePoint(String savePointName) throws SQLException {
        executeCommand("savepoint " + savePointName + ";");
    }

    public void createSavePoint(String savePointName) throws SQLException {
        executeCommand("rollback to savepoint " + savePointName + ";");
    }
    
    public LinkedHashMap<String, String> getColumnNamesAndTypes(String table_name) throws SQLException {
        //stat.executeUpdate("SELECT_COLUMNS sql FROM sqlite_master WHERE type = 'table'"); //ensuite faut parser --> old method to do that
        if (!containsTable(table_name)) {
            return null;
        }
        ResultSet rs = stat.executeQuery("SELECT * FROM '" + table_name + "' LIMIT 1");
        LinkedHashMap<String, String> columnNamesAndTypes = getColumnNamesAndTypesCurrentSearch(rs);
        try {
            rs.close();
        } catch (Exception e) {
            LogFrame2.printStackTrace(e);
        }
        return columnNamesAndTypes;
    }    
                                                
                    
'''

'''
data types
NULL. The value is a NULL value.

INTEGER. The value is a signed integer, stored in 1, 2, 3, 4, 6, or 8 bytes depending on the magnitude of the value.

REAL. The value is a floating point value, stored as an 8-byte IEEE floating point number.

TEXT. The value is a text string, stored using the database encoding (UTF-8, UTF-16BE or UTF-16LE).

BLOB. The value is a blob of data, stored exactly as it was input

Date and Time Datatype
SQLite does not have a storage class set aside for storing dates and/or times. Instead, the built-in Date And Time Functions of SQLite are capable of storing dates and times as TEXT, REAL, or INTEGER values:

TEXT as ISO8601 strings ("YYYY-MM-DD HH:MM:SS.SSS").
REAL as Julian day numbers, the number of days since noon in Greenwich on November 24, 4714 B.C. according to the proleptic Gregorian calendar.
INTEGER as Unix Time, the number of seconds since 1970-01-01 00:00:00 UTC.



https://www.sqlite.org/datatype3.html --> DOUBLE EXISTS BUT IS MAPPED TO REAL IN THE END --> USELESS TO SPECIFY IT !!! --> STAY SIMPLE
# maybe do a type translation automatically from the data --> in fact that is easy to do

FLOAT IS ALSO MAPPED TO REAL !!!

'''



def set_property(db_file, neo_data):
    # no data --> nothing to do
    if not neo_data:
        return
    try:

        db = TAsql(db_file)
        try:
            if db.exists('properties'):
                # db exists --> update it rather than recreating everything
                header, cols = db.run_SQL_command_and_get_results('SELECT * FROM properties',
                                                                  return_header=True)
                cols = cols[0]
                data = _to_dict(header, cols)
                data.update(neo_data)
            else:
                data = neo_data

            data = {k: [v] for k, v in data.items()}
            db.create_and_append_table('properties', data)
        except:
            traceback.print_exc()
            print('An error occurred while reading properties from the image or writing them to the database')
        finally:
            try:
                db.close()
            except:
                pass
    except:
        traceback.print_exc()
        print('Error could not insert new properties')

# returns the desired property from a TA db for the current file or None if not found
def get_property(db_file, property_name):
    val = None
    db = TAsql(db_file)
    try:
        val = db.run_SQL_command_and_get_results('SELECT '+property_name+' from properties')[0][0]
        if isinstance(val, str) and (val.lower() == 'none' or val.lower() == 'null' or val.strip() == ''):
            val=None
    except:
        # if anything goes wrong or table or col does not exist --> return None --> this is exactly what I want!!!
        pass
    db.close()
    return val

# def  get_registration(db_file):
#     val = None
#     db = TAsql(db_file)
#     try:
#         val = db.run_SQL_command_and_get_results('SELECT ' + property_name + ' from properties')[0][0]
#         if isinstance(val, str) and (val.lower() == 'none' or val.lower() == 'null' or val.strip() == ''):
#             val = None
#     except:
#         # if anything goes wrong or table or col does not exist --> return None --> this is exactly what I want!!!
#         pass
#     db.close()
#     return val


def get_properties_master_db(lst):
    database_list = smart_TA_list(lst, 'pyTA.db')
    for db_file in database_list:
        db = TAsql(db_file)
        if not 'properties' in db.get_tables():
            db.create_table('properties', ['voxel_z_over_x_ratio', 'time'], ['float', 'float'])
            # fill the row with Null
            # db.
            db.execute_command("INSERT INTO properties ('voxel_z_over_x_ratio', 'time') "
                               "SELECT NULL, NULL "
                               "WHERE NOT EXISTS (SELECT * FROM properties)"
                               )
        else:
            if db.isTableEmpty('properties'):
                db.execute_command("INSERT INTO properties ('voxel_z_over_x_ratio', 'time') "
                                   "SELECT NULL, NULL "
                                   "WHERE NOT EXISTS (SELECT * FROM properties)"
                                   )
        db.close()
    database = createMasterDB(lst, database_name='properties')
    
    # database.run_SQL_command_and_get_results('.schema properties')

    # .schema table_name
    # print('test', database)
    return database




# TODO reimplement an except
# get the properties search
def reinject_properties_to_TA_files(lst, master_db, indices_to_update=None):
    database_list = smart_TA_list(lst, 'pyTA.db')
    for iii, db_file in enumerate(database_list):
        if indices_to_update is not None:
            if iii not in indices_to_update:
                continue
        master_db.attach_table(db_file,'tmp')
        # attach tmp pyTA table to in mem master db
        master_db.execute_command('DROP TABLE IF EXISTS tmp.properties')


        # print(master_db.EXCEPT('properties', *['frame_nb', 'filename']))

        master_db.execute_command('CREATE TABLE tmp.properties AS SELECT '+master_db.EXCEPT('properties', *['frame_nb', 'filename'])+' FROM properties WHERE frame_nb ='+str(iii))
        # []
        # detach pyta table
        master_db.execute_command('DETACH DATABASE tmp')


# TODO also I would need handle polarity smarty but keep it channels --> TODO --> treat all of them as a single piece of data
def populate_table_content(db_name, prepend='#',
                           filtered_out_columns=['x', 'y', 'first_pixel', 'local_ID', 'pixel_within_cell', 'centroid',
                                                 'vx', 'cell_id',
                                                 'pixel_within_', 'perimeter_pixel_count', 'bond_cut_off', 'vertices', 'bonds', 'pixel_count']):  # TODO --> handle polarity nematic smartly --> need P1 and P2
    # return a list of col by names
    # with a few exceptions that would be plotted differently
    if not os.path.isfile(db_name):
        # file does not exist --> nothing to do
        return None
    table_content = []
    db = None
    try:
        db = TAsql(db_name)
        # get all the tables and populate it --> get anything but local ID first
        tables = db.get_tables()
        for table in tables:
            columns = db.get_table_column_names_and_types(table, return_colnames_only=True)

            cols_to_remove = []
            if filtered_out_columns is not None:
                for filter in filtered_out_columns:
                    for col in columns:
                        if col.startswith(filter) or col == filter:
                            # columns.remove(col)
                            # print('removing ', col)
                            cols_to_remove.append(col)
                            # break
                    columns = [col for col in columns if col not in cols_to_remove]
            for col in columns:
                table_content.append(('' if prepend is None else str(prepend))+ str(table) + '.' + str(col))
    except:
        traceback.print_exc()
        logger.error('error pouplating the database')
    finally:
        if db is not None:
            db.close()
        return table_content


# do a sql class to store all TA data --> see what is the best way to add all
# maybe do different classes
# maybe a map with each column title and a corresponding array of values would be a good idea in fact to get started
# or a row per row stuff and a header is also interesting --> really give it a try
# maybe also a typing of the table is a good idea
if __name__ == '__main__':
    import sys

    # --> could create the master db and then even further filter it
    # can I also do ins by sets --> think about it

    # NB in is same as
    #SELECT * FROM employees WHERE employee_id = 1 OR employee_id = 2 OR employee_id = 3 OR employee_id = 4;
    # SELECT * FROM employees WHERE first_name NOT IN ('Sarah', 'Jessica'); # maybe useful too

    if True:
        db = TAsql('/E/Sample_images/sample_images_PA/trash_test_mem/mini_different_nb_of_channels/focused_Series012/pyTA.db')
        # print(db.create_filtered_query('SELECT * from cells_2D', filtering_elements=[1,2,120],filter_name='local_ID'))
        print(db.create_filtered_query('SELECT * from cells_2D', filtering_elements_dict={'local_ID':[1,2,120]}))
        print(db.create_filtered_query('SELECT * from cells_2D', filtering_elements_dict={'local_ID':[1,2,120], 'cytoplasmic_area':[802]}))
        db.close()

        sys.exit(0)

    if True:
        new_properties = {'reg_x':16, 'reg_y':-32}
        set_property(
            '/E/Sample_images/sample_images_PA/trash_test_mem/mini_different_nb_of_channels/focused_Series012/pyTA.db',
            new_properties)  # ça marche!!!
        print(get_property(
            '/E/Sample_images/sample_images_PA/trash_test_mem/mini_different_nb_of_channels/focused_Series012/pyTA.db',
            'reg_x') + 1)  # ça marche!!!
        sys.exit(0)

    if False:
        lst = {'toto':'tutu', 'tata':None}
        print(', '.join((str(x) + ' ' + str(y)) if y is not None else str(x) + ' ' + 'Null' for x, y in lst.items()))
        import sys
        sys.exit(0)

        lst = [0, None, 123]
        print(_list_to_string(lst))


        print(', '.join(map(str, lst)))
        print(', '.join(map(lambda x: str(x) if x is not None else 'Null', lst)))
        print("'" + '\', \''.join(map(lambda x: str(x) if x is not None else 'Null', lst)) + "'")
        add_quotes = True
        print(', '.join(map(lambda x: "'"+str(x)+"'" if x is not None and add_quotes else str(x) if x is not None and not add_quotes else 'Null', lst)))


    if False:
        # use these properties to compute the real 3D area values --> maybe also store pc data in this

        print(get_property('/E/Sample_images/sample_images_PA/trash_test_mem/mini_different_nb_of_channels/focused_Series012/pyTA.db', 'time')+1) # ça marche!!!
        print(get_property('/E/Sample_images/sample_images_PA/trash_test_mem/mini_different_nb_of_channels/focused_Series012/pyTA.db', 'voxel_z_over_x_ratio'))
        import sys
        sys.exit(0)

    if False:
        tst_frequency_from_list()
        import sys

        sys.exit(0)

    if False:
        # all seems ok and in many aspects simpler than the java equivalent...
        sql_file = '/E/Sample_images/sample_images_PA/trash_test_mem/mini/test.db'
        table_name = 'test'
        table_columns = ['aaa', 'bbb', 'ccc', 'ddd', 'eee']
        table_cols_and_type = {'zzz': None, 'aaa': 'DOUBLE', 'bbb': 'INT',
                               'ccc': 'TEXT'}  # see all the types that exist and stick to that!!!

        one_row = [None, 10, 123., True, 'toto']

        print(_list_to_string(['10', '10.0', '20', 'TEXT1']))

        # magic table
        table_magic = {'zzz': [10, 20], 'aaa': [10., 22.], 'bbb': [20, 30],
                       'ccc': ['TEXT1', 'TEXT2']}

        print(_list_to_string(table_columns), table_columns)
        print(_list_to_string(table_cols_and_type), table_cols_and_type)

        print('line below will generate an error!')
        print(get_types_from_data(one_row))

        # now really create a table and then fill it --> TODO

        # test of all
        db = TAsql(sql_file)  # maybe if file is none --> store in mem ???
        db.create_and_append_table(table_name=table_name, datas=table_magic)
        # create and append table

        data = db.run_SQL_command_and_get_results('SELECT * FROM test WHERE ccc == "TEXT2"')
        print(data)  # --> just gives me the data row by row --> probably pandas offers me more flexibility!!!

        headers = db.get_table_header('test2')
        print('headers', headers)

        headers = db.get_table_header('test')
        print('headers', headers)

        print(db.exists('test 2312'))
        print(db.exists('test'))

        print('list of tables', db.get_tables())
        # [('table1',), ('table2',), ('table3',)]

        # sqlite3.Warning: You can only execute one statement at a time. -> see how I can concatenate the stuff
        # maybe just return the last data --> does make sense in fact
        # --> split the command and execute it
        print('test multi', db.run_SQL_command_and_get_results('SELECT * FROM test;'
                                                               'SELECT "aaa" FROM test;'))  # bug

        print('get col', db.get_column('test', 'aaa'), type(db.get_column('test', 'aaa')[
                                                                0]))  # --> the type is correct --> so in theory no need for casting but maybe if wrong type then it is needed

        # can use the stuff
        # can use the table headers TODO plots --> in fact that would be easy TODO I think
        # --> can easily add anything as a graph if I need it
        # add an edit for the images at the very end in the preview --> see how I can do that --> would be good if cell divisions and death could be edited there --> but need keep the cell relationship --> in fact easy just delete the table

        # print type

        # can do automatic plots for all
        # and maybe also offer color coding for all # --> either local or global --> in fact all is easy todo with my tool now!!!

        # for some columns the plots should be specific for others

        db.add_column('test', 'uuu', default_value=None)
        print(db.get_table_column_names_and_types('test'))

        db.get_min_max('test', 'aaa')

        db.clean()

        db.close()

        # try create a table
        # maybe also need typing of the data --> try

    if False:
        values = ['10', 20, 30, None, '10.2', '0.1', '0', 5]
        convert_digit_to_number = True
        if convert_digit_to_number:
            non_str = [val for val in values if not isinstance(val, str)]
            st = [val for val in values if isinstance(val, str)]
            st = [float(val) for val in st if val.isdigit()]  # ok but it does change the order --> should I care

            # could convert to None all the data that cannot be processed!!!

            print(values)
            print(non_str)
            print(st)

            # non_str2 = [val if not isinstance(val, str) else float(val) if val.isdigit() for val in values]
            # do it all manually cause too complex with list comprehension
            # a way of converting str to int
            for iii, v in enumerate(values):
                if isinstance(v, str):
                    # if v.isdigit():
                    #     values[iii]=float(v)
                    # else:
                    #     values[iii] = None
                    # this one is better as it should keep the type
                    if '.' in v:
                        try:
                            values[iii] = float(v)
                        except ValueError:
                            values[iii] = None
                    else:
                        try:
                            values[iii] = int(v)
                        except ValueError:
                            values[iii] = None

            print(values)

    # pbs will occur when files are deleted in preview --> prevent delete button from working
    # TODO populate the other based on that
    # concat table name to the column name
    # assume everything can be plotted directly except a few columns that I would name and plot differently --> TODO
    # add correspondance local cell id and track ID to the database
    if True:
        # nb some images have different nb of cols --> does not fit...
        # masterDB = createMasterDB(['/E/Sample_images/sample_images_PA/trash_test_mem/mini/focused_Series012.png','/E/Sample_images/sample_images_PA/trash_test_mem/mini/focused_Series014.png','/E/Sample_images/sample_images_PA/trash_test_mem/mini/focused_Series015.png','/E/Sample_images/sample_images_PA/trash_test_mem/mini/focused_Series016.png'])
        # masterDB = createMasterDB(
        #     ['/E/Sample_images/sample_images_PA/trash_test_mem/mini_different_nb_of_channels/focused_Series012.png',
        #      '/E/Sample_images/sample_images_PA/trash_test_mem/mini_different_nb_of_channels/focused_Series014.png',
        #      '/E/Sample_images/sample_images_PA/trash_test_mem/mini_different_nb_of_channels/focused_Series015.png',
        #      '/E/Sample_images/sample_images_PA/trash_test_mem/mini_different_nb_of_channels/focused_Series016.png',
        #      '/E/Sample_images/sample_images_PA/trash_test_mem/mini_different_nb_of_channels/focused_Series019.tif'])

        # masterDB = createMasterDB(loadlist('/E/Sample_images/sample_images_PA/trash_test_mem/mini_different_nb_of_channels/list.lst'))


        outputName = None
        # outputName = '/E/Sample_images/sample_images_pyta/surface_projection/masterDB.db'
        masterDB = createMasterDB(loadlist('/E/Sample_images/sample_images_pyta/surface_projection/list.lst'), outputName=outputName, force_track_cells_db_update=True)


        if masterDB is not None:
            # masterDB = createMasterDB(['/E/Sample_images/sample_images_PA/trash_test_mem/mini/focused_Series012.png','/E/Sample_images/sample_images_PA/trash_test_mem/mini/focused_Series012.png','/E/Sample_images/sample_images_PA/trash_test_mem/mini/focused_Series012.png','/E/Sample_images/sample_images_PA/trash_test_mem/mini/focused_Series012.png'])
            print(masterDB.get_tables())  # why None
            # print(masterDB.run_SQL_command_and_get_results('SELECT * FROM cells_2D;')) # that seems to work
            # print(masterDB.get_min_max(table_name, 'area'))
            # print(masterDB.get_min_max(table_name, 'area', freq=[0.15,0.05])) # 0.1 --> est ce que ce sont des pourcentages ??? --> dois-je changer ma conversion --> sinon 10% --> may  be a lot in fact --> ij fact ok
            # print(masterDB.get_table_column_names_and_types('bonds_2D'))

            # print(masterDB.print_query('SELECT * FROM vertices_2D;'))
            # print(masterDB.print_query('SELECT * FROM cells_2D;'))
            # print(masterDB.print_query('SELECT frame_nb, filename, local_ID, cytoplasmic_area, area FROM cells_2D;'))
            # print(masterDB.print_query('SELECT frame_nb, filename, local_ID, sum_px_int_vertices_included_ch1, sum_px_int_vertices_included_ch3 FROM bonds_2D;')) # amyebe there is a bug in the db --> really need fix it, it should contain None and it does not --> I am doing a mistake --> fix it!!!
            # print(masterDB.get_table_column_names_and_types('cells_2D'))
            # print(masterDB.get_table_column_names_and_types('bonds_2D')) #

            # print(masterDB.print_query('SELECT * FROM cells_2D NATURAL JOIN properties LIMIT 1'))


            # masterDB.drop_table('properties')
            # surprising that seems to work --> what if 0 DB inside
            # print(masterDB.print_query('SELECT * FROM cells_2D NATURAL JOIN properties'))  # ça a l'air de marcher mais va t'il y avoir des bugs ??? sinon facile à gerer je pense

            # how to make it work if the table does not exist at all --> did I take this into account --> # can I null it everywhere ????

            # print(masterDB.print_query('SELECT * FROM cell_tracks NATURAL JOIN properties'))
            # print(masterDB.print_query('SELECT * FROM cell_tracks'))
            print(masterDB.print_query('SELECT * FROM cells_2D NATURAL JOIN cells_3D NATURAL JOIN cell_tracks NATURAL JOIN properties'))  # ça a l'air de marcher mais va t'il y avoir des bugs ??? sinon facile à gerer je pense


            # print(masterDB.print_query('SELECT * FROM cells_2D NATURAL JOIN properties')) # ça a l'air de marcher mais va t'il y avoir des bugs ??? sinon facile à gerer je pense

            print(masterDB.get_min_max('bonds_2D', 'sum_px_int_vertices_included_ch1'))  # see how to exclude None values
            print(masterDB.get_min_max('bonds_2D', 'sum_px_int_vertices_included_ch1',
                                       ignore_None_and_string=False))  # see how to exclude None values
            print(masterDB.get_min_max('bonds_2D', 'sum_px_int_vertices_included_ch1', freq=[0.01, 0.01],
                                       ignore_None_and_string=False))  # see how to exclude None values
            print(masterDB.get_min_max('bonds_2D', 'sum_px_int_vertices_included_ch1', freq=[0.3, 0.01],
                                       ignore_None_and_string=False))  # see how to exclude None values
            print(masterDB.get_min_max('bonds_2D', 'sum_px_int_vertices_included_ch1', freq=[0.01, 0.01],
                                       ignore_None_and_string=True))  # see how to exclude None values
            # any way to print this ???


            # can I save an in mem table --> No

            masterDB.close()

    if True:
        sql_file = '/E/Sample_images/sample_images_PA/trash_test_mem/mini/focused_Series012/pyTA.db'

        print(populate_table_content(sql_file))

        table_name = 'cells_2D'

        db = TAsql(sql_file)  # maybe if file is none --> store in mem ???
        db.get_min_max(table_name, 'area')
        db.get_min_max(table_name, 'area', freq=[0.1, 0.1])
        print(db.get_table_column_names_and_types(table_name))  # what if two cols have same name ???
        db.clean()
        db.close()
