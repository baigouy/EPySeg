import sqlite3
import traceback
import os
import numpy as np
import math
import pandas
from prettytable import PrettyTable
import epyseg.SQLite_tools.tools
from epyseg.img import Img, has_metadata, _create_dir
from epyseg.ta.selections.selection import convert_coords_to_IDs
from epyseg.tools.early_stopper_class import early_stop
from epyseg.tools.logger import TA_logger  # logging
from epyseg.utils.loadlist import smart_TA_list,loadlist
from epyseg.ta.tracking.tools import smart_name_parser
import pandas as pd
from pathlib import Path
import pandas as pd
import os

logger = TA_logger()

# " need header"
def create_table_and_append_data(db_path, table_name, columns, data, column_types=None, temporary=False):
    """
    Creates an SQL table in a database and fills it with data.

    Args:
        db_path (str): The path to the database file.
        table_name (str): The name of the table to create.
        columns (list): A list of column names.
        data (dict): A dictionary containing the data to populate the table, where the keys represent column names and the values represent column data.
        column_types (list, optional): A list of column types. Defaults to None.
        temporary (bool, optional): Specifies whether the created table is temporary. Defaults to False.

    Raises:
        Exception: If an error occurs while creating the table or filling it with data, or when closing the database connection.

    Examples:
        # >>> columns = ['id', 'name', 'age']
        # >>> data = {
        # ...     'id': [1, 2, 3],
        # ...     'name': ['John', 'Jane', 'Alice'],
        # ...     'age': [25, 32, 41]
        # ... }
        # >>> create_table_and_append_data("mydb.db", "persons", columns, data)
        #
        # >>> columns = ['id', 'fruit']
        # >>> data = {
        # ...     'id': [1, 2, 3],
        # ...     'fruit': ['apple', 'banana', 'cherry']
        # ... }
        # >>> column_types = ['INTEGER', 'TEXT']
        # >>> create_table_and_append_data("mydb.db", "fruits", columns, data, column_types=column_types, temporary=True)

        # >>> columns = ['id', 'fruit']
        # >>> data = np.array( [['1', 'apple'], ['2', 'banana'], ['3', 'cherry']])
        # >>> column_types = ['INTEGER', 'TEXT']
        # >>> create_table_and_append_data("mydb.db", "fruits", columns, data, column_types=column_types, temporary=False)
    """

    db = None

    try:
        db = TAsql(db_path)

        # Drop the table if it already exists
        db.drop_table(table_name)

        # Prepare the column content for the CREATE TABLE statement
        col_content = columns
        if isinstance(col_content, list):
            if column_types is not None:
                # Combine the column names and types
                concat = list(zip(columns, column_types))
                concat = [str(name) + ' ' + str(type) for name, type in concat]
                col_content = _list_to_string(concat, add_quotes=False)
            else:
                col_content = _list_to_string(col_content, add_quotes=False)

        # Create the table with the specified columns and column types
        db.cur.execute(
            'CREATE' + (' TEMPORARY' if temporary else '') + ' TABLE ' + table_name + ' (' + col_content + ')')
        db.fill_table(table_name, data)

    except:
        traceback.print_exc()
        logger.error('Something went wrong...')

    finally:
        if db is not None:
            try:
                db.close()
            except:
                traceback.print_exc()

def get_table_columns(path_to_db, table_name):
    """
    Given a table name, prints the names of all the columns in the table.

    Args:
      path_to_db (str): The path to the SQLite database file.
      table_name (str): The name of the table.

    Returns:
      None

    Examples:
      >>> get_table_columns('/media/teamPrudhomme/EqpPrudhomme2/Benoit_pr_Benjamin/coccinelles/latest images_20230614/raw images/N4N4_29_M_1a/ladybug_seg.db', 'elytras_shape')[:2]
      ['areas_without_holes', 'areas_with_holes']
    """
    return epyseg.SQLite_tools.tools.get_table_columns(path_to_db, table_name)


def remove_dupes_from_table_and_overwrite_table(db_path, table_name):
    """
    Removes duplicates from a table and overwrites the table with the deduplicated data.

    Args:
        db_path (str): The path to the database file.
        table_name (str): The name of the table to deduplicate.

    Returns:
        list: The query results.

    # Examples:
    #     >>> remove_dupes_from_table_and_overwrite_table('database.db', 'my_table')
    #     [1, 2, 3, 4]
    """

    query_results = []
    db = None

    try:
        db = TAsql(db_path)

        headers, data_rows = db.run_SQL_command_and_get_results('SELECT DISTINCT * FROM ' + table_name, return_header=True)

        if not isinstance(data_rows, np.ndarray):
            data_rows = np.asarray(data_rows, dtype=object)
        data_rows = data_rows.T
        finally_formatted_table = {}

        for iii, header in enumerate(headers):
            try:
                finally_formatted_table[header] = data_rows[iii].tolist()
            except:
                logger.warning('no data to be added --> the column will be empty')
                finally_formatted_table[header] = []

        db.drop_table(table_name)
        db.create_and_append_table(table_name, finally_formatted_table)

    except:
        traceback.print_exc()
        logger.error('Something went wrong...')

    finally:
        if db is not None:
            try:
                db.close()
            except:
                traceback.print_exc()

        return query_results


# maybe put this some other place --> this saves a table data and header as csv

def save_data_to_csv(output_file_name, header, data):
    """
    Saves data to a CSV file.

    Args:
        output_file_name (str): The name of the output CSV file.
        header (list): The list of column headers.
        data (list): The list of data rows.

    # Examples:
    #     >>> header = ['Name', 'Age', 'Gender']
    #     >>> data = [['John', 25, 'Male'], ['Jane', 30, 'Female']]
    #     >>> save_data_to_csv('output.csv', header, data)
    """

    _create_dir(output_file_name)  # Create directory if it doesn't exist

    if header is not None and header:  # Check if header is provided and not empty
        db_df = pd.DataFrame(data=data if data else None, columns=header)  # Create a DataFrame with the data and header
        db_df.to_csv(output_file_name, index=False)  # Save the DataFrame to a CSV file without index
    else:
        open(output_file_name, 'w').close()  # Create an empty file if no header is provided


def prepend_to_content(content, thing_to_prepend, auto_convert_tuple_to_list=True):
    """
    Prepends a value to each element in a nested list or a single list.

    Args:
        content (list or tuple): The content to prepend the value to.
        thing_to_prepend (any): The value to prepend.
        auto_convert_tuple_to_list (bool): Flag to automatically convert tuples to lists.

    Returns:
        list or tuple: The modified content with the value prepended.

    Examples:
        >>> content = [['Apple', 'Banana'], ['Orange', 'Mango']]
        >>> prepend_to_content(content, 'Fruit')
        [['Fruit', 'Apple', 'Banana'], ['Fruit', 'Orange', 'Mango']]
    """

    if isinstance(content, list) and content:  # Check if content is a non-empty list
        if isinstance(content[0], (list, tuple)):  # Check if content is a nested list or tuple
            for iii, cur_content in enumerate(content):
                if auto_convert_tuple_to_list and isinstance(cur_content, tuple):
                    cur_content = list(cur_content)  # Convert tuple to list if auto conversion is enabled
                cur_content.insert(0, thing_to_prepend)  # Prepend the value to each element
                content[iii] = cur_content  # Update the modified element in the content list
        else:
            if auto_convert_tuple_to_list and isinstance(content, tuple):
                content = list(content)  # Convert tuple to list if auto conversion is enabled
            content.insert(0, thing_to_prepend)  # Prepend the value to the content list
    return content

def query_db_and_get_results(db_path, SQL_command_to_run):
    """
    Executes an SQL command on a database and retrieves the query results.

    Args:
        db_path (str): The path to the database.
        SQL_command_to_run (str): The SQL command to execute.

    Returns:
        list: The results of the SQL query.

    # Examples:
    #     >>> query_db_and_get_results('database.db', 'SELECT * FROM customers')
    #     [['John', 'Doe', 'john.doe@example.com'], ['Jane', 'Smith', 'jane.smith@example.com']]
    """

    from epyseg.ta.database.sql import TAsql  # Import the required module
    query_results = []  # Initialize an empty list to store the query results
    db = None

    try:
        db = TAsql(db_path)  # Create a TAsql object with the specified database path
        query_results = db.run_SQL_command_and_get_results(SQL_command_to_run, return_header=False)
        # Execute the SQL command and retrieve the results
    except:
        traceback.print_exc()  # Print the traceback in case of an exception
        logger.error('Something went wrong...')  # Log an error message
    finally:
        if db is not None:
            try:
                db.close()  # Close the database connection
            except:
                traceback.print_exc()  # Print the traceback in case of an exception
        return query_results  # Return the query results

def table_exists_in_db(db_path, table_name):
    """
    Checks if a table exists in a database.

    Args:
        db_path (str): The path to the database.
        table_name (str): The name of the table to check.

    Returns:
        bool: True if the table exists, False otherwise.

    # Examples:
    #     >>> table_exists_in_db('database.db', 'customers')
    #     True
    """
    from epyseg.ta.database.sql import TAsql
    db = None
    exists = False
    try:
        db = TAsql(db_path)
        exists = db.exists(table_name)
    except:
        traceback.print_exc()
        logger.error('Something went wrong...')
    finally:
        if db is not None:
            try:
                db.close()
            except:
                traceback.print_exc()
        return exists


# so maybe xlsx file is a good way of storing data in the end... --> TODO maybe too...
def table_to_xlsx_with_sheets(db_path, output_xlsx_file):
    """
    Converts SQL tables in a database to an Excel file with separate sheets for each table.

    Args:
        db_path (str): The path to the database.
        output_xlsx_file (str): The path to the output Excel file.

    Returns:
        None

    Examples:
        table_to_xlsx_with_sheets('database.db', 'output.xlsx')
    """
    # TODO maybe truncate tab name to 31 characters to avoid issues in Excel

    # read all the tables and store them to the xlsx db
    try:
        db = TAsql(db_path)
        tables = db.get_tables()

        if db is not None:
            try:
                db.close()
            except:
                traceback.print_exc()

        conn = sqlite3.connect(db_path)
        with pd.ExcelWriter(output_xlsx_file) as writer:
            for table in tables:
                df = pandas.read_sql_query('SELECT * from '+table, conn)
                df.to_excel(writer, sheet_name=table, index=False)

        conn.close()
    except:
        traceback.print_exc()
        logger.error('Something went wrong...')



class TAsql:
    def __init__(self, filename_or_connection=None, add_useful_missing_SQL_commands=True):
        """
        Initializes a TAsql object.

        Args:
            filename_or_connection (str or sqlite3.Connection): The filename or connection to the SQLite database.
            add_useful_missing_SQL_commands (bool): Whether to add useful missing SQL commands. Default is True.

        Returns:
            None
        """
        self.db_name = filename_or_connection
        if isinstance(filename_or_connection, sqlite3.Connection):
            self.con = filename_or_connection
            self.db_name = None
            logger.debug('Opened database from connection')
        elif filename_or_connection is None:
            # if no file name is specified, create an in-memory database
            self.con = sqlite3.connect(":memory:")
            self.db_name = ':memory:'
        else:
            if not os.path.exists(self.db_name):
                parent_dir = os.path.dirname(self.db_name)
                if not os.path.exists(parent_dir) and parent_dir != '':
                    # if parent folder does not exist, create it so that the db can be created too
                    os.makedirs(parent_dir, exist_ok=True)
            self.con = sqlite3.connect(self.db_name)
            logger.debug('Opened database: ' + str(self.db_name))

        if add_useful_missing_SQL_commands:
            self.add_useful_missing_SQL_commands()
        self.cur = self.con.cursor()

    def add_useful_missing_SQL_commands(self):
        """
        Adds useful missing SQL commands to the SQLite database connection.

        Args:
            None

        Returns:
            None
        """
        self.con.create_function("sin", 1, math.sin)
        self.con.create_function("atan2", 2, math.atan2)
        self.con.create_function("sqrt", 1, math.sqrt)

    def drop_table(self, table_name):
        """
        Drops a table from the SQLite database.

        Args:
            table_name (str): The name of the table to drop.

        Returns:
            None
        """
        self.cur.execute('DROP TABLE IF EXISTS ' + table_name)

    def create_and_append_table(self, table_name, datas, temporary=False):
        """
        Creates a new table in the SQLite database and appends data to it.

        Args:
            table_name (str): The name of the table to create.
            datas (dict): A dictionary containing the column names as keys and the data as values.
            temporary (bool, optional): Specifies whether the table is temporary. Defaults to False.

        Returns:
            None
        """
        # Create the table with the specified column names and types
        self.create_table(table_name, list(datas.keys()), column_types=get_types_from_data(datas), temporary=temporary)

        # Fill the table with data
        self.fill_table(table_name, datas)


    def create_table(self, table_name, columns, column_types=None, temporary=False):
        """
        Creates a new table in the SQLite database with the specified columns and column types.

        Args:
            table_name (str): The name of the table to create.
            columns (list): A list of column names.
            column_types (list, optional): A list of column types corresponding to the columns. Defaults to None.
            temporary (bool, optional): Specifies whether the table is temporary. Defaults to False.

        Returns:
            None
        """
        # TODO if types are specified --> need add it

        # Drop the table if it already exists
        self.drop_table(table_name)

        # Prepare the column content for the CREATE TABLE statement
        col_content = columns
        if isinstance(col_content, list):
            if column_types is not None:
                # Combine the column names and types
                concat = list(zip(columns, column_types))
                concat = [str(name) + ' ' + str(type) for name, type in concat]
                col_content = _list_to_string(concat, add_quotes=False)
            else:
                col_content = _list_to_string(col_content, add_quotes=False)

        # Create the table with the specified columns and column types
        self.cur.execute('CREATE' + (' TEMPORARY' if temporary else '') + ' TABLE ' + table_name + ' (' + col_content + ')')

    def fill_table(self, table_name, datas):
        """
        Fills the specified table in the SQLite database with the provided data.

        Args:
            table_name (str): The name of the table to fill.
            datas (dict or list): The data to insert into the table. If a dictionary is provided, the values will be inserted as rows in the table.

        Returns:
            None
        """
        if isinstance(datas, dict):
            # Convert the dictionary values to a list
            datas = list(datas.values())

            # Reorder the data into a more friendly format
            datas = np.array(datas, dtype=object).T.tolist()

        # TODO this is totally inefficient --> change the code some day to directly handle np.ndarray some day
        if isinstance(datas, np.ndarray):
            datas = datas.tolist()

        # Iterate over the data and insert into the table
        for data in datas:
            # Convert the data list to a string
            list_as_string = _list_to_string(data) # bug is here

            # Generate the INSERT INTO command
            COMMAND = "INSERT INTO " + table_name + " VALUES (" + list_as_string + ")"

            # Check if the command has empty values and insert a row of NULL values
            if COMMAND.endswith('()'):
                print('empty values creating an empty row filled with NULL')
                self.cur.execute("INSERT INTO " + table_name + " DEFAULT VALUES")
            else:
                self.cur.execute(COMMAND)

        # Commit the changes to the database
        self.con.commit()

    def exists(self, table_name, attached_table=None):
        """
        Checks if a table exists in the SQLite database.

        Args:
            table_name (str): The name of the table to check.
            attached_table (str, optional): The name of the attached table. Defaults to None.

        Returns:
            bool: True if the table exists, False otherwise.
        """
        if table_name is None:
            return None

        # Check if the table exists in the main database
        if not '.' in table_name:
            self.cur.execute("SELECT COUNT(name) FROM " + ('' if attached_table is None else str(
                attached_table) + '.') + "sqlite_master WHERE type='table' AND name='" + table_name + "';")
        else:
            # Split the table name to detect temporary tables
            master, table = table_name.split('.')
            self.cur.execute("SELECT COUNT(name) FROM " + str(
                master) + '.' + "sqlite_master WHERE type='table' AND name='" + table + "';")

        # If the count is 1, then the table exists
        if self.cur.fetchone()[0] == 1:
            return True

        return False

    def save_query_to_csv_file(self, sql_command, output_file_name):
        """
        Executes an SQL query and saves the results to a CSV file.

        Args:
            sql_command (str): The SQL command to execute.
            output_file_name (str): The name of the output CSV file.

        Returns:
            None
        """
        import pandas as pd

        # Execute the SQL query and store the results in a pandas DataFrame
        db_df = pd.read_sql_query(sql_command, self.con)

        # Save the DataFrame to a CSV file
        db_df.to_csv(output_file_name, index=False)

    def get_tables(self, force_lower_case=False):
        """
        Retrieves the list of table names in the database.

        Args:
            force_lower_case (bool): If True, forces the table names to be returned in lowercase.

        Returns:
            list: The list of table names in the database.
        """
        try:
            # Execute an SQL query to retrieve the table names from the database
            self.cur.execute("SELECT name FROM sqlite_master WHERE type='table';")
            query_result = self.cur.fetchall()
            # Unpack the query result to extract the table names
            names = self._unpack(query_result)
            if force_lower_case:
                # Convert table names to lowercase if specified
                names = _to_lower(names)
            return names
        except:
            # Something went wrong, assume no tables exist
            return None

    def close(self):
        """
        Closes the database connection.

        This method commits any pending changes and closes the connection to the database.
        """
        try:
            # Commit any pending changes before closing
            self.con.commit()
        except:
            # Ignore any exceptions that occur during commit
            pass

        logger.debug('Closing database: ' + str(self.db_name))
        # Close the database connection
        self.con.close()

    def get_table_header(self, tablename):
        """
        Retrieves the column names of a table.

        Args:
            tablename (str): The name of the table.

        Returns:
            list: The list of column names.
        """
        # Retrieve the column names using the get_table_column_names_and_types method
        header = self.get_table_column_names_and_types(table_name=tablename, return_colnames_only=True)

        # Return the column names
        return header

    def attach_table(self, dbName, nickName):
        """
        Attaches a new table to the current database.

        Args:
            dbName (str): The filename of the table to attach.
            nickName (str): The nickname for the attached table.

        Returns:
            None
        """
        if dbName is not None:
            # Replace backslashes with forward slashes in the filename
            dbName = dbName.replace('\\\\', '/').replace('\\', '/')

            # Execute the command to attach the table
            self.execute_command("ATTACH DATABASE '" + dbName + "' AS '" + nickName + "'")

    def detach_table(self, nickName):
        """
        Detaches a table from the current database.

        Args:
            nickName (str): The nickname of the table to detach.

        Returns:
            None
        """
        if nickName is not None:
            # Execute the command to detach the table
            self.execute_command("DETACH DATABASE '" + nickName + "'")

    def execute_command(self, SQL_command, warn_on_error=True):
        """
        Executes an SQL command on the database.

        Args:
            SQL_command (str): The SQL command to execute.
            warn_on_error (bool): If True, prints a warning message on error.

        Returns:
            None
        """
        if SQL_command is None:
            return
        try:
            # Execute the SQL command
            self.cur.execute(SQL_command)
            self.cur.fetchall()
            self.con.commit()
        except:
            if warn_on_error:
                traceback.print_exc()
                logger.error(
                    'error executing the following command:\n"' + str(SQL_command) + '"' + '\ntable name:' + str(
                        self.db_name))

    def EXCEPT(self, table_name, *columns_to_exclude):  # concatTableName=False,
        """
        Generates a list of column names for a table excluding specified columns.

        Args:
            table_name (str): The name of the table.
            *columns_to_exclude (str): Columns to exclude from the generated list.

        Returns:
            str: A comma-separated string of column names excluding the specified columns.
        """
        columns_to_exclude = [col.lower() for col in columns_to_exclude]
        columns = self.get_table_column_names_and_types(table_name, return_colnames_only=True)
        columns = [col.lower() for col in columns]
        columns = [col for col in columns if col not in columns_to_exclude]

        # Generate the comma-separated string of column names
        return ', '.join(columns)

    def run_SQL_command_and_get_results(self, SQL_command, return_header=False, warn_on_error=True):
        """
        Executes an SQL command and retrieves the results.

        Args:
            SQL_command (str): The SQL command to execute.
            return_header (bool): If True, returns the header along with the results.
            warn_on_error (bool): If True, prints a warning message on error.

        Returns:
            tuple: A tuple containing the results and optionally the header if return_header is True.
        """
        if SQL_command is None:
            if return_header:
                return None, None
            return None

        if SQL_command.count(';') > 1:
            # Split the command and execute the last part
            SQL_commands = SQL_command.strip().split(';')
            SQL_commands = [sql_command for sql_command in SQL_commands if sql_command.strip() != '']

            if len(SQL_commands) > 1:
                last_command = SQL_commands[-1]
                for command in SQL_commands[:-1]:
                    self.run_SQL_command_and_get_results(command, return_header=False, warn_on_error=warn_on_error)

                # Execute the last command and retrieve the results
                return self.run_SQL_command_and_get_results(last_command, return_header=return_header,
                                                            warn_on_error=warn_on_error)

        try:
            self.cur.execute(SQL_command)
            query_result = self.cur.fetchall()

            if return_header:
                headers = self._unpack(self.cur.description)
                return headers, query_result

            return query_result
        except:
            # Command failed (e.g., table does not exist)
            if warn_on_error:
                traceback.print_exc()

            if return_header:
                return None, None
            return None

    def clean(self):
        """
        Cleans the database by removing unnecessary data.

        This operation can lead to a strong size reduction in the database.

        Returns:
            None
        """
        logger.debug('Cleaning the database: ' + str(self.db_name))
        self.run_SQL_command_and_get_results("VACUUM;")

    def get_column(self, table_name, column_name, sort=None):
        """
        Retrieves the values of a specific column from a table.

        Args:
            table_name (str): The name of the table.
            column_name (str): The name of the column.
            sort (str): The sorting order of the results. Can be 'ASC' for ascending or 'DESC' for descending.

        Returns:
            list: The values of the specified column.
        """
        if not self.exists(table_name) or column_name is None:
            return None
        try:
            results = self.run_SQL_command_and_get_results('SELECT "' + column_name + '" FROM "' + table_name + '"' + (
                '' if sort is None else 'ORDER BY "' + column_name + '" ' + sort), return_header=False,
                                                           warn_on_error=False)
            results = self._unpack(results)
            return results
        except:
            return None

    def _unpack(self, lst):
        """
        Unpacks a list of tuples by extracting the first element from each tuple.

        Args:
            lst (list): The list of tuples to unpack.

        Returns:
            list: The unpacked list containing the first element of each tuple.
        """
        out = [elem[0] for elem in lst]
        return out

    def get_table_column_names_and_types(self, table_name, return_colnames_only=False, attached_table=None):
        """
        Retrieves the column names and types of a table.

        Args:
            table_name (str): The name of the table.
            return_colnames_only (bool): If True, returns only the column names. If False, returns a dictionary
                mapping column names to their corresponding types.
            attached_table (str): The name of the attached table, if applicable.

        Returns:
            list or dict: The column names and types of the table, depending on the value of return_colnames_only.
        """
        if not self.exists(table_name, attached_table=attached_table):
            return None

        if not '.' in table_name:
            cols_and_types = self.run_SQL_command_and_get_results('PRAGMA ' + ('' if attached_table is None else str(
                attached_table) + '.') + 'table_info("' + table_name + '");')  # does not return attached ones --> need another code
        else:
            master, table = table_name.split('.')
            cols_and_types = self.run_SQL_command_and_get_results(
                'PRAGMA ' + master + '.' + 'table_info("' + table + '");')  # does not return attached ones --> need another code

        if cols_and_types is None:
            return None

        cols_and_types = {col[1]: col[2] for col in cols_and_types}

        if return_colnames_only:
            return list(cols_and_types.keys())
        else:
            return cols_and_types

    def get_min_max(self, table_name, column_name, freq=None, ignore_None_and_string=True, force_numeric=False):
        """
        Retrieves the minimum and maximum values of a column in a table.

        Args:
            table_name (str): The name of the table.
            column_name (str): The name of the column.
            freq (float or list or tuple): The frequency or range of frequencies at which to sample the data. If None,
                the minimum and maximum values are returned. If a single float value is provided, it represents both the
                lower and upper frequency bounds. If a list or tuple of two float values is provided, they represent
                the lower and upper frequency bounds, respectively.
            ignore_None_and_string (bool): If True, ignores None and string values when calculating the minimum and maximum.
            force_numeric (bool): If True, converts values to numeric before calculating the minimum and maximum.

        Returns:
            tuple: A tuple containing the minimum and maximum values of the column.
        """
        if table_name is None or column_name is None:
            return None, None

        sorted_data = self.get_column(table_name, column_name, sort='ASC')
        if sorted_data is None:
            return None, None

        # Call sort_col_numpy function with the appropriate parameters
        return sort_col_numpy(sorted_data, freq=freq, ignore_None_and_string=ignore_None_and_string,
                              force_numeric=force_numeric, sort=False)

    def getNbRows(self, tableName):
        """
        Counts the number of rows in a table.

        Args:
            tableName (str): The name of the table.

        Returns:
            int: The number of rows in the table.
        """
        if not self.exists(tableName):
            return 0

        SQLQuery = "SELECT COUNT(*) from '" + tableName + "';"
        value = self.run_SQL_command_and_get_results(SQLQuery)

        try:
            return value[0][0]
        except:
            print('error')

        return 0

    def add_column(self, table_name, column_name, col_type=None, default_value='NULL'):
        """
        Adds a column to a table.

        Args:
            table_name (str): The name of the table.
            column_name (str): The name of the column to be added.
            col_type (str, optional): The data type of the column. Defaults to None.
            default_value (str, optional): The default value for the column. Defaults to 'NULL'.
        """
        if not self.exists(table_name):
            logger.error('Table ' + str(table_name) + ' does not exist.')
            return

        self.execute_command('ALTER TABLE ' + table_name + ' ADD COLUMN "' + column_name + '"' +
                             ('' if col_type is None else (' ' + str(col_type) + ' ')) +
                             ('' if default_value is None else '  DEFAULT ' + str(default_value)) + ';')

    def remove_column(self, table_name, column_name):
        """
        Removes a column from a table.

        Args:
            table_name (str): The name of the table.
            column_name (str): The name of the column to be removed.
        """
        if not self.exists(table_name):
            logger.error('Table ' + str(table_name) + ' does not exist.')
            return
        try:
            self.execute_command('ALTER TABLE ' + table_name + ' DROP COLUMN "' + column_name + '"')
        except:
            # If the column does not exist, there's nothing to drop, but it's not an error.
            logger.warning(
                'Column does not exist and could not be dropped: ' + str(table_name) + ' ' + str(column_name))
            pass

    def print_query(self, SQL_command):
        """
        Prints the results of a SQL query in a formatted table.

        Args:
            SQL_command (str): The SQL query to execute and print the results.
        """
        # Retrieve the header and data from the SQL query
        header, data = self.run_SQL_command_and_get_results(SQL_command, return_header=True)

        # Create a PrettyTable instance
        x = PrettyTable(header)

        # Add each row of data to the PrettyTable
        for row in data:
            x.add_row(row)

        # Print the PrettyTable
        print(x)

    def isTableEmpty(self, tableName):
        """
        Checks if a table is empty or does not exist.

        Args:
            tableName (str): The name of the table.

        Returns:
            bool: True if the table is empty or does not exist, False otherwise.
        """
        if not self.exists(tableName):
            return True

        if self.getNbRows(tableName) == 0:
            return True

        return False

    def create_filtered_query(self, SQL_command, filtering_elements_dict=None):
        """
        Creates a filtered query based on the provided SQL command and filtering elements.

        Args:
            SQL_command (str): The original SQL command to filter.
            filtering_elements_dict (dict): A dictionary where the keys are column names and the values are lists of elements
                to filter.

        Returns:
            str: The filtered SQL query.

        Examples:
            filtering_elements_dict = {'area': [120, 130], 'local_ID': [33]}
            create_filtered_query(SQL_command, filtering_elements_dict)
            This will return: "SELECT * FROM (SQL_command) WHERE area IN (120, 130) AND local_ID IN (33)"
        """
        if filtering_elements_dict is None:
            # Empty filter, return the default command
            return SQL_command

        if not filtering_elements_dict:
            return SQL_command

        initial_command = 'SELECT * FROM (' + SQL_command + ') WHERE '
        extra_command = ''

        for kkk, (k, v) in enumerate(filtering_elements_dict.items()):
            filters = ', '.join(str(f) for f in v)
            extra_command += ' ' + (' AND ' if kkk > 0 else '') + k + ' IN (%s)' % filters

        query = initial_command + extra_command
        return query

    def filter_by_clone(self, table_name, cell_label):
        """
        Filters a database based on a clone table.

        Args:
            table_name (str): The name of the clone table in the database.
            cell_label: The cell label image used to filter the database.

        Returns:
            list: The filtered local IDs.

        Examples:
            filter_by_clone('tracked_clone', cell_label)
            This will return a list of filtered local IDs based on the provided clone table and cell label image.
        """
        coords = []
        try:
            coords = self.run_SQL_command_and_get_results('SELECT * FROM ' + table_name, return_header=False)
        except:
            traceback.print_exc()

        local_ids = convert_coords_to_IDs(cell_label, coords, forbidden_colors=[0], return_image=False,
                                          new_color_to_give_to_cells_if_return_image=None)
        return local_ids
def any_to_numeric(values):
    """
    Converts values to numeric types if possible.

    Args:
        values (list): The list of values to convert.

    Returns:
        list: The converted values.

    Examples:
        values = ['123', '3.14', '456', '7.89']
        converted_values = any_to_numeric(values)
        This will convert the values in the list to their respective numeric types.
    """
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

def get_numeric_value(v):
    """
    Converts a value to a numeric type if possible.

    Args:
        v: The value to convert.

    Returns:
        The converted numeric value if conversion is successful, None otherwise.

    Examples:
        value = '3.14'
        converted_value = get_numeric_value(value)
        This will convert the value to a float type.
    """
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


import numpy as np

def sort_col_numpy(unsorted_data, freq=None, ignore_None_and_string=True, force_numeric=False, sort=True):
    """
    Sorts a column of data using NumPy and computes the minimum and maximum values.

    Args:
        unsorted_data: The column of data to sort.
        freq: The frequency range to consider for computing the minimum and maximum values.
        ignore_None_and_string: Flag to ignore None and string values during sorting.
        force_numeric: Flag to force conversion of values to numeric types.
        sort: Flag to indicate whether to perform sorting.

    Returns:
        The minimum and maximum values of the sorted column.

    Examples:
        data = [3, 1, 2]
        min_val, max_val = sort_col_numpy(data)
        This will sort the data and compute the minimum and maximum values.
    """
    if unsorted_data is None:
        return None, None

    sorted_data = unsorted_data

    if force_numeric:
        sorted_data = any_to_numeric(sorted_data)

    if ignore_None_and_string:
        sorted_data = [val for val in sorted_data if val is not None or isinstance(val, str)]

    if sort:
        sorted_data = np.sort(unsorted_data)

    length = len(sorted_data)
    min_val = sorted_data[0]
    max_val = sorted_data[-1]

    if freq is None or freq == 0.:
        return min_val, max_val
    else:
        if isinstance(freq, list) or isinstance(freq, tuple):
            if len(freq) == 1:
                lower = freq[0]
                upper = freq[0]
            else:
                lower = freq[0]
                upper = freq[1]
        else:
            lower = freq
            upper = freq

    if lower > 0.:
        idx = round(lower * length)
        try:
            min_val = sorted_data[idx]
        except:
            pass

    if upper > 0.:
        idx = round(upper * length)
        try:
            max_val = sorted_data[-idx]
        except:
            pass

    return min_val, max_val



def update_db_properties_using_image_properties(input_file, ta_path):
    """
    Update the database properties using the image properties.

    Args:
        input_file (str): The path to the input image file.
        ta_path (str): The path to the database directory.

    Raises:
        Exception: If an error occurs while reading properties from the image or writing them to the database.
    """
    try:
        tmp = Img(input_file)  # Create an Img object from the input image file
        if has_metadata(tmp):  # Check if the image has metadata
            db_path = os.path.join(ta_path, 'pyTA.db')  # Construct the path to the database file
            db = TAsql(db_path)  # Create a TAsql object with the database path
            try:
                # Initialize variables for storing the image properties
                voxel_size_x = None
                voxel_size_y = None
                voxel_size_z = None
                voxel_z_over_x_ratio = None
                time = None
                creation_time = None

                # Extract the image properties from the metadata
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

                # Create a dictionary with the image properties
                neo_data = {'voxel_size_x': voxel_size_x,
                            'voxel_size_y': voxel_size_y,
                            'voxel_size_z': voxel_size_z,
                            'voxel_z_over_x_ratio': voxel_z_over_x_ratio,
                            'time': time,
                            'creation_time': creation_time}

                if db.exists('properties'):
                    # The 'properties' table exists in the database, update the existing data
                    header, cols = db.run_SQL_command_and_get_results('SELECT * FROM properties', return_header=True)
                    cols = cols[0]
                    data = _to_dict(header, cols)  # Convert the retrieved data to a dictionary
                    for key in list(neo_data.keys()):
                        if neo_data[key] is None:
                            if key not in data:
                                data[key] = neo_data[key]
                        else:
                            data[str(key)] = neo_data[key]
                else:
                    # The 'properties' table does not exist, use the new data as it is
                    data = neo_data

                data = {k: [v] for k, v in data.items()}  # Convert the data dictionary to a dictionary of lists
                db.create_and_append_table('properties', data)  # Create or append the 'properties' table with the data
            except:
                traceback.print_exc()
                print('An error occurred while reading properties from the image or writing them to the database')
            finally:
                try:
                    db.close()  # Close the database connection
                except:
                    pass
        del tmp  # Delete the temporary Img object
    except:
        traceback.print_exc()
        print('Error could not save image properties to the TA database')

def _to_dict(header, col):
    """
    Convert the header and column data into a dictionary.

    Args:
        header (list): The header containing the column names.
        col (list): The column data.

    Returns:
        dict: A dictionary where the column names are the keys and the column data are the values.
    """
    dct = {}
    for iii, he in enumerate(header):
        dct[he] = col[iii]  # Map the column name to its corresponding data value
    return dct


def _list_to_string(lst, add_quotes=True):
    """
    Convert a list or dictionary to a string representation.

    Args:
        lst (list or dict): The list or dictionary to be converted.
        add_quotes (bool, optional): Indicates whether quotes should be added around string values (default: True).

    Returns:
        str: The string representation of the list or dictionary.

    Examples:
        >>> lst = [1, 2, 3]
        >>> _list_to_string(lst)
        "'1', '2', '3'"

        >>> lst = ['apple', 'banana', 'cherry']
        >>> _list_to_string(lst)
        "'apple', 'banana', 'cherry'"

        >>> dct = {'Name': 'John', 'Age': 25, 'Country': 'USA'}
        >>> _list_to_string(dct)
        'Name John, Age 25, Country USA'
    """
    if isinstance(lst, list):
        return ', '.join(
            map(
                lambda x: "'" + str(x) + "'" if x is not None and add_quotes else str(x) if x is not None and not add_quotes else 'Null',
                lst
            )
        )
    elif isinstance(lst, dict):
        return ', '.join(
            (str(x) + ' ' + str(y)) if y is not None else str(x) + ' ' + 'Null' for x, y in lst.items()
        )
    else:
        return list

# TODO get master db --> TODO --> required for global plots --> easy way of doing things in fact, I love it

# maybe allow unknown
def get_type(value):
    """
    Determine the data type of a value.

    Args:
        value (any): The value to determine the data type.

    Returns:
        str: The data type of the value.

    Examples:
        >>> get_type(42)
        'INTEGER'

        >>> get_type(3.14)
        'FLOAT'

        >>> get_type('Hello, world!')
        'TEXT'

        >>> get_type(True)
        'BOOLEAN'
    """
    if isinstance(value, bool):
        return 'BOOLEAN'
    if isinstance(value, str):
        return 'TEXT'
    if isinstance(value, int):
        return 'INTEGER'
    if isinstance(value, float):
        return 'FLOAT'
    if isinstance(value, list):
        return 'TEXT'
    if isinstance(value, np.int64) or isinstance(value, np.uint64) \
            or isinstance(value, np.int32) or isinstance(value, np.uint32) \
            or isinstance(value, np.int8) or isinstance(value, np.uint8):
        return 'INTEGER'
    if isinstance(value, Img):
        return 'INTEGER'
    if isinstance(value, tuple) or isinstance(value, np.ndarray):
        return 'TEXT'

    print('error type not supported:', type(value), value)
    return 'TEXT'


def get_types_from_data(one_row_of_data):
    """
    Get the data types from a row of data.

    Args:
        one_row_of_data (dict or list): A row of data represented as a dictionary or list.

    Returns:
        list: A list of data types corresponding to the values in the row of data.

    Examples:
        >>> get_types_from_data({'name': 'John', 'age': 30, 'height': 180.5})
        ['TEXT', 'INTEGER', 'FLOAT']

    """
    types = []
    if isinstance(one_row_of_data, dict):
        for k, v in one_row_of_data.items():
            if v is None:
                types.append('TEXT')
                continue

            if not isinstance(v, list):
                types.append(get_type(v))
            else:
                success = False
                for vv in v:
                    if vv is not None:
                        types.append(get_type(vv))
                        success = True
                        break
                if not success:
                    types.append('TEXT')
    else:
        for data in one_row_of_data:
            types.append(get_type(data))

    return types


def _to_lower(lst):
    """
    Convert a list of elements to lowercase strings.

    Args:
        lst (list): A list of elements.

    Returns:
        list: A new list with elements converted to lowercase strings.

    Examples:
        >>> _to_lower(['Apple', 'Banana', 'Orange'])
        ['apple', 'banana', 'orange']

        >>> _to_lower(['Hello', 'WORLD'])
        ['hello', 'world']
    """
    return [str(elm).lower() for elm in lst]


# NB this will fail when some command returns for some of the samples a null output --> think if there can be a workaround
# MEGA TODO also I should always return the header and make sure it's identical to the previous one, because if it's not the case the aggregation is not possible --> then maybe set the row to None and later replace it with the appropriate nb of Null for example !!! --> think about that
def combine_single_file_queries(lst, sql_command, table_names=None, db_name='pyTA.db', return_header=False, prepend_frame_nb=True, prepend_file_name=True, output_filename=None):
    """
    Combine the results of single file queries into a master data set.

    Args:
        lst (list): List of file paths to query.
        sql_command (str): SQL command to execute for each file.
        table_names (str or list, optional): Names of tables to include in the query. Defaults to None.
        db_name (str, optional): Name of the database file. Defaults to 'pyTA.db'.
        return_header (bool, optional): Flag to indicate whether to return the header. Defaults to False.
        prepend_frame_nb (bool, optional): Flag to indicate whether to prepend the frame number to the data. Defaults to True.
        prepend_file_name (bool, optional): Flag to indicate whether to prepend the file name to the data. Defaults to True.
        output_filename (str, optional): Name of the output file to save the combined data. Defaults to None.

    Returns:
        tuple or list: If `output_filename` is None and `return_header` is False, returns the combined data as a list.
                      If `output_filename` is None and `return_header` is True, returns the header and combined data as a tuple.
                      If `output_filename` is provided, saves the combined data to the specified file and returns None.

    # Examples:
    #     >>> lst = ['file1.db', 'file2.db', 'file3.db']
    #     >>> sql_command = 'SELECT * FROM data'
    #     >>> combine_single_file_queries(lst, sql_command, table_names='my_table', output_filename='combined_data.csv')
    #
    #     >>> lst = ['file1.db', 'file2.db', 'file3.db']
    #     >>> sql_command = 'SELECT * FROM data WHERE value > 10'
    #     >>> header, data = combine_single_file_queries(lst, sql_command, return_header=True)
    #     >>> print(header)
    #     >>> print(data)
    """

    if lst is not None and lst and isinstance(lst, list):
        # merged output
        master_data = []
        header = None
        for iii, file in enumerate(lst):
            # Parse the database file path
            db_path = smart_name_parser(file, db_name)
            # Create a TAsql instance for the database
            db = TAsql(db_path)
            try:
                final_command = sql_command

                if table_names is not None:
                    if isinstance(table_names, list):
                        # Iterate over the table names and check if they exist in the database
                        for table_name in table_names:
                            if db.exists(table_name):
                                final_command += table_name
                                break
                    else:
                        final_command += table_names

                # Execute the SQL command and get the results
                out = db.run_SQL_command_and_get_results(final_command, return_header=iii == 0)
                if isinstance(out, tuple):
                    header, data = out
                    if prepend_file_name:
                        header = prepend_to_content(header, 'filename')
                    if prepend_frame_nb:
                        header = prepend_to_content(header, 'frame_nb')
                else:
                    data = out
                if prepend_file_name:
                    data = prepend_to_content(data, file)
                if prepend_frame_nb:
                    data = prepend_to_content(data, iii)

                # Append the data to the master data list
                master_data.extend(data)
            except:
                traceback.print_exc()
            finally:
                # Close the database connection
                db.close()

        if not master_data:
            master_data = None
        if output_filename is not None:
            # Save the combined data to the output file
            save_data_to_csv(output_filename, header, master_data)
        else:
            if return_header:
                return header, master_data
            return master_data
    else:
        logger.error('No input list -> nothing to do...')
        if return_header:
            return None, None
        return None


# TODO KEEP MEGA URGENT do force_track_cells_db_update for bonds and vertices when the tracking will be implemented.
# TODO --> find a way to handle clones here
def createMasterDB(lst, outputName=None, progress_callback=None, database_name=None, force_track_cells_db_update=False, db_name='pyTA.db'):
    """
    Creates a master database by combining data from multiple database files.

    Args:
        lst (list): List of input filenames or database file paths.
        outputName (str, optional): Name of the output database file. If provided, the existing file will be removed
                                    before creating a new one. Defaults to None.
        progress_callback (function, optional): Callback function to track the progress of the database creation.
                                                Defaults to None.
        database_name (str, optional): Name of a specific database to include. Only the tables from this database will
                                       be added to the master database. Defaults to None.
        force_track_cells_db_update (bool, optional): If True, forces an update to the 'cell_tracks' table in the
                                                     database. Defaults to False.
        db_name (str, optional): Name of the database. Defaults to 'pyTA.db'.

    Returns:
        TAsql: The master database object.

    Examples:
        lst = ['db1.db', 'db2.db', 'db3.db']
        createMasterDB(lst, outputName='master.db')

        This example will create a master database named 'master.db' by combining the tables from 'db1.db', 'db2.db',
        and 'db3.db'. The existing 'master.db' file, if any, will be removed before creating the new database.
    """
    masterDB = None
    try:
        frame_nb = "frame_nb"
        fileName = "filename"

        # Create or remove the output file if specified
        if outputName is not None:
            try:
                os.remove(outputName)
            except OSError:
                pass

        # Generate a list of database filenames
        database_list = smart_TA_list(lst, db_name)

        # Create the masterDB object
        masterDB = TAsql(filename_or_connection=outputName, add_useful_missing_SQL_commands=False)

        # Iterate over each database file
        if database_list is not None:
            for l, db_l in enumerate(database_list):
                try:
                    # Check if the process should be stopped
                    if early_stop.stop:
                        return

                    # Update progress if a callback function is provided
                    if progress_callback is not None:
                        progress_callback.emit(int((l / len(database_list)) * 100))
                    else:
                        logger.info(str((l / len(database_list)) * 100) + '%')
                except:
                    pass

                # Open the database file
                dbHandler = None
                try:
                    dbHandler = TAsql(filename_or_connection=db_l)

                    # Get all tables in the database
                    tables = dbHandler.get_tables()

                    if force_track_cells_db_update:
                        force_update = False

                        # Check if 'cell_tracks' table is missing or empty
                        if 'cell_tracks' not in tables or dbHandler.isTableEmpty('cell_tracks'):
                            force_update = True

                        if not force_update:
                            try:
                                # Check if 'track_id' column is empty
                                data = dbHandler.run_SQL_command_and_get_results('SELECT track_id FROM cell_tracks LIMIT 1')
                                if data == None or data[0][0] == None:
                                    force_update = True
                            except:
                                force_update = True

                            if force_update:
                                # Drop 'cell_tracks' table and create a new one
                                dbHandler.drop_table('cell_tracks')
                                dbHandler.execute_command('CREATE TABLE cell_tracks AS SELECT local_id as local_id, CAST(NULL AS INTEGER) AS track_id FROM cells_2D')

                    masterDBTables = masterDB.get_tables()

                    # Iterate over each table in the database
                    for string in tables:
                        if database_name is not None:
                            if database_name.lower() != string.lower():
                                continue
                        if string not in masterDBTables:  # (!masterDBTables.contains(string)) {
                            tableHeaderAndType = dbHandler.get_table_column_names_and_types(string)
                            extra = {}
                            extra[frame_nb] = 'INTEGER'
                            extra[fileName] = 'TEXT'

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

                        fixed_cols = set(_to_lower(cols.keys()))
                        fixed_cols.add(frame_nb)
                        fixed_cols.add(fileName)

                        fixed_master = set(_to_lower(cols_master.keys()))

                        # make it case insensitive
                        cols_master = {k.lower(): v for k, v in
                                       cols_master.items()}  # TODO maybe replace by a case insensitive dict
                        cols = {k.lower(): v for k, v in cols.items()}

                        if not fixed_cols == fixed_master:
                            present_in_master_but_missing_in_cur = fixed_master - fixed_cols
                            present_in_cur_but_missing_in_master = fixed_cols - fixed_master

                            if present_in_master_but_missing_in_cur:
                                # Create a temporary table for adding missing columns
                                masterDB.drop_table('pytaTMP')
                                masterDB.execute_command('CREATE TABLE pytaTMP AS SELECT * from ' + str(DB_to_read))

                                for col in present_in_master_but_missing_in_cur:
                                    # Add missing columns to the temporary table
                                    masterDB.add_column('pytaTMP', col, col_type=cols_master[col])

                                DB_to_read = 'pytaTMP'

                            for col in present_in_cur_but_missing_in_master:
                                # Add missing columns to the masterDB table
                                masterDB.add_column(string, col, col_type=cols[col])

                        name = smart_name_parser(lst[l], ordered_output=['short'])[0]

                        # Insert data from the database table into the masterDB table
                        masterDB.execute_command(
                            "INSERT INTO '" + str(string) + "' SELECT " + str(l) + " AS '" + str(
                                frame_nb) + "', '" + str(
                                name) + "' AS '" + str(fileName) + "', * FROM " + str(DB_to_read))

                        # Drop the temporary table
                        masterDB.drop_table('pytaTMP')

                except:
                    traceback.print_exc()
                finally:
                    if dbHandler is not None:
                        # Detach the database table and close the connection
                        masterDB.detach_table('tmp')
                        dbHandler.close()

    except:
        traceback.print_exc()

    finally:
        # print(outputName)

        # Clean up and close the masterDB if an outputName is provided
        if masterDB is not None and outputName is not None:
            masterDB.clean()
            masterDB.close()
        else:
            return masterDB


def set_property(db_file, neo_data):
    """
    Sets properties in a database table using data from Neo.

    Args:
        db_file (str): The path to the database file.
        neo_data (dict): The dictionary containing the Neo data.

    Returns:
        None

    Raises:
        Exception: If an error occurs while inserting new properties.

    # Examples:
    #     >>> db_file = 'data.db'
    #     >>> neo_data = {'key1': 'value1', 'key2': 'value2', 'key3': 'value3'}
    #     >>> set_property(db_file, neo_data)
    """
    if not neo_data:
        return

    try:
        db = TAsql(db_file)

        try:
            if db.exists('properties'):
                header, cols = db.run_SQL_command_and_get_results('SELECT * FROM properties', return_header=True)
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
    """
    Retrieves a property value from a database table.

    Args:
        db_file (str): The path to the database file.
        property_name (str): The name of the property to retrieve.

    Returns:
        str or None: The value of the property, or None if it does not exist or an error occurs.

    # Examples:
    #     >>> db_file = 'data.db'
    #     >>> property_name = 'key1'
    #     >>> result = get_property(db_file, property_name)
    #     >>> print(result)
    #     'value1'
    """
    val = None
    db = TAsql(db_file)

    try:
        val = db.run_SQL_command_and_get_results('SELECT ' + property_name + ' from properties')[0][0]

        if isinstance(val, str) and (val.lower() == 'none' or val.lower() == 'null' or val.strip() == ''):
            val = None
    except:
        # If anything goes wrong or the table or column does not exist, return None
        pass

    db.close()
    return val


def get_properties_master_db(lst):
    """
    Retrieves properties from multiple databases and creates a master database.

    Args:
        lst (list): The list of database names.

    Returns:
        TAsql: The master database containing the properties.

    # Examples:
    #     >>> lst = ['db1', 'db2', 'db3']
    #     >>> result = get_properties_master_db(lst)
    #     >>> print(result)
    #     <TAsql object at 0x...>
    """
    database_list = smart_TA_list(lst, 'pyTA.db')

    if database_list is not None:
        for db_file in database_list:
            db = TAsql(db_file)

            if not 'properties' in db.get_tables():
                db.create_table('properties', ['voxel_z_over_x_ratio', 'time'], ['float', 'float'])
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

    return database

def reinject_properties_to_TA_files(lst, master_db, indices_to_update=None):
    """
    Reinjects properties from the master database to individual TA files.

    Args:
        lst (list): The list of TA file names.
        master_db (TAsql): The master database containing the properties.
        indices_to_update (list or None): The indices of the TA files to update. If None, update all files.

    Returns:
        None

    # Examples:
    #     >>> lst = ['db1', 'db2', 'db3']
    #     >>> master_db = TAsql('master.db')
    #     >>> indices_to_update = [0, 2]
    #     >>> reinject_properties_to_TA_files(lst, master_db, indices_to_update)
    """
    database_list = smart_TA_list(lst, 'pyTA.db')

    for iii, db_file in enumerate(database_list):
        if indices_to_update is not None:
            if iii not in indices_to_update:
                continue

        master_db.attach_table(db_file, 'tmp')

        master_db.execute_command('DROP TABLE IF EXISTS tmp.properties')

        master_db.execute_command('CREATE TABLE tmp.properties AS SELECT ' +
                                  master_db.EXCEPT('properties', *['frame_nb', 'filename']) +
                                  ' FROM properties WHERE frame_nb =' + str(iii))

        master_db.execute_command('DETACH DATABASE tmp')

# TODO also I would need handle polarity smarty but keep it channels --> TODO --> treat all of them as a single piece of data

def populate_table_content(db_name, prepend='#', filtered_out_columns=['x', 'y', 'first_pixel', 'local_ID', 'pixel_within_cell', 'centroid', 'vx', 'cell_id', 'pixel_within_', 'perimeter_pixel_count', 'bond_cut_off', 'vertices', 'bonds', 'pixel_count']):
    """
    Populates the content of a table in a database.

    Args:
        db_name (str): The name of the database file.
        prepend (str): The string to prepend to each table and column name. Default is '#'.
        filtered_out_columns (list): The list of column names to filter out. Default is a list of column names.

    Returns:
        list: The list of table and column names.

    # Examples:
    #     >>> db_name = 'data.db'
    #     >>> prepend = '#'
    #     >>> filtered_out_columns = ['x', 'y', 'first_pixel', 'local_ID']
    #     >>> result = populate_table_content(db_name, prepend, filtered_out_columns)
    #     >>> print(result)
    #     ['#table1.column1', '#table1.column2', '#table2.column1', '#table2.column2']
    """
    if not os.path.isfile(db_name):
        return None

    table_content = []
    db = None

    try:
        db = TAsql(db_name)
        tables = db.get_tables()

        for table in tables:
            columns = db.get_table_column_names_and_types(table, return_colnames_only=True)
            cols_to_remove = []

            if filtered_out_columns is not None:
                for filter in filtered_out_columns:
                    for col in columns:
                        if col.startswith(filter) or col == filter:
                            cols_to_remove.append(col)

                    columns = [col for col in columns if col not in cols_to_remove]

            for col in columns:
                table_content.append(('' if prepend is None else str(prepend)) + str(table) + '.' + str(col))
    except:
        traceback.print_exc()
        logger.error('Error populating the database')
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

    if True:
        columns = ['id', 'fruit']
        data = np.array( [['1', 'apple'], ['2', 'banana'], ['3', 'cherry']])
        column_types = ['INTEGER', 'TEXT']
        create_table_and_append_data("mydb.db", "fruits", columns, data, column_types=column_types, temporary=False)
        sys.exit(0)


    if True:
        table_to_xlsx_with_sheets('/E/Sample_images/FISH/Sample_transcription_dot_detection/manue_manually_segmented_images/training_set/tests/full_real_analysis_tst1/Sarah/kniR-405_knrlR-488_kniD-565_kniD-633/Image 23_Stitch/FISH.db', '/E/Sample_images/FISH/Sample_transcription_dot_detection/manue_manually_segmented_images/training_set/tests/full_real_analysis_tst1/Sarah/kniR-405_knrlR-488_kniD-565_kniD-633/Image 23_Stitch/FISH.xlsx')


        sys.exit(0)

    if True:

        # print(os.path.exists('/E/Sample_images/FISH/Sample_transcription_dot_detection/manue_manually_segmented_images/training_set/tests/full_real_analysis_tst1/Benoit R1 R6/R1 565 R6 633 f W5 NS3/FISHcopie.db'))
        remove_dupes_from_table_and_overwrite_table('/E/Sample_images/FISH/Sample_transcription_dot_detection/manue_manually_segmented_images/training_set/tests/full_real_analysis_tst1/Benoit R1 R6/R1 565 R6 633 f W5 NS3/FISH (copie).db', 'human_curated_distances_3D')
        remove_dupes_from_table_and_overwrite_table('/E/Sample_images/FISH/Sample_transcription_dot_detection/manue_manually_segmented_images/training_set/tests/full_real_analysis_tst1/Benoit R1 R6/R1 565 R6 633 f W5 NS3/FISH (copie).db', 'human_curated_distances_3D_chromatic_aberrations_corrected')


        sys.exit(0)

    if False:
        print(table_exists_in_db('/E/Sample_images/FISH/Sample_transcription_dot_detection/manue_manually_segmented_images/training_set/tests/full_real_analysis_tst1/tmp/R1 565 R6 633 f  S1/FISH.db', 'human_curated_distances_3D'))

        print(table_exists_in_db('/E/Sample_images/FISH/Sample_transcription_dot_detection/manue_manually_segmented_images/training_set/tests/full_real_analysis_tst1/tmp/R1 565 R6 633 f  S1/FISH.db', 'breaking_bad'))

        sys.exit(0)

    # --> could create the master db and then even further filter it
    # can I also do ins by sets --> think about it

    # NB in is same as
    #SELECT * FROM employees WHERE employee_id = 1 OR employee_id = 2 OR employee_id = 3 OR employee_id = 4;
    # SELECT * FROM employees WHERE first_name NOT IN ('Sarah', 'Jessica'); # maybe useful too

    if False:
        db = TAsql('/E/Sample_images/sample_images_PA/mini_different_nb_of_channels/focused_Series012/pyTA.db')
        # print(db.create_filtered_query('SELECT * from cells_2D', filtering_elements=[1,2,120],filter_name='local_ID'))
        print(db.create_filtered_query('SELECT * from cells_2D', filtering_elements_dict={'local_ID':[1,2,120]}))
        print(db.create_filtered_query('SELECT * from cells_2D', filtering_elements_dict={'local_ID':[1,2,120], 'cytoplasmic_area':[802]}))
        db.close()

        sys.exit(0)

    if False:
        new_properties = {'reg_x':16, 'reg_y':-32}
        set_property(
            '/E/Sample_images/sample_images_PA/mini_different_nb_of_channels/focused_Series012/pyTA.db',
            new_properties)  # ça marche!!!
        print(get_property(
            '/E/Sample_images/sample_images_PA/mini_different_nb_of_channels/focused_Series012/pyTA.db',
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

        print(get_property('/E/Sample_images/sample_images_PA/mini_different_nb_of_channels/focused_Series012/pyTA.db', 'time')+1) # ça marche!!!
        print(get_property('/E/Sample_images/sample_images_PA/mini_different_nb_of_channels/focused_Series012/pyTA.db', 'voxel_z_over_x_ratio'))
        import sys
        sys.exit(0)


    if True:
        # do a real mount maybe ???

        # all seems ok and in many aspects simpler than the java equivalent...
        # sql_file = '/E/Sample_images/sample_images_PA/mini/test.db'
        # sql_file = '/run/user/1000/gvfs/smb-share:server=teamdfs2,share=teamprudhomme/EqpPrudHomme2/To be analyzed/221123 R1y R6y/R1 565 R6 633 f  S3/test.db' # test for sqlite on samba --> there is a bug on a samba drive --> how can I fix that
        sql_file = '/media/eqpPrudhomme/EqpPrudHomme2/To be analyzed/test.db' # FINALLY GOT IT TO WORK --> use the nobrl option in mount CIFS # test for sqlite on samba --> there is a bug on a samba drive --> how can I fix that # TO GET IT TO WORK I NEED THE 'NOBRL' command !!!
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

        import sys
        sys.exit(0)

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
        # masterDB = createMasterDB(['/E/Sample_images/sample_images_PA/mini/focused_Series012.png','/E/Sample_images/sample_images_PA/mini/focused_Series014.png','/E/Sample_images/sample_images_PA/mini/focused_Series015.png','/E/Sample_images/sample_images_PA/mini/focused_Series016.png'])
        # masterDB = createMasterDB(
        #     ['/E/Sample_images/sample_images_PA/mini_different_nb_of_channels/focused_Series012.png',
        #      '/E/Sample_images/sample_images_PA/mini_different_nb_of_channels/focused_Series014.png',
        #      '/E/Sample_images/sample_images_PA/mini_different_nb_of_channels/focused_Series015.png',
        #      '/E/Sample_images/sample_images_PA/mini_different_nb_of_channels/focused_Series016.png',
        #      '/E/Sample_images/sample_images_PA/mini_different_nb_of_channels/focused_Series019.tif'])

        # masterDB = createMasterDB(loadlist('/E/Sample_images/sample_images_PA/mini_different_nb_of_channels/list.lst'))


        outputName = None
        # outputName = '/E/Sample_images/sample_images_pyta/surface_projection/masterDB.db'
        masterDB = createMasterDB(loadlist('/E/Sample_images/sample_images_pyta/surface_projection/list.lst'), outputName=outputName, force_track_cells_db_update=True)


        if masterDB is not None:
            # masterDB = createMasterDB(['/E/Sample_images/sample_images_PA/mini/focused_Series012.png','/E/Sample_images/sample_images_PA/mini/focused_Series012.png','/E/Sample_images/sample_images_PA/mini/focused_Series012.png','/E/Sample_images/sample_images_PA/mini/focused_Series012.png'])
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
        sql_file = '/E/Sample_images/sample_images_PA/mini/focused_Series012/pyTA.db'

        print(populate_table_content(sql_file))

        table_name = 'cells_2D'

        db = TAsql(sql_file)  # maybe if file is none --> store in mem ???
        db.get_min_max(table_name, 'area')
        db.get_min_max(table_name, 'area', freq=[0.1, 0.1])
        print(db.get_table_column_names_and_types(table_name))  # what if two cols have same name ???
        db.clean()
        db.close()
