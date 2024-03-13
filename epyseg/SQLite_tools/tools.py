# see also pandas_tools/tools as there is some overlap between the two
import os.path
import sqlite3

def set_voxel_size(db_path, voxel_size):
    """
    Inserts a list or tuple of three floats into an SQLite database table named 'voxel_size'
    with columns 'vz', 'vy', and 'vx'.

    Args:
        db_path (str): The path to the SQLite database file.
        voxel_size (list or tuple): A list or tuple containing three floats.

    Returns:
        None.
    """

    # delete table if exists
    drop_table_if_exists(db_path,'voxel_size', verbose=False)

    if voxel_size is None:
        return

    # Connect to the SQLite database
    conn = sqlite3.connect(db_path)

    # Create a cursor object
    c = conn.cursor()

    # Insert the values into the 'voxel_size' table
    c.execute('CREATE TABLE voxel_size ("vz", "vy", "vx")')
    c.execute('INSERT INTO voxel_size (vz, vy, vx) VALUES (?, ?, ?)', voxel_size)

    # Commit the transaction
    conn.commit()

    # Close the cursor and database connection
    c.close()
    conn.close()

def get_voxel_size(db_path):
    """
        Retrieves the voxel size values from an SQLite database file.

        Args:
            db_path (str): The path to the SQLite database file.

        Returns:
            list: A list containing the voxel size values as floats.

        # Examples:
        #     If the 'voxel_size' table in the 'example.db' database file contains the values (1.0, 2.0, 3.0), then:
        #
        #     >>> get_voxel_size('/E/Sample_images/FISH/Sample_transcription_dot_detection/manue_manually_segmented_images/training_set/tests/full_real_analysis_tst1/Coloc X2 2048/230314 Coloc X2 ZH-2A m 2048_2023_03_14__17_43_51(1)/FISH.db')
        #     [1.0, 2.0, 3.0]
        #
        #     If the 'voxel_size' table does not exist in the database file:
        #
        #     >>> get_voxel_size('/E/Sample_images/FISH/Sample_transcription_dot_detection/manue_manually_segmented_images/training_set/tests/full_real_analysis_tst1/Coloc X2 2048/230314 Coloc X2 ZH-2A m 2048_2023_03_14__17_43_51(1)/FISH.db')
        #     None
        #
        #     If the database file does not exist:
        #
        #     >>> get_voxel_size('nonexistent.db')
        #     None
    """
    if not os.path.exists(db_path):
        return None

    # Connect to the SQLite database
    conn = sqlite3.connect(db_path)

    # Create a cursor object
    c = conn.cursor()

    # Select the row from the 'voxel_size' table
    try:
        c.execute('SELECT * FROM voxel_size')
        row = c.fetchone()

        # Convert the row to a list
        voxel_size_list = list(row)
    except:
        # most likely table does not exist --> return None
        voxel_size_list = None

    # Close the cursor and database connection
    c.close()
    conn.close()

    return voxel_size_list


def drop_table_if_exists(db_path, table_name, verbose=True):
    """
    Drops an SQLite table if it exists in the specified database file.

    Args:
        db_path (str): The path to the SQLite database file.
        table_name (str): The name of the table to drop.

    Returns:
        None.
    """

    if not os.path.exists(db_path):
        if verbose:
            print('sqlite file does not exist -−> skipping')
        return

    # Connect to the SQLite database
    conn = sqlite3.connect(db_path)

    # Create a cursor object
    c = conn.cursor()

    # Check if the table exists
    c.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?", (table_name,))
    result = c.fetchone()

    if result:
        # Drop the table if it exists
        c.execute(f"DROP TABLE {table_name}")

    # Commit the transaction
    conn.commit()

    # Close the cursor and database connection
    c.close()
    conn.close()

def get_column_from_sqlite_table(db_path, table_name, column_identifier):
    """
    Retrieves a column from an SQLite table by name or position.

    Args:
        db_path (str): The path to the SQLite database file.
        table_name (str): The name of the table.
        column_identifier (str or int): The column name or position (starting from 0).

    Returns:
        list: The column data as a list.

    Raises:
        ValueError: If the column_identifier is invalid (neither a string nor an integer).

    """
    # NOT TESTED WITH INDEX −−> TODO especially with negative indices

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Fetch the column data by name or position
    if isinstance(column_identifier, str):
        cursor.execute(f"SELECT {column_identifier} FROM {table_name}")
    elif isinstance(column_identifier, int):
        cursor.execute(f"PRAGMA table_info({table_name})")
        columns = cursor.fetchall()
        column_name = columns[column_identifier][1]
        cursor.execute(f"SELECT {column_name} FROM {table_name}")
    else:
        conn.close()
        raise ValueError("Invalid column_identifier. Must be either a string (column name) or an integer (position).")

    column_data = cursor.fetchall()
    column_data = [row[0] for row in column_data]

    conn.close()

    return column_data


def get_tables(path_to_db):
    """
    Retrieves the names of all tables in a SQLite database.

    Args:
        path_to_db (str): The path to the SQLite database file.

    Returns:
        list: A list of table names in the database.

    # Examples:
    #     >>> get_tables('/media/teamPrudhomme/EqpPrudhomme2/Benoit_pr_Benjamin/coccinelles/latest images_20230614/raw images/N4N4_29_M_1a/ladybug_seg.db')[:2]
    #     ['elytras_shape', 'elytras_without_spots_gray']
    """
    # Establish a connection to the SQLite database
    conn = sqlite3.connect(path_to_db)

    # Create a cursor object to execute SQL queries
    cursor = conn.cursor()

    # Execute a SQL query to retrieve the names of all tables in the database
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")

    # Fetch all the rows returned by the query and store them in the 'tables' variable
    tables = cursor.fetchall()

    # Close the database connection
    conn.close()

    # Extract the table names from the fetched rows and return them as a list
    return [table[0] for table in tables]


def get_table_columns(path_to_db, table_name, force_lower_case=False):
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

    # connect to the database
    conn = sqlite3.connect(path_to_db)

    # create a cursor object
    cur = conn.cursor()

    # execute a query to get the column names
    cur.execute("PRAGMA table_info({})".format(table_name))

    # fetch the results and print the column names
    cols = []
    results = cur.fetchall()
    for row in results:
        # print(row)
        # print('TADA', row[1].strip("'"))
        cols.append(row[1].strip("'"))

    if force_lower_case:
        if cols:
            cols = [col.lower() for col in cols]

    # close the cursor and connection
    cur.close()
    conn.close()
    return cols


'''
to be added
def set_synchronous(sync):
    """
    Sets the synchronous setting for a SQLite database connection.
    """
    # connect to the database
    conn = sqlite3.connect('my_database.db')

    # create a cursor object
    cur = conn.cursor()

    # execute the PRAGMA statement to set the synchronous setting
    cur.execute("PRAGMA synchronous = {};".format("NORMAL" if sync else "OFF"))

    # commit the changes and close the cursor and connection
    conn.commit()
    cur.close()
    conn.close()

def set_memory(sync):
    """
    Sets the journal mode of a SQLite database connection to either MEMORY or OFF.
    """
    # connect to the database
    conn = sqlite3.connect('my_database.db')

    # create a cursor object
    cur = conn.cursor()

    # execute the PRAGMA statement to set the journal mode
    cur.execute("PRAGMA journal_mode = {};".format("MEMORY" if sync else "OFF"))

    # commit the changes and close the cursor and connection
    conn.commit()
    cur.close()
    conn.close()

def update_table_by_hidden_index(table_name, row_index, data):
    """
    Updates a row in a SQLite database table using a hidden index.
    """
    try:
        # connect to the database
        conn = sqlite3.connect('my_database.db')

        # create a cursor object
        cur = conn.cursor()

        # check if table exists
        if not contains_table(table_name):
            print(f"Table {table_name} does not exist.")
            return False

        # get the columns of the table
        cols = get_columns(table_name)

        # check if the row index is valid
        if row_index < 1:
            print("HIDDEN_TABLE_INDEX < 1 --> error")
            return False

        # build the SQL command to update the table
        sql_command = f"UPDATE \"{table_name}\" SET "
        first = True
        for col_name, val in data.items():
            if col_name.lower() not in cols:
                print(f"Column {col_name} does not exist --> skipping")
                continue
            val_str = "NULL" if val is None else str(val)
            if first:
                sql_command += f"\"{col_name}\" = \"{val_str}\""
                first = False
            else:
                sql_command += f", \"{col_name}\" = \"{val_str}\""
        sql_command += f" WHERE HIDDEN_TABLE_INDEX = {row_index};"

        # execute the SQL command and commit the changes
        cur.execute(sql_command)
        conn.commit()

        # close the cursor and connection
        cur.close()
        conn.close()
    except Exception as e:
        print(f"Error updating table: {e}")
        return False

    return True



def update_table_by_row_index(table_name, row_index, data):
    """
    Updates a row in a SQLite database table using a row index.
    """
    try:
        # connect to the database
        conn = sqlite3.connect('my_database.db')

        # create a cursor object
        cur = conn.cursor()

        # check if table exists
        if not contains_table(table_name):
            print(f"Table {table_name} does not exist.")
            return False

        # get the columns of the table
        cols = get_columns(table_name)

        # check if the row index is valid
        if row_index < 1:
            print("row idx < 1 --> error")
            return False

        # build the SQL command to update the table
        sql_command = f"UPDATE \"{table_name}\" SET "
        first = True
        for col_name, val in data.items():
            if col_name.lower() not in cols:
                print(f"Column {col_name} does not exist --> skipping")
                continue
            val_str = "NULL" if val is None else str(val)
            if first:
                sql_command += f"\"{col_name}\" = \"{val_str}\""
                first = False
            else:
                sql_command += f", \"{col_name}\" = \"{val_str}\""
        sql_command += f" WHERE ROWID = {row_index};"

        # execute the SQL command and commit the changes
        cur.execute(sql_command)
        conn.commit()

        # close the cursor and connection
        cur.close()
        conn.close()
    except Exception as e:
        print(f"Error updating table: {e}")
        return False

    return True

def update_column(table_name, column_name, col_data):
    """
    Updates a column in a SQLite database table.
    """
    try:
        # connect to the database
        conn = sqlite3.connect('my_database.db')

        # create a cursor object
        cur = conn.cursor()

        # check if table exists
        if not contains_table(table_name):
            print(f"Table {table_name} does not exist.")
            return

        # build the SQL command to update the column
        count = 1
        for val in col_data:
            val_str = "NULL" if val is None else str(val)
            sql_command = f"UPDATE \"{table_name}\" SET \"{column_name}\" = \"{val_str}\" WHERE ROWID = {count};"
            cur.execute(sql_command)
            count += 1

        # commit the changes and close the cursor and connection
        conn.commit()
        cur.close()
        conn.close()
    except Exception as e:
        print(f"Error updating column: {e}")
        return

def append_to_existing_table_or_create_it_if_it_does_not_exist(table_name, data):
    """
    Appends data to an existing table in a SQLite database or creates the table if it does not exist.
    """
    try:
        # connect to the database
        conn = sqlite3.connect('my_database.db')

        # create a cursor object
        cur = conn.cursor()

        # check if table exists
        if not contains_table(table_name):
            # create the table with columns based on the keys of the data dictionary
            headers = generate_headers_with_type_from_key_set(data, ",")
            create_table(table_name, headers)
        else:
            # check if all columns exist, create them if necessary
            for key, val in data.items():
                if not col_exists(table_name, key):
                    data_type = get_data_type(val)
                    add_column(table_name, key, data_type)

        # add the data to the table
        add_to_table_fast_object(table_name, data)

        # commit the changes and close the cursor and connection
        conn.commit()
        cur.close()
        conn.close()
    except Exception as e:
        print(f"Error appending to table: {e}")
        return

def rename_table(old_table_name, new_table_name):
    """
    Renames a table in a SQLite database.
    """
    try:
        # connect to the database
        conn = sqlite3.connect('my_database.db')

        # create a cursor object
        cur = conn.cursor()

        # check if new table name already exists
        if contains_table(new_table_name):
            print(f"Table {new_table_name} already exists.")
            return

        # rename the table
        sql_query = f"ALTER TABLE \"{old_table_name}\" RENAME TO \"{new_table_name}\";"
        cur.execute(sql_query)

        # commit the changes and close the cursor and connection
        conn.commit()
        cur.close()
        conn.close()
    except Exception as e:
        print(f"Error renaming table: {e}")
        return        

def add_row(table_name):
    """
    Adds an empty row to a table in a SQLite database.
    """
    try:
        # connect to the database
        conn = sqlite3.connect('my_database.db')

        # create a cursor object
        cur = conn.cursor()

        # add the empty row to the table
        sql_query = f"INSERT INTO \"{table_name}\" DEFAULT VALUES;"
        cur.execute(sql_query)

        # commit the changes and close the cursor and connection
        conn.commit()
        cur.close()
        conn.close()
    except Exception as e:
        print(f"Error adding row: {e}")
        return


def add_column(table_name, new_col_name, data_type=None):
    """
    Adds a new column to a table in a SQLite database.
    """
    try:
        # connect to the database
        conn = sqlite3.connect('my_database.db')

        # create a cursor object
        cur = conn.cursor()

        # check if column already exists
        if col_exists(table_name, new_col_name):
            print(f"Column {new_col_name} already exists in table {table_name}.")
            return

        # add the new column to the table
        if data_type is not None:
            sql_query = f"ALTER TABLE \"{table_name}\" ADD COLUMN \"{new_col_name}\" {data_type};"
        else:
            sql_query = f"ALTER TABLE \"{table_name}\" ADD COLUMN \"{new_col_name}\";"
        cur.execute(sql_query)

        # commit the changes and close the cursor and connection
        conn.commit()
        cur.close()
        conn.close()
    except Exception as e:
        print(f"Error adding column: {e}")
        return


def swap_2_rows(table_name, row1, row2):
    """
    Swaps the positions of two rows in a table in a SQLite database.
    """
    try:
        # connect to the database
        conn = sqlite3.connect('my_database.db')

        # create a cursor object
        cur = conn.cursor()

        # swap the positions of the rows
        sql_query = f"UPDATE \"{table_name}\" SET HIDDEN_TABLE_INDEX = \"tmp\" WHERE HIDDEN_TABLE_INDEX = {row1};"
        sql_query += f"UPDATE \"{table_name}\" SET HIDDEN_TABLE_INDEX = {row1} WHERE HIDDEN_TABLE_INDEX = {row2};"
        sql_query += f"UPDATE \"{table_name}\" SET HIDDEN_TABLE_INDEX = {row2} WHERE HIDDEN_TABLE_INDEX = \"tmp\";"
        cur.execute(sql_query)

        # commit the changes and close the cursor and connection
        conn.commit()
        cur.close()
        conn.close()
    except Exception as e:
        print(f"Error swapping rows: {e}")
        return


def drop_table(table_name):
    """
    Drops a table from a SQLite database.
    """
    try:
        # connect to the database
        conn = sqlite3.connect('my_database.db')

        # create a cursor object
        cur = conn.cursor()

        # drop the table
        sql_query = f"DROP TABLE IF EXISTS \"{table_name}\";"
        cur.execute(sql_query)

        # commit the changes and close the cursor and connection
        conn.commit()
        cur.close()
        conn.close()
    except Exception as e:
        print(f"Error dropping table: {e}")
        return


def replace_nulls_with_SQL_NULL():
    """
    Replaces all occurrences of the string 'null' with the SQL NULL value in all columns of all tables in a SQLite database.
    """
    try:
        # connect to the database
        conn = sqlite3.connect('my_database.db')

        # create a cursor object
        cur = conn.cursor()

        # replace nulls with SQL NULL in all columns of all tables
        tables = get_tables()
        for table in tables:
            columns = get_columns(table)
            for column in columns:
                sql_query = f"UPDATE \"{table}\" SET \"{column}\" = REPLACE(\"{column}\", 'null', NULL) WHERE \"{column}\" = 'null';"
                cur.execute(sql_query)

        # commit the changes and close the cursor and connection
        conn.commit()
        cur.close()
        conn.close()
    except Exception as e:
        print(f"Error replacing nulls: {e}")
        return


def replace_true_with_SQL_TRUE():
    """
    Replaces all occurrences of the string 'true' with the SQL TRUE value in all columns of all tables in a SQLite database.
    """
    try:
        # connect to the database
        conn = sqlite3.connect('my_database.db')

        # create a cursor object
        cur = conn.cursor()

        # replace trues with SQL TRUE in all columns of all tables
        tables = get_tables()
    for table in tables:
        columns = get_columns(table)
        for column in columns:
            sql_query = f"UPDATE \"{table}\" SET \"{column}\" = REPLACE(\"{column}\", 'true', TRUE) WHERE \"{column}\" = 'true';"
            cur.execute(sql_query)

        # commit the changes and close the cursor and connection
    conn.commit()
    cur.close()
    conn.close()

except Exception as e:
print(f"Error replacing trues: {e}")
return


def replace_NaN_with_SQL_NULL():
    """
    Replaces all occurrences of the string 'NaN' with the SQL NULL value in all columns of all tables in a SQLite database.
    """
    try:
        # connect to the database
        conn = sqlite3.connect('my_database.db')

        # create a cursor object
        cur = conn.cursor()

        # replace NaNs with SQL NULL in all columns of all tables
        tables = get_tables()
        for table in tables:
            columns = get_columns(table)
            for column in columns:
                sql_query = f"UPDATE \"{table}\" SET \"{column}\" = REPLACE(\"{column}\", 'NaN', NULL) WHERE \"{column}\" = 'NaN';"
                cur.execute(sql_query)

        # commit the changes and close the cursor and connection
        conn.commit()
        cur.close()
        conn.close()
    except Exception as e:
        print(f"Error replacing NaNs: {e}")
        return


def replace_string_in_column(column_name, string_to_replace, replacement_val):
    """
    Replaces all occurrences of a string in a column of all tables in a SQLite database.
    """
    try:
        # connect to the database
        conn = sqlite3.connect('my_database.db')

        # create a cursor object
        cur = conn.cursor()

        # replace the string in the column of all tables
        tables = get_tables()
        for table in tables:
            columns = get_columns(table)
            if column_name not in columns:
                continue
            sql_query = f"UPDATE \"{table}\" SET \"{column_name}\" = REPLACE(\"{column_name}\", '{string_to_replace}', '{replacement_val}') WHERE \"{column_name}\" LIKE '%{string_to_replace}%';"
            cur.execute(sql_query)

        # commit the changes and close the cursor and connection
        conn.commit()
        cur.close()
        conn.close()
    except Exception as e:
        print(f"Error replacing string: {e}")
        return

def mergeSeveralDatabases(list_of_databases, force_matching_of_partially_overlapping_databases, *tables2merge):
    for string in list_of_databases:
        attach_db = f"ATTACH DATABASE {string} AS tmp"
        executeCommand(attach_db)
        for string1 in tables2merge:
            if containsTable(string1):
                insert_all = f"INSERT INTO {string1} SELECT * FROM tmp.{string1}"
                executeCommand(insert_all)
            else:
                insert_all = f"CREATE TABLE {string1} AS SELECT * FROM tmp.{string1}"
                executeCommand(insert_all)
        detach_db = "DETACH DATABASE tmp"
        executeCommand(detach_db)

def SELECT_ALL(table_name):
    return f"SELECT * FROM TABLE {table_name}"

def SELECT_COLUMNS(table_name, *columns_to_select):
    columns = createSelection(*columns_to_select)
    return f"SELECT {columns} FROM TABLE {table_name}"

def SELECT_COLUMN(table_name, columns_to_select):
    return f"SELECT {columns_to_select} FROM TABLE {table_name}"

def CREATE_TABLE_AS_SELECT(newTableName, SQLCommandUsedForCreation):
    command = f"CREATE TABLE '{newTableName}' AS {SQLCommandUsedForCreation}"
    if not SQLCommandUsedForCreation.strip().endswith(";"):
        command += ";"
    return command

def RENAME_TABLE(orig_table_name, new_table_name):
    return f"ALTER TABLE '{orig_table_name}' RENAME TO '{new_table_name}';"

def createSelection(*cols):
    out = ""
    for string in cols:
        out += string + ","
    if out.endswith(","):
        out = out[:-1]
    return out

def createSelection(table_name, *cols):
    out = ""
    for string in cols:
        out += f"{table_name}.{string},"
    if out.endswith(","):
        out = out[:-1]
    return out


def SELECT_ALL_EXCEPT(table_name, *columns_to_exclude):
    names = [str(column).strip() for column in columns_to_exclude]
    cols = EXCEPT(table_name, names)
    return SELECT_COLUMN(table_name, cols)


def EXCEPT(table_name, *columns_to_exclude):
    return EXCEPT(table_name, True, columns_to_exclude)


def getAllColumnsAsASingleString(table_name):
    col_name = getColumns(table_name)
    out = ""
    for col_name1 in col_name:
        string = col_name1.strip()
        out += f" {table_name}.{string},"
    if out.endswith(","):
        out = out[:-1]
    return out


def SELECT_ALL_EXCEPT(table_name, columns_to_exclude):
    return SELECT_ALL_EXCEPT(table_name, *columns_to_exclude)


def NODUPES(table_name1, table_name2):
    col_name1 = getColumns(table_name1)
    col_name2 = getColumns(table_name2)
    col_name2 = list(set(col_name2) - set(col_name1))
    columns = ""
    for cur_name in col_name1:
        columns += f" {table_name1}.{cur_name},"
    for cur_name in col_name2:
        columns += f" {table_name2}.{cur_name},"
    if columns.endswith(","):
        columns = columns[:-1]
    columns = f"SELECT {columns} FROM TABLE {table_name1},{table_name2}"
    return columns


def palette2DB(table_name, palette):
    if len(palette) > 260:
        palette_fused = PaletteCreator().fused_RGB_palette(palette)
    else:
        palette_fused = palette
    values = []
    header = ["gray_value", "corresponding_RGB_color", "R_only", "G_only", "B_only"]
    values.append(header)
    for i in range(len(palette_fused)):
        row = []
        RGB = palette_fused[i]
        red = (RGB >> 16) & 0xFF
        green = (RGB >> 8) & 0xFF
        blue = RGB & 0xFF
        row.append(i)
        row.append(RGB)
        row.append(red)
        row.append(green)
        row.append(blue)
        values.append(row)
    arrayListObjects2SQL(table_name, values)

'''


if __name__ == '__main__':

    if True:
        set_voxel_size('/E/Sample_images/FISH/Sample_transcription_dot_detection/manue_manually_segmented_images/training_set/tests/full_real_analysis_tst1/Coloc X2 2048/230314 Coloc X2 ZH-2A m 2048_2023_03_14__17_43_51(1)/FISH.db', (14,15,16))
        import sys
        sys.exit(0)

    if True:
        voxel_size =get_voxel_size(
            '/E/Sample_images/FISH/Sample_transcription_dot_detection/manue_manually_segmented_images/training_set/tests/full_real_analysis_tst1/Coloc X2 2048/230314 Coloc X2 ZH-2A m 2048_2023_03_14__17_43_51(1)/FISH.db')
        print(voxel_size)
        print(voxel_size[0]*2)

        import sys
        sys.exit(0)

    my_string = "'areas_without_holes'"
    clean_string = my_string.strip("'")
    print(clean_string)
    print(get_tables('/media/teamPrudhomme/EqpPrudhomme2/Benoit_pr_Benjamin/coccinelles/latest images_20230614/raw images/N4N4_29_M_1a/ladybug_seg.db'))