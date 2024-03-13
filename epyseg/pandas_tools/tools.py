# Pandas related tools/methods

# in vs code only ctrl + F5 works for running and having the path
# DEV NOTE TWO LINES BELOW KEEP FIX FOR RUNNING IN VS CODE §!!!!!!!!! --> very dirty though
#import sys
#sys.path.append('/home/aigouy/mon_prog/Python/epyseg_pkg/')

import matplotlib.pyplot as plt
import sqlite3
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

from epyseg.files.tools import open_file_with_default_handler


def add_cumulative_sum(df):
    """
    Modify a DataFrame so that each row contains the cumulative sum of all rows above it.

    Args:
        df (pd.DataFrame): The input DataFrame.

    Returns:
        pd.DataFrame: The modified DataFrame with cumulative sums.

    Examples:
        >>> data = {'A': [1, 2, 3, 4]}
        >>> df = pd.DataFrame(data)
        >>> df_cumulative = add_cumulative_sum(df)
        >>> print(df_cumulative)
            A
        0   1
        1   3
        2   6
        3  10
    """
    # Use the cumsum() function along the rows (axis=0) to calculate cumulative sums
    df_cumulative = df.cumsum(axis=0)

    return df_cumulative


def remove_columns_with_string(df, substring):
    """
    Remove all columns from a DataFrame that contain a specified substring in their name.

    Args:
        df (pd.DataFrame): The input DataFrame.
        substring (str): The substring to search for in column names.

    Returns:
        pd.DataFrame: The DataFrame with columns containing the substring removed.

    Examples:
        >>> data = {'fruit_flies_count': [10, 20, 30], 'apple_count': [5, 15, 25]}
        >>> df = pd.DataFrame(data)
        >>> df_filtered = remove_columns_with_string(df, 'flies')
        >>> print(df_filtered)
           apple_count
        0            5
        1           15
        2           25
    """
    # Get a list of column names to keep (those that do not contain the specified substring)
    columns_to_keep = [col for col in df.columns if substring not in col]

    # Create a new DataFrame with the selected columns
    df_filtered = df[columns_to_keep]

    return df_filtered

def append_row_to_df(df, row_content):
    df.loc[len(df)] = row_content
    return df # do I even need to return the df



def numpy_array_to_pandas_df(my_array, header=None):
    """
    loads a numpy array to a pandas df.

    Args:
        my_array (numpy.ndarray): The numpy array to save.
        header_names (list, optional): A list of header names for the pandas df. If not provided, no header will be added.
    """
    # Convert the numpy array to a pandas DataFrame
    df = pd.DataFrame(my_array,columns=header)

    # Add headers if provided
    # if header is not None:
        # if not isinstance(header, np.ndarray):
        #     header = np.asarray(header)
        # df.columns = header

    # Save the DataFrame to a CSV file
    # df.to_csv(file_path, index=False)

    return df

def create_empty_table_like(df):
    # Create an empty DataFrame with the same column names
    empty_df = pd.DataFrame(columns=df.columns)
    return empty_df


def add_column(df, new_column_name):
    # Add a new empty column named 'NewEmptyColumn'
    df[new_column_name] = pd.Series()
    return df

def create_empty_df(header):
    return numpy_array_to_pandas_df(None,header=header)

def insert_column_at_position(df, index_position, new_column_name ):
    # Insert the new column at the specified index position
    df.insert(index_position, new_column_name, value=None)
    return df

def prepend_column(df, col_name, col_values):
    """
    Prepends a new column to a pandas DataFrame.

    Args:
        df (pandas.DataFrame): The DataFrame to prepend the column to.
        col_name (str): The name of the new column to prepend.
        col_values (list): A list of values to populate the new column.
    """
    # Use the insert() method to prepend the new column to the DataFrame
    df.insert(loc=0, column=col_name, value=col_values)
    return df

def get_column_factors(df, column):
    """
    Returns a list of all the unique values in the specified column of the DataFrame,
    using either the column name or the value as an index if it is an integer.

    Parameters:
        - df: The DataFrame containing the data.
        - column: The name or index of the column.

    Returns:
        A list of unique values in the specified column.

    Examples:
        >>> df = pd.DataFrame({'A': [1, 2, 3, 4], 'B': [1, 2, 2, 3]})
        >>> get_column_factors(df, 'B')
        array([1, 2, 3])
    """
    # If the column is specified as an index (integer), get the corresponding column name
    if isinstance(column, int):
        column = df.columns[column]

    # Retrieve the unique values from the specified column
    factors = df[column].unique()

    # Return the list of unique values
    return factors


def get_combined_factors(df, columns):
    """
    Returns a list of all the unique values in the specified columns of the DataFrame,
    using either the column names or the values as indices if they are integers.

    Parameters:
        - df: The DataFrame containing the data.
        - columns: A list of column names or indices.

    Returns:
        A list of unique values from the specified columns.

    Examples:
        >>> df = pd.DataFrame({'A': [1, 2, 3, 4], 'B': [1, 2, 2, 3], 'C': ['X', 'Y', 'Z', 'Z']})
        >>> get_combined_factors(df, ['A', 1, 'C'])
        array([1, 2, 3, 4, 'X', 'Y', 'Z'], dtype=object)
    """
    # Convert column indices to column names if they are integers
    column_names = [df.columns[col] if isinstance(col, int) else col for col in columns]

    # Concatenate the specified columns into a single series
    combined_series = pd.concat([df[name] for name in column_names])

    # Retrieve the unique values from the combined series
    factors = combined_series.unique()

    # Return the list of unique values
    return factors

def strip_column(df, col_name_or_idx):
    if isinstance(col_name_or_idx, str):
        df[col_name_or_idx] = df[col_name_or_idx].str.strip()
    else:
        # Strip leading and trailing whitespace from the specified column
        df.iloc[:, col_name_or_idx] = df.iloc[:, col_name_or_idx].str.strip()
    return df


def diff_dataframes(df1, df2):
    """
    Returns a DataFrame with only the rows that differ between the two input DataFrames.

    Parameters:
        - df1: The first DataFrame.
        - df2: The second DataFrame.

    Returns:
        A DataFrame containing the differing rows between df1 and df2.

    Examples:
        >>> df1 = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
        >>> df2 = pd.DataFrame({'A': [1, 2, 4], 'B': [4, 5, 7]})
        >>> diff_dataframes(df1, df2)
           A  B      _merge
        2  3  6   left_only
        3  4  7  right_only
    """

    # Merge the two DataFrames based on all columns, with an indicator column added
    merged = pd.merge(df1, df2, how='outer', indicator=True)

    # Filter the merged DataFrame to only include rows that are not present in both DataFrames
    diff = merged[merged['_merge'] != 'both']

    # Return the resulting DataFrame with differing rows
    return diff

def open_as_df(file_path, sep='\t', header=None):
    # Read the file into a DataFrame with a header
    df = pd.read_csv(file_path, delimiter=sep, header=header)
    return df


def filter_out_rows(df1, other_dfs):
    """
    Filter out rows from a DataFrame that are contained in one or more other DataFrames.

    Parameters:
    - df1 (pandas.DataFrame): The main DataFrame from which rows will be filtered.
    - other_dfs (list of pandas.DataFrame): A list of other DataFrames to check for containment.

    Returns:
    - pandas.DataFrame: A DataFrame containing only the rows from df1 that are not contained in any of the other DataFrames.

    Example:
    >>> import pandas as pd
    >>> data1 = {'ID': [1, 2, 3, 4, 5, 6], 'Value': ['A', 'B', 'C', 'D', 'E', 'G']}
    >>> df1 = pd.DataFrame(data1)
    >>> data2 = {'ID': [2, 4], 'Value': ['B', 'D']}
    >>> df2 = pd.DataFrame(data2)
    >>> data3 = {'ID': [1, 3, 5], 'Value': ['A', 'C', 'E']}
    >>> df3 = pd.DataFrame(data3)
    >>> other_dfs = [df2, df3]
    >>> filtered_df = filter_out_rows(df1, other_dfs)
    >>> print(filtered_df)
       ID Value
    0   6     G
    """
    # Initialize the filtered DataFrame with df1
    filtered_df = df1

    if not isinstance(other_dfs, list):
        other_dfs = [other_dfs]

    # Loop through each DataFrame in other_dfs
    for other_df in other_dfs:
        # Use the isin() method to check if rows in df1 are contained in the current other DataFrame
        mask = df1.isin(other_df.to_dict(orient='list')).all(axis=1)

        # Filter out the rows based on the mask
        filtered_df = filtered_df[~mask]

    # Reset the index of the filtered DataFrame
    filtered_df.reset_index(drop=True, inplace=True)

    return filtered_df


def rename_column(df, index, new_name):
    return df.rename(columns={df.columns[index]: new_name})

def combine_dataframes(dataframes):
    """
    Combines a list of dataframes into a single master dataframe.

    Parameters:
        - dataframes: A list of pandas.DataFrame objects to combine.

    Returns:
        A master dataframe that combines all the input dataframes.

    Examples:
        >>> df1 = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
        >>> df2 = pd.DataFrame({'A': [4, 5, 6], 'B': [7, 8, 9]})
        >>> combine_dataframes([df1, df2])
           A  B
        0  1  4
        1  2  5
        2  3  6
        3  4  7
        4  5  8
        5  6  9
    """

    # Combine the dataframes into a single master dataframe
    master_df = pd.concat(dataframes, ignore_index=True)

    return master_df


def load_table_to_pandas_df(path_to_db, table_name):
    """
    Loads a SQL table into a pandas DataFrame from the specified SQLite database file.

    Parameters:
        - path_to_db: The path to the SQLite database file.
        - table_name: The name of the table to load.

    Returns:
        A pandas DataFrame containing the data from the specified table.

    # Examples:
    #     >>> df = load_table_to_pandas_df('mydatabase.db', 'mytable')
    """

    # Create a connection to the SQLite database
    conn = sqlite3.connect(path_to_db)

    # Load the SQL table into a pandas DataFrame
    df = pd.read_sql('SELECT * FROM ' + table_name, conn)

    # Close the connection to the database
    conn.close()

    return df


def get_blob_columns(df):
    """
    Returns a list of column names from the DataFrame that have object (string) data type.

    Parameters:
        - df: The DataFrame from which to extract the blob columns.

    Returns:
        A list of column names that have object data type.

    Examples:
        >>> df = pd.DataFrame({'A': [1, 2, 3], 'B': ['abc', 'def', 'ghi'], 'C': [True, False, True]})
        >>> get_blob_columns(df)
        ['B']
    """
    blob_columns = df.select_dtypes(include=['object']).columns.tolist()
    return blob_columns


def dict_of_tables_to_SQL(tables_dict, output_sql_file):
    """
    Saves a dictionary of DataFrames as separate tables in an SQLite database file.

    Parameters:
        - tables_dict: A dictionary containing table names as keys and DataFrames as values.
        - output_sql_file: The path to the output SQLite database file.

    Returns:
        None

    # Examples:
    #     >>> df1 = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
    #     >>> df2 = pd.DataFrame({'A': [4, 5, 6], 'B': [7, 8, 9]})
    #     >>> tables_dict = {'table1': df1, 'table2': df2}
    #     >>> dict_of_tables_to_SQL(tables_dict, 'output.db')
    """

    # Create a connection to the SQLite database
    conn = sqlite3.connect(output_sql_file)

    # Loop through the dictionary and save each DataFrame as a table in the SQL file
    for table_name, df in tables_dict.items():
        df.to_sql(table_name, conn, if_exists='replace')

    # Close the connection to the database
    conn.close()


def add_dataframe_to_sql(df, db_file, table_name):
    """
    Adds a Pandas DataFrame to an SQL database as a new table.

    Parameters:
        - df: The DataFrame to be added to the database.
        - db_file: The name and path of the SQL database file.
        - table_name: The name to be given to the new table in the database.

    Returns:
        None

    # Examples:
    #     >>> df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
    #     >>> add_dataframe_to_sql(df, 'mydatabase.db', 'mytable')
    """
    # Connect to the SQL database
    conn = sqlite3.connect(db_file)

    # Use the to_sql() method to add the DataFrame to the database
    df.to_sql(table_name, conn, if_exists='replace')

    # Close the database connection
    conn.close()


def name_index_column(df, name_to_use='index_label'):
    """
    Sets the name of the index column in the DataFrame.

    Parameters:
        - df: The DataFrame in which the index column name will be set.
        - name_to_use: The name to assign to the index column (default is 'index_label').

    Returns:
        The DataFrame with the updated index column name.

    Examples:
        >>> import re
        >>> df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
        >>> re.sub(r'\s+', '', str(name_index_column(df, 'new_index')))
        'ABnew_index014125236'
    """

    # Set the name of the index column in the DataFrame
    df.index.name = name_to_use

    # Return the updated DataFrame
    return df


def kmeans_clustering(df, n_clusters, exclude_non_numeric_columns=True, verbose=False):
    """
    Perform K-means clustering on a Pandas DataFrame.

    Parameters:
        - df: The DataFrame to cluster.
        - n_clusters: The number of clusters to create.
        - exclude_non_numeric_columns: Boolean value indicating whether non-numeric columns should be excluded from clustering (default is True).

    Returns:
        The input DataFrame with an additional column 'cluster', containing the cluster labels for each row.

    Examples:
        >>> df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
        >>> kmeans_clustering(df, 2)
           A  B  cluster
        0  1  4        1
        1  2  5        0
        2  3  6        0
    """
    if exclude_non_numeric_columns:
        # create a numpy array with only the numeric columns
        X = df.select_dtypes(include=[np.number]).values
    else:
        X = df.values  # convert DataFrame to a numpy array

    # create a KMeans object and fit it to the data
    kmeans = KMeans(n_clusters=n_clusters, random_state=42,n_init='auto')
    df['cluster'] = kmeans.fit_predict(X)

    if True:
        # perform KMeans clustering and obtain the transformed distances
        distances = kmeans.fit_transform(X)
        # compute the correlation between each column and the transformed distances
        correlations = df.corrwith(pd.DataFrame(distances))

        # sort the correlations in descending order of absolute value
        sorted_correlations = correlations.abs().sort_values(ascending=False)

        # print the sorted correlations
    if verbose:
        print("Feature correlations with KMeans distances in descending order:")
        print(sorted_correlations)

    if False:
        # get the distances of each sample to each cluster center
        distances = kmeans.transform(X)
        # convert the distances array to a DataFrame with column names
        X = pd.DataFrame(distances, columns=[f"distance_{i}" for i in range(distances.shape[1])])

        # compute the mean distance for each column
        mean_distances = X.mean(axis=0)

        # create a dictionary of column names and their corresponding mean distances
        distance_dict = dict(zip(X.columns, mean_distances))

        # sort the dictionary by mean distance in ascending order
        sorted_distances = sorted(distance_dict.items(), key=lambda x: x[1])

        # print the sorted dictionary
        print("Feature distances in ascending order:")
        for feature, distance in sorted_distances:
            print(f"{feature}: {distance:.3f}")

    return df


def pca_analysis(df, exclude_non_numeric_columns=True):
    """
    Perform PCA clustering on a DataFrame and determine the most important feature.

    Parameters:
        - df: a pandas DataFrame containing the data to be analyzed.
        - exclude_non_numeric_columns: Boolean value indicating whether non-numeric columns should be excluded from PCA analysis (default is True).

    Returns:
        - A tuple containing the explained variance ratio for each component,
          the component matrix showing the contribution of each original feature to each principal component,
          and the name of the most important feature.

    Examples:
        >>> df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
        >>> pca_analysis(df)
        (array([1.00000000e+00, 2.81351049e-34]),           A         B
        0  0.707107  0.707107
        1  0.707107 -0.707107, 'A')
    """
    if exclude_non_numeric_columns:
        # select only the numeric columns
        numeric_df = df.select_dtypes(include=[np.number])
    else:
        numeric_df = df

    # perform PCA on the numeric DataFrame
    pca = PCA()
    components = pca.fit_transform(numeric_df.values)

    # get the explained variance ratio for each component
    explained_variance_ratio = pca.explained_variance_ratio_

    # create a component matrix DataFrame
    component_matrix = pd.DataFrame(pca.components_, columns=numeric_df.columns)

    # determine the most important feature based on the first principal component
    most_important_feature = component_matrix.iloc[0].idxmax()

    return explained_variance_ratio, component_matrix, most_important_feature

def pca_analysis2(df, n_clusters=2):
    """
    Perform PCA clustering on a DataFrame and determine the most important feature.

    Parameters:
        - df: a pandas DataFrame containing the data to be analyzed.
        - n_clusters: the number of clusters to use in KMeans clustering.

    Returns:
        - A tuple containing the explained variance ratio for each component,
          the component matrix showing the contribution of each original feature to each principal component,
          the name of the most important feature,
          and a pandas Series containing the cluster labels for each sample in the input DataFrame.
    """
    # select only the numeric columns
    numeric_df = df.select_dtypes(include=[np.number])

    # drop any rows that contain missing values or non-numeric values
    clean_df = numeric_df.dropna().apply(pd.to_numeric, errors='coerce')

    # instantiate the PCA model with n_components=2
    pca = PCA(n_components=2)

    # fit the model to the data and transform the data
    pca.fit(clean_df)
    transformed = pca.transform(clean_df)

    # get the explained variance ratio for each component
    explained_variance_ratio = pca.explained_variance_ratio_

    # compute the contribution of each original feature to each principal component
    component_matrix = pd.DataFrame(pca.components_, columns=clean_df.columns, index=['PC1', 'PC2'])

    # instantiate the KMeans model with the specified number of clusters
    kmeans = KMeans(n_clusters=n_clusters,n_init='auto')

    # fit the model to the transformed data and predict the cluster labels
    kmeans.fit(transformed)
    labels = kmeans.predict(transformed)

    # add the cluster labels as a new column in the input DataFrame
    df['Cluster'] = labels

    # determine the most important feature
    most_important_feature = component_matrix.abs().idxmax(axis=1)

    if True:
        # create a scatter plot of the PCA components colored by cluster label
        plt.scatter(transformed[:, 0], transformed[:, 1], c=labels)
        plt.xlabel('PC1')
        plt.ylabel('PC2')
        plt.title('PCA Plot')
        plt.show()

    # return the results as a tuple
    results = (explained_variance_ratio, component_matrix, most_important_feature, labels)

    return results


def split_column(df, col_name, new_col_names, strip='()'):
    """
    Split a column in a pandas DataFrame into two new columns and drop the original column.

    Parameters:
        - df: a pandas DataFrame containing the column to split.
        - col_name: a string specifying the name of the column to split.
        - new_col_names: a tuple of two strings specifying the names of the new columns to create.

    Returns:
        - A pandas DataFrame containing the input DataFrame with the original column dropped and
          two new columns added containing the split values of the original column.
    """
    # split the column into two new columns
    df[new_col_names] = df[col_name].str.strip(strip).str.split(',', expand=True).astype(float)

    # drop the original column
    df = df.drop(col_name, axis=1)

    return df


def to_clipboard(df, index=False, header=True):
    """
    Copy the contents of a Pandas DataFrame to the system clipboard.

    Parameters:
        df : pandas.DataFrame
            The DataFrame to be copied to the clipboard.
        index : bool, optional
            Whether to include the index column in the copied data. Default is False.
        header : bool, optional
            Whether to include the column headers in the copied data. Default is True.

    Returns:
        None

    Examples:
        >>> import pandas as pd
        >>> df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6], 'C': [7, 8, 9]})
        >>> to_clipboard(df)
    """
    return df.to_clipboard(index=index, header=header)

def set_df_starting_index(df, start_index):
    df.index = range(start_index, start_index + len(df))
    return df

def nullify_df_indices(df):
    df.index = None
    return df

def copy_df_and_fill_withvalue(df, value):
    # Create a copy of the DataFrame
    df_copy = df.copy()
    df_copy = df_copy.replace(df_copy.values, value)
    return df_copy

def check_and_remove_duplicate_indices(df):
    # Check if indices are unique
    if df.index.duplicated().any():
        # Remove duplicates and keep the first instance
        df = df[~df.index.duplicated(keep='first')]

    return df

def merge_df_and_keep_indices(*dfs, remove_na=True):
    # result_df = pd.merge(df1, df2, left_index=True, right_index=True, how='outer')
    # Merge the single-column DataFrames into a single DataFrame with n columns
    result_df = pd.concat(dfs, axis=1, join='outer')
    if remove_na:
        result_df.fillna('', inplace=True)
    return result_df

if __name__ == '__main__':

    if False:
        import pandas as pd

        # Example DataFrames with different indices
        # Your n single-column DataFrames with different indices
        df1 = pd.DataFrame({'Column1': ['A', 'T', 'G']}, index=[1, 2, 3])
        df2 = pd.DataFrame({'Column2': ['C', 50, 60]}, index=[4, 5, 6])
        df3 = pd.DataFrame({'Column3': [70, 80, 90]}, index=[12, 8, 8])

        # Merge the single-column DataFrames into a single DataFrame with n columns
        # result_df = pd.concat([df1, df2, df3], axis=1, join='outer')

        # print(result_df)
        # Merge DataFrames on indices
        # merged_df = pd.merge(df1, df2, left_index=True, right_index=True, how='outer')
        merged_df = merge_df_and_keep_indices(df1, df2, df3)



        # Display the merged DataFrame
        print(merged_df)

        import sys
        sys.exit(0)

    if True:
        print(create_empty_df(['test1','test2', 'tutu']))

        import sys
        sys.exit(0)


    from epyseg.img import Img
    if True:
        files = ['/media/teamPrudhomme/EqpPrudhomme2/Benoit_pr_Benjamin/coccinelles/latest images_20230614/raw images/effet_sexe_29.db', '/media/teamPrudhomme/EqpPrudhomme2/Benoit_pr_Benjamin/coccinelles/latest images_20230614/raw images/effet_sexe_18.db', '/media/teamPrudhomme/EqpPrudhomme2/Benoit_pr_Benjamin/coccinelles/ANALYZE_LATER/25°N4N45F_25°N4N42M_25°R16F_25°R17M_29°R26inds-26-05-2023_014/ladybug_seg.db']
        corresponding_image = [None, None, '/media/teamPrudhomme/EqpPrudhomme2/Benoit_pr_Benjamin/coccinelles/ANALYZE_LATER/25°N4N45F_25°N4N42M_25°R16F_25°R17M_29°R26inds-26-05-2023_014.tif']
        idx = 0

        conn = sqlite3.connect(files[idx])

        img = None
        if corresponding_image[idx] is not None:
            img = Img(corresponding_image[idx])


        df = pd.read_sql_query('SELECT * FROM elytras_shape', conn)
        # df = pd.read_sql_query('SELECT * FROM ' + table_name, conn)
        if idx == 2: # TODO --> remove this line
            df = df.loc[df['cluster'].isin([1, 2])]

        old = df.copy()
        # use the drop() method to remove the column
        forbidden_columns = ['index', 'cluster','local_ID']
        for col in forbidden_columns:
            try:
                df = df.drop(col, axis=1)
            except:
                print('no ' + col+' column --> ignoring')

        print(df.columns)
        print(df.head())

        # print(pca_analysis(df))
        print(pca_analysis2(df))

        # df = kmeans_clustering(df, 2)

        print(df)
        old['new_cluster']=df['Cluster']

        print(old)
        # group the DataFrame by the pairs of values in columns A and B and count the number of occurrences of each pair
        counts = old.groupby(['cluster', 'new_cluster']).size().reset_index(name='Count')

        # print the counts for each pair
        print(counts)

        # group the DataFrame by the 'cluster' column and compute the max value of 'Count' for each group
        max_counts = counts.groupby('cluster')['Count'].max()

        # group the DataFrame by the 'cluster' column and compute the sum of 'Count' for each group
        sum_counts = counts.groupby('cluster')['Count'].sum()

        # compute the ratio of max counts to sum counts
        ratios = max_counts / sum_counts

        # print the resulting ratios
        print(ratios)

        old = split_column(old, 'centroids', ['centroid_y','centroid_x'], strip='()')

        print(old['centroid_y'])

        # define a dictionary that maps cluster values to colors
        color_map = {0: 'purple', 1: 'blue', 2: 'red', 3: 'green', 4: 'yellow', 5: 'cyan'}

        # print(df.head())
        conn.close()

        if img is not None:
            img = img/img.max()
            # plot the image
            fig, ax = plt.subplots()
            ax.imshow(img)
            # define a colormap
            cmap = 'viridis'

            # plot the points over the image
            for i, row in old.iterrows():
                cluster = row['new_cluster']
                y = row['centroid_y']
                x = row['centroid_x']
                ax.scatter(x, y, c=color_map[cluster])

            # add a colorbar
            fig.colorbar(ax.collections[0], ax=ax)
            # add a legend
            # ax.legend()

            # show the plot
            plt.show()

        import sys
        sys.exit(0)


    if True:
        # Create a sample DataFrame
        data = {'A': [1, 2, 3], 'B': [4, 5, 6]}
        df = pd.DataFrame(data)
        print("Before naming the index column:")
        print(df)

        # Name the index column as 'ID'
        df_named = name_index_column(df, 'ID')

        print("After naming the index column:")
        print(df_named)
        

    if True:
        # Create a list of dataframes
        df1 = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
        df2 = pd.DataFrame({'A': [4, 5, 6], 'B': [7, 8, 9]})
        df3 = pd.DataFrame({'A': [7, 8, 9], 'B': [10, 11, 12]})
        df_list = [df1, df2, df3]

        # Call the combine_dataframes() method to create a master dataframe
        master_df = combine_dataframes(df_list)

        # Print the master dataframe
        print(master_df)

        import sys
        sys.exit(0)


    if True:
        # Create two sample DataFrames
        df1 = pd.DataFrame({'A': [1, 2, 3], 'B': ['foo', 'bar', 'baz']})
        df2 = pd.DataFrame({'A': [1, 3, 4], 'B': ['foo', 'baz', 'qux']})

        # Call the diff_dataframes() method to get the rows that differ between the two DataFrames
        diff = diff_dataframes(df1, df2)

        print(diff)

        import sys
        sys.exit(0)

    # Create a sample DataFrame
    df = pd.DataFrame({'col1': ['foo', 'bar', 'baz', 'foo', 'qux', 'bar'], 'col2': [1, 2, 3, 4, 5, 6]})

    # Call the get_column_factors() method with a column name
    factors1 = get_column_factors(df, 'col1')
    print(factors1)

    # Call the get_column_factors() method with an integer index
    factors2 = get_column_factors(df, 1)
    print(factors2)

    # Create a sample DataFrame
    df = pd.DataFrame({'col1': ['foo', 'bar', 'baz', 'foo', 'qux', 'bar'],
                       'col2': [1, 2, 3, 4, 5, 6],
                       'col3': ['baz', 'qux', 'foo', 'baz', 'bar', 'qux']})

    # Call the get_combined_factors() method with column names
    factors1 = get_combined_factors(df, ['col1', 'col3'])
    print(factors1)

    # Call the get_combined_factors() method with integer indices
    factors2 = get_combined_factors(df, [0, 2])
    print(factors2)


    # Call the get_combined_factors() method with mixed columns
    factors2 = get_combined_factors(df, ['col1', 2])
    print(factors2)

