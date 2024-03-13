# will contain a series of tools to better handle dicts

import numpy as np
import json

def string_to_dict(string):
    """
    Converts a string representation of a dictionary back to a real dictionary.

    Args:
        string (str): The string representation of a dictionary.

    Returns:
        dict: The converted dictionary.

    Examples:
        >>> string = "{'key1': 'value1', 'key2': 'value2'}"
        >>> result = string_to_dict(string)
        >>> print(result)
        {'key1': 'value1', 'key2': 'value2'}
    """
    if string is None:
        return None

    # Replace single quotes with double quotes and 'None' with 'null'
    string = string.replace("'", '"').replace('None', 'null')

    # Convert the string to a dictionary using json.loads
    return json.loads(string)

def invert_keys_and_values(input_dict, warn_on_dupes=True):
    """
    Swaps keys and values of a dictionary.

    Args:
        input_dict (dict): The input dictionary.
        warn_on_dupes (bool, optional): Whether to warn if there are duplicated values in the dictionary. Default is True.

    Returns:
        dict: The dictionary with keys and values inverted.

    Examples:
        >>> input_dict = {'A': 1, 'B': 2, 'C': 3}
        >>> result = invert_keys_and_values(input_dict)
        >>> print(result)
        {1: 'A', 2: 'B', 3: 'C'}

        >>> input_dict = {'A': 1, 'B': 1, 'C': 3}
        >>> result = invert_keys_and_values(input_dict, warn_on_dupes=True)
        Warning: some values in the dictionary are duplicated. Keys and values cannot be inverted safely.
        >>> print(result)
        {1: 'B', 3: 'C'}
    """
    # Create a new dictionary with keys and values swapped using a dictionary comprehension
    output_dict = {value:key for key, value in input_dict.items()}

    # Check if warning on duplicates is enabled
    if warn_on_dupes:
        # Check if the length of the output dictionary is not equal to the length of the input dictionary
        if len(output_dict) != len(input_dict):
            # Print a warning message if there are duplicated values in the dictionary
            print('Warning: some values in the dictionary are duplicated. Keys and values cannot be inverted safely.')

    # Return the resulting dictionary with inverted keys and values
    return output_dict


def dict_to_numpy(input_dict):
    """
    Converts a dictionary to a NumPy array.

    Args:
        input_dict (dict): The input dictionary.

    Returns:
        numpy.ndarray: The NumPy array containing key-value pairs as items.

    Examples:
        >>> input_dict = {'key1': 1, 'key2': 2, 'key3': 3}
        >>> result = dict_to_numpy(input_dict)
        >>> print(result)
        [['key1' '1']
         ['key2' '2']
         ['key3' '3']]

    """
    return np.array(list(input_dict.items()))

def get_values_in_dict_matching_a_search_list(input_dict, search_list, ignore_not_found=True, value_for_not_found_if_not_ignored=None):
    """
    Finds values in a dictionary that match a search list.

    Args:
        input_dict (dict): The input dictionary to search.
        search_list (list): The list of values to search for in the dictionary.
        ignore_not_found (bool): Flag to ignore search values not found in the dictionary. Defaults to True.
        value_for_not_found_if_not_ignored: Value to use for search values not found if ignore_not_found is False.

    Returns:
        list: A list of values from the dictionary that match the search list.

    Examples:
        >>> input_dict = {'key1': 1, 'key2': 2, 'key3': 3}
        >>> search_list = ['key2', 'key3']
        >>> result = get_values_in_dict_matching_a_search_list(input_dict, search_list)
        >>> print(result)
        [2, 3]
    """
    if ignore_not_found:
        # Find values in the input_dict that match the search_list and ignore not found values
        output = [input_dict[search] for search in search_list if search in input_dict.keys()]
    else:
        # Find values in the input_dict that match the search_list and handle not found values with a specified value
        output = [input_dict[search] if search in input_dict.keys() else value_for_not_found_if_not_ignored for search in search_list]
    return output


def prepend_dict(dictionary, new_key, new_value):
    """
    Prepend a new key-value pair to a dictionary.

    Args:
        dictionary (dict): The dictionary to modify.
        new_key (hashable): The key to prepend.
        new_value (Any): The value to associate with the new key.

    Returns:
        dict: A new dictionary containing the prepended key-value pair and the original key-value pairs.

    Examples:
        >>> dictionary = {'key1': 'value1', 'key2': 'value2'}
        >>> new_key = 'key0'
        >>> new_value = 'value0'
        >>> result = prepend_dict(dictionary, new_key, new_value)
        >>> print(result)
        {'key0': 'value0', 'key1': 'value1', 'key2': 'value2'}
    """
    new_dict = {new_key: new_value}
    new_dict.update(dictionary)
    return new_dict


def _to_dict(input):
    """
    Convert a list of key-value pairs to a dictionary.

    Args:
        input (list): List of key-value pairs.

    Returns:
        dict: A dictionary created from the input list.

    Examples:
        >>> input = [('key1', 'value1'), ('key2', 'value2')]
        >>> result = _to_dict(input)
        >>> print(result)
        {'key1': 'value1', 'key2': 'value2'}
    """
    # Create an empty dictionary
    dct = {}

    # Iterate over each key-value pair in the input list
    for k, v in input:
        # Assign the value to the key in the dictionary
        dct[k] = v

    # Return the resulting dictionary
    return dct



def prepend_txt_to_all_dict_keys(dictionary, text):
    """
    Prepend a given text to all keys of a dictionary.

    Args:
        dictionary (dict): The dictionary to modify.
        text (str): The text to prepend to the keys.

    Returns:
        dict: A new dictionary with modified keys.

    Examples:
        >>> dictionary = {'key1': 'value1', 'key2': 'value2'}
        >>> result = prepend_txt_to_all_dict_keys(dictionary, 'prefix_')
        >>> print(result)
        {'prefix_key1': 'value1', 'prefix_key2': 'value2'}
    """
    # Use a dictionary comprehension to create a new dictionary with modified keys
    return {text + key: value for key, value in dictionary.items()}

def sort_dict_by_keys(dictionary):
    """
    Sort a dictionary by its keys in ascending order.

    Args:
        dictionary (dict): The dictionary to sort.

    Returns:
        dict: The sorted dictionary.

    Examples:
        >>> dictionary = {'c': 3, 'a': 1, 'b': 2}
        >>> result = sort_dict_by_keys(dictionary)
        >>> print(result)
        {'a': 1, 'b': 2, 'c': 3}
    """
    # Sort the dictionary by keys using the sorted() function and convert it back to a dictionary
    sorted_dict = dict(sorted(dictionary.items()))

    # Return the sorted dictionary
    return sorted_dict

def sort_dict_by_values(input_dict, ascending=True):
    """
    Sort a dictionary by its values in ascending or descending order.

    Args:
        input_dict (dict): The input dictionary to be sorted.
        ascending (bool, optional): If True, sort in ascending order; if False, sort in descending order.
                                   Default is True (ascending).

    Returns:
        dict: A new dictionary with the same keys but sorted by values based on the specified order.

    Examples:
        >>> my_dict = {'apple': 3, 'banana': 1, 'cherry': 2}
        >>> ascending_sorted_dict = sort_dict_by_values(my_dict, ascending=True)
        >>> descending_sorted_dict = sort_dict_by_values(my_dict, ascending=False)
        >>> print(ascending_sorted_dict)
        {'banana': 1, 'cherry': 2, 'apple': 3}
        >>> print(descending_sorted_dict)
        {'apple': 3, 'cherry': 2, 'banana': 1}
    """
    sorted_dict = dict(sorted(input_dict.items(), key=lambda item: item[1], reverse=not ascending))
    return sorted_dict


if __name__ == '__main__':

    if True:
        # Example dictionary
        my_dict = {'a': 1, 'b': 2, 'c': 3}
        print(my_dict)

        # New key and value to prepend
        new_key = 'x'
        new_value = 0

        # Prepend the new key-value pair to the dictionary
        new_dict = prepend_dict(my_dict, new_key, new_value)

        # Print the resulting dictionary
        print(new_dict)

    if True:
        # Example dictionary header
        header = {'Name': 'John', 'Age': 30, 'Gender': 'Male'}
        print(header)
        # Text to prepend to keys
        text = 'Prefix_'

        # Prepend the text to all keys in the dictionary header
        new_header = prepend_txt_to_all_dict_keys(header, text)

        # Print the resulting dictionary header
        print(new_header)