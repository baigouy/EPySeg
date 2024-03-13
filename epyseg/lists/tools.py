# contains a set of useful methods for list and list operations
import itertools
import itertools
from itertools import combinations


def combine_lists(list1, list2):
    combined_list = [(item1, item2) for item1 in list1 for item2 in list2]
    return combined_list

def get_list_combinatorial(lst, repack_all_to_lists_if_not_lists=True, return_list=True):
    """
    Creates all possible combinations of the elements in a list.

    Args:
        lst (list): The input list.
        repack_all_to_lists_if_not_lists (bool): If True, converts non-list elements to lists.
        return_list (bool): If True, returns the combinations as a list of lists.

    Returns:
        list: The list of all possible combinations.

    Examples:
        >>> lst = ['D', ['1P', 'hinge_undefined'], '1P']
        >>> result = get_list_combinatorial(lst)
        >>> print(result)
        [['D', '1P', '1P'], ['D', 'hinge_undefined', '1P']]
    """
    optimized_list = lst

    if repack_all_to_lists_if_not_lists:
        optimized_list = [[item] if not isinstance(item, list) else item for item in lst]

    tmp = list(itertools.product(*optimized_list))

    if return_list:
        tmp = [list(el) for el in tmp]

    return tmp


def list_contains_sublist(lst):
    """
    Checks if a list contains any sublists.

    Args:
        lst (list): The input list.

    Returns:
        bool: True if the list contains sublists, False otherwise.

    Examples:
        >>> lst = ['a', 'b', ['c', 'd']]
        >>> result = list_contains_sublist(lst)
        >>> print(result)
        True
    """
    for el in lst:
        if isinstance(el, list):
            return True

    return False

def get_consecutive_file_pairs(file_paths):
    """
    Given a list of file paths, returns a list of tuples,
    where each tuple contains the consecutive pair of file paths.

    Args:
        file_paths (list): The list of file paths.

    Returns:
        list: The list of tuples containing consecutive file pairs.

    Examples:
        >>> file_paths = ['file1.txt', 'file2.txt', 'file3.txt']
        >>> result = get_consecutive_file_pairs(file_paths)
        >>> print(result)
        [('file1.txt', 'file2.txt'), ('file2.txt', 'file3.txt')]
    """
    return [(file_paths[i], file_paths[i+1]) for i in range(len(file_paths)-1)]

def flatten_list(input_list):
    """
    Flattens a list containing nested tuples or other elements.

    Args:
        input_list (list): The input list to be flattened.

    Returns:
        list: A new list with nested elements flattened.

    Examples:
        >>> input_list = ['toto', 'tutu', ('bob', 'beb'), 'tata']
        >>> flattened_list = flatten_list(input_list)
        >>> print(flattened_list)
        ['toto', 'tutu', 'bob', 'beb', 'tata']
    """
    # Flatten the list using a list comprehension
    flattened_list = [item for sublist in input_list for item in (sublist if isinstance(sublist, tuple) else [sublist])]
    return flattened_list

def create_all_possible_pairs(single_cutters):
    """
    Create all possible pairs of keys from a dictionary of single-cut restriction enzymes.

    Args:
        single_cutters (dict): A dictionary where keys represent enzyme names and values
                              are associated information for single-cut restriction enzymes.

    Returns:
        list: A list of tuples containing all unique pairs of enzyme names from the dictionary.

    Examples:
        >>> single_cutters = {'EcoRI': {'site': 'GAATTC', 'recognition_site': 'G^AATTC', 'cut_site': 1},
        ...                   'BamHI': {'site': 'GGATCC', 'recognition_site': 'G^GATCC', 'cut_site': 1}}
        >>> pairs = create_all_possible_pairs(single_cutters)
        >>> print(pairs)
        [('EcoRI', 'BamHI')]
        >>> single_cutters = ['EcoRI','BamHI']
        >>> pairs = create_all_possible_pairs(single_cutters)
        >>> print(pairs)
        [('EcoRI', 'BamHI')]
    """
    if isinstance(single_cutters, dict):
        single_cutters =single_cutters.keys()
    key_pairs = [pair for pair in combinations(single_cutters, 2)]
    return key_pairs


if __name__ == '__main__':
    if True:
        import os

        # list of file paths as strings
        file_paths = ["/path/to/file1.txt", "/path/to/file2.txt", "/path/to/file3.txt"]

        print(get_consecutive_file_pairs(file_paths))

    test = ['D', ['1P', 'hinge_undefined'], '1P']
    print(list_contains_sublist(test))  # True
    combi = get_list_combinatorial(test)
    print(combi)  # [['D', '1P', '1P'], ['D', 'hinge_undefined', '1P']]
    print(list_contains_sublist(combi[0]))  # False
