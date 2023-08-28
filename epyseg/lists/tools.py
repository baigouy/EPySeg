# contains a set of useful methods for list and list operations
import itertools
import itertools

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
