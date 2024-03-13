def difference(set1, set2):
    """
    Returns the elements of set1 that are not in set2.

    Args:
        set1 (set or iterable): The first set or iterable.
        set2 (set or iterable): The second set or iterable.

    Returns:
        set: The set of elements from set1 that are not in set2.

    Examples:
        >>> set1 = {1, 2, 3}
        >>> set2 = {2, 3, 4}
        >>> difference(set1, set2)
        {1}

    """
    if not isinstance(set1, set):
        set1 = set(set1)
    return set1.difference(set2)

def differences(set1, set2):
    """
    Returns the elements of set1 and set2 that are not present in the other set.

    Args:
        set1 (set or iterable): The first set or iterable.
        set2 (set or iterable): The second set or iterable.

    Returns:
        set: The set of elements from set1 and set2 that are not present in the other set.

    Examples:
        >>> set1 = {1, 2, 3}
        >>> set2 = {2, 3, 4}
        >>> differences(set1, set2)
        {1, 4}

    """
    dif1 = difference(set1, set2)
    dif2 = difference(set2, set1)
    return dif1.union(dif2)

def intersection(set1, set2):
    """
    Returns the intersection of set1 and set2.

    Args:
        set1 (set or iterable): The first set or iterable.
        set2 (set or iterable): The second set or iterable.

    Returns:
        set: The set of elements that are common to both set1 and set2.

    Examples:
        >>> set1 = {1, 2, 3}
        >>> set2 = {2, 3, 4}
        >>> intersection(set1, set2)
        {2, 3}

    """
    if not isinstance(set1, set):
        set1 = set(set1)
    return set1.intersection(set2)

def union_no_dupes(*sets):
    """
    Returns the union of multiple sets, removing duplicate elements.

    Args:
        *sets (sets or iterables): Multiple sets or iterables.

    Returns:
        set: The union of all sets, excluding duplicate elements.

    Examples:
        >>> set1 = {1, 2, 3}
        >>> set2 = {2, 3, 4}
        >>> set3 = {3, 4, 5}
        >>> union_no_dupes(set1, set2, set3)
        {1, 2, 3, 4, 5}

    """
    output_set = set(sets[0])
    for st in sets[1:]:
        output_set = output_set.union(st)
    return output_set

def common(*sts):
    first = set(sts[0])
    common_elements = first.intersection(*sts[1:])
    return common_elements

if __name__ == '__main__':
    set1 = {1, 2, 3}
    set2 = {2, 3, 4}

    print(intersection(set1, set2))  # Output: {2, 3}
    print(union_no_dupes(set1, set2))  # Output: {1, 2, 3, 4}
    print(differences(set1, set2))  # Output: {1, 4}
    print(difference(set1, set2))

    set3 = {1,2}
    print('common', common(set1, set2, set3))




