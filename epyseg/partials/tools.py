from functools import partial


def dummy():
    print('test')

def dummy2():
    print('test2')

def get_method_name(partial_func):
    """
    Returns the name of the wrapped function in a partial object.

    Args:
        partial_func (functools.partial): The partial object.

    Returns:
        str: The name of the wrapped function.

    Examples:
        >>> tst = partial(dummy)
        >>> print(get_method_name(tst))
        'dummy'
    """
    return partial_func.func.__name__

def is_partial(func):
    """
    Checks if an object is a partial object.

    Args:
        func: The object to check.

    Returns:
        bool: True if the object is a partial object, False otherwise.

    Examples:
        >>> tst = partial(dummy)
        >>> print(is_partial(tst))
        True
    """
    return isinstance(func, partial)

def are_function_names_matching(func1, func2):
    """
    Compares the names of two function objects or partial objects and returns True if they match, False otherwise.

    Args:
        func1: The first function object or partial object.
        func2: The second function object or partial object.

    Returns:
        bool: True if the names match, False otherwise.

    Examples:
        >>> print(are_function_names_matching(dummy, dummy2))
        False
        >>> print(are_function_names_matching(dummy, dummy))
        True
        >>> tst = partial(dummy)
        >>> print(are_function_names_matching(tst, dummy))
        True
    """
    if is_partial(func1):
        func1 = get_method_name(func1)
    else:
        func1 = func1.__name__
    if is_partial(func2):
        func2 = get_method_name(func2)
    else:
        func2 = func2.__name__
    return func1 == func2

if __name__ == '__main__':

    tst = partial(dummy)


    print('is partial:', is_partial(tst))
    print('get method name:', get_method_name(tst))
    print('are function names matching:', are_function_names_matching(dummy, dummy2))
    print('are function names matching:', are_function_names_matching(dummy, dummy))
    print('are function names matching:', are_function_names_matching(tst, dummy))
