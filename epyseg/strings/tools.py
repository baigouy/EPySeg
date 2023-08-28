"""
A collection of methods to manipulate strings

"""

def format_string_with_numbers(*args, auto_add_underscore_to_text_if_missing=True, padding_for_all_numbers=None):
    """
    Formats a list of strings and numbers into a single string, with optional automatic addition of underscores to text
    and padding for all numbers.

    Args:
        *args: A variable-length list of strings and numbers to format.
        auto_add_underscore_to_text_if_missing (bool): Whether to automatically add an underscore to the end of any
            string arguments that don't already end with one. Defaults to True.
        padding_for_all_numbers (int or float): If provided, all numbers will be formatted with this many digits using
            leading zeros (for integers) or trailing zeros (for floats). If an integer is provided, the padding will be
            applied to all numbers, while if a float is provided, the padding will be determined by the number of digits
            after the decimal point. Defaults to None.

    Returns:
        str: The formatted string.

    Examples:
        >>> format_string_with_numbers('toto', 10, 'bob', 12, padding_for_all_numbers=3)
        'toto_010_bob_012'
        >>> format_string_with_numbers('toto', 0.33255121121, 'bob', 12, padding_for_all_numbers=0.2)
        'toto_0.33_bob_12.00'
        >>> format_string_with_numbers('toto', 0.33255121121, 'bob', 12, padding_for_all_numbers='{:0>9.2f}')
        'toto_000000.33_bob_000012.00'
    """
    if args is None:
        return None

    # Check if no arguments were passed
    if not args:
        return ''

    # Check if padding for numbers was provided and format it accordingly
    if isinstance(padding_for_all_numbers, (int, float)):
        if isinstance(padding_for_all_numbers, int):
            padding_for_all_numbers = '{:0' + str(padding_for_all_numbers) + 'd}'
        else:
            # Remove leading zeros from decimal numbers
            my_string = str(padding_for_all_numbers)
            if my_string.startswith('0'):
                my_string = my_string[1:]
            padding_for_all_numbers = '{:' + my_string + 'f}'

    output_string = ''
    for arg in args:
        # If the argument is a string, add an underscore to the end if needed
        if isinstance(arg, str):
            if auto_add_underscore_to_text_if_missing and not arg.endswith('_'):
                arg += '_'
            output_string += arg
        else:
            # If padding is provided, format the number with the specified padding
            if padding_for_all_numbers is not None:
                output_string += padding_for_all_numbers.format(arg)
            else:
                output_string += str(arg)

            # Add an underscore to the end of the number if needed
            if auto_add_underscore_to_text_if_missing:
                output_string += '_'

    # Remove the last underscore if it exists
    if auto_add_underscore_to_text_if_missing and output_string.endswith('_'):
        output_string = output_string[:-1]

    return output_string

if __name__ == '__main__':
    print(format_string_with_numbers('toto', 10, 'bob', 12, padding_for_all_numbers=3)) # toto_010_bob_012
    print(format_string_with_numbers('toto', 0.33255121121, 'bob', 12, padding_for_all_numbers=0.2)) # toto_0.33_bob_12.00
    print(format_string_with_numbers('toto', 0.33255121121, 'bob', 12, padding_for_all_numbers='{:0>9.2f}')) # toto_000000.33_bob_000012.00

    # print(help(tools))
    # print(format_string_with_numbers.__doc__)
    help(format_string_with_numbers)