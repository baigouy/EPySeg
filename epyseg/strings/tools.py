"""
A collection of methods to manipulate strings

"""
import textwrap
import re

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

def write_wrapped_text_to_file(file_path, text, line_width=60):
    """
    Write wrapped text to a file, ensuring lines are no longer than a specified width.

    Parameters:
    - file_path (str): The path to the file where the wrapped text will be written.
    - text (str): The input text to be wrapped and written to the file.
    - line_width (int, optional): The maximum width for each line. Default is 60.

    Returns:
    None
    """
    # Wrap the text to a maximum line width of 60 characters
    wrapped_text = textwrap.fill(text, width=line_width)

    # Write the wrapped text to the file
    with open(file_path, 'w') as file:
        file.write(wrapped_text)


def wrap_text(text, line_width=60):
    """
    Wrap the input text to ensure lines are no longer than a specified width.

    Parameters:
    - text (str): The input text to be wrapped.
    - line_width (int, optional): The maximum width for each line. Default is 60.

    Returns:
    str: The wrapped text with lines no longer than the specified width.

    Example:
    >>> wrapped_text = wrap_text("This is a long line that needs to be wrapped to fit within 60 characters per line.")
    >>> print(wrapped_text)
    This is a long line that needs to be wrapped to fit within
    60 characters per line.
    """
    wrapped_text = textwrap.fill(text, width=line_width)
    return wrapped_text

def capitalize_first_letter(input_string):
    # return input_string.capitalize() # DEV NOTE KEEP this is really bad as it removes inner capitalized --> really useless to me then
    try:
        return input_string[:1].upper() + input_string[1:]
    except:
        pass
    return input_string.capitalize() # if it failed roll back to that


def replace_ignore_case(original_string, old_substring, new_substring):
    pattern = re.compile(re.escape(old_substring), re.IGNORECASE)
    modified_string = pattern.sub(new_substring, original_string)
    return modified_string


# do the text chunk splitting here
def chunk_text_with_overlap(text, chunk_size_words, overlap_words):
    """
    Split the input text into chunks of specified size with an overlap.

    Args:
    - text (str): The input text to be split into chunks.
    - chunk_size_words (int): The size of each chunk in terms of words.
    - overlap_words (int): The overlap size between adjacent chunks in terms of words.

    Returns:
    - list: A list containing the text chunks.
    """
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size_words, len(words))
        chunk = ' '.join(words[start:end])
        chunks.append(chunk)
        start += chunk_size_words - overlap_words
    return chunks



if __name__ == '__main__':
    test= 'CGTGGTCCAAAGCAACCCCGACTACCTGAACGTGGCCAACTATGCGCCCACTCCTGCTCCTTGGTGGTTCAACTGAACTGTGGACAGCAATTAGCATAAATTAGGTTGTA'
    print(wrap_text(test))

    print(format_string_with_numbers('toto', 10, 'bob', 12, padding_for_all_numbers=3)) # toto_010_bob_012
    print(format_string_with_numbers('toto', 0.33255121121, 'bob', 12, padding_for_all_numbers=0.2)) # toto_0.33_bob_12.00
    print(format_string_with_numbers('toto', 0.33255121121, 'bob', 12, padding_for_all_numbers='{:0>9.2f}')) # toto_000000.33_bob_000012.00

    # Example usage:
    input_string = "Hello txt, replace TXT with python."
    old_substring = "txt"
    new_substring = "<i>python</i>"

    result = replace_ignore_case(input_string, old_substring, new_substring)
    print(result)
    # print(help(tools))
    # print(format_string_with_numbers.__doc__)
    # help(format_string_with_numbers)