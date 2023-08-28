import re

class RegexChecker:
    @staticmethod
    def is_regex(string):
        '''
        Checks if a string is a regular expression.

        Args:
            string (str): The input string to be checked.

        Returns:
            bool: True if the string is a regular expression, False otherwise.

        Examples:
            >>> RegexChecker.is_regex('abc')
            False
            >>> RegexChecker.is_regex('[a-z]+')
            True
            >>> RegexChecker.is_regex(r'abc')
            False
            >>> RegexChecker.is_regex(r'[a-z]+')
            True
        '''
        if any(c in string for c in ['[', ']', '(', ')', '{', '}', '.', '*', '+', '?', '|', '\\']):
            pattern = re.compile(string)
            return True
        else:
            return False

if __name__ == '__main__':
    print('is regex', RegexChecker.is_regex('abc'))  # Output: False
    print('is regex', RegexChecker.is_regex('[a-z]+'))  # Output: True
    print('is regex', RegexChecker.is_regex(r'abc'))  # Output: False
    print('is regex', RegexChecker.is_regex(r'[a-z]+'))  # Output: True
