import traceback

# have nice colored command line outputs (can be very useful when using command line tools)
# http://en.wikipedia.org/wiki/ANSI_escape_code

STRING_TO_ANSI = {
        'black': 30,
        'red': 31,
        'green': 32,
        'yellow': 33,
        'blue': 34,
        'magenta': 35,
        'cyan': 36,
        'white': 37,
       }

__ANSI_STYLEs = {
        'Bold':1,
        'Faint':2,
        'Italic':3,
        'Underline':4,
        'B': 1,
        'F': 2,
        'I': 3,
        'U': 4,
        '<B>': 1, # check if this is html
        '<I>': 3,
        '<U>': 4,
        }

# add string to bg
# adds the letter values of ANSI
# adds bright colors to ANSI
__tmp = {}
for color, value in STRING_TO_ANSI.items():
    __tmp[color[0]]=value
    __tmp['l'+color[0]] = value + 60
    __tmp['b'+color[0]] = value + 60
    __tmp['bg' + color[0]] = value + 10
    __tmp['lbg' + color[0]] = value + 60 + 10
    __tmp['bbg' + color[0]] = value + 60 + 10

# add light or bright colors
for color, value in STRING_TO_ANSI.items():
    __tmp['light_'+color]=value + 60
    __tmp['bright_'+color]=value + 60
    __tmp['foreground_' + color] = value
    __tmp['fg' + color] = value
    __tmp['background_' + color] = value + 10
    __tmp['bg' + color] = value + 10
    __tmp['light_background_' + color] = value + 60 + 10
    __tmp['bright_background_' + color] = value + 60 + 10
    __tmp['bg' + color] = value + 10
    __tmp['lbg' + color] = value + 60 + 10
    __tmp['bbg' + color] = value + 60 + 10

STRING_TO_ANSI.update(__tmp)
del __tmp
STRING_TO_ANSI.update(__ANSI_STYLEs)

def demo_all_ansi():
    """
    Demonstrates ANSI color codes by printing each color code with its corresponding number.

    This function iterates through ANSI color codes (0-108) and prints each code along with its number
    to demonstrate how different colors can be displayed using ANSI escape sequences.

    Args:
        None

    Returns:
        None
        ...
    """
    import sys
    for i in range(11):
        for j in range(10):
            n = 10 * i + j
            if n > 108:
                break
            sys.stdout.write(f"\x1b[{n}m {n:3d}\x1b[0m")
        print()

def create_ansi(ansi_value, text=''):
    """
     Create ANSI-formatted text by applying ANSI color codes to the input text.

     This function takes an ANSI color code or a list of codes and applies them to the input text.
     It returns the text with the specified color codes applied.

     Args:
         ansi_value (int, str, or list): ANSI color code(s) to apply. Can be an integer, a string
             representing a color name, or a list of integers/strings.
         text (str, optional): The input text to which ANSI codes will be applied. Defaults to an empty string.

     Returns:
         str: The input text with ANSI color codes applied.

     """
    if isinstance(ansi_value, (int,str)):
        ansi_value = [ansi_value]
    output=""
    for ansi in ansi_value:
        if isinstance(ansi, str):
            try:
                ansi = STRING_TO_ANSI[ansi]
            except:
                print('unsupported ANSI',ansi)
                traceback.print_exc()
                continue
        output+=f'\x1b[{ansi}m'
    output+=text
    output+="\x1b[0m"
    return output

def print_raw(text):
    """
      Print text raw without special characters interpreted.

      This function prints the input text without interpreting special characters or ANSI escape sequences.
      It allows viewing the ANSI text as a raw string rather than as colored text.

      Args:
          text (str): The input text to print.

      Returns:
          None

      """
    print(repr(text))

if __name__ == '__main__':
    # prompt = input(create_ansi('g', '>>>You:\n'))
    # print(create_ansi('br', "\n>>>ChatGPT:\n" + prompt))

    print("Hello\x1b[K") # Clear line
    print("\x1b[2J\x1b[HCleared screen") # Clear screen

    # VERY GOOD −−> MAKE USE OF THAT CAUSE VERY USEFUL WITH THE COMMANDE LINE

    print(create_ansi(9, 'test')) # --> barre le texte
    print_raw(create_ansi(9, 'test')) # '\x1b[9mtest\x1b[0m'

    print(create_ansi(37, 'test'))  # --> en gris
    print_raw(create_ansi(37, 'test'))  # --> '\x1b[37mtest\x1b[0m'

    # combinaison des deux
    print('\x1b[37m\x1b[9mtest\x1b[0m') #very good so it's fairly easy to combine

    print(create_ansi([9,37,52], 'test')) # cree un text encadré # peut etre faire pareil avec des strings et aussi du html to be faster
    # I could convert html to ansi --> would be cool too
    # print(create_ansi([9, 'bright_yellow', 52],'test'))  # cree un text encadré # peut etre faire pareil avec des strings et aussi du html to be faster
    print(create_ansi([9, 'ly', 52],'test'))  # cree un text encadré # peut etre faire pareil avec des strings et aussi du html to be faster
    print(create_ansi([9, 'lbgr','ly', 52],'test'))  # cree un text encadré jaune surligné en rouge
    print(create_ansi(['B','lbgr','ly'],'test'))  # cree un text encadré jaune surligné en rouge
    print(create_ansi([9, 'B'],'test'))  # cree un text encadré jaune surligné en rouge
    print(create_ansi([9, 'I'],'test'))  # cree un text encadré jaune surligné en rouge
    print(create_ansi([9, 'I','B'],'test')) # bold italic barre

    demo_all_ansi()