from colorful.ansi import rgb_to_ansi16


class ColoredPrompt:
    COLORS = {
        'black': '\033[30m',
        'red': '\033[31m',
        'green': '\033[32m',
        'yellow': '\033[33m',
        'blue': '\033[34m',
        'magenta': '\033[35m',
        'cyan': '\033[36m',
        'white': '\033[37m',
        'bright_black': '\033[90m',
        'bright_red': '\033[91m',
        'bright_green': '\033[92m',
        'bright_yellow': '\033[93m',
        'bright_blue': '\033[94m',
        'bright_magenta': '\033[95m',
        'bright_cyan': '\033[96m',
        'bright_white': '\033[97m',
        'b': '\033[30m',
        'r': '\033[31m',
        'g': '\033[32m',
        'y': '\033[33m', # plus kaki que yellow
        'b': '\033[34m',
        'm': '\033[35m',
        'c': '\033[36m',
        'w': '\033[37m', #it's rather gray --> but that's easily visible
    }
    RESET = '\033[0m'

    @staticmethod
    def get(prompt, color='red'):
        if color is None or color=='':
            color='\033[30m'
        if color.startswith('\033['):
            color_code = color
        elif color.startswith('#'):
            # convert HTML color to ansi
            r,g,b = ColoredPrompt.html_to_rgb(color)
            ansi_color=rgb_to_ansi16(r, g, b)

            L = (0.2126 * r + 0.7152 * g + 0.0722 * b)/255.
            # print('L',L)
            if L>0.5:
                ansi_color+=60

            print(ansi_color)

            color_code = '\033['+str(ansi_color)+'m'

        else:
            color_code = ColoredPrompt.COLORS.get(color.lower(), ColoredPrompt.COLORS[color.lower()])
        return f"{color_code}{prompt}{ColoredPrompt.RESET}"

    @staticmethod
    def html_to_rgb(html_color):
        # Remove the '#' character from the input string
        html_color = html_color.lstrip('#')

        # Convert the hex string to RGB values
        r = int(html_color[0:2], 16)
        g = int(html_color[2:4], 16)
        b = int(html_color[4:6], 16)

        return r, g, b

if __name__ == '__main__':

    for color in ColoredPrompt.COLORS.keys():
        print(color,"-->", ColoredPrompt.get("Enter your command: ", color))

    print(color,"-->", ColoredPrompt.get("Enter your command: ", '\033[30m'))
    print(color,"-->", ColoredPrompt.get("Enter your command: ", '#FFFFFF'))
    print(color,"-->", ColoredPrompt.get("Enter your command: ", '#FF0000'))
    print(color,"-->", ColoredPrompt.get("Enter your command: ", '#00FF00'))
    print(color,"-->", ColoredPrompt.get("Enter your command: ", '#0000FF'))
