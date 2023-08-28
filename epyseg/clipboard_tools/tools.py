import pyperclip


def to_clip(txt):
    pyperclip.copy(txt)

def get_clip():
    return pyperclip.paste()