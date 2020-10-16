import webbrowser
from PyQt5.QtCore import QPoint
from PyQt5.QtWidgets import QToolTip
from markdown import markdown

import os

def open_tmp_web_page(html):
    import tempfile
    import webbrowser
    with tempfile.NamedTemporaryFile('w', delete=True, suffix='.html') as f:
        url = 'file://' + f.name
        f.write(html)
        webbrowser.open(url)

def markdown_file_to_html(filepath):
    path = filepath
    if not os.path.exists(path):
        this_dir, _ = os.path.split(__file__)
        path = os.path.join(this_dir, path)
        # print(path, os.path.exists(path))
    data = ''
    with open(path, 'r') as file:
        data = file.read()
    return markdown_to_html(data)

def markdown_to_html(text):
    html = markdown(text)
    return html

def open_md(file):
    # file2display = file
    if not os.path.exists(file):
        # print(file2display)
        open_tmp_web_page(markdown_file_to_html(file))

def browse_tip(file_or_text='unknown help'):
    if file_or_text.lower().endswith('.md') and not file_or_text.lower().startswith('http'):
        open_md(file_or_text)
    else:
        webbrowser.open(file_or_text)

