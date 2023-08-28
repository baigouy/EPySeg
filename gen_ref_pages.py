# TODO -> make it execute the command and open the file maybe --> think about it or rather to a shell script that calls this then runs the other command

import os.path
import sys
import traceback
from pathlib import Path
import shutil
import mkdocs_gen_files

# maybe add
# options:
# show_source: false
# heading_level: 2

from epyseg.ta.tracking.tools import smart_name_parser

nav = mkdocs_gen_files.Nav()

# Get the directory path of the current file
current_file_dir = os.path.dirname(os.path.abspath(__file__))

try:
    # Use the shutil.rmtree() function to delete the folder and its contents
    shutil.rmtree(current_file_dir+'/docs/epyseg_pkg')
except:
    pass

# copy README.md to /docs/index.md
shutil.copy(current_file_dir+'/README.md', current_file_dir+'/docs/'+'index.md')
shutil.copy(smart_name_parser(current_file_dir,'parent')+'/tissue_analyzer'+'/README.md', current_file_dir+'/docs/TA/README.md')
# sys.exit(0)

current_file_dir+='/epyseg'

parent = smart_name_parser(smart_name_parser(current_file_dir,'parent'),'parent')


for path in sorted(Path(current_file_dir).rglob("*.py")):
    module_path = path.relative_to(parent).with_suffix("")
    doc_path = path.relative_to(parent).with_suffix(".md")
    # doc_path = path.relative_to(".").with_suffix(".md")
    full_doc_path = Path(".", doc_path)
    # print(full_doc_path)

    # print(doc_path.as_posix())

    # print(full_doc_path)

    parts = tuple(module_path.parts)

    if parts[-1] == "__init__":
        parts = parts[:-1]
    elif parts[-1] == "__main__":
        continue
    elif parts[-1] == "setup":
        continue
    try:
        nav[parts] = doc_path.as_posix()  #

        # print(full_doc_path.absolute())
        #
        # if os.path.exists(full_doc_path.absolute()):
        #     print(full_doc_path.absolute())
        #     os.remove(full_doc_path.absolute())

        # print(".".join(parts))
        # print(full_doc_path.absolute())
        if True:
            with mkdocs_gen_files.open(full_doc_path, "w") as fd:
                ident = ".".join(parts)
                fd.write(f"::: {ident}")

        mkdocs_gen_files.set_edit_path(full_doc_path, path)
    except:
        traceback.print_exc()

with mkdocs_gen_files.open("reference.md", "w") as nav_file:  #
    nav_file.writelines(nav.build_literate_nav())  #
