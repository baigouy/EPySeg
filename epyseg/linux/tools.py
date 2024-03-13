import os
import subprocess
import shutil
import traceback

from epyseg.ta.tracking.tools import smart_name_parser


def create_linux_desktop_shortcut(script_path, shortcut_name, shortcut_description="", icon_path="", store_in_application_folder=True):
    # MEGA TODO --> improve the code --> if the icon already exists in the folder --> change its name or offer overwrite it or offer rename it and apply the change --> appart from that not so bad !!!

    if not store_in_application_folder:
        try:
            desktop_folder_name = subprocess.check_output(["xdg-user-dir", "DESKTOP"], universal_newlines=True).strip()
        except subprocess.CalledProcessError:
            # Fallback to a default name if the command fails
            desktop_folder_name = "Desktop"
        # Fallback to a default name if the command fails
        desktop_path = os.path.join(os.path.expanduser("~"), desktop_folder_name)
        desktop_file_path = f"{desktop_path}/{shortcut_name}.desktop"
    else:
        desktop_file_path = f"{os.path.join(os.path.expanduser('~'),'.local/share/applications/')}/{shortcut_name}.desktop"
    neo_icon_path=None

    try:
        if icon_path is not None and icon_path and 'applications' in desktop_file_path:
            neo_icon_path = smart_name_parser(desktop_file_path, 'parent') + '/icons'
            neo_icon_path = os.path.join(neo_icon_path, smart_name_parser(icon_path, 'short'))
            os.makedirs(os.path.dirname(neo_icon_path), exist_ok=True)
            try:
                if icon_path != neo_icon_path: # no need to duplicate the image if already there
                    shutil.copy2(icon_path, neo_icon_path)
            except:
                traceback.print_exc()
    except:
        traceback.print_exc()

    content = f"""[Desktop Entry]
Version=1.0
Type=Application
Terminal=true
Exec={script_path}
Name={shortcut_name}
Comment={shortcut_description}
Icon={icon_path if neo_icon_path is None else neo_icon_path}
"""

    with open(desktop_file_path, "w") as desktop_file:
        desktop_file.write(content)



    print(f"Desktop shortcut created at: {desktop_file_path}")
    # move to application folder and run change stuff


if __name__ == '__main__':

    # TODO ADD 'sudo lshw -C display' --> command to show nvidia
    # sudo ubuntu-drivers install ? https://www.cyberciti.biz/faq/ubuntu-linux-install-nvidia-driver-latest-proprietary-driver/

    # Example usage:
    script_path = "/home/aigouy/miniconda3/bin/python /home/aigouy/mon_prog/Python/epyseg_pkg/personal/hiC_microC_tmp/EVO_BLASTER_GUI.py"
    shortcut_name = "EVO Blaster"
    shortcut_description = "Runs local blast and identifies the most conserved gene in every species"
    # icon_path = "/home/aigouy/Téléchargements/logo1.png"
    icon_path = "/home/aigouy/.local/share/applications/icons/logo1.png"

    #/home/aigouy/.local/share/applications/


    create_linux_desktop_shortcut(script_path, shortcut_name, shortcut_description, icon_path)
