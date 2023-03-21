import os
import json

# for latex code
import matplotlib.pyplot as plt
from io import BytesIO

from PyQt5.QtWidgets import QDialog, QLabel, QPushButton, QVBoxLayout

PATHGUISETUP = os.path.join(os.getcwd(), "gui_setup.json")


def create_setup_file():
    if not os.path.exists(PATHGUISETUP):
        geometry_of_window = {"width": 800,
                              "height": 500,
                              "x_pos": 500,
                              "y_pos": 500
                              }
        basic_setup = {"Rainbow": geometry_of_window,
                       "Rainbow: Definition der Geometrie": geometry_of_window,
                       "Rainbow: Definition der Öffnungen": geometry_of_window,
                       "Rainbow: Berechnungsparameter": geometry_of_window,
                       "Rainbow: Festlegen der Stützstellen": geometry_of_window,
                       "Rainbow: Definition von Einzellasten": geometry_of_window,
                       "Rainbow: Impressum": {"width": 326,
                                              "height": 266,
                                              "x_pos": 499,
                                              "y_pos": 519
                                                },
                       "basic_color_code": {
                           "1": [
                               "#0008fc",
                               "F.P"
                           ],
                           "2": [
                               "#eb4034",
                               "L.U.E"
                           ],
                           "3": [
                               "#ebba34",
                               "L.O.E"
                           ],
                           "4": [
                               "#bdeb34",
                               "R.U.E"
                           ],
                           "5": [
                               "#34eb5c",
                               "R.O.E"
                           ],
                           "6": [
                               "#34d3eb",
                               "R.U"
                           ],
                           "7": [
                               "#a534eb",
                               "R.L"
                           ],
                           "8": [
                               "#bada55",
                               "R.O"
                           ],
                           "9": [
                               "#ff80ed",
                               "R.R"
                           ],
                           "11": [
                               "#f1c806",
                               "\u00d6.U.L"
                           ],
                           "12": [
                               "#053410",
                               "\u00d6.U.R"
                           ],
                           "13": [
                               "#98c472",
                               "\u00d6.O.L"
                           ],
                           "14": [
                               "#9b7b64",
                               "\u00d6.O.R"
                           ],
                           "15": [
                               "#eb610d",
                               "Anker Kunstoff Materialgesetz 1"
                           ],
                           "16": [
                               "#806da1",
                               "Anker Kunstoff Materialgesetz 2"
                           ],
                           "17": [
                               "#72881f",
                               "Anker Metall Materialgesetz 1"
                           ],
                           "18": [
                               "#95a5c6",
                               "Anker Metall Materialgesetz 2"
                           ],
                       },

                       "Materialgesetze":
                           {
                               "Kunsstoff1" : {
                                   "Cx": 1000,
                                   "Cy": 1000,
                               },

                               "Kunsstoff2": {
                                   "Cx": 5000,
                                   "Cy": 5000
                               },

                               "Metall1": {
                                   "u": [5],
                                   "F": [7]
                               },

                               "Metall2": {
                                   "u": [5],
                                   "F": [7]
                               }
                           },

                       "DefaultSettings":
                           {
                               "Emod":  30,
                               "Tempdiff": 50,
                               "Tempgrad": 5,
                               "Alphat": 0.00001
                           }

                       }
        with open(PATHGUISETUP, "w") as file:
            json.dump(basic_setup, file, indent=4)


def change_setup_file(window_name: str, width: int, height: int, x_pos: int, y_pos: int):
    create_setup_file()
    with open(PATHGUISETUP, "r") as file:
        data = json.load(file)
    data[window_name] = {"width": width,
                         "height": height,
                         "x_pos": x_pos,
                         "y_pos": y_pos + 50
                         }
    with open(PATHGUISETUP, "w") as file:
        json.dump(data, file, indent=4)


def get_setup_file(window_name: str):
    with open(PATHGUISETUP, "r") as file:
        data = json.load(file)
    return data[window_name]

def change_settings(settings_name: str, value):
    with open(PATHGUISETUP, "r") as file:
        data = json.load(file)
    data[settings_name] = value
    with open(PATHGUISETUP, "w") as file:
        json.dump(data, file, indent=4)

class RainbowErrorDialog(QDialog):

    def __init__(self, problem_text: str, dialog_title: str = "Rainbow: Error"):
        super().__init__()

        self.setWindowTitle(dialog_title)

        l_problem_text = QLabel(problem_text)

        button_close = QPushButton("Close")
        button_close.clicked.connect(self.close)

        main_layout = QVBoxLayout()
        main_layout.addWidget(l_problem_text)
        main_layout.addWidget(button_close)

        self.setLayout(main_layout)
        self.show()


def tex2svg(formula, fontsize=11, dpi=300):
    """Render TeX formula to SVG.
    Args:
        formula (str): TeX formula.
        fontsize (int, optional): Font size.
        dpi (int, optional): DPI.
    Returns:
        str: SVG render.
    """

    fig = plt.figure(figsize=(0.01, 0.01))
    fig.text(0, 0, r'${}$'.format(formula), fontsize=fontsize)

    output = BytesIO()
    fig.savefig(output, dpi=dpi, transparent=True, format='svg',
                bbox_inches='tight', pad_inches=0.0)
    plt.close(fig)

    output.seek(0)
    return output.read()

if __name__ == '__main__':
    change_setup_file("start_window", 5, 3, 7, 9)
