import json
import sys

# Setting Backend for matplotlib
import matplotlib.pyplot as plt
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QIcon, QPixmap
from PyQt5.QtSvg import QSvgWidget
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QHBoxLayout, QVBoxLayout, QLabel, QLineEdit, QComboBox, \
    QGridLayout, QSpinBox, QTableWidget, QHeaderView, QCheckBox, QFileDialog, QGroupBox

from Gui.gui_tools import create_setup_file, change_setup_file, get_setup_file, RainbowErrorDialog, tex2svg
from rainbow.Datacontroller import setupData, loadData, Wall
from rainbow.SolveGrid import calculate

version = "1.0 "


class BasicWindowRainbow(QWidget):

    def __init__(self, name: str):
        self.name = name
        super().__init__()
        self.setWindowTitle(version + name)
        self.setWindowIcon(QIcon("Gui/images/regenbogen.png"))

    def closeEvent(self, event):
        change_setup_file(self.name, self.width(), self.height(), self.pos().x(), self.pos().y())
        event.accept()

    def load_window_settings(self):
        data = get_setup_file(self.name)
        self.setGeometry(data["x_pos"], data["y_pos"], data["width"], data["height"])


class WelcomeWindow(BasicWindowRainbow):

    def __init__(self):
        super().__init__("Rainbow")

        # Data (Start) ################################################################################################

        self.DATA = setupData()

        # window basics ################################################################################################
        self.Settingswindow = None
        self.Defwindow = None
        # Buttons ######################################################################################################

        pb_start = QPushButton("Start")
        pb_start.clicked.connect(self.start_rainbow)

        pb_template = QPushButton("Auswahl einer Vorlage")
        pb_template.clicked.connect(self.chose_template)

        pb_name = QPushButton("Impressum")
        pb_name.clicked.connect(self.settings)

        # Images #######################################################################################################

        pm_welcome = QPixmap("Gui/images/Startbild.png")
        pm_tu_logo = QPixmap("Gui/images/TUK_LOGO_FELD_OBEN_4C.jpg")
        scale_tu_logo = pm_tu_logo.scaled(256, 64)

        l_welcome_image = QLabel()
        l_welcome_image.setPixmap(pm_welcome)
        l_welcome_image.setAlignment(Qt.AlignCenter)

        l_tu_logo_image = QLabel()
        l_tu_logo_image.setPixmap(scale_tu_logo)

        # Layout #######################################################################################################

        v_box_buttons = QVBoxLayout()
        v_box_buttons.addWidget(pb_start)
        v_box_buttons.addWidget(pb_template)
        v_box_buttons.addWidget(pb_name)

        h_box = QHBoxLayout()
        h_box.addWidget(l_tu_logo_image)
        h_box.addLayout(v_box_buttons)

        self.main_layout = QVBoxLayout()
        self.main_layout.addWidget(l_welcome_image)
        self.main_layout.addLayout(h_box)

        self.setLayout(self.main_layout)

        ################################################################################################################
        create_setup_file()
        self.load_window_settings()
        self.show()

    def start_rainbow(self):
        if self.Settingswindow is not None:
            self.DATA["Kunststoff1"] = self.Settingswindow.data["Kunststoff1"]
            self.DATA["Kunststoff2"] = self.Settingswindow.data["Kunststoff2"]

        self.close()
        self.Defwindow = DefinitionOfTheGeometry(data=self.DATA)

    def chose_template(self):
        fname, _ = QFileDialog.getOpenFileName(self, 'Auswahl der Vorlage', filter="*.json")
        l_file_name = QLabel("Vorlage: " + fname)
        self.main_layout.addWidget(l_file_name)
        self.DATA = loadData(fname)
    
    def settings(self):
        self.wind = RainbowName()

class RainbowName(BasicWindowRainbow):

    def __init__(self):
        super().__init__(name="Rainbow: Impressum")
        
        line_1 = QLabel("Technische Universität Kaiserslautern")
        line_2 = QLabel("Fachbereich Bauingenieurwesen")
        line_3 = QLabel("Fachgebiet Massivbau und Baukonstruktion")
        line_4 = QLabel("Professur Baukonstruktion und Fertigteilbau")
        line_empty = QLabel("")
        line_5 = QLabel("Stefan Carstens, M.Eng.")
        line_6 = QLabel("Dipl.-Ing. Fabian Penkert")

        layout_main = QVBoxLayout()
        layout_main.addWidget(line_1)
        layout_main.addWidget(line_2)
        layout_main.addWidget(line_3)
        layout_main.addWidget(line_4)
        layout_main.addWidget(line_empty)
        layout_main.addWidget(line_5)
        layout_main.addWidget(line_6)

        self.setLayout(layout_main)
        self.load_window_settings()
        self.show()

class DefinitionOfTheGeometry(BasicWindowRainbow):

    def __init__(self, data: dict):
        super().__init__(name="Rainbow: Definition der Geometrie")

        self.DATA = data

        # window basics ################################################################################################
        self.open_window = None
        self.error_window = None
        self.cal_para_window = None

        # Input ########################################################################################################
        # print(self.DATA["main"]["laenge"])
        self.ql_length = QSpinBox()
        self.ql_length.setMaximum(2147483647)
        self.ql_length.setValue(self.DATA["main"]["laenge"])

        self.ql_height = QSpinBox()
        self.ql_height.setMaximum(2147483647)
        self.ql_height.setValue(self.DATA["main"]["hoehe"])

        self.ql_thick_facing_layer = QSpinBox()
        self.ql_thick_facing_layer.setMaximum(2147483647)
        self.ql_thick_facing_layer.setValue(self.DATA["vorsatzschale"]["dicke"])

        self.ql_thick_core_layer = QSpinBox()
        self.ql_thick_core_layer.setMaximum(2147483647)

        self.ql_thick_support_layer = QSpinBox()
        self.ql_thick_support_layer.setMaximum(2147483647)
        self.ql_thick_support_layer.setValue(self.DATA["tragschale"]["dicke"])

        # Labels #######################################################################################################

        l_length = QLabel("Länge in mm")
        l_height = QLabel("Höhe in mm")
        l_thick_facing_layer = QLabel("Dicke der Vorsatzschale in mm")
        l_thick_core_layer = QLabel("Dicke der Kernschicht in mm")
        l_thick_support_layer = QLabel("Dicke der Tragschale in mm")

        # Units ########################################################################################################

        unit_length = QSvgWidget()
        unit_length.load(tex2svg(r"L \enspace in \enspace mm"))

        unit_height = QSvgWidget()
        unit_height.load(tex2svg(r"H \enspace in \enspace mm"))

        # Buttons ######################################################################################################

        pb_set_openings = QPushButton("Definition der Öffnungen")
        pb_set_openings.clicked.connect(self.set_opening)

        pb_check_settings_go_next = QPushButton("Weiter")
        pb_check_settings_go_next.clicked.connect(self.check_values)

        # Layout #######################################################################################################

        v_box_input = QVBoxLayout()
        v_box_input.addWidget(self.ql_length)
        v_box_input.addWidget(self.ql_height)
        v_box_input.addWidget(self.ql_thick_facing_layer)
        
        v_box_input.addWidget(self.ql_thick_support_layer)

        v_box_label = QVBoxLayout()

        length_box = QHBoxLayout()
        length_box.addWidget(l_length)
        v_box_label.addLayout(length_box)

        height_box = QHBoxLayout()
        height_box.addWidget(l_height)
        v_box_label.addLayout(height_box)

        v_box_label.addWidget(l_thick_facing_layer)
        v_box_label.addWidget(l_thick_support_layer)

        h_box_input_with_label = QHBoxLayout()
        h_box_input_with_label.addLayout(v_box_label)
        h_box_input_with_label.addLayout(v_box_input)

        h_box_buttons = QHBoxLayout()
        h_box_buttons.addWidget(pb_set_openings)
        h_box_buttons.addWidget(pb_check_settings_go_next)

        main_layout = QVBoxLayout()
        main_layout.addLayout(h_box_input_with_label)
        main_layout.addLayout(h_box_buttons)

        self.setLayout(main_layout)
        self.load_window_settings()
        self.show()

    def set_opening(self):
        if self.open_window is not None:
            self.DATA["main"]["oeffnung"] = self.open_window.DATA

        self.open_window = DefinitionOfOpenings(self.DATA["main"]["oeffnung"])

    def check_values(self):

        if self.ql_height.value() == 0:
            self.error_window = RainbowErrorDialog("Ungültige Höhe")
            return

        if self.ql_length.value() == 0:
            self.error_window = RainbowErrorDialog("Ungültige Länge")
            return

        if self.ql_thick_facing_layer.value() == 0:
            self.error_window = RainbowErrorDialog("Ungültige Dicke der Vorsatzschale")
            return


        if self.ql_thick_support_layer.value() == 0:
            self.error_window = RainbowErrorDialog("Ungültige Dicke der Tragschale")
            return
        self.DATA["main"]["hoehe"] = self.ql_height.value()
        self.DATA["main"]["laenge"] = self.ql_length.value()
        self.DATA["vorsatzschale"]["dicke"] = self.ql_thick_facing_layer.value()
        self.DATA["tragschale"]["dicke"] = self.ql_thick_support_layer.value()

        if self.open_window is not None:
            self.DATA["main"]["oeffnung"] = self.open_window.DATA

        # print(self.DATA)
        self.cal_para_window = DefinitionOfCalculationParametersWindow(self.DATA)
        self.close()


class DefinitionOfOpenings(BasicWindowRainbow):

    def __init__(self, data):
        super().__init__("Rainbow: Definition der Öffnungen")

        self.DATA = data
        self.current_id_opening = 0

        # Table #####################################################################################################

        self.main_table = QTableWidget()
        self.main_table.setColumnCount(4)
        self.main_table.setHorizontalHeaderLabels(["Linke untere Ecke x:",
                                                   "Linke untere Ecke y:",
                                                   "Länge der Öffnung:",
                                                   "Höhe der Öffnung:"])

        # Buttons ######################################################################################################

        pb_add_new = QPushButton("+")
        pb_add_new.clicked.connect(self.add_new_point)

        pb_del_point = QPushButton("-")
        pb_del_point.clicked.connect(self.remove_point)

        pb_finish = QPushButton("Überprüfen der Eingaben")
        pb_finish.clicked.connect(self.check_input)

        # Layout #######################################################################################################

        l_controll = QVBoxLayout()
        l_controll.addWidget(pb_add_new)
        l_controll.addWidget(pb_del_point)
        l_controll.addWidget(pb_finish)

        main_layout = QHBoxLayout()
        main_layout.addWidget(self.main_table)
        main_layout.addLayout(l_controll)
        self.setLayout(main_layout)

        self.load_points()
        self.load_window_settings()
        self.show()

    def load_points(self):
        for opening in self.DATA:
            # adding new Row for all existing openings
            row_c = self.main_table.rowCount()
            self.main_table.setRowCount(row_c + 1)

            x_pos = QSpinBox()
            x_pos.setMaximum(2147483647)
            x_pos.setValue(opening["xstart"])
            self.main_table.setCellWidget(row_c, 0, x_pos)  # set a new row

            y_pos = QSpinBox()
            y_pos.setMaximum(2147483647)
            y_pos.setValue(opening["ystart"])
            self.main_table.setCellWidget(row_c, 1, y_pos)

            length = QSpinBox()
            length.setMaximum(2147483647)
            length.setValue(opening["xlaenge"])
            self.main_table.setCellWidget(row_c, 2, length)

            height = QSpinBox()
            height.setMaximum(2147483647)
            height.setValue(opening["ylaenge"])
            self.main_table.setCellWidget(row_c, 3, height)

        # stretch table size so you can read content
        head = self.main_table.horizontalHeader()
        head.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        head.setSectionResizeMode(1, QHeaderView.ResizeToContents)
        head.setSectionResizeMode(2, QHeaderView.ResizeToContents)
        head.setSectionResizeMode(3, QHeaderView.Stretch)

    def add_new_point(self):
        # add new Row to table
        row_c = self.main_table.rowCount()  # first set new row count
        self.main_table.setRowCount(row_c + 1)

        x_pos = QSpinBox()
        x_pos.setMaximum(2147483647)
        self.main_table.setCellWidget(row_c, 0, x_pos)  # set a new row

        y_pos = QSpinBox()
        y_pos.setMaximum(2147483647)
        self.main_table.setCellWidget(row_c, 1, y_pos)

        length = QSpinBox()
        length.setMaximum(2147483647)
        self.main_table.setCellWidget(row_c, 2, length)

        height = QSpinBox()
        height.setMaximum(2147483647)
        self.main_table.setCellWidget(row_c, 3, height)

        # stretch table size so you can read content
        head = self.main_table.horizontalHeader()
        head.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        head.setSectionResizeMode(1, QHeaderView.ResizeToContents)
        head.setSectionResizeMode(2, QHeaderView.ResizeToContents)
        head.setSectionResizeMode(3, QHeaderView.Stretch)

    def remove_point(self):
        self.main_table.removeRow(self.main_table.currentRow())

    def check_input(self):
        # first remove old Data
        self.DATA = []

        # fill DATA with Table content
        for k in range(self.main_table.rowCount()):
            self.DATA.append({
                "xstart": self.main_table.cellWidget(k, 0).value(),
                "ystart": self.main_table.cellWidget(k, 1).value(),
                "xlaenge": self.main_table.cellWidget(k, 2).value(),
                "ylaenge": self.main_table.cellWidget(k, 3).value()
            })

        self.close()


class DefinitionOfCalculationParametersWindow(BasicWindowRainbow):

    def __init__(self, data: dict):
        super().__init__(name="Rainbow: Berechnungsparameter")

        # Data #########################################################################################################

        self.DATA = data
        # self.visualizer = GridVisualizer({})
        self.support_point_window = None

        self.error_dialog = None
        self.show_grid_window = None
        self.def_window = None

        # Labels #######################################################################################################

        l_count_support_point_m = QLabel("Anzahl der Stützstellen vertikal")
        l_count_support_point_n = QLabel("Anzahl der Stützstellen horizontal")
        l_plastic_1 = QLabel("Steigikeit Verbindungsmittel")
        l_plastic_1_cx = QLabel("Cx in N / mm: ")
        l_plastic_1_cy = QLabel("Cy in N / mm: ")
        l_plastic_1_cz = QLabel("Cz in N / mm: ")
        l_Panel_construction = QLabel("Lagerungsart der Vorsatzschale")

        # Spinbox ######################################################################################################

        self.sb_count_support_point_m = QSpinBox()
        self.sb_count_support_point_m.setMaximum(2147483647)
        self.sb_count_support_point_m.setValue(self.DATA["main"]["gittervertikal"])

        self.sb_count_support_point_n = QSpinBox()
        self.sb_count_support_point_n.setMaximum(2147483647)
        self.sb_count_support_point_n.setValue(self.DATA["main"]["gitterhorizontal"])

        self.ql_plastic_1_x = QLineEdit(str(self.DATA["Kunststoff1"]["Cx"]))
        self.ql_plastic_1_y = QLineEdit(str(self.DATA["Kunststoff1"]["Cy"]))
        self.ql_plastic_1_z = QLineEdit(str(self.DATA["Kunststoff1"]["Cz"]))

        # Buttons ######################################################################################################

        pb_check_settings = QPushButton("Weiter")
        pb_check_settings.clicked.connect(self.check_settings)

        pb_show_grid = QPushButton("Anzeigen des Gitters")
        pb_show_grid.clicked.connect(self.show_grid)

        pb_set_support_points = QPushButton("Festlegen der Anker")
        pb_set_support_points.clicked.connect(self.set_support_points)

        # Combobox #####################################################################################################

        self.cb_set_Panel_construction = QComboBox()
        self.cb_set_Panel_construction.addItems([#"Auswahl: Tafelbauweise",
                                                 "aufstehende Vorsatzschale",
                                                 "frei hängende Vorsatzschale"])
        index = self.cb_set_Panel_construction.findText(self.DATA["main"]["Tafelbauweise"], Qt.MatchFixedString)
        if index >= 0:
            self.cb_set_Panel_construction.setCurrentIndex(index)
        else:
            raise ValueError("In der Vorlage ist ein Fehler bei der Auswahl der Bauweise")

        # Layout #######################################################################################################

        main_layout = QGridLayout()

        main_layout.addWidget(l_count_support_point_m, 0, 0)
        main_layout.addWidget(self.sb_count_support_point_m, 0, 1)

        main_layout.addWidget(l_count_support_point_n, 1, 0)
        main_layout.addWidget(self.sb_count_support_point_n, 1, 1)

        main_layout.addWidget(l_Panel_construction, 2,0)
        main_layout.addWidget(self.cb_set_Panel_construction, 2, 1)

        main_layout.addWidget(l_plastic_1, 3, 0)
        main_layout.addWidget(l_plastic_1_cx, 4, 0)
        main_layout.addWidget(self.ql_plastic_1_x, 4, 1)
        main_layout.addWidget(l_plastic_1_cy, 5, 0)
        main_layout.addWidget(self.ql_plastic_1_y, 5, 1)
        main_layout.addWidget(l_plastic_1_cz, 6, 0)
        main_layout.addWidget(self.ql_plastic_1_z, 6, 1)

        main_layout.addWidget(pb_set_support_points, 7, 0)

        main_layout.addWidget(pb_show_grid, 8, 0)
        main_layout.addWidget(pb_check_settings, 8, 1)

        self.setLayout(main_layout)

        self.load_window_settings()
        self.show()

    def check_settings(self):
        if self.check_support_points_and_create_grid() == 0:
            return
        self.DATA["main"]["gittervertikal"] = self.sb_count_support_point_m.value()
        self.DATA["main"]["gitterhorizontal"] = self.sb_count_support_point_n.value()
        self.DATA["Kunststoff1"]["Cx"] = float(self.ql_plastic_1_x.text())
        self.DATA["Kunststoff1"]["Cy"] = float(self.ql_plastic_1_y.text())
        self.DATA["Kunststoff1"]["Cz"] = float(self.ql_plastic_1_z.text())
        if self.cb_set_Panel_construction.currentText() == "Auswahl: Tafelbauweise":
            self.error_dialog = RainbowErrorDialog("Bitte erst die Tafelbauweise festlegen")
            return

        if self.support_point_window is not None:
            self.DATA["main"]["Anker"] = self.support_point_window.DATA

        self.DATA["main"]["Tafelbauweise"] = self.cb_set_Panel_construction.currentText()

        self.def_window = DefinitionFrontWallWindow(self.DATA)
        self.close()

    def show_grid(self):
        if self.check_support_points_and_create_grid() == 0:
            return
        self.DATA["main"]["gittervertikal"] = self.sb_count_support_point_m.value()
        self.DATA["main"]["gitterhorizontal"] = self.sb_count_support_point_n.value()
        if self.support_point_window is not None:
            self.DATA["main"]["Anker"] = self.support_point_window.DATA

        wall = Wall(self.DATA)
        grid = wall.grid()

        fig, axs = plt.subplots(2)
        axs[0].imshow(grid)
        axs[0].set_xlabel("Gitterpunkte in x-Richtung", fontsize=12)
        axs[0].set_ylabel("Gitterpunkte in y-Richtung", fontsize=12)
        axs[1].set_aspect(1)
        axs[1].set_xlabel("Wandelement y-Richtung in mm", fontsize=12)
        axs[1].set_ylabel("Wandelement x-Richtung in mm", fontsize=12)

        for row in range(self.DATA["main"]["gittervertikal"]):
            for col in range(self.DATA["main"]["gitterhorizontal"]):
                if grid[row, col] != 0:
                    axs[1].plot(row * wall.deltay, col * wall.deltax, "bo")

        plt.show()


    def set_support_points(self):
        if self.support_point_window is not None:
            self.DATA["main"]["Anker"] = self.support_point_window.DATA
        self.support_point_window = DefinitionOfSupportPointsWindow(self.DATA["main"]["Anker"])

    def check_support_points_and_create_grid(self):
        if self.sb_count_support_point_m.value() == 0:
            self.error_dialog = RainbowErrorDialog("Bitte erst Anzahl an Stützstellen festlegen in m")
            return 0
        elif self.sb_count_support_point_n.value() == 0:
            self.error_dialog = RainbowErrorDialog("Bitte erst Anzahl an Stützstellen festlegen in n")
            return 0

class DefinitionOfSupportPointsWindow(BasicWindowRainbow):

    def __init__(self, data: list):
        super().__init__(name="Rainbow: Festlegen der Stützstellen")

        self.DATA = data

        # Table ########################################################################################################

        self.main_table = QTableWidget()

        self.main_table.setColumnCount(2)
        self.main_table.setHorizontalHeaderLabels(["X Pos", "Y Pos"])

        # Button #######################################################################################################

        pb_add_new_point = QPushButton("+")
        pb_add_new_point.clicked.connect(self.add_new_point)

        pb_del_point = QPushButton("-")
        pb_del_point.clicked.connect(self.del_point)

        pb_finish = QPushButton("Finish")
        pb_finish.clicked.connect(self.check_data)

        # Layout #######################################################################################################

        l_controll = QVBoxLayout()
        l_controll.addWidget(pb_add_new_point)
        l_controll.addWidget(pb_del_point)
        l_controll.addWidget(pb_finish)

        main_layout = QHBoxLayout()
        main_layout.addWidget(self.main_table)
        main_layout.addLayout(l_controll)

        self.load_points()
        self.setLayout(main_layout)
        self.show()

    def load_points(self):

        for point in self.DATA:

            row_c = self.main_table.rowCount()  # first set new row count
            self.main_table.setRowCount(row_c + 1)

            x_pos = QSpinBox()
            x_pos.setMaximum(2147483647)
            x_pos.setValue(point["x"])
            self.main_table.setCellWidget(row_c, 0, x_pos)

            y_pos = QSpinBox()
            y_pos.setMaximum(2147483647)
            y_pos.setValue(point["y"])
            self.main_table.setCellWidget(row_c, 1, y_pos)

    def add_new_point(self):
        # add new Row to table
        row_c = self.main_table.rowCount()  # first set new row count
        self.main_table.setRowCount(row_c + 1)

        x_pos = QSpinBox()
        x_pos.setMaximum(2147483647)
        self.main_table.setCellWidget(row_c, 0, x_pos)

        y_pos = QSpinBox()
        y_pos.setMaximum(2147483647)
        self.main_table.setCellWidget(row_c, 1, y_pos)

        head = self.main_table.horizontalHeader()
        head.setSectionResizeMode(0, QHeaderView.ResizeToContents)

        head.setSectionResizeMode(1, QHeaderView.Stretch)


    def del_point(self):
        # self.size_table += -1
        self.main_table.removeRow(self.main_table.currentRow())

    def check_data(self):
        self.DATA = []

        for k in range(self.main_table.rowCount()):
            self.DATA.append({"x": self.main_table.cellWidget(k, 0).value(),
                              "y": self.main_table.cellWidget(k, 1).value()})

        self.close()


class DefinitionFrontWallWindow(BasicWindowRainbow):

    def __init__(self, data: dict):
        super().__init__(name="Rainbow: Definition der Vorsatzschale")

        self.DATA = data
        self.choseload = None

        # Groupbox #####################################################################################################

        groupbox_facing = QGroupBox("Vorsatzschale")
        lay_facing = QGridLayout()
        groupbox_facing.setLayout(lay_facing)

        groupbox_support = QGroupBox("Tragschale")
        lay_support = QGridLayout()
        groupbox_support.setLayout(lay_support)

        # Labels facing ################################################################################################

        l_emod_facing = QLabel("Elastizitaetsmodul E in kN/m²")
        lay_facing.addWidget(l_emod_facing, 0, 0)

        # Labels support ###############################################################################################

        l_emod_support = QLabel("Elastizitaetsmodul E in kN/m²")
        lay_support.addWidget(l_emod_support, 0, 0)

        # Input facing  ################################################################################################

        self.sb_emod_facing = QSpinBox()
        self.sb_emod_facing.setMaximum(2147483647)
        self.sb_emod_facing.setValue(int(self.DATA["vorsatzschale"]["Elastizitaetsmodul"] / 1e3))
        lay_facing.addWidget(self.sb_emod_facing, 0, 1)


        # Input support ################################################################################################

        self.sb_emod_support = QSpinBox()
        self.sb_emod_support.setMaximum(2147483647)
        self.sb_emod_support.setValue(int(self.DATA["tragschale"]["Elastizitaetsmodul"] / 1e3))
        lay_support.addWidget(self.sb_emod_support, 0, 1)

        # Buttons ######################################################################################################

        pb_check = QPushButton("Weiter")
        pb_check.clicked.connect(self.check)

        # Layout #######################################################################################################


        main_layout = QGridLayout()
        main_layout.addWidget(groupbox_facing, 0, 0)
        main_layout.addWidget(groupbox_support, 0, 1)
        main_layout.addWidget(pb_check, 1, 1)

        self.setLayout(main_layout)
        self.show()

    def check(self):
        self.DATA["vorsatzschale"]["Elastizitaetsmodul"] = self.sb_emod_facing.value() * 1e3

        self.DATA["tragschale"]["Elastizitaetsmodul"] = self.sb_emod_support.value() * 1e3


        self.close()
        self.choseload = ChoseLoadWindow(self.DATA)


class ChoseLoadWindow(BasicWindowRainbow):

    def __init__(self, data: dict):
        super().__init__(name="Rainbow: Auswahl der Last")

        self.DATA = data
        self.checklist = {}
        self.settingsWindow = None

        # Labels #######################################################################################################

        l_wind = QLabel("Windbelastung")
        l_earth = QLabel("Erddruck")
        l_temp_grad = QLabel("Temperaturgradient")
        l_temp_diff = QLabel("Temperaturdifferenz")

        # Images #######################################################################################################

        pm_wind = QPixmap("Gui/images/Windlast.png")
        l_image_wind = QLabel()
        l_image_wind.setPixmap(pm_wind)

        pm_earth = QPixmap("Gui/images/Erddruck.png")
        l_image_earth = QLabel()
        l_image_earth.setPixmap(pm_earth)

        pm_temp_grad = QPixmap("Gui/images/Temperaturgrad.png")
        l_images_temp_grad = QLabel()
        l_images_temp_grad.setPixmap(pm_temp_grad)

        pm_temp_diff = QPixmap("Gui/images/Temperaturdiff.png")
        l_images_temp_diff = QLabel()
        l_images_temp_diff.setPixmap(pm_temp_diff)

        # Checkbox #####################################################################################################

        self.cb_wind = QCheckBox()
        self.cb_earth = QCheckBox()
        self.cb_temp_diff = QCheckBox()
        self.cb_temp_grad = QCheckBox()


        # Button #######################################################################################################

        pb_check = QPushButton("Weiter")
        pb_check.clicked.connect(self.check)

        # Load #########################################################################################################



        # Layout #######################################################################################################

        # lay_wind = QVBoxLayout()
        # lay_wind.addWidget(l_wind)
        # lay_wind.addWidget(l_image_wind)
        # lay_wind.addWidget(self.cb_wind)

        self.lay_force = QGridLayout()
        self.lay_force.addWidget(l_wind, 0, 0)
        self.lay_force.addWidget(l_image_wind, 1, 0)
        self.lay_force.addWidget(self.cb_wind, 2, 0)

        self.lay_force.addWidget(l_earth, 0, 1)
        self.lay_force.addWidget(l_image_earth, 1, 1)
        self.lay_force.addWidget(self.cb_earth, 2, 1)

        self.lay_force.addWidget(l_temp_diff, 0, 2)
        self.lay_force.addWidget(l_images_temp_diff, 1, 2)
        self.lay_force.addWidget(self.cb_temp_diff, 2, 2)

        self.lay_force.addWidget(l_temp_grad, 0, 3)
        self.lay_force.addWidget(l_images_temp_grad, 1, 3)
        self.lay_force.addWidget(self.cb_temp_grad, 2, 3)




        main_layout = QVBoxLayout()
        main_layout.addLayout(self.lay_force)
        main_layout.addWidget(pb_check)

        if self.DATA["main"]["belastung"]["wind"] != 0:
            self.cb_wind.setChecked(True)

        if self.DATA["main"]["belastung"]["erdlastunten"] != 0 or self.DATA["main"]["belastung"]["erdlastoben"] != 0:
            self.cb_earth.setChecked(True)

        if self.DATA["main"]["belastung"]["tempdiff"] != 0:
            self.cb_temp_diff.setChecked(True)

        if self.DATA["main"]["belastung"]["tempgrad"] != 0:
            self.cb_temp_grad.setChecked(True)

        self.setLayout(main_layout)
        self.show()

    def check(self):
        self.checklist["wind"] = self.cb_wind.isChecked()
        self.checklist["earth"] = self.cb_earth.isChecked()
        self.checklist["temp_diff"] = self.cb_temp_diff.isChecked()
        self.checklist["temp_grad"] = self.cb_temp_grad.isChecked()

        self.close()
        self.settingsWindow = SettingsLoadWindow(self.DATA, self.checklist)


class SettingsLoadWindow(BasicWindowRainbow):

    def __init__(self, data: dict, checklist: dict):
        super().__init__(name="Rainbow: Einstellen der Last")

        self.DATA = data
        self.checklist = checklist
        self.single_load_window = None

        # linear load
        l_load = QLabel("Windlast in N/mm²")
        self.sb_load = QLineEdit(str(self.DATA["main"]["belastung"]["wind"]))

        # Trapezoid load
        l_top_load = QLabel("Erdlast oben in N/mm")
        self.sb_top_load = QLineEdit(str(self.DATA["main"]["belastung"]["erdlastoben"]))

        l_bot_load = QLabel("Erdlast unten in N/mm")
        self.sb_bot_load = QLineEdit(str(self.DATA["main"]["belastung"]["erdlastunten"]))

        # Temp diff
        l_temp_diff = QLabel("Temperaturdifferenz in K")
        self.sb_temp_diff = QLineEdit(str(self.DATA["main"]["belastung"]["tempdiff"]))

        # Temp grad
        l_temp_grad = QLabel("Temperaturgradient in K")
        self.sb_temp_grad = QLineEdit(str(self.DATA["main"]["belastung"]["tempgrad"]))

        # alpha star
        l_alpha_t = QLabel("Ausdehnungskoeffizient in 1/K")
        self.sb_alpha_t = QLineEdit(str(self.DATA["main"]["belastung"]["Ausdehnungskoeffizient"] * 1e6))

        # load vertical
        l_load_vertical_facing = QLabel("Wichte Vorsatzschale in N/mm³")
        self.sb_load_vertical_facing = QLineEdit(str(self.DATA["vorsatzschale"]["Wichte"]))
        l_load_vertical_support = QLabel("Wichte Tragschale in N/mm³")
        self.sb_load_vertical_support = QLineEdit(str(self.DATA["tragschale"]["Wichte"]))

        # Layout #######################################################################################################

        pb_calc = QPushButton("Berechnen")
        pb_calc.clicked.connect(self.calc)

        main_layout = QGridLayout()
        if self.checklist["wind"]:
            main_layout.addWidget(l_load, 0, 0)
            main_layout.addWidget(self.sb_load, 0, 1)

        if self.checklist["earth"]:
            main_layout.addWidget(l_top_load, 1, 0)
            main_layout.addWidget(self.sb_top_load, 1, 1)
            main_layout.addWidget(l_bot_load, 2, 0)
            main_layout.addWidget(self.sb_bot_load, 2, 1)

        if self.checklist["temp_diff"]:
            main_layout.addWidget(l_temp_diff, 3, 0)
            main_layout.addWidget(self.sb_temp_diff, 3, 1)

        if self.checklist["temp_grad"]:
            main_layout.addWidget(l_temp_grad, 4, 0)
            main_layout.addWidget(self.sb_temp_grad, 4, 1)

        if self.checklist["temp_grad"] or self.checklist["temp_diff"]:
            main_layout.addWidget(l_alpha_t, 5, 0)
            main_layout.addWidget(self.sb_alpha_t, 5, 1)

        main_layout.addWidget(l_load_vertical_facing, 6, 0)
        main_layout.addWidget(self.sb_load_vertical_facing, 6, 1)
        main_layout.addWidget(l_load_vertical_support, 7, 0)
        main_layout.addWidget(self.sb_load_vertical_support, 7, 1)

        main_layout.addWidget(pb_calc, 9, 0)

        self.setLayout(main_layout)
        self.show()

    def calc(self):
        if self.checklist["wind"]:
            self.DATA["main"]["belastung"]["wind"] = float(self.sb_load.text())

        if self.checklist["earth"]:
            self.DATA["main"]["belastung"]["erdlastoben"] = float(self.sb_top_load.text())
            self.DATA["main"]["belastung"]["erdlastunten"] = float(self.sb_bot_load.text())

        if self.checklist["temp_diff"]:
            self.DATA["main"]["belastung"]["tempdiff"] = float(self.sb_temp_diff.text())

        if self.checklist["temp_grad"]:
            self.DATA["main"]["belastung"]["tempgrad"] = float(self.sb_temp_grad.text())

        if self.checklist["temp_diff"] or self.checklist["temp_grad"]:
            self.DATA["main"]["belastung"]["Ausdehnungskoeffizient"] = float(self.sb_alpha_t.text()) / 1e6

        self.DATA["vorsatzschale"]["Wichte"] = float(self.sb_load_vertical_facing.text())
        self.DATA["tragschale"]["Wichte"] = float(self.sb_load_vertical_support.text())

        if self.single_load_window is not None:
            self.DATA["main"]["belastung"]["einzellast"] = self.single_load_window.DATA

        wall = Wall(self.DATA)
        single_load_grid_x, single_load_grid_y, single_load_grid_z = wall.single_load_grid()

        with open("Gui/ergbnisse/Parameter.json", "w") as file:
            json.dump(self.DATA, file, indent=4)

        calculate(wall.grid(),
                  single_load_grid_x,
                  single_load_grid_y,
                  single_load_grid_z,
                  wall.deltax,
                  wall.deltay,
                  self.DATA["vorsatzschale"]["Elastizitaetsmodul"],
                  self.DATA["tragschale"]["Elastizitaetsmodul"],
                  self.DATA["vorsatzschale"]["dicke"],
                  self.DATA["tragschale"]["dicke"],
                  self.DATA["main"]["v"],
                  self.DATA["main"]["belastung"]["wind"],
                  {"Kunststoff1": self.DATA["Kunststoff1"], "Kunststoff2": self.DATA["Kunststoff2"]},
                  self.DATA["main"]["belastung"]["erdlastoben"],
                  self.DATA["main"]["belastung"]["erdlastunten"],
                  self.DATA["main"]["belastung"]["tempdiff"],
                  self.DATA["main"]["belastung"]["tempgrad"],
                  self.DATA["main"]["belastung"]["Ausdehnungskoeffizient"],
                  self.DATA["main"]["hoehe"],
                  self.DATA["main"]["laenge"],
                  self.DATA["vorsatzschale"]["Wichte"],
                  self.DATA["tragschale"]["Wichte"],
                  self.DATA["main"]["Tafelbauweise"],
                  10,
                  self.DATA["vorsatzschale"],
                  self.DATA["tragschale"]
                  )
        self.close()



        print("Fertig")



if __name__ == '__main__':
    app = QApplication(sys.argv)
    welcome = WelcomeWindow()
    app.exec_()
