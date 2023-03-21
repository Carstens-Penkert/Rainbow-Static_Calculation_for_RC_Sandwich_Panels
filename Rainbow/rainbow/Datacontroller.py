import numpy as np
import json
import matplotlib.pyplot as plt


class Wall:

    def __init__(self, data):
        self.Data = data

    def grid(self):
        simple_grid = np.ones((self.Data["main"]["gittervertikal"], self.Data["main"]["gitterhorizontal"]))

        # set corners and edges
        simple_grid[self.Data["main"]["gittervertikal"] - 1, 0] = 2
        simple_grid[0, 0] = 3
        simple_grid[self.Data["main"]["gittervertikal"] - 1, self.Data["main"]["gitterhorizontal"] - 1] = 4
        simple_grid[0, self.Data["main"]["gitterhorizontal"] - 1] = 5

        simple_grid[self.Data["main"]["gittervertikal"] - 1, 1: self.Data["main"]["gitterhorizontal"] - 1] = 6
        simple_grid[1:self.Data["main"]["gittervertikal"] - 1, 0] = 7
        simple_grid[0, 1:  self.Data["main"]["gitterhorizontal"] - 1] = 8
        simple_grid[1:self.Data["main"]["gittervertikal"] - 1, self.Data["main"]["gitterhorizontal"] - 1] = 9

        for opening in self.Data["main"]["oeffnung"]:

            xstart_point = int(opening["xstart"] / self.deltax)
            yend_point = (self.Data["main"]["gittervertikal"] - 1) - int(opening["ystart"] / self.deltay) + 1

            xend_point = int((opening["xstart"] + opening["xlaenge"]) / self.deltax) + 1
            ystart_point = (self.Data["main"]["gittervertikal"] - 1) - (int((opening["ystart"] + opening["ylaenge"]) /
                                                                            self.deltay))
            simple_grid[ystart_point:yend_point, xstart_point: xend_point] = 0

            if xstart_point == 1:
                raise ValueError(
                    "Der X start Punkt liegt auf dem Rand der Wand. Entweder den Start weiter nach rechts verschieben oder das Gitter feiner waehlen")

            try:
                simple_grid[ystart_point: yend_point, xstart_point - 1] = 9
            except IndexError:
                pass

            try:
                simple_grid[ystart_point: yend_point, xend_point] = 7
            except IndexError:
                pass

            try:
                simple_grid[ystart_point - 1, xstart_point:xend_point] = 6
            except IndexError:
                pass

            try:
                simple_grid[yend_point, xstart_point:xend_point] = 8
            except IndexError:
                pass

            # Fall 1: Öffnung an der Linken Seite
            if xstart_point == 0:
                try:
                    simple_grid[ystart_point - 1, xstart_point] = 2
                except IndexError:
                    pass
                try:
                    simple_grid[yend_point, xstart_point] = 3
                except IndexError:
                    pass

            # Fall 2: Öffnung an der Rechten Seite
            if xend_point == self.Data["main"]["gitterhorizontal"]:
                try:
                    simple_grid[ystart_point - 1, xend_point - 1] = 4
                except IndexError:
                    pass
                try:
                    simple_grid[yend_point, xend_point - 1] = 5
                except IndexError:
                    pass

            # Fall 3: Öffnung an der unteren Seite
            if yend_point == self.Data["main"]["gittervertikal"]:
                try:
                    simple_grid[yend_point - 1, xstart_point - 1] = 4
                except IndexError:
                    pass
                try:
                    simple_grid[yend_point - 1, xend_point] = 2
                except IndexError:
                    pass

            # Fall 4: Öffnung an der oberen Seite
            if ystart_point == 0:
                try:
                    simple_grid[ystart_point, xstart_point - 1] = 5
                except IndexError:
                    pass
                try:
                    simple_grid[ystart_point, xend_point] = 3
                except IndexError:
                    pass

        # placing support points

        for point in self.Data["main"]["Anker"]:
            x = int(point["x"] / self.deltax)
            y = (self.Data["main"]["gittervertikal"]) - int(point["y"] / self.deltay + 1)

            simple_grid[y, x] = 15

        return simple_grid

    def single_load_grid(self):

        simple_grid_x = np.zeros((self.Data["main"]["gittervertikal"], self.Data["main"]["gitterhorizontal"]))
        simple_grid_y = np.zeros((self.Data["main"]["gittervertikal"], self.Data["main"]["gitterhorizontal"]))
        simple_grid_z = np.zeros((self.Data["main"]["gittervertikal"], self.Data["main"]["gitterhorizontal"]))

        for single_load in self.Data["main"]["belastung"]["einzellast"]:
            x = int(single_load["x"] / self.deltax)
            y = (self.Data["main"]["gittervertikal"]) - int(single_load["y"] / self.deltay + 1)

            simple_grid_x[y, x] = single_load["wertx"]
            simple_grid_y[y, x] = single_load["werty"]
            simple_grid_z[y, x] = single_load["wertz"]

        return simple_grid_x, simple_grid_y, simple_grid_z

    @property
    def deltay(self):
        return self.Data["main"]["hoehe"] / (self.Data["main"]["gittervertikal"] - 1)

    @property
    def deltax(self):
        return self.Data["main"]["laenge"] / (self.Data["main"]["gitterhorizontal"] - 1)


def setupData():
    data = {
        "main": {
            "laenge": 0,
            "hoehe": 0,
            "belastung": {
                "wind": 0,
                "tempgrad": 0,
                "tempdiff": 0,
                "erdlastoben": 0,
                "erdlastunten": 0,
                "wichtevorsatzschale": 0,
                "wichtetragschale": 0,
                "Ausdehnungskoeffizient": 0,
                "einzellast": [],
            },
            "v": 0.2,
            "gitterhorizontal": 0,
            "gittervertikal": 0,
            "oeffnung": [],
            "Tafelbauweise": "aufstehende Vorsatzschale",
            "Anker": [],
        },
        "vorsatzschale": {
            "dicke": 0,
            "Elastizitaetsmodul": 0,
            "Wichte": 25e-6,
            "Zylinderdruckfestigkeit": 0,
            "Zugbruchdehnung": 0,
            "Elastizitaetsmodul Betonstahl": 0,
            "Betondeckung obere Bewehrung": 0,
            "Querschnittsflaeche obere Bewhrung": 0,
            "Betondeckung untere Bewehrung": 0,
            "Querschnittsflaeche untere Bewhrung": 0,
            "Fliesgrenze Betonstahl": 0
        },
        "tragschale": {
            "dicke": 0,
            "Elastizitaetsmodul": 0,
            "Wichte": 25e-6,
            "Zylinderdruckfestigkeit": 0,
            "Zugbruchdehnung": 0,
            "Elastizitaetsmodul Betonstahl": 0,
            "Betondeckung obere Bewehrung": 0,
            "Querschnittsflaeche obere Bewhrung": 0,
            "Betondeckung untere Bewehrung": 0,
            "Querschnittsflaeche untere Bewhrung": 0,
            "Fliesgrenze Betonstahl": 0
        },
        "Kunststoff1": {"Cx": 0, "Cy": 0, "Cz": 0},
        "Kunststoff2": {"Cx": 0, "Cy": 0, "Cz": 0},

    }
    return data


def loadData(path: str):
    with open(path, "r") as file:
        data = json.load(file)
    return data


if __name__ == "__main__":
    #  test = setupData()
    #  test["main"]["hoehe"] = 1300
    #  test["main"]["laenge"] = 1300
    #  test["main"]["gitterhorizontal"] = 21
    #  test["main"]["gittervertikal"] = 21
    #  # test["main"]["Anker"].append({"xstart": 500,
    #  #                                 "ystart": 500,
    #  #                                 "xlaenge": 1000,
    #  #                                 "ylaenge": 1000,})
    # #wall = Wall(test)
    # #wall.grid()
    #
    #  test["main"]["Anker"].append({"x": 150,
    #                                "y": 150,
    #                                "Ankertyp": "Kunststoffanker",
    #                                "Gesetz": "Kunststoff Gesetz 1"})
    #  with open("../datafile.json", "w") as file:
    #      json.dump(test, file, indent=4)

    test = loadData("../Gui/ergbnisse/Parameter.json")
    # test["main"]["Anker"] = [test["main"]["Anker"][1]]
    wall = Wall(test)
    # print(wall.grid())
    plt.imshow(wall.grid())
    plt.show()
