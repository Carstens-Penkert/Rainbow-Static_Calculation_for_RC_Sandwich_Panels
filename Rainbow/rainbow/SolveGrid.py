import numpy as np
import pyomo.environ as pyo
import matplotlib.pyplot as plt
import pandas as pd
from rainbow.Datasaving import save_file


def calculate(wall,
              single_load_x,
              single_load_y,
              single_load_z,
              delta_x,
              delta_y,
              E_facing_wall,
              E_support_wall,
              t_facing_wall,
              t_support_wall,
              v,
              q_wind,
              material_laws: dict,
              q_top=0,
              q_bot=0,
              temp_grad=0,
              temp_diff=0,
              alpha_t=0,
              height=0,
              length=0,
              q_eigenlast_facing=25e-6,
              q_eigenlast_support=25e-6,
              build_type="aufstehende Vorsatzschale",
              count_force_level=1,
              data_facing_wall={},
              data_support_wall={}):

    # force top to bot
    if q_bot == 0 and q_top == 0:
        force = np.zeros_like(wall, dtype=np.float)
    else:
        step_size = (q_bot - q_top) / wall.shape[0]
        force = np.mgrid[q_top:q_bot:step_size, 0:wall.shape[1]][0]

    for row in range(wall.shape[0]):
        for col in range(wall.shape[1]):
            force[row, col] += q_wind  # N / mm ** 2



    index_anker = np.where((wall == 15) | (wall == 16))

    m, n = wall.shape

    # vorschale
    D_facing_wall = (E_facing_wall * t_facing_wall ** 3) / (12 * (1 - v ** 2))
    Ds_facing_wall = (E_facing_wall * t_facing_wall) / (1 - v ** 2)
    M_star_facing_wall = (alpha_t * E_facing_wall * t_facing_wall ** 2 * temp_grad) / 6
    N_star_facing_wall = 1 * (alpha_t * temp_diff * Ds_facing_wall )

    # tragschale
    D_support_wall = (E_support_wall * t_support_wall ** 3) / (12 * (1 - v ** 2))
    Ds_support_wall = (E_support_wall * t_support_wall) / (1 - v ** 2)
    M_star_support_wall = (alpha_t * E_support_wall * t_support_wall ** 2 * temp_grad) / 6
    N_star_support_wall = 1 * (alpha_t * temp_diff * Ds_support_wall)

    # Hier wird die Funktion definiert die Minimiert werden soll
    def obj(model):

        ################################################################################################################
        # w facing wall
        ################################################################################################################

        # Ableitungen mit Differenzenquotient zu erst 2 Ableitungen dann noch die ersten
        w2x_facingwall = np.zeros_like(wall).tolist()
        w2y_facingwall = np.zeros_like(wall).tolist()
        w2xy_facingwall = np.zeros_like(wall).tolist()

        for row in model.row:

            for col in model.col:

                wall_type = wall[row, col]

                if not wall_type == 0:

                    # w2x

                    if wall_type in [1, 15, 8, 6, 11, 12, 13, 14, 16]:  # 2.14

                        w2x_facingwall[row][col] = (model.shifts_facing_wall[row, col + 1] - 2 * model.shifts_facing_wall[row, col] + model.shifts_facing_wall[
                            row, col - 1]) / (delta_x ** 2)

                    elif wall_type in [9, 5, 4]:  # 2.17

                        w2x_facingwall[row][col] = 0

                    else:  # 2.21

                        w2x_facingwall[row][col] = 0

                    # w2y

                    if wall_type in [1, 15, 9, 7, 11, 12, 13, 14, 16]:  # 2.15

                        w2y_facingwall[row][col] = (model.shifts_facing_wall[row + 1, col] - 2 * model.shifts_facing_wall[row, col] + model.shifts_facing_wall[
                            row - 1, col]) / (delta_y ** 2)

                    elif wall_type in [8, 5, 3]:  # 2.25

                        w2y_facingwall[row][col] = 0

                    else:  # 2.29

                        w2y_facingwall[row][col] = 0

                    # w2xy

                    if wall_type in [1, 15, 16]:  # 2.13

                        w2xy_facingwall[row][col] = (model.shifts_facing_wall[row + 1, col + 1] - model.shifts_facing_wall[row - 1, col + 1] -
                                                     model.shifts_facing_wall[
                                              row + 1, col - 1] + model.shifts_facing_wall[row - 1, col - 1]) / (
                                                 4 * delta_x * delta_y)

                    elif wall_type == 9:  # 2.16

                        w2xy_facingwall[row][col] = (2 * model.shifts_facing_wall[row + 1, col] - 2 * model.shifts_facing_wall[row - 1, col] - 2 *
                                                     model.shifts_facing_wall[
                                              row + 1, col - 1] + 2 * model.shifts_facing_wall[row - 1, col - 1]) / (
                                                 4 * delta_x * delta_y)

                    elif wall_type == 7:  # 2.20

                        w2xy_facingwall[row][col] = (2 * model.shifts_facing_wall[row - 1, col] - 2 * model.shifts_facing_wall[row + 1, col] - 2 *
                                                     model.shifts_facing_wall[
                                              row - 1, col + 1] + 2 * model.shifts_facing_wall[row + 1, col + 1]) / (
                                                 4 * delta_x * delta_y)

                    elif wall_type == 8:  # 2.24

                        w2xy_facingwall[row][col] = (2 * model.shifts_facing_wall[row, col - 1] - 2 * model.shifts_facing_wall[row, col + 1] - 2 *
                                                     model.shifts_facing_wall[
                                              row + 1, col - 1] + 2 * model.shifts_facing_wall[row + 1, col + 1]) / (
                                                 4 * delta_x * delta_y)

                    elif wall_type == 6:  # 2.28

                        w2xy_facingwall[row][col] = (2 * model.shifts_facing_wall[row, col + 1] - 2 * model.shifts_facing_wall[row, col - 1] - 2 *
                                                     model.shifts_facing_wall[
                                              row - 1, col + 1] + 2 * model.shifts_facing_wall[row - 1, col - 1]) / (
                                                 4 * delta_x * delta_y)

                    elif wall_type == 5:  # 2.32

                        w2xy_facingwall[row][col] = 0

                    elif wall_type == 3:  # 2.35

                        w2xy_facingwall[row][col] = 0

                    elif wall_type == 4:  # 2.36

                        w2xy_facingwall[row][col] = 0

                    elif wall_type == 2:  # 2.37

                        w2xy_facingwall[row][col] = 0

                    elif wall_type == 11:  # 2.40
                        w2xy_facingwall[row][col] = (2 * model.shifts_facing_wall[row - 1, col] - 2 * model.shifts_facing_wall[row - 1, col + 1] -
                                                     model.shifts_facing_wall[
                                              row + 1, col - 1] + model.shifts_facing_wall[row + 1, col + 1]) / (
                                                 4 * delta_x * delta_y)

                    elif wall_type == 12:  # 2.41
                        w2xy_facingwall[row][col] = (2 * model.shifts_facing_wall[row - 1, col - 1] - model.shifts_facing_wall[row - 1, col] -
                                                     model.shifts_facing_wall[
                                              row + 1, col - 1] + model.shifts_facing_wall[row + 1, col + 1]) / (
                                                 4 * delta_x * delta_y)

                    elif wall_type == 13:  # 2.38
                        w2xy_facingwall[row][col] = (2 * model.shifts_facing_wall[row + 1, col + 1] - model.shifts_facing_wall[row - 1, col + 1] - 2 *
                                                     model.shifts_facing_wall[row + 1, col] + model.shifts_facing_wall[row - 1, col - 1]) / (
                                                 4 * delta_x * delta_y)

                    elif wall_type == 14:  # 2.39
                        w2xy_facingwall[row][col] = (2 * model.shifts_facing_wall[row + 1, col] - 2 * model.shifts_facing_wall[row + 1, col - 1] -
                                                     model.shifts_facing_wall[
                                              row - 1, col + 1] + model.shifts_facing_wall[row - 1, col - 1]) / (
                                                 4 * delta_x * delta_y)

        w_x_facingwall = np.zeros_like(wall).tolist()
        w_y_facingwall = np.zeros_like(wall).tolist()
        w_x_supportwall = np.zeros_like(wall).tolist()
        w_y_supportwall = np.zeros_like(wall).tolist()

        for row in model.row:

            for col in model.col:

                wall_type = wall[row, col]
                if wall_type != 0:
                    # u_x
                    if wall_type in [9, 4, 5]:
                        w_x_facingwall[row][col] = (model.shifts_facing_wall[row, col] -
                                                      model.shifts_facing_wall[
                                                          row, col - 1]) / delta_x
                        w_x_supportwall[row][col] = (model.shifts_support_wall[row, col] -
                                                    model.shifts_support_wall[
                                                        row, col - 1]) / delta_x
                    else:
                        w_x_facingwall[row][col] = (model.shifts_facing_wall[row, col + 1] -
                                                      model.shifts_facing_wall[
                                                          row, col]) / delta_x
                        w_x_supportwall[row][col] = (model.shifts_support_wall[row, col + 1] -
                                                    model.shifts_support_wall[
                                                        row, col]) / delta_x

                    # u_y
                    if wall_type in [6, 2, 4]:
                        w_y_facingwall[row][col] = (model.shifts_facing_wall[row, col] -
                                                      model.shifts_facing_wall[
                                                          row - 1, col]) / delta_y
                        w_y_supportwall[row][col] = (model.shifts_support_wall[row, col] -
                                                      model.shifts_support_wall[
                                                          row - 1, col]) / delta_y
                    else:
                        w_y_facingwall[row][col] = (model.shifts_facing_wall[row + 1, col] -
                                                      model.shifts_facing_wall[
                                                          row, col]) / delta_y
                        w_y_supportwall[row][col] = (model.shifts_support_wall[row + 1, col] -
                                                      model.shifts_support_wall[
                                                          row, col]) / delta_y

        # Bestimmen der Momente
        mx_facingwall = np.zeros_like(wall).tolist()
        my_facingwall = np.zeros_like(wall).tolist()
        mxy_facingwall = np.zeros_like(wall).tolist()

        for row in model.row:

            for col in model.col:

                wall_type = wall[row, col]

                if not wall_type == 0:
                    mx_facingwall[row][col] = - D_facing_wall * (w2x_facingwall[row][col] + v * w2y_facingwall[row][col]) - M_star_facing_wall / (1 - v)
                    my_facingwall[row][col] = - D_facing_wall * (w2y_facingwall[row][col] + v * w2x_facingwall[row][col]) - M_star_facing_wall / (1 - v)
                    mxy_facingwall[row][col] = -D_facing_wall * (1 - v) * w2xy_facingwall[row][col]

        # Festlegen der Arbeit
        innerwork_facingwall = np.zeros_like(wall).tolist()
        outwork_facingwall = np.zeros_like(wall).tolist()

        diff = np.zeros_like(wall).tolist()

        for row in model.row:

            for col in model.col:

                wall_type = wall[row, col]

                if not wall_type == 0:

                    if wall_type in [2, 3, 4, 5]:   # Ecke Platt
                        d_x_and_d_y = (delta_x * delta_y) / 2
                    elif wall_type == 6 or wall_type == 8:    # Rand unten bzw oben
                        d_x_and_d_y = delta_x * (delta_y / 2)
                    elif wall_type == 7 or wall_type == 9:  # Rand Links bzw. Rechts
                        d_x_and_d_y = (delta_x / 2) * delta_y
                    elif wall_type in [11, 12, 13, 14]: # Ecken öffnung
                        d_x_and_d_y = (3 / 4) * (delta_x * delta_y)
                    else:
                        d_x_and_d_y = delta_x * delta_y

                    innerwork_facingwall[row][col] = -0.5 * (
                            mx_facingwall[row][col] * (w2x_facingwall[row][col]) + my_facingwall[row][col] * (w2y_facingwall[row][col] ) + 2 * mxy_facingwall[row][col] * w2xy_facingwall[row][
                        col]) * d_x_and_d_y

                    outwork_facingwall[row][col] =  1 * force[row][col] * (model.shifts_facing_wall[row, col] + model.v_shifts_support_wall[row, col]) * d_x_and_d_y
                    diff[row][col] = innerwork_facingwall[row][col] - outwork_facingwall[row][col]

        sum_obj = 0
        for row in model.row:

            for col in model.col:

                wall_type = wall[row, col]

                if not wall_type == 0:
                    sum_obj += diff[row][col]

        ################################################################################################################
        # w support wall
        ################################################################################################################

        # Ableitung für die Tragschale (2 Ableitung) mit hilfe des Differenzenqoutient
        w2x_support_wall = np.zeros_like(wall).tolist()
        w2y_support_wall = np.zeros_like(wall).tolist()
        w2xy_support_wall = np.zeros_like(wall).tolist()

        for row in model.row:

            for col in model.col:

                wall_type = wall[row, col]

                if not wall_type == 0:

                    # w2x

                    if wall_type in [1, 15, 8, 6, 11, 12, 13, 14, 16]:  # 2.14

                        w2x_support_wall[row][col] = (model.shifts_support_wall[row, col + 1] - 2 *
                                                      model.shifts_support_wall[row, col] + model.shifts_support_wall[
                                                          row, col - 1]) / (delta_x ** 2)

                    elif wall_type in [9, 5, 4]:  # 2.17

                        w2x_support_wall[row][col] = 0

                    else:  # 2.21

                        w2x_support_wall[row][col] = 0

                    # w2y

                    if wall_type in [1, 15, 9, 7, 11, 12, 13, 14, 16]:  # 2.15

                        w2y_support_wall[row][col] = (model.shifts_support_wall[row + 1, col] - 2 *
                                                      model.shifts_support_wall[row, col] + model.shifts_support_wall[
                                                          row - 1, col]) / (delta_y ** 2)

                    elif wall_type in [8, 5, 3]:  # 2.25

                        w2y_support_wall[row][col] = 0

                    else:  # 2.29

                        w2y_support_wall[row][col] = 0

                    # w2xy

                    if wall_type in [1, 15, 16]:  # 2.13

                        w2xy_support_wall[row][col] = (model.shifts_support_wall[row + 1, col + 1] -
                                                       model.shifts_support_wall[row - 1, col + 1] -
                                                       model.shifts_support_wall[
                                                           row + 1, col - 1] + model.shifts_support_wall[
                                                           row - 1, col - 1]) / (
                                                              4 * delta_x * delta_y)

                    elif wall_type == 9:  # 2.16

                        w2xy_support_wall[row][col] = (2 * model.shifts_support_wall[row + 1, col] - 2 *
                                                       model.shifts_support_wall[row - 1, col] - 2 *
                                                       model.shifts_support_wall[
                                                           row + 1, col - 1] + 2 * model.shifts_support_wall[
                                                           row - 1, col - 1]) / (
                                                              4 * delta_x * delta_y)

                    elif wall_type == 7:  # 2.20

                        w2xy_support_wall[row][col] = (2 * model.shifts_support_wall[row - 1, col] - 2 *
                                                       model.shifts_support_wall[row + 1, col] - 2 *
                                                       model.shifts_support_wall[
                                                           row - 1, col + 1] + 2 * model.shifts_support_wall[
                                                           row + 1, col + 1]) / (
                                                              4 * delta_x * delta_y)

                    elif wall_type == 8:  # 2.24

                        w2xy_support_wall[row][col] = (2 * model.shifts_support_wall[row, col - 1] - 2 *
                                                       model.shifts_support_wall[row, col + 1] - 2 *
                                                       model.shifts_support_wall[
                                                           row + 1, col - 1] + 2 * model.shifts_support_wall[
                                                           row + 1, col + 1]) / (
                                                              4 * delta_x * delta_y)

                    elif wall_type == 6:  # 2.28

                        w2xy_support_wall[row][col] = (2 * model.shifts_support_wall[row, col + 1] - 2 *
                                                       model.shifts_support_wall[row, col - 1] - 2 *
                                                       model.shifts_support_wall[
                                                           row - 1, col + 1] + 2 * model.shifts_support_wall[
                                                           row - 1, col - 1]) / (
                                                              4 * delta_x * delta_y)

                    elif wall_type == 5:  # 2.32

                        w2xy_support_wall[row][col] = 0

                    elif wall_type == 3:  # 2.35

                        w2xy_support_wall[row][col] = 0

                    elif wall_type == 4:  # 2.36

                        w2xy_support_wall[row][col] = 0

                    elif wall_type == 2:  # 2.37

                        w2xy_support_wall[row][col] = 0

                    elif wall_type == 11:  # 2.40
                        w2xy_support_wall[row][col] = (2 * model.shifts_support_wall[row - 1, col] - 2 *
                                                       model.shifts_support_wall[row - 1, col + 1] -
                                                       model.shifts_support_wall[
                                                           row + 1, col - 1] + model.shifts_support_wall[
                                                           row + 1, col + 1]) / (
                                                              4 * delta_x * delta_y)

                    elif wall_type == 12:  # 2.41
                        w2xy_support_wall[row][col] = (2 * model.shifts_support_wall[row - 1, col - 1] -
                                                       model.shifts_support_wall[row - 1, col] -
                                                       model.shifts_support_wall[
                                                           row + 1, col - 1] + model.shifts_support_wall[
                                                           row + 1, col + 1]) / (
                                                              4 * delta_x * delta_y)

                    elif wall_type == 13:  # 2.38
                        w2xy_support_wall[row][col] = (2 * model.shifts_support_wall[row + 1, col + 1] -
                                                       model.shifts_support_wall[row - 1, col + 1] - 2 *
                                                       model.shifts_support_wall[row + 1, col] +
                                                       model.shifts_support_wall[row - 1, col - 1]) / (
                                                              4 * delta_x * delta_y)

                    elif wall_type == 14:  # 2.39
                        w2xy_support_wall[row][col] = (2 * model.shifts_support_wall[row + 1, col] - 2 *
                                                       model.shifts_support_wall[row + 1, col - 1] -
                                                       model.shifts_support_wall[
                                                           row - 1, col + 1] + model.shifts_support_wall[
                                                           row - 1, col - 1]) / (
                                                              4 * delta_x * delta_y)

        # Hier wird das Moment bestimmt
        mx_support_wall = np.zeros_like(wall).tolist()
        my_support_wall = np.zeros_like(wall).tolist()
        mxy_support_wall = np.zeros_like(wall).tolist()

        for row in model.row:

            for col in model.col:

                wall_type = wall[row, col]

                if not wall_type == 0:
                    mx_support_wall[row][col] = - D_support_wall * (
                                w2x_support_wall[row][col] + v * w2y_support_wall[row][col]) - M_star_support_wall / (1 - v)
                    my_support_wall[row][col] = - D_support_wall * (
                                w2y_support_wall[row][col] + v * w2x_support_wall[row][col]) - M_star_support_wall / (1 - v)
                    mxy_support_wall[row][col] = -D_support_wall * (1 - v) * w2xy_support_wall[row][col]

        # Hier wird dann die Arbeit aufgestellt
        innerwork_support_Wall = np.zeros_like(wall).tolist()
        outwork_support_wall = np.zeros_like(wall).tolist()

        diff = np.zeros_like(wall).tolist()

        for row in model.row:

            for col in model.col:

                wall_type = wall[row, col]

                if not wall_type == 0:

                    if wall_type in [2, 3, 4, 5]:  # Ecke Platt
                        d_x_and_d_y = (delta_x * delta_y) / 2
                    elif wall_type == 6 or wall_type == 8:  # Rand unten bzw oben
                        d_x_and_d_y = delta_x * (delta_y / 2)
                    elif wall_type == 7 or wall_type == 9:  # Rand Links bzw. Rechts
                        d_x_and_d_y = (delta_x / 2) * delta_y
                    elif wall_type in [11, 12, 13, 14]:  # Ecken öffnung
                        d_x_and_d_y = (3 / 4) * (delta_x * delta_y)
                    else:
                        d_x_and_d_y = delta_x * delta_y

                    innerwork_support_Wall[row][col] = -0.5 * (
                            mx_support_wall[row][col] * (w2x_support_wall[row][col]) + my_support_wall[row][col] * (
                    w2y_support_wall[row][col]) + 2 * mxy_support_wall[row][col] * w2xy_support_wall[row][
                                col]) * d_x_and_d_y

                    diff[row][col] = innerwork_support_Wall[row][col]



        for row in model.row:

            for col in model.col:

                wall_type = wall[row, col]

                if not wall_type == 0:
                    sum_obj += diff[row][col]

        ################################################################################################################
        # u and v facing wall
        ################################################################################################################

        # Scheibentheorie

        # Verschiebungen U

        # Hier werden dann die Ableitung der Verschiebungen in X bzw Y Richtung bestimmt
        u_x_facing_wall = np.zeros_like(wall).tolist()  # Epsilonx
        v_y_facing_wall = np.zeros_like(wall).tolist()  # Epsilony

        u_y_facing_wall = np.zeros_like(wall).tolist()  # Gamma = u_y + v_x
        v_x_facing_wall = np.zeros_like(wall).tolist()

        for row in model.row:

            for col in model.col:

                wall_type = wall[row, col]
                if wall_type != 0:
                    # u_x
                    if wall_type in [9, 4, 5]:
                        u_x_facing_wall[row][col] = (model.u_shifts_facing_wall[row, col] - model.u_shifts_facing_wall[row, col - 1]) / delta_x
                    else:
                        u_x_facing_wall[row][col] = (model.u_shifts_facing_wall[row, col + 1] - model.u_shifts_facing_wall[
                            row, col]) / delta_x
                    # v_x
                    if wall_type in [9, 4, 5]:
                        v_x_facing_wall[row][col] = (model.v_shifts_facing_wall[row, col] - model.v_shifts_facing_wall[row, col - 1]) / delta_x
                    else:
                        v_x_facing_wall[row][col] = (model.v_shifts_facing_wall[row, col + 1] - model.v_shifts_facing_wall[
                            row, col]) / delta_x
                    # u_y
                    if wall_type in [6, 2, 4]:
                        u_y_facing_wall[row][col] = (model.u_shifts_facing_wall[row, col] - model.u_shifts_facing_wall[row - 1, col]) / delta_y
                    else:
                        u_y_facing_wall[row][col] = (model.u_shifts_facing_wall[row + 1, col] - model.u_shifts_facing_wall[
                            row, col]) / delta_y
                    # v_y
                    if wall_type in [6, 2, 4]:
                        v_y_facing_wall[row][col] = (model.v_shifts_facing_wall[row, col] - model.v_shifts_facing_wall[row - 1, col]) / delta_y
                    else:
                        v_y_facing_wall[row][col] = (model.v_shifts_facing_wall[row + 1, col] - model.v_shifts_facing_wall[
                            row, col]) / delta_y

        # Formaenderungsarbeit

        # Schnittgrößen aus Verschiebungen u und v (Normalkraft)
        nx_facing_wall = np.zeros_like(wall).tolist()
        ny_facing_wall = np.zeros_like(wall).tolist()
        nxy_facing_wall = np.zeros_like(wall).tolist()

        for row in model.row:
            for col in model.col:
                nx_facing_wall[row][col] = Ds_facing_wall * ((u_x_facing_wall[row][col] - alpha_t * temp_diff) + v * (v_y_facing_wall[row][col]- alpha_t * temp_diff))
                ny_facing_wall[row][col] = Ds_facing_wall * ((v_y_facing_wall[row][col] - alpha_t * temp_diff) + v * (u_x_facing_wall[row][col]- alpha_t * temp_diff))

                # nxy
                nxy_facing_wall[row][col] = Ds_facing_wall * ((1 - v) / 2) * (u_y_facing_wall[row][col] + v_x_facing_wall[row][col])

        # Aufstellen der Arbeit
        formingwork_facing_wall = np.zeros_like(wall).tolist()
        work_plate_facing_wall = np.zeros_like(wall).tolist()

        for row in model.row:

            for col in model.col:
                formingwork_facing_wall[row][col] = 0.5 * ((u_x_facing_wall[row][col] - alpha_t * temp_diff) * nx_facing_wall[row][col] +
                                                           (v_y_facing_wall[row][col] - alpha_t * temp_diff) * ny_facing_wall[row][col] +
                                                           (u_y_facing_wall[row][col]  + v_x_facing_wall[row][col] ) * nxy_facing_wall[row][col]) * delta_x * delta_y  # Formänderungsarbeit
                work_plate_facing_wall[row][col] =t_facing_wall * q_eigenlast_facing * delta_x * delta_y * model.v_shifts_facing_wall[row, col]  # Arbeit Platte

                sum_obj += formingwork_facing_wall[row][col] + work_plate_facing_wall[row][col]



        ################################################################################################################
        # u and v support wall
        ################################################################################################################

        # Scheibentheorie

        # Verschiebungen U

        u_x_support_wall = np.zeros_like(wall).tolist()  # Epsilonx
        v_y_support_wall = np.zeros_like(wall).tolist()  # Epsilony

        u_y_support_wall = np.zeros_like(wall).tolist()  # Gamma = u_y + v_x
        v_x_support_wall = np.zeros_like(wall).tolist()

        for row in model.row:

            for col in model.col:

                wall_type = wall[row, col]
                if wall_type != 0:
                    # u_x
                    if wall_type in [9, 4, 5]:
                        u_x_support_wall[row][col] = (model.u_shifts_support_wall[row, col] - model.u_shifts_support_wall[
                            row, col - 1]) / delta_x
                    else:
                        u_x_support_wall[row][col] = (model.u_shifts_support_wall[row, col + 1] -
                                                     model.u_shifts_support_wall[
                                                         row, col]) / delta_x
                    # v_x
                    if wall_type in [9, 4, 5]:
                        v_x_support_wall[row][col] = (model.v_shifts_support_wall[row, col] - model.v_shifts_support_wall[
                            row, col - 1]) / delta_x
                    else:
                        v_x_support_wall[row][col] = (model.v_shifts_support_wall[row, col + 1] -
                                                     model.v_shifts_support_wall[
                                                         row, col]) / delta_x
                    # u_y
                    if wall_type in [6, 2, 4]:
                        u_y_support_wall[row][col] = (model.u_shifts_support_wall[row, col] - model.u_shifts_support_wall[
                            row - 1, col]) / delta_y
                    else:
                        u_y_support_wall[row][col] = (model.u_shifts_support_wall[row + 1, col] -
                                                     model.u_shifts_support_wall[
                                                         row, col]) / delta_y
                    # v_y
                    if wall_type in [6, 2, 4]:
                        v_y_support_wall[row][col] = (model.v_shifts_support_wall[row, col] - model.v_shifts_support_wall[
                            row - 1, col]) / delta_y
                    else:
                        v_y_support_wall[row][col] = (model.v_shifts_support_wall[row + 1, col] -
                                                     model.v_shifts_support_wall[
                                                         row, col]) / delta_y

        # Formaenderungsarbeit

        # Schnittgrößen aus Verschiebungen u und v

        nx_support_wall = np.zeros_like(wall).tolist()
        ny_support_wall = np.zeros_like(wall).tolist()
        nxy_support_wall = np.zeros_like(wall).tolist()

        for row in model.row:
            for col in model.col:
                nx_support_wall[row][col] = Ds_support_wall * (
                            u_x_support_wall[row][col] + v * v_y_support_wall[row][col]) #- (N_star_support_wall)
                ny_support_wall[row][col] = Ds_support_wall * (
                            v_y_support_wall[row][col] + v * u_x_support_wall[row][col]) #- (N_star_support_wall)

                # nxy
                nxy_support_wall[row][col] = Ds_support_wall * ((1 - v) / 2) * (
                            u_y_support_wall[row][col] + v_x_support_wall[row][col])

        formingwork_support_wall = np.zeros_like(wall).tolist()
        work_plate_support_wall = np.zeros_like(wall).tolist()

        for row in model.row:

            for col in model.col:

                formingwork_support_wall[row][col] = 0.5 * (
                            (u_x_support_wall[row][col]) * nx_support_wall[row][col] + (
                                v_y_support_wall[row][col]) * ny_support_wall[row][
                                col] + (u_y_support_wall[row][col] + v_x_support_wall[row][col]) *
                            nxy_support_wall[row][col]) * delta_x * delta_y  # Formänderungsarbeit

                work_plate_support_wall[row][col] = t_support_wall * q_eigenlast_support * delta_x * delta_y * model.v_shifts_support_wall[row, col]  # Arbeit Platte

                sum_obj += formingwork_support_wall[row][col] + work_plate_support_wall[row][col]


                if wall[row][col] == 15:  # Kunststoff material Gesetz 1
                    sum_obj += 0.5 * ((model.shifts_support_wall[row, col] - model.shifts_facing_wall[row, col]) ** 2) * material_laws["Kunststoff1"]["Cz"]
                    sum_obj += 0.5  * material_laws["Kunststoff1"]["Cx"] *\
                               (model.u_shifts_facing_wall[row,col] + (w_x_facingwall[row][col] * t_facing_wall / 2) -
                                (model.u_shifts_support_wall[row,col] + (w_x_supportwall[row][col] * t_support_wall / 2))) ** 2
                    sum_obj += 0.5 * \
                           material_laws["Kunststoff1"]["Cy"] * (model.v_shifts_facing_wall[row,col] + (w_y_facingwall[row][col] * t_facing_wall / 2)
                                                                 - (model.v_shifts_support_wall[row,col] + (w_y_supportwall[row][col] * t_support_wall / 2))) ** 2

        ################################################################################################################
        # single load facing wall
        ################################################################################################################
        # Die Arbeit für einzel Lasten wird aufgestellt

        return sum_obj

    # Festlegen der Nebenbedingung das keine Löcher entstehen können
    def constraint_compatibility_facing_wall(model, row_main, col_main):

        if wall[row_main, col_main] == 0:
            return model.u_shifts_facing_wall[row_main, col_main] == 0

        u_x = np.zeros_like(wall).tolist()  # Epsilonx
        v_y = np.zeros_like(wall).tolist()  # Epsilony

        u_y = np.zeros_like(wall).tolist()  # Gamma = u_y + v_x
        v_x = np.zeros_like(wall).tolist()

        for row in model.row:

            for col in model.col:

                # Verschiebungen nur bei Wawndstücl möglich nicht bei Fenster Ränder getrennt betrachten
                # Verschiebung einfach für Normale Punkte
                if wall[row, col] in [1, 15, 16]:
                    u_x[row][col] = (model.u_shifts_facing_wall[row, col + 1] - model.u_shifts_facing_wall[row, col - 1]) / (2 * delta_x)
                    u_y[row][col] = (model.u_shifts_facing_wall[row - 1, col] - model.u_shifts_facing_wall[row + 1, col]) / (2 * delta_y)

                    v_x[row][col] = (model.v_shifts_facing_wall[row, col + 1] - model.v_shifts_facing_wall[row, col - 1]) / (2 * delta_x)
                    v_y[row][col] = (model.v_shifts_facing_wall[row - 1, col] - model.v_shifts_facing_wall[row + 1, col]) / (2 * delta_y)

                # Rand Links
                if wall[row][col] == 7:
                    u_x[row][col] = (model.u_shifts_facing_wall[row, col + 1] - model.u_shifts_facing_wall[row, col]) / delta_x
                    u_y[row][col] = (model.u_shifts_facing_wall[row - 1, col] - model.u_shifts_facing_wall[row + 1, col]) / (2 * delta_y)

                    v_x[row][col] = (model.v_shifts_facing_wall[row, col + 1] - model.v_shifts_facing_wall[row, col]) / delta_x
                    v_y[row][col] = (model.v_shifts_facing_wall[row - 1, col] - model.v_shifts_facing_wall[row + 1, col]) / (2 * delta_y)

                # Rand Rechts
                elif wall[row][col] == 9:
                    u_x[row][col] = (model.u_shifts_facing_wall[row, col] - model.u_shifts_facing_wall[row, col - 1]) / delta_x
                    u_y[row][col] = (model.u_shifts_facing_wall[row - 1, col] - model.u_shifts_facing_wall[row + 1, col]) / (2 * delta_y)

                    v_x[row][col] = (model.v_shifts_facing_wall[row, col] - model.v_shifts_facing_wall[row, col - 1]) / delta_x
                    v_y[row][col] = (model.v_shifts_facing_wall[row - 1, col] - model.v_shifts_facing_wall[row + 1, col]) / (2 * delta_y)


                # Rand Oben
                elif wall[row][col] == 8:
                    u_x[row][col] = (model.u_shifts_facing_wall[row, col + 1] - model.u_shifts_facing_wall[row, col - 1]) / (2 * delta_x)
                    u_y[row][col] = (model.u_shifts_facing_wall[row + 1, col] - model.u_shifts_facing_wall[row, col]) / delta_y

                    v_x[row][col] = (model.v_shifts_facing_wall[row, col + 1] - model.v_shifts_facing_wall[row, col - 1]) / (2 * delta_x)
                    v_y[row][col] = (model.v_shifts_facing_wall[row + 1, col] - model.v_shifts_facing_wall[row, col]) / delta_y

                # Rand unten
                elif wall[row][col] == 6:
                    u_x[row][col] = (model.u_shifts_facing_wall[row, col + 1] - model.u_shifts_facing_wall[row, col - 1]) / (2 * delta_x)
                    u_y[row][col] = (model.u_shifts_facing_wall[row, col] - model.u_shifts_facing_wall[row - 1, col]) / delta_y

                    v_x[row][col] = (model.v_shifts_facing_wall[row, col + 1] - model.v_shifts_facing_wall[row, col - 1]) / (2 * delta_x)
                    v_y[row][col] = (model.v_shifts_facing_wall[row, col] - model.v_shifts_facing_wall[row - 1, col]) / delta_y

                # Linke obere Ecke
                elif wall[row][col] in [3, 12]:
                    u_x[row][col] = (model.u_shifts_facing_wall[row, col + 1] - model.u_shifts_facing_wall[row, col]) / delta_x
                    u_y[row][col] = (model.u_shifts_facing_wall[row + 1, col] - model.u_shifts_facing_wall[row, col]) / delta_y

                    v_x[row][col] = (model.v_shifts_facing_wall[row, col + 1] - model.v_shifts_facing_wall[row, col]) / delta_x
                    v_y[row][col] = (model.v_shifts_facing_wall[row + 1, col] - model.v_shifts_facing_wall[row, col]) / delta_y

                # Linke untere Ecke
                elif wall[row][col] in [2, 14]:
                    u_x[row][col] = (model.u_shifts_facing_wall[row, col + 1] - model.u_shifts_facing_wall[row, col]) / delta_x
                    u_y[row][col] = (model.u_shifts_facing_wall[row, col] - model.u_shifts_facing_wall[row - 1, col]) / delta_y

                    v_x[row][col] = (model.v_shifts_facing_wall[row, col + 1] - model.v_shifts_facing_wall[row, col]) / delta_x
                    v_y[row][col] = (model.v_shifts_facing_wall[row, col] - model.v_shifts_facing_wall[row - 1, col]) / delta_y

                # Rechte obere Ecke
                elif wall[row][col] in [5, 11]:
                    u_x[row][col] = (model.u_shifts_facing_wall[row, col] - model.u_shifts_facing_wall[row, col - 1]) / delta_x
                    u_y[row][col] = (model.u_shifts_facing_wall[row + 1, col] - model.u_shifts_facing_wall[row, col]) / delta_y

                    v_x[row][col] = (model.v_shifts_facing_wall[row, col] - model.v_shifts_facing_wall[row, col - 1]) / delta_x
                    v_y[row][col] = (model.v_shifts_facing_wall[row + 1, col] - model.v_shifts_facing_wall[row, col]) / delta_y

                # Rechte untere Ecke
                elif wall[row][col] in [4, 13]:
                    u_x[row][col] = (model.u_shifts_facing_wall[row, col] - model.u_shifts_facing_wall[row, col - 1]) / delta_x
                    u_y[row][col] = (model.u_shifts_facing_wall[row, col] - model.u_shifts_facing_wall[row - 1, col]) / delta_y

                    v_x[row][col] = (model.v_shifts_facing_wall[row, col] - model.v_shifts_facing_wall[row, col - 1]) / delta_x
                    v_y[row][col] = (model.v_shifts_facing_wall[row, col] - model.v_shifts_facing_wall[row - 1, col]) / delta_y

        epsilonx_y = np.zeros_like(wall).tolist()
        epsilony_x = np.zeros_like(wall).tolist()
        gamma_x = np.zeros_like(wall).tolist()
        gamma = np.zeros_like(wall).tolist()

        for row in model.row:
            for col in model.col:
                gamma_x[row][col] = u_y[row][col] + v_x[row][col]

        for row in model.row:

            for col in model.col:

                # Verschiebungen nur bei Wawndstücl möglich nicht bei Fenster Ränder getrennt betrachten
                # Verschiebung einfach für Normale Punkte
                if wall[row, col] in [1, 15, 16]:
                    epsilony_x[row][col] = (v_y[row][col + 1] - v_y[row][col - 1]) / (2 * delta_x)
                    epsilonx_y[row][col] = (u_x[row - 1][col] - u_x[row + 1][col]) / (2 * delta_y)

                    gamma_x[row][col] = (gamma[row][col + 1] - gamma[row][col - 1]) / (2 * delta_x)

                # Rand Links
                if wall[row][col] == 7:
                    epsilony_x[row][col] = (v_y[row][col + 1] - v_y[row][col]) / delta_x
                    epsilonx_y[row][col] = (u_x[row - 1][col] - u_x[row + 1][col]) / (2 * delta_y)

                    gamma_x[row][col] = (gamma[row][col + 1] - gamma[row][col]) / delta_x

                # Rand Rechts
                elif wall[row][col] == 9:
                    epsilony_x[row][col] = (v_y[row][col] - v_y[row][col - 1]) / delta_x
                    epsilonx_y[row][col] = (u_x[row - 1][col] - u_x[row + 1][col]) / (2 * delta_y)

                    gamma_x[row][col] = (gamma[row][col] - gamma[row][col - 1]) / delta_x


                # Rand Oben
                elif wall[row][col] == 8:
                    epsilony_x[row][col] = (v_y[row][col + 1] - v_y[row][col - 1]) / (2 * delta_x)
                    epsilonx_y[row][col] = (u_x[row + 1][col] - u_x[row][col]) / delta_y

                    gamma_x[row][col] = (gamma[row][col + 1] - gamma[row][col - 1]) / (2 * delta_x)

                # Rand unten
                elif wall[row][col] == 6:
                    epsilony_x[row][col] = (v_y[row][col + 1] - v_y[row][col - 1]) / (2 * delta_x)
                    epsilonx_y[row][col] = (u_x[row][col] - u_x[row - 1][col]) / delta_y

                    gamma_x[row][col] = (gamma[row][col + 1] - gamma[row][col - 1]) / (2 * delta_x)

                # Linke obere Ecke
                elif wall[row][col] in [3, 12]:
                    epsilony_x[row][col] = (v_y[row][col + 1] - v_y[row][col]) / delta_x
                    epsilonx_y[row][col] = (u_x[row + 1][col] - u_x[row][col]) / delta_y

                    gamma_x[row][col] = (gamma[row][col + 1] - gamma[row][col]) / delta_x

                # Linke untere Ecke
                elif wall[row][col] in [2, 14]:
                    epsilony_x[row][col] = (v_y[row][col + 1] - v_y[row][col]) / delta_x
                    epsilonx_y[row][col] = (u_x[row][col] - u_x[row - 1][col]) / delta_y

                    gamma_x[row][col] = (gamma[row][col + 1] - gamma[row][col]) / delta_x

                # Rechte obere Ecke
                elif wall[row][col] in [5, 11]:
                    epsilony_x[row][col] = (v_y[row][col] - v_y[row][col - 1]) / delta_x
                    epsilonx_y[row][col] = (u_x[row + 1][col] - u_x[row][col]) / delta_y

                    gamma_x[row][col] = (gamma[row][col] - gamma[row][col - 1]) / delta_x

                # Rechte untere Ecke
                elif wall[row][col] in [4, 13]:
                    epsilony_x[row][col] = (v_y[row][col] - v_y[row][col - 1]) / delta_x
                    epsilonx_y[row][col] = (u_x[row][col] - u_x[row - 1][col]) / delta_y

                    gamma_x[row][col] = (gamma[row][col] - gamma[row][col - 1]) / delta_x

        epsilonx__y = np.zeros_like(wall).tolist()
        epsilony__x = np.zeros_like(wall).tolist()
        gamma_x_y = np.zeros_like(wall).tolist()

        for row in model.row:

            for col in model.col:

                # Verschiebungen nur bei Wawndstücl möglich nicht bei Fenster Ränder getrennt betrachten
                # Verschiebung einfach für Normale Punkte
                if wall[row, col] in [1, 15, 16]:
                    epsilony__x[row][col] = (epsilony_x[row][col + 1] - epsilony_x[row][col - 1]) / (2 * delta_x)
                    epsilonx__y[row][col] = (epsilonx_y[row - 1][col] - epsilonx_y[row + 1][col]) / (2 * delta_y)

                    gamma_x_y[row][col] = (gamma_x[row][col + 1] - gamma_x[row][col - 1]) / (2 * delta_x)

                # Rand Links
                if wall[row][col] == 7:
                    epsilony__x[row][col] = (epsilony_x[row][col + 1] - epsilony_x[row][col]) / delta_x
                    epsilonx__y[row][col] = (epsilonx_y[row - 1][col] - epsilonx_y[row + 1][col]) / (2 * delta_y)

                    gamma_x_y[row][col] = (gamma_x[row][col + 1] - gamma_x[row][col]) / delta_x

                # Rand Rechts
                elif wall[row][col] == 9:
                    epsilony__x[row][col] = (epsilony_x[row][col] - epsilony_x[row][col - 1]) / delta_x
                    epsilonx__y[row][col] = (epsilonx_y[row - 1][col] - epsilonx_y[row + 1][col]) / (2 * delta_y)

                    gamma_x_y[row][col] = (gamma_x[row][col] - gamma_x[row][col - 1]) / delta_x


                # Rand Oben
                elif wall[row][col] == 8:
                    epsilony__x[row][col] = (epsilony_x[row][col + 1] - epsilony_x[row][col - 1]) / (2 * delta_x)
                    epsilonx__y[row][col] = (epsilonx_y[row + 1][col] - epsilonx_y[row][col]) / delta_y

                    gamma_x_y[row][col] = (gamma_x[row][col + 1] - gamma_x[row][col - 1]) / (2 * delta_x)

                # Rand unten
                elif wall[row][col] == 6:
                    epsilony__x[row][col] = (epsilony_x[row][col + 1] - epsilony_x[row][col - 1]) / (2 * delta_x)
                    epsilonx__y[row][col] = (epsilonx_y[row][col] - epsilonx_y[row - 1][col]) / delta_y

                    gamma_x_y[row][col] = (gamma_x[row][col + 1] - gamma_x[row][col - 1]) / (2 * delta_x)

                # Linke obere Ecke
                elif wall[row][col] in [3, 12]:
                    epsilony__x[row][col] = (epsilony_x[row][col + 1] - epsilony_x[row][col]) / delta_x
                    epsilonx__y[row][col] = (epsilonx_y[row + 1][col] - epsilonx_y[row][col]) / delta_y

                    gamma_x_y[row][col] = (gamma_x[row][col + 1] - gamma_x[row][col]) / delta_x

                # Linke untere Ecke
                elif wall[row][col] in [2, 14]:
                    epsilony__x[row][col] = (epsilony_x[row][col + 1] - epsilony_x[row][col]) / delta_x
                    epsilonx__y[row][col] = (epsilonx_y[row][col] - epsilonx_y[row - 1][col]) / delta_y

                    gamma_x_y[row][col] = (gamma_x[row][col + 1] - gamma_x[row][col]) / delta_x

                # Rechte obere Ecke
                elif wall[row][col] in [5, 11]:
                    epsilony__x[row][col] = (epsilony_x[row][col] - epsilony_x[row][col - 1]) / delta_x
                    epsilonx__y[row][col] = (epsilonx_y[row + 1][col] - epsilonx_y[row][col]) / delta_y

                    gamma_x_y[row][col] = (gamma_x[row][col] - gamma_x[row][col - 1]) / delta_x

                # Rechte untere Ecke
                elif wall[row][col] in [4, 13]:
                    epsilony__x[row][col] = (epsilony_x[row][col] - epsilony_x[row][col - 1]) / delta_x
                    epsilonx__y[row][col] = (epsilonx_y[row][col] - epsilonx_y[row - 1][col]) / delta_y

                    gamma_x_y[row][col] = (gamma_x[row][col] - gamma_x[row][col - 1]) / delta_x

        return (epsilonx__y[row_main][col_main] + epsilony__x[row_main][col_main] - gamma_x_y[row_main][col_main]) == 0

    def constraint_compatibility_support_wall(model, row_main, col_main):

        if wall[row_main, col_main] == 0:
            return model.u_shifts_support_wall[row_main, col_main] == 0

        u_x = np.zeros_like(wall).tolist()  # Epsilonx
        v_y = np.zeros_like(wall).tolist()  # Epsilony

        u_y = np.zeros_like(wall).tolist()  # Gamma = u_y + v_x
        v_x = np.zeros_like(wall).tolist()

        for row in model.row:

            for col in model.col:

                # Verschiebungen nur bei Wawndstücl möglich nicht bei Fenster Ränder getrennt betrachten
                # Verschiebung einfach für Normale Punkte
                if wall[row, col] in [1, 15, 16]:
                    u_x[row][col] = (model.u_shifts_support_wall[row, col + 1] - model.u_shifts_support_wall[row, col - 1]) / (2 * delta_x)
                    u_y[row][col] = (model.u_shifts_support_wall[row - 1, col] - model.u_shifts_support_wall[row + 1, col]) / (2 * delta_y)

                    v_x[row][col] = (model.v_shifts_support_wall[row, col + 1] - model.v_shifts_support_wall[row, col - 1]) / (2 * delta_x)
                    v_y[row][col] = (model.v_shifts_support_wall[row - 1, col] - model.v_shifts_support_wall[row + 1, col]) / (2 * delta_y)

                # Rand Links
                if wall[row][col] == 7:
                    u_x[row][col] = (model.u_shifts_support_wall[row, col + 1] - model.u_shifts_support_wall[row, col]) / delta_x
                    u_y[row][col] = (model.u_shifts_support_wall[row - 1, col] - model.u_shifts_support_wall[row + 1, col]) / (2 * delta_y)

                    v_x[row][col] = (model.v_shifts_support_wall[row, col + 1] - model.v_shifts_support_wall[row, col]) / delta_x
                    v_y[row][col] = (model.v_shifts_support_wall[row - 1, col] - model.v_shifts_support_wall[row + 1, col]) / (2 * delta_y)

                # Rand Rechts
                elif wall[row][col] == 9:
                    u_x[row][col] = (model.u_shifts_support_wall[row, col] - model.u_shifts_support_wall[row, col - 1]) / delta_x
                    u_y[row][col] = (model.u_shifts_support_wall[row - 1, col] - model.u_shifts_support_wall[row + 1, col]) / (2 * delta_y)

                    v_x[row][col] = (model.v_shifts_support_wall[row, col] - model.v_shifts_support_wall[row, col - 1]) / delta_x
                    v_y[row][col] = (model.v_shifts_support_wall[row - 1, col] - model.v_shifts_support_wall[row + 1, col]) / (2 * delta_y)


                # Rand Oben
                elif wall[row][col] == 8:
                    u_x[row][col] = (model.u_shifts_support_wall[row, col + 1] - model.u_shifts_support_wall[row, col - 1]) / (2 * delta_x)
                    u_y[row][col] = (model.u_shifts_support_wall[row + 1, col] - model.u_shifts_support_wall[row, col]) / delta_y

                    v_x[row][col] = (model.v_shifts_support_wall[row, col + 1] - model.v_shifts_support_wall[row, col - 1]) / (2 * delta_x)
                    v_y[row][col] = (model.v_shifts_support_wall[row + 1, col] - model.v_shifts_support_wall[row, col]) / delta_y

                # Rand unten
                elif wall[row][col] == 6:
                    u_x[row][col] = (model.u_shifts_support_wall[row, col + 1] - model.u_shifts_support_wall[row, col - 1]) / (2 * delta_x)
                    u_y[row][col] = (model.u_shifts_support_wall[row, col] - model.u_shifts_support_wall[row - 1, col]) / delta_y

                    v_x[row][col] = (model.v_shifts_support_wall[row, col + 1] - model.v_shifts_support_wall[row, col - 1]) / (2 * delta_x)
                    v_y[row][col] = (model.v_shifts_support_wall[row, col] - model.v_shifts_support_wall[row - 1, col]) / delta_y

                # Linke obere Ecke
                elif wall[row][col] in [3, 12]:
                    u_x[row][col] = (model.u_shifts_support_wall[row, col + 1] - model.u_shifts_support_wall[row, col]) / delta_x
                    u_y[row][col] = (model.u_shifts_support_wall[row + 1, col] - model.u_shifts_support_wall[row, col]) / delta_y

                    v_x[row][col] = (model.v_shifts_support_wall[row, col + 1] - model.v_shifts_support_wall[row, col]) / delta_x
                    v_y[row][col] = (model.v_shifts_support_wall[row + 1, col] - model.v_shifts_support_wall[row, col]) / delta_y

                # Linke untere Ecke
                elif wall[row][col] in [2, 14]:
                    u_x[row][col] = (model.u_shifts_support_wall[row, col + 1] - model.u_shifts_support_wall[row, col]) / delta_x
                    u_y[row][col] = (model.u_shifts_support_wall[row, col] - model.u_shifts_support_wall[row - 1, col]) / delta_y

                    v_x[row][col] = (model.v_shifts_support_wall[row, col + 1] - model.v_shifts_support_wall[row, col]) / delta_x
                    v_y[row][col] = (model.v_shifts_support_wall[row, col] - model.v_shifts_support_wall[row - 1, col]) / delta_y

                # Rechte obere Ecke
                elif wall[row][col] in [5, 11]:
                    u_x[row][col] = (model.u_shifts_support_wall[row, col] - model.u_shifts_support_wall[row, col - 1]) / delta_x
                    u_y[row][col] = (model.u_shifts_support_wall[row + 1, col] - model.u_shifts_support_wall[row, col]) / delta_y

                    v_x[row][col] = (model.v_shifts_support_wall[row, col] - model.v_shifts_support_wall[row, col - 1]) / delta_x
                    v_y[row][col] = (model.v_shifts_support_wall[row + 1, col] - model.v_shifts_support_wall[row, col]) / delta_y

                # Rechte untere Ecke
                elif wall[row][col] in [4, 13]:
                    u_x[row][col] = (model.u_shifts_support_wall[row, col] - model.u_shifts_support_wall[row, col - 1]) / delta_x
                    u_y[row][col] = (model.u_shifts_support_wall[row, col] - model.u_shifts_support_wall[row - 1, col]) / delta_y

                    v_x[row][col] = (model.v_shifts_support_wall[row, col] - model.v_shifts_support_wall[row, col - 1]) / delta_x
                    v_y[row][col] = (model.v_shifts_support_wall[row, col] - model.v_shifts_support_wall[row - 1, col]) / delta_y

        epsilonx_y = np.zeros_like(wall).tolist()
        epsilony_x = np.zeros_like(wall).tolist()
        gamma_x = np.zeros_like(wall).tolist()
        gamma = np.zeros_like(wall).tolist()

        for row in model.row:
            for col in model.col:
                gamma_x[row][col] = u_y[row][col] + v_x[row][col]

        for row in model.row:

            for col in model.col:

                # Verschiebungen nur bei Wawndstücl möglich nicht bei Fenster Ränder getrennt betrachten
                # Verschiebung einfach für Normale Punkte
                if wall[row, col] in [1, 15, 16]:
                    epsilony_x[row][col] = (v_y[row][col + 1] - v_y[row][col - 1]) / (2 * delta_x)
                    epsilonx_y[row][col] = (u_x[row - 1][col] - u_x[row + 1][col]) / (2 * delta_y)

                    gamma_x[row][col] = (gamma[row][col + 1] - gamma[row][col - 1]) / (2 * delta_x)

                # Rand Links
                if wall[row][col] == 7:
                    epsilony_x[row][col] = (v_y[row][col + 1] - v_y[row][col]) / delta_x
                    epsilonx_y[row][col] = (u_x[row - 1][col] - u_x[row + 1][col]) / (2 * delta_y)

                    gamma_x[row][col] = (gamma[row][col + 1] - gamma[row][col]) / delta_x

                # Rand Rechts
                elif wall[row][col] == 9:
                    epsilony_x[row][col] = (v_y[row][col] - v_y[row][col - 1]) / delta_x
                    epsilonx_y[row][col] = (u_x[row - 1][col] - u_x[row + 1][col]) / (2 * delta_y)

                    gamma_x[row][col] = (gamma[row][col] - gamma[row][col - 1]) / delta_x


                # Rand Oben
                elif wall[row][col] == 8:
                    epsilony_x[row][col] = (v_y[row][col + 1] - v_y[row][col - 1]) / (2 * delta_x)
                    epsilonx_y[row][col] = (u_x[row + 1][col] - u_x[row][col]) / delta_y

                    gamma_x[row][col] = (gamma[row][col + 1] - gamma[row][col - 1]) / (2 * delta_x)

                # Rand unten
                elif wall[row][col] == 6:
                    epsilony_x[row][col] = (v_y[row][col + 1] - v_y[row][col - 1]) / (2 * delta_x)
                    epsilonx_y[row][col] = (u_x[row][col] - u_x[row - 1][col]) / delta_y

                    gamma_x[row][col] = (gamma[row][col + 1] - gamma[row][col - 1]) / (2 * delta_x)

                # Linke obere Ecke
                elif wall[row][col] in [3, 12]:
                    epsilony_x[row][col] = (v_y[row][col + 1] - v_y[row][col]) / delta_x
                    epsilonx_y[row][col] = (u_x[row + 1][col] - u_x[row][col]) / delta_y

                    gamma_x[row][col] = (gamma[row][col + 1] - gamma[row][col]) / delta_x

                # Linke untere Ecke
                elif wall[row][col] in [2, 14]:
                    epsilony_x[row][col] = (v_y[row][col + 1] - v_y[row][col]) / delta_x
                    epsilonx_y[row][col] = (u_x[row][col] - u_x[row - 1][col]) / delta_y

                    gamma_x[row][col] = (gamma[row][col + 1] - gamma[row][col]) / delta_x

                # Rechte obere Ecke
                elif wall[row][col] in [5, 11]:
                    epsilony_x[row][col] = (v_y[row][col] - v_y[row][col - 1]) / delta_x
                    epsilonx_y[row][col] = (u_x[row + 1][col] - u_x[row][col]) / delta_y

                    gamma_x[row][col] = (gamma[row][col] - gamma[row][col - 1]) / delta_x

                # Rechte untere Ecke
                elif wall[row][col] in [4, 13]:
                    epsilony_x[row][col] = (v_y[row][col] - v_y[row][col - 1]) / delta_x
                    epsilonx_y[row][col] = (u_x[row][col] - u_x[row - 1][col]) / delta_y

                    gamma_x[row][col] = (gamma[row][col] - gamma[row][col - 1]) / delta_x

        epsilonx__y = np.zeros_like(wall).tolist()
        epsilony__x = np.zeros_like(wall).tolist()
        gamma_x_y = np.zeros_like(wall).tolist()

        for row in model.row:

            for col in model.col:

                # Verschiebungen nur bei Wawndstücl möglich nicht bei Fenster Ränder getrennt betrachten
                # Verschiebung einfach für Normale Punkte
                if wall[row, col] in [1, 15, 16]:
                    epsilony__x[row][col] = (epsilony_x[row][col + 1] - epsilony_x[row][col - 1]) / (2 * delta_x)
                    epsilonx__y[row][col] = (epsilonx_y[row - 1][col] - epsilonx_y[row + 1][col]) / (2 * delta_y)

                    gamma_x_y[row][col] = (gamma_x[row][col + 1] - gamma_x[row][col - 1]) / (2 * delta_x)

                # Rand Links
                if wall[row][col] == 7:
                    epsilony__x[row][col] = (epsilony_x[row][col + 1] - epsilony_x[row][col]) / delta_x
                    epsilonx__y[row][col] = (epsilonx_y[row - 1][col] - epsilonx_y[row + 1][col]) / (2 * delta_y)

                    gamma_x_y[row][col] = (gamma_x[row][col + 1] - gamma_x[row][col]) / delta_x

                # Rand Rechts
                elif wall[row][col] == 9:
                    epsilony__x[row][col] = (epsilony_x[row][col] - epsilony_x[row][col - 1]) / delta_x
                    epsilonx__y[row][col] = (epsilonx_y[row - 1][col] - epsilonx_y[row + 1][col]) / (2 * delta_y)

                    gamma_x_y[row][col] = (gamma_x[row][col] - gamma_x[row][col - 1]) / delta_x


                # Rand Oben
                elif wall[row][col] == 8:
                    epsilony__x[row][col] = (epsilony_x[row][col + 1] - epsilony_x[row][col - 1]) / (2 * delta_x)
                    epsilonx__y[row][col] = (epsilonx_y[row + 1][col] - epsilonx_y[row][col]) / delta_y

                    gamma_x_y[row][col] = (gamma_x[row][col + 1] - gamma_x[row][col - 1]) / (2 * delta_x)

                # Rand unten
                elif wall[row][col] == 6:
                    epsilony__x[row][col] = (epsilony_x[row][col + 1] - epsilony_x[row][col - 1]) / (2 * delta_x)
                    epsilonx__y[row][col] = (epsilonx_y[row][col] - epsilonx_y[row - 1][col]) / delta_y

                    gamma_x_y[row][col] = (gamma_x[row][col + 1] - gamma_x[row][col - 1]) / (2 * delta_x)

                # Linke obere Ecke
                elif wall[row][col] in [3, 12]:
                    epsilony__x[row][col] = (epsilony_x[row][col + 1] - epsilony_x[row][col]) / delta_x
                    epsilonx__y[row][col] = (epsilonx_y[row + 1][col] - epsilonx_y[row][col]) / delta_y

                    gamma_x_y[row][col] = (gamma_x[row][col + 1] - gamma_x[row][col]) / delta_x

                # Linke untere Ecke
                elif wall[row][col] in [2, 14]:
                    epsilony__x[row][col] = (epsilony_x[row][col + 1] - epsilony_x[row][col]) / delta_x
                    epsilonx__y[row][col] = (epsilonx_y[row][col] - epsilonx_y[row - 1][col]) / delta_y

                    gamma_x_y[row][col] = (gamma_x[row][col + 1] - gamma_x[row][col]) / delta_x

                # Rechte obere Ecke
                elif wall[row][col] in [5, 11]:
                    epsilony__x[row][col] = (epsilony_x[row][col] - epsilony_x[row][col - 1]) / delta_x
                    epsilonx__y[row][col] = (epsilonx_y[row + 1][col] - epsilonx_y[row][col]) / delta_y

                    gamma_x_y[row][col] = (gamma_x[row][col] - gamma_x[row][col - 1]) / delta_x

                # Rechte untere Ecke
                elif wall[row][col] in [4, 13]:
                    epsilony__x[row][col] = (epsilony_x[row][col] - epsilony_x[row][col - 1]) / delta_x
                    epsilonx__y[row][col] = (epsilonx_y[row][col] - epsilonx_y[row - 1][col]) / delta_y

                    gamma_x_y[row][col] = (gamma_x[row][col] - gamma_x[row][col - 1]) / delta_x

        return (epsilonx__y[row_main][col_main] + epsilony__x[row_main][col_main] - gamma_x_y[row_main][col_main]) == 0

    # Mit dieser Nebenbedingung soll die Trivial Lösung (verschiebung gleich null) verhindert werden
    def constraint_moment(model, row_main, col_main):
        w2x_facingwall = np.zeros_like(wall).tolist()
        w2y_facingwall = np.zeros_like(wall).tolist()
        w2xy_facingwall = np.zeros_like(wall).tolist()

        for row in model.row:

            for col in model.col:

                wall_type = wall[row, col]

                if not wall_type == 0:

                    # w2x

                    if wall_type in [1, 15, 8, 6, 11, 12, 13, 14, 16]:  # 2.14

                        w2x_facingwall[row][col] = (model.shifts_facing_wall[row, col + 1] - 2 *
                                                    model.shifts_facing_wall[row, col] + model.shifts_facing_wall[
                                                        row, col - 1]) / (delta_x ** 2)

                    elif wall_type in [9, 5, 4]:  # 2.17

                        w2x_facingwall[row][col] = 0

                    else:  # 2.21

                        w2x_facingwall[row][col] = 0

                    # w2y

                    if wall_type in [1, 15, 9, 7, 11, 12, 13, 14, 16]:  # 2.15

                        w2y_facingwall[row][col] = (model.shifts_facing_wall[row + 1, col] - 2 *
                                                    model.shifts_facing_wall[row, col] + model.shifts_facing_wall[
                                                        row - 1, col]) / (delta_y ** 2)

                    elif wall_type in [8, 5, 3]:  # 2.25

                        w2y_facingwall[row][col] = 0

                    else:  # 2.29

                        w2y_facingwall[row][col] = 0

                    # w2xy

                    if wall_type in [1, 15, 16]:  # 2.13

                        w2xy_facingwall[row][col] = (model.shifts_facing_wall[row + 1, col + 1] -
                                                     model.shifts_facing_wall[row - 1, col + 1] -
                                                     model.shifts_facing_wall[
                                                         row + 1, col - 1] + model.shifts_facing_wall[
                                                         row - 1, col - 1]) / (
                                                            4 * delta_x * delta_y)

                    elif wall_type == 9:  # 2.16

                        w2xy_facingwall[row][col] = (2 * model.shifts_facing_wall[row + 1, col] - 2 *
                                                     model.shifts_facing_wall[row - 1, col] - 2 *
                                                     model.shifts_facing_wall[
                                                         row + 1, col - 1] + 2 * model.shifts_facing_wall[
                                                         row - 1, col - 1]) / (
                                                            4 * delta_x * delta_y)

                    elif wall_type == 7:  # 2.20

                        w2xy_facingwall[row][col] = (2 * model.shifts_facing_wall[row - 1, col] - 2 *
                                                     model.shifts_facing_wall[row + 1, col] - 2 *
                                                     model.shifts_facing_wall[
                                                         row - 1, col + 1] + 2 * model.shifts_facing_wall[
                                                         row + 1, col + 1]) / (
                                                            4 * delta_x * delta_y)

                    elif wall_type == 8:  # 2.24

                        w2xy_facingwall[row][col] = (2 * model.shifts_facing_wall[row, col - 1] - 2 *
                                                     model.shifts_facing_wall[row, col + 1] - 2 *
                                                     model.shifts_facing_wall[
                                                         row + 1, col - 1] + 2 * model.shifts_facing_wall[
                                                         row + 1, col + 1]) / (
                                                            4 * delta_x * delta_y)

                    elif wall_type == 6:  # 2.28

                        w2xy_facingwall[row][col] = (2 * model.shifts_facing_wall[row, col + 1] - 2 *
                                                     model.shifts_facing_wall[row, col - 1] - 2 *
                                                     model.shifts_facing_wall[
                                                         row - 1, col + 1] + 2 * model.shifts_facing_wall[
                                                         row - 1, col - 1]) / (
                                                            4 * delta_x * delta_y)

                    elif wall_type == 5:  # 2.32

                        w2xy_facingwall[row][col] = 0

                    elif wall_type == 3:  # 2.35

                        w2xy_facingwall[row][col] = 0

                    elif wall_type == 4:  # 2.36

                        w2xy_facingwall[row][col] = 0

                    elif wall_type == 2:  # 2.37

                        w2xy_facingwall[row][col] = 0

                    elif wall_type == 11:  # 2.40
                        w2xy_facingwall[row][col] = (2 * model.shifts_facing_wall[row - 1, col] - 2 *
                                                     model.shifts_facing_wall[row - 1, col + 1] -
                                                     model.shifts_facing_wall[
                                                         row + 1, col - 1] + model.shifts_facing_wall[
                                                         row + 1, col + 1]) / (
                                                            4 * delta_x * delta_y)

                    elif wall_type == 12:  # 2.41
                        w2xy_facingwall[row][col] = (2 * model.shifts_facing_wall[row - 1, col - 1] -
                                                     model.shifts_facing_wall[row - 1, col] -
                                                     model.shifts_facing_wall[
                                                         row + 1, col - 1] + model.shifts_facing_wall[
                                                         row + 1, col + 1]) / (
                                                            4 * delta_x * delta_y)

                    elif wall_type == 13:  # 2.38
                        w2xy_facingwall[row][col] = (2 * model.shifts_facing_wall[row + 1, col + 1] -
                                                     model.shifts_facing_wall[row - 1, col + 1] - 2 *
                                                     model.shifts_facing_wall[row + 1, col] + model.shifts_facing_wall[
                                                         row - 1, col - 1]) / (
                                                            4 * delta_x * delta_y)

                    elif wall_type == 14:  # 2.39
                        w2xy_facingwall[row][col] = (2 * model.shifts_facing_wall[row + 1, col] - 2 *
                                                     model.shifts_facing_wall[row + 1, col - 1] -
                                                     model.shifts_facing_wall[
                                                         row - 1, col + 1] + model.shifts_facing_wall[
                                                         row - 1, col - 1]) / (
                                                            4 * delta_x * delta_y)

        mx_facingwall = np.zeros_like(wall).tolist()
        my_facingwall = np.zeros_like(wall).tolist()
        mxy_facingwall = np.zeros_like(wall).tolist()

        for row in model.row:

            for col in model.col:

                wall_type = wall[row, col]

                if not wall_type == 0:
                    mx_facingwall[row][col] = - D_facing_wall * (
                                w2x_facingwall[row][col] + v * w2y_facingwall[row][col]) - M_star_facing_wall / (1 - v)
                    my_facingwall[row][col] = - D_facing_wall * (
                                w2y_facingwall[row][col] + v * w2x_facingwall[row][col]) - M_star_facing_wall / (1 - v)
                    mxy_facingwall[row][col] = -D_facing_wall * (1 - v) * w2xy_facingwall[row][col]

        mx2x_facingwall = np.zeros_like(wall).tolist()
        my2y_facingwall = np.zeros_like(wall).tolist()
        mxy2xy_facingwall = np.zeros_like(wall).tolist()

        for row in model.row:

            for col in model.col:

                wall_type = wall[row, col]

                if not wall_type == 0:

                    # w2x

                    if wall_type in [1, 15, 8, 6, 11, 12, 13, 14, 16]:  # 2.14

                        mx2x_facingwall[row][col] = (mx_facingwall[row][ col + 1] - 2 *
                                                    mx_facingwall[row][col] + mx_facingwall[
                                                        row][ col - 1]) / (delta_x ** 2)

                    # w2y

                    if wall_type in [1, 15, 9, 7, 11, 12, 13, 14, 16]:  # 2.15

                        my2y_facingwall[row][col] = (my_facingwall[row + 1][ col] - 2 *
                                                    my_facingwall[row][ col] + my_facingwall[
                                                        row - 1][col]) / (delta_y ** 2)

                    # w2xy

                    if wall_type in [1, 15, 16]:  # 2.13

                        mxy2xy_facingwall[row][col] = (mxy_facingwall[row + 1][ col + 1] -
                                                     mxy_facingwall[row - 1][ col + 1] -
                                                     mxy_facingwall[
                                                         row + 1][ col - 1] + mxy_facingwall[
                                                         row - 1][ col - 1]) / (
                                                            4 * delta_x * delta_y)

                    elif wall_type == 9:  # 2.16

                        mxy2xy_facingwall[row][col] = (2 * mxy_facingwall[row + 1][ col] - 2 *
                                                     mxy_facingwall[row - 1][ col] - 2 *
                                                     mxy_facingwall[
                                                         row + 1][ col - 1] + 2 * mxy_facingwall[
                                                         row - 1][ col - 1]) / (
                                                            4 * delta_x * delta_y)

                    elif wall_type == 7:  # 2.20

                        mxy2xy_facingwall[row][col] = (2 * mxy_facingwall[row - 1][ col] - 2 *
                                                     mxy_facingwall[row + 1][ col] - 2 *
                                                     mxy_facingwall[
                                                         row - 1][ col + 1] + 2 * mxy_facingwall[
                                                         row + 1][ col + 1]) / (
                                                            4 * delta_x * delta_y)

                    elif wall_type == 8:  # 2.24

                        mxy2xy_facingwall[row][col] = (2 * mxy_facingwall[row][ col - 1] - 2 *
                                                     mxy_facingwall[row][ col + 1] - 2 *
                                                     mxy_facingwall[
                                                         row + 1][ col - 1] + 2 * mxy_facingwall[
                                                         row + 1][ col + 1]) / (
                                                            4 * delta_x * delta_y)

                    elif wall_type == 6:  # 2.28

                        mxy2xy_facingwall[row][col] = (2 * mxy_facingwall[row][ col + 1] - 2 *
                                                     mxy_facingwall[row][ col - 1] - 2 *
                                                     mxy_facingwall[
                                                         row - 1][ col + 1] + 2 * mxy_facingwall[
                                                         row - 1][ col - 1]) / (
                                                            4 * delta_x * delta_y)

                    elif wall_type == 11:  # 2.40
                        mxy2xy_facingwall[row][col] = (2 * mxy_facingwall[row - 1][ col] - 2 *
                                                     mxy_facingwall[row - 1][ col + 1] -
                                                     mxy_facingwall[
                                                         row + 1][ col - 1] + mxy_facingwall[
                                                         row + 1][ col + 1]) / (
                                                            4 * delta_x * delta_y)

                    elif wall_type == 12:  # 2.41
                        mxy2xy_facingwall[row][col] = (2 * mxy_facingwall[row - 1][ col - 1] -
                                                     mxy_facingwall[row - 1][ col] -
                                                     mxy_facingwall[
                                                         row + 1][ col - 1] + mxy_facingwall[
                                                         row + 1][ col + 1]) / (
                                                            4 * delta_x * delta_y)

                    elif wall_type == 13:  # 2.38
                        mxy2xy_facingwall[row][col] = (2 * mxy_facingwall[row + 1][ col + 1] -
                                                     mxy_facingwall[row - 1][ col + 1] - 2 *
                                                     mxy_facingwall[row + 1][ col] + mxy_facingwall[
                                                         row - 1][ col - 1]) / (
                                                            4 * delta_x * delta_y)

                    elif wall_type == 14:  # 2.39
                        mxy2xy_facingwall[row][col] = (2 * mxy_facingwall[row + 1][ col] - 2 *
                                                     mxy_facingwall[row + 1][ col - 1] -
                                                     mxy_facingwall[
                                                         row - 1][ col + 1] + mxy_facingwall[
                                                         row - 1][ col - 1]) / (
                                                            4 * delta_x * delta_y)

        return 0 == mx2x_facingwall[row_main][col_main] + 2 * mxy2xy_facingwall[row_main][col_main] + my2y_facingwall[row_main][col_main] +\
               force[row][col] + single_load_z[row_main][col_main] / (delta_x * delta_y) +\
               (model.shifts_facing_wall[row_main, col_main] - model.shifts_support_wall[row_main, col_main]) / (delta_x * delta_y) * material_laws["Kunststoff1"]["Cz"]

    # Geometrische Ranbedigung für die Verschiebungen der beiden Schalen
    # Für aufstehende Vorsatzschale
    def constraint_facing_wall_bot_u(model, index_col):
        return model.u_shifts_facing_wall[m-1, index_col] == 0

    def constraint_facing_wall_bot_v(model, index_col):
        return model.v_shifts_facing_wall[m-1, index_col] == 0

    # Freihängende Vorsatzschale

    def constraint_panel_construct_top_w(model, index_col):
        return model.shifts_support_wall[0, index_col] == 0

    def constraint_panel_construct_bot_w(model, index_col):
        return model.shifts_support_wall[m-1, index_col] == 0

    def constraint_panel_construct_bot_u(model, index_col):
        return model.u_shifts_support_wall[m-1, index_col] == 0

    def constraint_panel_construct_bot_v(model, index_col):
        return model.v_shifts_support_wall[m-1, index_col] == 0


########################################################################################################################
# Setting up Model #####################################################################################################
########################################################################################################################


    # Hier wird das Pyomo Modell dann gebaut
    model = pyo.ConcreteModel()

    model.row = pyo.RangeSet(0, m - 1)
    model.col = pyo.RangeSet(0, n - 1)

    model.shifts_facing_wall = pyo.Var(model.row, model.col, domain=pyo.Reals, bounds=(-50, 50), initialize=0)
    model.shifts_support_wall = pyo.Var(model.row, model.col, domain=pyo.Reals, bounds=(-50, 50),
                                        initialize=0)       # 100

    # Scheibentheorie Verzehrungen
    model.u_shifts_facing_wall = pyo.Var(model.row, model.col, domain=pyo.Reals, bounds=(-2147483647, 2147483647), initialize=0)
    model.v_shifts_facing_wall = pyo.Var(model.row, model.col, domain=pyo.Reals, bounds=(-2147483647, 2147483647), initialize=0)

    model.u_shifts_support_wall = pyo.Var(model.row, model.col, domain=pyo.Reals, bounds=(-2147483647, 2147483647),
                                          initialize=0)
    model.v_shifts_support_wall = pyo.Var(model.row, model.col, domain=pyo.Reals, bounds=(-2147483647, 2147483647),
                                          initialize=0)

    model.index_counter = pyo.RangeSet(0, len(index_anker[0]) - 1)

    model.constraint_compatibility_facing_wall = pyo.Constraint(model.row, model.col,
                                                                rule=constraint_compatibility_facing_wall)
    model.constraint_compatibility_support_wall = pyo.Constraint(model.row, model.col,
                                                                 rule=constraint_compatibility_support_wall)

    model.constraint_moment = pyo.Constraint(model.row, model.col, rule=constraint_moment)

    # Randbedingungen
    if build_type == "aufstehende Vorsatzschale":
        model.constraint_facing_wall_bot_u = pyo.Constraint(model.col, rule=constraint_facing_wall_bot_u)
        model.constraint_facing_wall_bot_v = pyo.Constraint(model.col, rule=constraint_facing_wall_bot_v)

    model.constraint_panel_construct_top_w = pyo.Constraint(model.col, rule=constraint_panel_construct_top_w)
    model.constraint_panel_construct_bot_w = pyo.Constraint(model.col, rule=constraint_panel_construct_bot_w)
    model.constraint_panel_construct_bot_u = pyo.Constraint(model.col, rule=constraint_panel_construct_bot_u)
    model.constraint_panel_construct_bot_v = pyo.Constraint(model.col, rule=constraint_panel_construct_bot_v)


    model.obj = pyo.Objective(rule=obj)
    # instance = model

    # Lösen des Models
    opt = pyo.SolverFactory('ipopt')
    # results = opt.solve(model, tee=True)
    # opt.options['print_level'] = 12
    opt.options['output_file'] = "my_logfile.txt"
    opt.options['max_iter'] = 1000
    results = opt.solve(model, tee=True)

########################################################################################################################
# Showing results ######################################################################################################
########################################################################################################################
    # Erstellen der Ausgabe
    w_shifts = []
    for row in range(m):
        mini = []
        for col in range(n):
            mini.append(model.shifts_facing_wall[row, col].value)
        w_shifts.append(mini)
    w_shifts = np.array(w_shifts)
    w_shifts_facing_wall = w_shifts.copy()

    u_shifts = np.zeros_like(w_shifts)
    v_shifts = np.zeros_like(w_shifts)

    for row in range(m):
        for col in range(n):
            u_shifts[row, col] = model.u_shifts_facing_wall[row, col].value
            v_shifts[row, col] = model.v_shifts_facing_wall[row, col].value

    results.write()
    model.display("result_pyomo.txt")

    save_file(w_shifts, "Vorsatzschale/w", "Verschiebung in mm", index_anker)

    save_file(u_shifts, "Vorsatzschale/u", "Verschiebung in mm", index_anker)

    save_file(v_shifts, "Vorsatzschale/v", "Verschiebung in mm", index_anker)

    w2x = np.zeros_like(wall).tolist()
    w2y = np.zeros_like(wall).tolist()
    w2xy = np.zeros_like(wall).tolist()

    for row in range(m):

        for col in range(n):

            wall_type = wall[row, col]

            if not wall_type == 0:

                # w2x

                if wall_type in [1, 15, 8, 6, 11, 12, 13, 14, 16]:  # 2.14

                    w2x[row][col] = (w_shifts[row, col + 1] - 2 * w_shifts[row, col] + w_shifts[
                        row, col - 1]) / (delta_x ** 2)

                elif wall_type in [9, 5, 4]:  # 2.17

                    w2x[row][col] = 0

                else:  # 2.21

                    w2x[row][col] = 0

                # w2y

                if wall_type in [1, 15, 9, 7, 11, 12, 13, 14, 16]:  # 2.15

                    w2y[row][col] = (w_shifts[row + 1, col] - 2 * w_shifts[row, col] + w_shifts[
                        row - 1, col]) / (delta_y ** 2)

                elif wall_type in [8, 5, 3]:  # 2.25

                    w2y[row][col] = 0

                else:  # 2.29

                    w2y[row][col] = 0

                # w2xy

                if wall_type in [1, 15, 16]:  # 2.13

                    w2xy[row][col] = (w_shifts[row + 1, col + 1] - w_shifts[row - 1, col + 1] -
                                      w_shifts[
                                          row + 1, col - 1] + w_shifts[row - 1, col - 1]) / (
                                             4 * delta_x * delta_y)

                elif wall_type == 9:  # 2.16

                    w2xy[row][col] = (2 * w_shifts[row + 1, col] - 2 * w_shifts[row - 1, col] - 2 *
                                      w_shifts[
                                          row + 1, col - 1] + 2 * w_shifts[row - 1, col - 1]) / (
                                             4 * delta_x * delta_y)

                elif wall_type == 7:  # 2.20

                    w2xy[row][col] = (2 * w_shifts[row - 1, col] - 2 * w_shifts[row + 1, col] - 2 *
                                      w_shifts[
                                          row - 1, col + 1] + 2 * w_shifts[row + 1, col + 1]) / (
                                             4 * delta_x * delta_y)

                elif wall_type == 8:  # 2.24

                    w2xy[row][col] = (2 * w_shifts[row, col - 1] - 2 * w_shifts[row, col + 1] - 2 *
                                      w_shifts[
                                          row + 1, col - 1] + 2 * w_shifts[row + 1, col + 1]) / (
                                             4 * delta_x * delta_y)

                elif wall_type == 6:  # 2.28

                    w2xy[row][col] = (2 * w_shifts[row, col + 1] - 2 * w_shifts[row, col - 1] - 2 *
                                      w_shifts[
                                          row - 1, col + 1] + 2 * w_shifts[row - 1, col - 1]) / (
                                             4 * delta_x * delta_y)

                elif wall_type == 5:  # 2.32

                    w2xy[row][col] = 0

                elif wall_type == 3:  # 2.35

                    w2xy[row][col] = 0

                elif wall_type == 4:  # 2.36

                    w2xy[row][col] = 0

                elif wall_type == 2:  # 2.37

                    w2xy[row][col] = 0

                elif wall_type == 11:  # 2.40
                    w2xy[row][col] = (2 * w_shifts[row - 1, col] - 2 * w_shifts[row - 1, col + 1] -
                                      w_shifts[
                                          row + 1, col - 1] + w_shifts[row + 1, col + 1]) / (
                                             4 * delta_x * delta_y)

                elif wall_type == 12:  # 2.41
                    w2xy[row][col] = (2 * w_shifts[row - 1, col - 1] - w_shifts[row - 1, col] -
                                      w_shifts[
                                          row + 1, col - 1] + w_shifts[row + 1, col + 1]) / (
                                             4 * delta_x * delta_y)

                elif wall_type == 13:  # 2.38
                    w2xy[row][col] = (2 * w_shifts[row + 1, col + 1] - w_shifts[row - 1, col + 1] - 2 *
                                      w_shifts[row + 1, col] + w_shifts[row - 1, col - 1]) / (
                                             4 * delta_x * delta_y)

                elif wall_type == 14:  # 2.39
                    w2xy[row][col] = (2 * w_shifts[row + 1, col] - 2 * w_shifts[row + 1, col - 1] -
                                      w_shifts[
                                          row - 1, col + 1] + w_shifts[row - 1, col - 1]) / (
                                             4 * delta_x * delta_y)

    mx = np.zeros_like(wall).tolist()
    my = np.zeros_like(wall).tolist()
    mxy = np.zeros_like(wall).tolist()

    for row in range(m):

        for col in range(n):

            wall_type = wall[row, col]

            if not wall_type == 0:
                mx[row][col] = - D_facing_wall * (w2x[row][col] + v * w2y[row][col]) - M_star_facing_wall / (1 - v)
                my[row][col] = - D_facing_wall * (w2y[row][col] + v * w2x[row][col]) - M_star_facing_wall / (1 - v)
                mxy[row][col] = -D_facing_wall * (1 - v) * w2xy[row][col]

    save_file(mx, "Vorsatzschale/mx", "mx in N/mm", index_anker)
    save_file(my, "Vorsatzschale/my", "my in N/mm", index_anker)
    save_file(mxy, "Vorsatzschale/mxy", "mxy in N/mm", index_anker)

    qx = np.zeros_like(wall).tolist()
    qy = np.zeros_like(wall).tolist()

    for row in range(m):
        for col in range(n):

            # MItte der Platte
            if wall[row][col] in [1, 15, 16]:

                qx[row][col] = (- D_facing_wall * (
                        w2x[row][col + 1] - w2x[row][col - 1] + w2y[row][col + 1] - w2y[row][col - 1])) / (
                                       2 * delta_x)
                qy[row][col] = (- D_facing_wall * (
                        w2x[row + 1][col] - w2x[row - 1][col] + w2y[row + 1][col] - w2y[row - 1][col])) / (
                                       2 * delta_y)

            # Linke untere Ecke
            elif wall[row][col] in [2, 14]:
                qx[row][col] = (- D_facing_wall * (w2x[row][col + 1] - w2x[row][col] + w2y[row][col + 1] - w2y[row][col])) / delta_x
                qy[row][col] = (- D_facing_wall * (w2x[row][col] - w2x[row - 1][col] + w2y[row][col] - w2y[row - 1][col])) / delta_y

            # Linke obere Ecke
            elif wall[row][col] in [3, 12]:
                qx[row][col] = (- D_facing_wall * (w2x[row][col + 1] - w2x[row][col] + w2y[row][col + 1] - w2y[row][col])) / delta_x
                qy[row][col] = (- D_facing_wall * (w2x[row + 1][col] - w2x[row][col] + w2y[row + 1][col] - w2y[row][col])) / delta_y

            # Rechte untere Ecke
            elif wall[row][col] in [4, 13]:
                qx[row][col] = (- D_facing_wall * (w2x[row][col] - w2x[row][col - 1] + w2y[row][col] - w2y[row][col - 1])) / delta_x
                qy[row][col] = (- D_facing_wall * (w2x[row][col] - w2x[row - 1][col] + w2y[row][col] - w2y[row - 1][col])) / delta_y

            # Rechte obere Ecke
            elif wall[row][col] in [5, 11]:
                qx[row][col] = (- D_facing_wall * (
                        w2x[row][col] - w2x[row][col - 1] + w2y[row][col] - w2y[row][col - 1])) / delta_x
                qy[row][col] = (- D_facing_wall * (
                        w2x[row + 1][col] - w2x[row][col] + w2y[row + 1][col] - w2y[row][col])) / delta_y

            # Rand unten
            elif wall[row][col] == 6:
                qx[row][col] = (- D_facing_wall * (
                        w2x[row][col + 1] - w2x[row][col - 1] + w2y[row][col + 1] - w2y[row][col - 1])) / (
                                       2 * delta_x)
                qy[row][col] = (- D_facing_wall * (w2x[row][col] - w2x[row - 1][col] + w2y[row][col] - w2y[row - 1][col])) / delta_y

            # Rand Links
            elif wall[row][col] == 7:
                qx[row][col] = (- D_facing_wall * (w2x[row][col + 1] - w2x[row][col] + w2y[row][col + 1] - w2y[row][col])) / delta_x
                qy[row][col] = (- D_facing_wall * (
                        w2x[row + 1][col] - w2x[row - 1][col] + w2y[row + 1][col] - w2y[row - 1][col])) / (
                                       2 * delta_y)

            # Rand oben
            elif wall[row][col] == 8:
                qx[row][col] = (- D_facing_wall * (
                        w2x[row][col + 1] - w2x[row][col - 1] + w2y[row][col + 1] - w2y[row][col - 1])) / (
                                       2 * delta_x)
                qy[row][col] = (- D_facing_wall * (w2x[row + 1][col] - w2x[row][col] + w2y[row + 1][col] - w2y[row][col])) / delta_y

            # Rand Rechts
            elif wall[row][col] == 9:
                qx[row][col] = (- D_facing_wall * (w2x[row][col] - w2x[row][col - 1] + w2y[row][col] - w2y[row][col - 1])) / delta_x
                qy[row][col] = (- D_facing_wall * (
                        w2x[row + 1][col] - w2x[row - 1][col] + w2y[row + 1][col] - w2y[row - 1][col])) / (
                                       2 * delta_y)

    u_x = np.zeros_like(wall).tolist()  # Epsilonx
    v_y = np.zeros_like(wall).tolist()  # Epsilony

    u_y = np.zeros_like(wall).tolist()  # Gamma = u_y + v_x
    v_x = np.zeros_like(wall).tolist()

    for row in model.row:

        for col in model.col:
            wall_type = wall[row, col]
            if wall_type != 0:
                # u_x
                if wall_type in [9, 4, 5]:
                    u_x[row][col] = (model.u_shifts_facing_wall[row, col].value - model.u_shifts_facing_wall[
                        row, col - 1].value) / delta_x
                else:
                    u_x[row][col] = (model.u_shifts_facing_wall[row, col + 1].value - model.u_shifts_facing_wall[
                        row, col].value) / delta_x
                # v_x
                if wall_type in [9, 4, 5]:
                    v_x[row][col] = (model.v_shifts_facing_wall[row, col].value - model.v_shifts_facing_wall[
                        row, col - 1].value) / delta_x
                else:
                    v_x[row][col] = (model.v_shifts_facing_wall[row, col + 1].value - model.v_shifts_facing_wall[
                        row, col].value) / delta_x
                # u_y
                if wall_type in [6, 2, 4]:
                    u_y[row][col] = (model.u_shifts_facing_wall[row, col].value - model.u_shifts_facing_wall[
                        row - 1, col].value) / delta_y
                else:
                    u_y[row][col] = (model.u_shifts_facing_wall[row + 1, col].value - model.u_shifts_facing_wall[
                        row, col].value) / delta_y
                # v_y
                if wall_type in [6, 2, 4]:
                    v_y[row][col] = (model.v_shifts_facing_wall[row, col].value - model.v_shifts_facing_wall[
                        row - 1, col].value) / delta_y
                else:
                    v_y[row][col] = (model.v_shifts_facing_wall[row + 1, col].value - model.v_shifts_facing_wall[
                        row, col].value) / delta_y

    nx = np.zeros_like(wall).tolist()
    ny = np.zeros_like(wall).tolist()
    nxy = np.zeros_like(wall).tolist()

    for row in model.row:
        for col in model.col:
            nx[row][col] = Ds_facing_wall * ((u_x[row][col] - alpha_t * temp_diff) + v * (v_y[row][col] - alpha_t * temp_diff))
            ny[row][col] = Ds_facing_wall * ((v_y[row][col] - alpha_t * temp_diff) + v * (u_x[row][col] - alpha_t * temp_diff))

            nxy[row][col] = Ds_facing_wall * ((1 - v) / 2) * (u_y[row][col] + v_x[row][col])

    save_file(u_x, "Vorsatzschale/u_x", "u_x in N/mm", index_anker)
    save_file(v_y, "Vorsatzschale/v_y", "v_y in N/mm", index_anker)
    save_file(qx, "Vorsatzschale/qx", "qx in N", index_anker)
    save_file(qy, "Vorsatzschale/qy", "qy in N", index_anker)
    save_file(nx, "Vorsatzschale/nx", "nx in N/mm", index_anker)
    save_file(ny, "Vorsatzschale/ny", "ny in N/mm", index_anker)
    save_file(nxy, "Vorsatzschale/nxy", "nxy in N/mm", index_anker)

    u_y = np.zeros_like(wall).tolist()  # Gamma = u_y + v_x
    v_x = np.zeros_like(wall).tolist()
    gamma = np.zeros_like(wall).tolist()
    formingwork = np.zeros_like(wall).tolist()
    part1 = np.zeros_like(wall).tolist()
    part2 = np.zeros_like(wall).tolist()
    part3 = np.zeros_like(wall).tolist()

    for row in model.row:

        for col in model.col:

            # Verschiebungen nur bei Wawndstücl möglich nicht bei Fenster Ränder getrennt betrachten
            # Verschiebung einfach für Normale Punkte
            if wall[row, col] in [1, 15, 16]:
                u_y[row][col] = (model.u_shifts_facing_wall[row - 1, col].value - model.u_shifts_facing_wall[row + 1, col].value) / (
                            2 * delta_y)
                v_x[row][col] = (model.v_shifts_facing_wall[row, col + 1].value - model.v_shifts_facing_wall[row, col - 1].value) / (
                            2 * delta_x)

            # Rand Links
            if wall[row][col] == 7:
                u_y[row][col] = (model.u_shifts_facing_wall[row - 1, col].value - model.u_shifts_facing_wall[row + 1, col].value) / (
                            2 * delta_y)
                v_x[row][col] = (model.v_shifts_facing_wall[row, col + 1].value - model.v_shifts_facing_wall[row, col].value) / delta_x

            # Rand Rechts
            elif wall[row][col] == 9:
                u_y[row][col] = (model.u_shifts_facing_wall[row - 1, col].value - model.u_shifts_facing_wall[row + 1, col].value) / (
                            2 * delta_y)
                v_x[row][col] = (model.v_shifts_facing_wall[row, col].value - model.v_shifts_facing_wall[row, col - 1].value) / delta_x


            # Rand Oben
            elif wall[row][col] == 8:
                u_y[row][col] = (model.u_shifts_facing_wall[row + 1, col].value - model.u_shifts_facing_wall[row, col].value) / delta_y

                v_x[row][col] = (model.v_shifts_facing_wall[row, col + 1].value - model.v_shifts_facing_wall[row, col - 1].value) / (
                            2 * delta_x)

            # Rand unten
            elif wall[row][col] == 6:
                u_y[row][col] = (model.u_shifts_facing_wall[row, col].value - model.u_shifts_facing_wall[row - 1, col].value) / delta_y
                v_x[row][col] = (model.v_shifts_facing_wall[row, col + 1].value - model.v_shifts_facing_wall[row, col - 1].value) / (
                            2 * delta_x)

            # Linke obere Ecke
            elif wall[row][col] in [3, 12]:
                u_y[row][col] = (model.u_shifts_facing_wall[row + 1, col].value - model.u_shifts_facing_wall[row, col].value) / delta_y
                v_x[row][col] = (model.v_shifts_facing_wall[row, col + 1].value - model.v_shifts_facing_wall[row, col].value) / delta_x

            # Linke untere Ecke
            elif wall[row][col] in [2, 14]:
                u_y[row][col] = (model.u_shifts_facing_wall[row, col].value - model.u_shifts_facing_wall[row - 1, col].value) / delta_y
                v_x[row][col] = (model.v_shifts_facing_wall[row, col + 1].value - model.v_shifts_facing_wall[row, col].value) / delta_x

            # Rechte obere Ecke
            elif wall[row][col] in [5, 11]:
                u_y[row][col] = (model.u_shifts_facing_wall[row + 1, col].value - model.u_shifts_facing_wall[row, col].value) / delta_y
                v_x[row][col] = (model.v_shifts_facing_wall[row, col].value - model.v_shifts_facing_wall[row, col - 1].value) / delta_x

            # Rechte untere Ecke
            elif wall[row][col] in [4, 13]:
                u_y[row][col] = (model.u_shifts_facing_wall[row, col].value - model.u_shifts_facing_wall[row - 1, col].value) / delta_y
                v_x[row][col] = (model.v_shifts_facing_wall[row, col].value - model.v_shifts_facing_wall[row, col - 1].value) / delta_x

    for row in model.row:

        for col in model.col:
            gamma[row][col] = u_y[row][col] + v_x[row][col]
    save_file(gamma, "Vorsatzschale/gamma", "gamma in N/mm", index_anker)

    ####################################################################################################################
    # Plot support wall
    ####################################################################################################################

    w_shifts = []
    for row in range(m):
        mini = []
        for col in range(n):
            mini.append(model.shifts_support_wall[row, col].value)
        w_shifts.append(mini)
    w_shifts = np.array(w_shifts)

    u_shifts = np.zeros_like(w_shifts)
    v_shifts = np.zeros_like(w_shifts)

    for row in range(m):
        for col in range(n):
            u_shifts[row, col] = model.u_shifts_support_wall[row, col].value
            v_shifts[row, col] = model.v_shifts_support_wall[row, col].value

    save_file(w_shifts, "Tragschale/w", "Verschiebung in mm", index_anker)

    save_file(u_shifts, "Tragschale/u", "Verschiebung in mm", index_anker)

    save_file(v_shifts, "Tragschale/v", "Verschiebung in mm", index_anker)

    w2x = np.zeros_like(wall).tolist()
    w2y = np.zeros_like(wall).tolist()
    w2xy = np.zeros_like(wall).tolist()

    for row in range(m):

        for col in range(n):

            wall_type = wall[row, col]

            if not wall_type == 0:

                # w2x

                if wall_type in [1, 15, 8, 6, 11, 12, 13, 14, 16]:  # 2.14

                    w2x[row][col] = (w_shifts[row, col + 1] - 2 * w_shifts[row, col] + w_shifts[
                        row, col - 1]) / (delta_x ** 2)

                elif wall_type in [9, 5, 4]:  # 2.17

                    w2x[row][col] = 0

                else:  # 2.21

                    w2x[row][col] = 0

                # w2y

                if wall_type in [1, 15, 9, 7, 11, 12, 13, 14, 16]:  # 2.15

                    w2y[row][col] = (w_shifts[row + 1, col] - 2 * w_shifts[row, col] + w_shifts[
                        row - 1, col]) / (delta_y ** 2)

                elif wall_type in [8, 5, 3]:  # 2.25

                    w2y[row][col] = 0

                else:  # 2.29

                    w2y[row][col] = 0

                # w2xy

                if wall_type in [1, 15, 16]:  # 2.13

                    w2xy[row][col] = (w_shifts[row + 1, col + 1] - w_shifts[row - 1, col + 1] -
                                      w_shifts[
                                          row + 1, col - 1] + w_shifts[row - 1, col - 1]) / (
                                             4 * delta_x * delta_y)

                elif wall_type == 9:  # 2.16

                    w2xy[row][col] = (2 * w_shifts[row + 1, col] - 2 * w_shifts[row - 1, col] - 2 *
                                      w_shifts[
                                          row + 1, col - 1] + 2 * w_shifts[row - 1, col - 1]) / (
                                             4 * delta_x * delta_y)

                elif wall_type == 7:  # 2.20

                    w2xy[row][col] = (2 * w_shifts[row - 1, col] - 2 * w_shifts[row + 1, col] - 2 *
                                      w_shifts[
                                          row - 1, col + 1] + 2 * w_shifts[row + 1, col + 1]) / (
                                             4 * delta_x * delta_y)

                elif wall_type == 8:  # 2.24

                    w2xy[row][col] = (2 * w_shifts[row, col - 1] - 2 * w_shifts[row, col + 1] - 2 *
                                      w_shifts[
                                          row + 1, col - 1] + 2 * w_shifts[row + 1, col + 1]) / (
                                             4 * delta_x * delta_y)

                elif wall_type == 6:  # 2.28

                    w2xy[row][col] = (2 * w_shifts[row, col + 1] - 2 * w_shifts[row, col - 1] - 2 *
                                      w_shifts[
                                          row - 1, col + 1] + 2 * w_shifts[row - 1, col - 1]) / (
                                             4 * delta_x * delta_y)

                elif wall_type == 5:  # 2.32

                    w2xy[row][col] = 0

                elif wall_type == 3:  # 2.35

                    w2xy[row][col] = 0

                elif wall_type == 4:  # 2.36

                    w2xy[row][col] = 0

                elif wall_type == 2:  # 2.37

                    w2xy[row][col] = 0

                elif wall_type == 11:  # 2.40
                    w2xy[row][col] = (2 * w_shifts[row - 1, col] - 2 * w_shifts[row - 1, col + 1] -
                                      w_shifts[
                                          row + 1, col - 1] + w_shifts[row + 1, col + 1]) / (
                                             4 * delta_x * delta_y)

                elif wall_type == 12:  # 2.41
                    w2xy[row][col] = (2 * w_shifts[row - 1, col - 1] - w_shifts[row - 1, col] -
                                      w_shifts[
                                          row + 1, col - 1] + w_shifts[row + 1, col + 1]) / (
                                             4 * delta_x * delta_y)

                elif wall_type == 13:  # 2.38
                    w2xy[row][col] = (2 * w_shifts[row + 1, col + 1] - w_shifts[row - 1, col + 1] - 2 *
                                      w_shifts[row + 1, col] + w_shifts[row - 1, col - 1]) / (
                                             4 * delta_x * delta_y)

                elif wall_type == 14:  # 2.39
                    w2xy[row][col] = (2 * w_shifts[row + 1, col] - 2 * w_shifts[row + 1, col - 1] -
                                      w_shifts[
                                          row - 1, col + 1] + w_shifts[row - 1, col - 1]) / (
                                             4 * delta_x * delta_y)

    mx = np.zeros_like(wall).tolist()
    my = np.zeros_like(wall).tolist()
    mxy = np.zeros_like(wall).tolist()

    for row in range(m):

        for col in range(n):

            wall_type = wall[row, col]

            if not wall_type == 0:
                mx[row][col] = - D_support_wall * (w2x[row][col] + v * w2y[row][col]) - M_star_support_wall / (1 - v)
                my[row][col] = - D_support_wall * (w2y[row][col] + v * w2x[row][col]) - M_star_support_wall / (1 - v)
                mxy[row][col] = -D_support_wall * (1 - v) * w2xy[row][col]

    save_file(mx, "Tragschale/mx", "mx in N/mm", index_anker)
    save_file(my, "Tragschale/my", "my in N/mm", index_anker)
    save_file(mxy, "Tragschale/mxy", "mxy in N/mm", index_anker)

    qx = np.zeros_like(wall).tolist()
    qy = np.zeros_like(wall).tolist()

    for row in range(m):
        for col in range(n):

            # MItte der Platte
            if wall[row][col] in [1, 15, 16]:

                qx[row][col] = (- D_support_wall * (
                        w2x[row][col + 1] - w2x[row][col - 1] + w2y[row][col + 1] - w2y[row][col - 1])) / (
                                       2 * delta_x)
                qy[row][col] = (- D_support_wall * (
                        w2x[row + 1][col] - w2x[row - 1][col] + w2y[row + 1][col] - w2y[row - 1][col])) / (
                                       2 * delta_y)

            # Linke untere Ecke
            elif wall[row][col] in [2, 14]:
                qx[row][col] = (- D_support_wall * (w2x[row][col + 1] - w2x[row][col] + w2y[row][col + 1] - w2y[row][col])) / delta_x
                qy[row][col] = (- D_support_wall * (w2x[row][col] - w2x[row - 1][col] + w2y[row][col] - w2y[row - 1][col])) / delta_y

            # Linke obere Ecke
            elif wall[row][col] in [3, 12]:
                qx[row][col] = (- D_support_wall * (w2x[row][col + 1] - w2x[row][col] + w2y[row][col + 1] - w2y[row][col])) / delta_x
                qy[row][col] = (- D_support_wall * (w2x[row + 1][col] - w2x[row][col] + w2y[row + 1][col] - w2y[row][col])) / delta_y

            # Rechte untere Ecke
            elif wall[row][col] in [4, 13]:
                qx[row][col] = (- D_support_wall * (w2x[row][col] - w2x[row][col - 1] + w2y[row][col] - w2y[row][col - 1])) / delta_x
                qy[row][col] = (- D_support_wall * (w2x[row][col] - w2x[row - 1][col] + w2y[row][col] - w2y[row - 1][col])) / delta_y

            # Rechte obere Ecke
            elif wall[row][col] in [5, 11]:
                qx[row][col] = (- D_support_wall * (
                        w2x[row][col] - w2x[row][col - 1] + w2y[row][col] - w2y[row][col - 1])) / delta_x
                qy[row][col] = (- D_support_wall * (
                        w2x[row + 1][col] - w2x[row][col] + w2y[row + 1][col] - w2y[row][col])) / delta_y

            # Rand unten
            elif wall[row][col] == 6:
                qx[row][col] = (- D_support_wall * (
                        w2x[row][col + 1] - w2x[row][col - 1] + w2y[row][col + 1] - w2y[row][col - 1])) / (
                                       2 * delta_x)
                qy[row][col] = (- D_support_wall * (w2x[row][col] - w2x[row - 1][col] + w2y[row][col] - w2y[row - 1][col])) / delta_y

            # Rand Links
            elif wall[row][col] == 7:
                qx[row][col] = (- D_support_wall * (w2x[row][col + 1] - w2x[row][col] + w2y[row][col + 1] - w2y[row][col])) / delta_x
                qy[row][col] = (- D_support_wall * (
                        w2x[row + 1][col] - w2x[row - 1][col] + w2y[row + 1][col] - w2y[row - 1][col])) / (
                                       2 * delta_y)

            # Rand oben
            elif wall[row][col] == 8:
                qx[row][col] = (- D_support_wall * (
                        w2x[row][col + 1] - w2x[row][col - 1] + w2y[row][col + 1] - w2y[row][col - 1])) / (
                                       2 * delta_x)
                qy[row][col] = (- D_support_wall * (w2x[row + 1][col] - w2x[row][col] + w2y[row + 1][col] - w2y[row][col])) / delta_y

            # Rand Rechts
            elif wall[row][col] == 9:
                qx[row][col] = (- D_support_wall * (w2x[row][col] - w2x[row][col - 1] + w2y[row][col] - w2y[row][col - 1])) / delta_x
                qy[row][col] = (- D_support_wall * (
                        w2x[row + 1][col] - w2x[row - 1][col] + w2y[row + 1][col] - w2y[row - 1][col])) / (
                                       2 * delta_y)

    u_x = np.zeros_like(wall).tolist()  # Epsilonx
    v_y = np.zeros_like(wall).tolist()  # Epsilony

    u_y = np.zeros_like(wall).tolist()  # Gamma = u_y + v_x
    v_x = np.zeros_like(wall).tolist()

    for row in model.row:

        for col in model.col:
            wall_type = wall[row, col]
            if wall_type != 0:
                # u_x
                if wall_type in [9, 4, 5]:
                    u_x[row][col] = (model.u_shifts_support_wall[row, col].value - model.u_shifts_support_wall[
                        row, col - 1].value) / delta_x
                else:
                    u_x[row][col] = (model.u_shifts_support_wall[row, col + 1].value - model.u_shifts_support_wall[
                        row, col].value) / delta_x
                # v_x
                if wall_type in [9, 4, 5]:
                    v_x[row][col] = (model.v_shifts_support_wall[row, col].value - model.v_shifts_support_wall[
                        row, col - 1].value) / delta_x
                else:
                    v_x[row][col] = (model.v_shifts_support_wall[row, col + 1].value - model.v_shifts_support_wall[
                        row, col].value) / delta_x
                # u_y
                if wall_type in [6, 2, 4]:
                    u_y[row][col] = (model.u_shifts_support_wall[row, col].value - model.u_shifts_support_wall[
                        row - 1, col].value) / delta_y
                else:
                    u_y[row][col] = (model.u_shifts_support_wall[row + 1, col].value - model.u_shifts_support_wall[
                        row, col].value) / delta_y
                # v_y
                if wall_type in [6, 2, 4]:
                    v_y[row][col] = (model.v_shifts_support_wall[row, col].value - model.v_shifts_support_wall[
                        row - 1, col].value) / delta_y
                else:
                    v_y[row][col] = (model.v_shifts_support_wall[row + 1, col].value - model.v_shifts_support_wall[
                        row, col].value) / delta_y

    nx = np.zeros_like(wall).tolist()
    ny = np.zeros_like(wall).tolist()
    nxy = np.zeros_like(wall).tolist()

    for row in model.row:
        for col in model.col:
            nx[row][col] = Ds_support_wall * (u_x[row][col] + v * v_y[row][col])
            ny[row][col] = Ds_support_wall * (v_y[row][col] + v * u_x[row][col])

            nxy[row][col] = Ds_support_wall * ((1 - v) / 2) * (u_y[row][col] + v_x[row][col])

    save_file(u_x, "Tragschale/u_x", "u_x in N/mm", index_anker)
    save_file(v_y, "Tragschale/v_y", "v_y in N/mm", index_anker)
    save_file(qx, "Tragschale/qx", "qx in N", index_anker)
    save_file(qy, "Tragschale/qy", "qy in N", index_anker)
    save_file(nx, "Tragschale/nx", "nx in N/mm", index_anker)
    save_file(ny, "Tragschale/ny", "ny in N/mm", index_anker)
    save_file(nxy, "Tragschale/nxy", "nxy in N/mm", index_anker)

    u_y = np.zeros_like(wall).tolist()  # Gamma = u_y + v_x
    v_x = np.zeros_like(wall).tolist()
    gamma = np.zeros_like(wall).tolist()
    formingwork = np.zeros_like(wall).tolist()
    part1 = np.zeros_like(wall).tolist()
    part2 = np.zeros_like(wall).tolist()
    part3 = np.zeros_like(wall).tolist()

    for row in model.row:

        for col in model.col:

            # Verschiebungen nur bei Wawndstück möglich nicht bei Fenster Ränder getrennt betrachten
            # Verschiebung einfach für normale Punkte
            if wall[row, col] in [1, 15, 16]:
                u_y[row][col] = (model.u_shifts_support_wall[row - 1, col].value - model.u_shifts_support_wall[
                    row + 1, col].value) / (
                                        2 * delta_y)
                v_x[row][col] = (model.v_shifts_support_wall[row, col + 1].value - model.v_shifts_support_wall[
                    row, col - 1].value) / (
                                        2 * delta_x)

            # Rand Links
            if wall[row][col] == 7:
                u_y[row][col] = (model.u_shifts_support_wall[row - 1, col].value - model.u_shifts_support_wall[
                    row + 1, col].value) / (
                                        2 * delta_y)
                v_x[row][col] = (model.v_shifts_support_wall[row, col + 1].value - model.v_shifts_support_wall[
                    row, col].value) / delta_x

            # Rand Rechts
            elif wall[row][col] == 9:
                u_y[row][col] = (model.u_shifts_support_wall[row - 1, col].value - model.u_shifts_support_wall[
                    row + 1, col].value) / (
                                        2 * delta_y)
                v_x[row][col] = (model.v_shifts_support_wall[row, col].value - model.v_shifts_support_wall[
                    row, col - 1].value) / delta_x


            # Rand Oben
            elif wall[row][col] == 8:
                u_y[row][col] = (model.u_shifts_support_wall[row + 1, col].value - model.u_shifts_support_wall[
                    row, col].value) / delta_y

                v_x[row][col] = (model.v_shifts_support_wall[row, col + 1].value - model.v_shifts_support_wall[
                    row, col - 1].value) / (
                                        2 * delta_x)

            # Rand unten
            elif wall[row][col] == 6:
                u_y[row][col] = (model.u_shifts_support_wall[row, col].value - model.u_shifts_support_wall[
                    row - 1, col].value) / delta_y
                v_x[row][col] = (model.v_shifts_support_wall[row, col + 1].value - model.v_shifts_support_wall[
                    row, col - 1].value) / (
                                        2 * delta_x)

            # Linke obere Ecke
            elif wall[row][col] in [3, 12]:
                u_y[row][col] = (model.u_shifts_support_wall[row + 1, col].value - model.u_shifts_support_wall[
                    row, col].value) / delta_y
                v_x[row][col] = (model.v_shifts_support_wall[row, col + 1].value - model.v_shifts_support_wall[
                    row, col].value) / delta_x

            # Linke untere Ecke
            elif wall[row][col] in [2, 14]:
                u_y[row][col] = (model.u_shifts_support_wall[row, col].value - model.u_shifts_support_wall[
                    row - 1, col].value) / delta_y
                v_x[row][col] = (model.v_shifts_support_wall[row, col + 1].value - model.v_shifts_support_wall[
                    row, col].value) / delta_x

            # Rechte obere Ecke
            elif wall[row][col] in [5, 11]:
                u_y[row][col] = (model.u_shifts_support_wall[row + 1, col].value - model.u_shifts_support_wall[
                    row, col].value) / delta_y
                v_x[row][col] = (model.v_shifts_support_wall[row, col].value - model.v_shifts_support_wall[
                    row, col - 1].value) / delta_x

            # Rechte untere Ecke
            elif wall[row][col] in [4, 13]:
                u_y[row][col] = (model.u_shifts_support_wall[row, col].value - model.u_shifts_support_wall[
                    row - 1, col].value) / delta_y
                v_x[row][col] = (model.v_shifts_support_wall[row, col].value - model.v_shifts_support_wall[
                    row, col - 1].value) / delta_x

    for row in model.row:

        for col in model.col:
            gamma[row][col] = u_y[row][col] + v_x[row][col]
            formingwork[row][col] = 0.5 * (
                    (u_x[row][col]) * nx[row][col] +
                    (v_y[row][col]) * ny[row][col] +
                    (u_y[row][col] + v_x[row][col]) * nxy[row][col]) * delta_x * delta_y  # Formänderungsarbeit
            part1[row][col] = 0.5 * (u_x[row][col]) * nx[row][col] * delta_x * delta_y
            part2[row][col] = 0.5 * (v_y[row][col]) * ny[row][col] * delta_x * delta_y
            part3[row][col] = 0.5 * (u_y[row][col] + v_x[row][col]) * nxy[row][col] * delta_x * delta_y

    save_file(gamma, "Tragschale/gamma", "gamma in N/mm", index_anker)
    plt.imshow(gamma)
    plt.colorbar(label='')
    plt.savefig("Gui/ergbnisse/Tragschale/gamma.jpg")
    plt.clf()

    anker_force_z = np.zeros_like(wall)
    for row in model.row:
        for col in model.col:
            if wall[row][col] == 15:
                anker_force_z[row][col] = (model.shifts_facing_wall[row, col].value - model.shifts_support_wall[row, col].value) * material_laws["Kunststoff1"]["Cz"]
    plt.imshow(anker_force_z)
    plt.colorbar(label='')
    plt.savefig("Gui/ergbnisse/Verbindungsmittel/Verbindungsmittel_z.jpg")
    plt.clf()
    pd.DataFrame(anker_force_z).to_csv("Gui/ergbnisse/Verbindungsmittel/Verbindungsmittel_z.csv", header=None, index=None)

    anker_force_x = np.zeros_like(wall)
    for row in model.row:
        for col in model.col:
            if wall[row][col] == 15:
                anker_force_x[row][col] = (model.u_shifts_facing_wall[row, col].value - model.u_shifts_support_wall[
                    row, col].value) * material_laws["Kunststoff1"]["Cx"]
    plt.imshow(anker_force_x)
    plt.colorbar(label='')
    plt.savefig("Gui/ergbnisse/Verbindungsmittel/Verbindungsmittel_x.jpg")
    plt.clf()
    pd.DataFrame(anker_force_x).to_csv("Gui/ergbnisse/Verbindungsmittel/Verbindungsmittel_x.csv", header=None, index=None)

    anker_force_y = np.zeros_like(wall)
    for row in model.row:
        for col in model.col:
            if wall[row][col] == 15:
                anker_force_y[row][col] = (model.v_shifts_facing_wall[row, col].value - model.v_shifts_support_wall[
                    row, col].value) * material_laws["Kunststoff1"]["Cy"]
    plt.imshow(anker_force_y)
    plt.colorbar(label='')
    plt.savefig("Gui/ergbnisse/Verbindungsmittel/Verbindungsmittel_y.jpg")
    plt.clf()
    pd.DataFrame(anker_force_y).to_csv("Gui/ergbnisse/Verbindungsmittel/Verbindungsmittel_y.csv", header=None, index=None)
