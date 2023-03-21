import pandas as pd
import matplotlib.pyplot as plt


def save_file(value, name, text, index_anker):
    path = "Gui/ergbnisse/" + name

    pd.DataFrame(value).to_csv(path + ".csv", header=None, index=None)
    plt.imshow(value)
    for index in range(len(index_anker[0])):
        plt.plot(index_anker[1][index], index_anker[0][index], 'o', color='red')
    plt.colorbar(label=text)
    plt.savefig(path + ".jpg")
    plt.clf()