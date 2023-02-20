import pandas as pd
import matplotlib.pyplot as plt


def get_graph(plot_ear_mean=False):
    """
    This method creates the graph of all the EAR we get and computes the mean value
    :return: graph
    """

    data = pd.read_csv('data.csv')
    y = data["EAR"]
    x = data["ITERATION"]
    if plot_ear_mean:
        plt.style.use('fivethirtyeight')
        plt.cla()

        plt.plot(x, y, label='EAR')
        plt.legend(loc='upper left')
        plt.tight_layout()
        plt.tight_layout()
        plt.show()

    mean = 0
    for value in y:
        mean += value

    if len(y) == 0:
        #value which tells me that the length of y is null. In this case no value has been detected, which means that,
        #for instance, a hand was covering the camera, so both open eyes values and closed eyes values have been registered
        return -1
    else:
        return mean/len(y)

