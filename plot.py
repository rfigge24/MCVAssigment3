import numpy as np
import matplotlib.pyplot as plt


def plotPersonPaths(personcenters):
    for i ,person in enumerate(personcenters):
        i -= 1
        color = [0,0,0]
        if i >= 0:
            color[i] = 1
        plt.scatter(person[:][0],person[:][1], c=[color])
    plt.xlabel('X'),plt.ylabel('Y')
    plt.show()