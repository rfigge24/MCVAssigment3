import numpy as np
import matplotlib.pyplot as plt


def plotPersonPaths(personcenters):
    """
    This function plots the walking paths of the persons by scattering the stored personcenters.
    """
    personcenters = np.array(personcenters)
    for i ,person in enumerate(personcenters):
        i -= 1
        color = [0,0,0]
        if i >= 0:
            color[i] = 1
        #-1 * to mirror the y axis back
        plt.scatter(person[:,0],-1 * person[:,1], c=[color],marker=".")
    plt.xlabel('X'),plt.ylabel('Y')
    plt.xlim([-100,200])
    plt.ylim([-100,200])
    plt.show()

