import matplotlib.pyplot as plt


def make_scatter(projection, filename=None, y=None):
    plt.figure()
    if y is not None:
        plt.scatter(*projection.T, c=y, cmap=plt.cm.Spectral)
    else:
        plt.scatter(*projection.T)
    if filename:
        plt.savefig(filename)
        plt.close()
    else:
        plt.show()
