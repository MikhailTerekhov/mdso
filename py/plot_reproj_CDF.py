import sys
import numpy as np
import matplotlib.pyplot as plt

def main(argv):
    if (len(argv) != 2):
        print("I need just one argument -- name of the file to process")
        return 0
    f = open(argv[1])
    val = np.array([float(x) for x in f.readline().split()])
    prob = np.linspace(0., 1., num=len(val))
    fig, ax = plt.subplots()
    ax.plot(val, prob)
    plt.xlim(0, 1)

    plt.show()


if __name__ == "__main__":
    sys.exit(main(sys.argv))
