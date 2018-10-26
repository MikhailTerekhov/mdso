import sys
import numpy as np
import matplotlib.pyplot as plt

def main(argv):
    if (len(argv) != 2):
        print("I need just one argument -- dso output directory")
        return 0
    out_dir = argv[1]
    f = open(out_dir + '/reproj_err.txt')
    val = np.array([float(x) for x in f.readline().split()])
    prob = np.arange(0., 1., 1. / len(val))
    fig, ax = plt.subplots()
    ax.plot(val, prob)
    #  plt.xlim(0, 10)

    plt.show()


if __name__ == "__main__":
    sys.exit(main(sys.argv))
