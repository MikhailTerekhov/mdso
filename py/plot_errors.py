import sys
import numpy as np
import matplotlib.pyplot as plt

def main(argv):
    if (len(argv) < 2):
        print("I need at least one argument -- names of files to process")
        return 0
    
    fig, ax = plt.subplots()
    for fname in argv[1:]:
        f = open(fname)
        val = np.array([float(x) for x in f.readline().split()])
        x = np.linspace(0., 1., num=len(val))
        ax.plot(x, val)
    plt.ylim(0, 1)

    plt.show()


if __name__ == "__main__":
    sys.exit(main(sys.argv))
