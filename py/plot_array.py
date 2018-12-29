import sys
import numpy as np
import matplotlib.pyplot as plt

def main(argv):
    if (len(argv) < 2):
        print("I need at least one argument -- names of files to process")
        return 0
    
    fig, ax = plt.subplots()
    for fnum, fname in enumerate(argv[1:]):
        f = open(fname)
        val = np.array([int(x) for x in f.readline().split()])
        count = (max(val) + 1) * [0]
        val.sort()
        for v in val:
            count[v] += 1
        ax.bar(np.arange(len(count)), count)
    
    ax.legend()
    plt.show()


if __name__ == "__main__":
    sys.exit(main(sys.argv))
