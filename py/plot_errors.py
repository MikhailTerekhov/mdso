import sys
import numpy as np
import matplotlib.pyplot as plt

labels = ['ошибки в ключевых точках',
          'ошибки после интерполяции',
          'ошибки после поиска вдоль эпиполярной кривой']

def main(argv):
    if (len(argv) < 4):
        print("I need at least three arguments -- two for y-borders and then names of files to process")
        return 0
    
    y0 = float(argv[1])
    y1 = float(argv[2])
    print(f"y0={y0} y1={y1}")

    fig, ax = plt.subplots()
    for fnum, fname in enumerate(argv[3:]):
        f = open(fname)
        val = np.array([float(x) for x in f.readline().split()])
        val.sort()
        print(f"val:\n{val}")
        print(f"median={val[len(val) // 2]}")
        print(f"len={len(val)}")
        x = np.linspace(0., 1., num=len(val))
        ax.plot(val, x, label=labels[fnum])
    plt.xlim(y0, y1)

    ax.legend()
    plt.show()


if __name__ == "__main__":
    sys.exit(main(sys.argv))
