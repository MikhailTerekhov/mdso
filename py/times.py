import sys
import numpy as np
import matplotlib.pyplot as plt

def pairify(lst):
    return [(lst[i], lst[i + 1]) for i in range(0, len(lst) // 2 * 2 - 2, 2)]

def lvl(s):
    return int(s.split()[-1][1:])
def tm(s):
    return int(s.split()[-1])

def main(argv):
    if (len(argv) < 2):
        print("I need at least one argument -- names of files to process")
        return 0
    
    f = open(argv[1])
    s = f.readlines()
    times = [(lvl(sl), tm(st)) for (sl, st) in pairify(s)]
    print(f'some:\n{times[0:2]}')
    total = len(times) // 6
    print(f'total={total}')
    sums = [0] * 6
    for (l, t) in times:
        sums[l] += t
    print(f'sums={sums}')
    avg = [(s / total) * 1e-3 for s in sums]


    plt.bar(list(range(6)), avg)
    plt.xlabel('pyramid level')
    plt.ylabel('average time (ms)')
    plt.show()


if __name__ == "__main__":
    sys.exit(main(sys.argv))
