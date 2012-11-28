"""
Plot the sparsity pattern of arrays
"""

#SORT = True # slowww
SORT = False

from matplotlib.pyplot import figure, show
from scipy.io import mmread, mminfo
import numpy

def swap(A, i, j):
    tmp = numpy.copy(A[i])
    A[i] = A[j]
    A[j] = tmp

def main(argv):
    assert len(argv) == 2, "Usage: python %s matrix.mm" % argv[0]
    mm_filename = argv[1]

    fig = figure()
    ax = fig.add_subplot(111)

    x = mmread(mm_filename)

    if SORT:
        print "Re-arranging rows."
        xd = x.todense()
        m, n = xd.shape
        for i in range(0,m):
            for j in range(i,m):
                if xd[j].tolist() > xd[i].tolist():
                    swap(xd, i, j)
        x = xd

    ax.spy(x, markersize=1)
    show()

if __name__ == '__main__':
    import sys
    main(sys.argv)
