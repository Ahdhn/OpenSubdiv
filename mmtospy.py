"""
Plot the sparsity pattern of arrays
"""

#SORT = True # slowww
SORT = False

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from scipy.io import mmread, mminfo
import numpy

def swap(A, i, j):
    tmp = numpy.copy(A[i])
    A[i] = A[j]
    A[j] = tmp

def main(argv):
    assert len(argv) == 2, "Usage: python %s matrix.mm" % argv[0]
    mm_filename = argv[1]
    x = mmread(mm_filename)

    w, h = matplotlib.figure.figaspect(x)
    fig = plt.figure(figsize=(w,h))
    ax = fig.add_subplot(111)

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
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)

    extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    fig.savefig("spy_%s.pdf" % mm_filename[:-3], bbox_inches=extent)

if __name__ == '__main__':
    import sys
    main(sys.argv)
