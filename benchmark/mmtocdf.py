#!/usr/bin/env python2.7

"""
Plot the cdf of row nnz's
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from scipy.io import mmread, mminfo
from scipy.sparse import *
import numpy

def numsmaller(A, val):
    return len([i for i in A if i < val])

def main(argv):
    assert len(argv) == 2, "Usage: ./%s NAME.mm" % argv[0]
    mm_filename = argv[1]
    print "Reading COO matrix..."
    coo = mmread(mm_filename)
    print "Converting to CSR..."
    x = csr_matrix( coo )

    print "Gathering row sizes..."
    rowsizes = [x[i][:].getnnz() for i in range( x.shape[0] )]
    biggest, smallest, nitems = max(rowsizes), min(rowsizes), float(len(rowsizes))

    xs = [i for i in range(smallest, biggest+1)]
    cdf = [ numsmaller(rowsizes, i)/nitems for i in range(smallest, biggest+1) ]

    print "Plotting..."
    plt.plot(xs, cdf)
    plt.title("CDF of Row Nnz for %s Model" % mm_filename[:-3].capitalize())
    plt.xlabel("NNZ")
    plt.ylabel("Fraction < x")
    plt.savefig("cdf_%s.pdf" % mm_filename[:-3])
    print "Done."

if __name__ == '__main__':
    import sys
    main(sys.argv)
