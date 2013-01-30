#!/usr/bin/env python2.7

"""
Plot the distribution of row nnz's
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from scipy.io import mmread, mminfo
from scipy.sparse import *
import numpy

def main(argv):
    assert len(argv) == 2, "Usage: ./%s NAME.mm" % argv[0]
    mm_filename = argv[1]
    print "Reading COO matrix..."
    coo = mmread(mm_filename)
    print "Converting to CSR..."
    x = csr_matrix( coo )

    print "Gathering row sizes..."
    rowsizes = [x[i][:].getnnz() for i in range( x.shape[0] )]

    print "Plotting..."
    nbins = max(rowsizes) - min(rowsizes)
    plt.hist(rowsizes, bins=nbins)
    plt.title("Row nnz Distribution for %s Model" % mm_filename[:-3].capitalize())
    plt.xlabel("Row NNZ")
    plt.ylabel("Count")
    plt.savefig("%s.pdf" % mm_filename[:-3])
    print "Done."

if __name__ == '__main__':
    import sys
    main(sys.argv)
