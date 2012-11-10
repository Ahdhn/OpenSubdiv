"""
Plot the sparsity pattern of arrays
"""

from sys import argv
assert len(argv) == 2, "Usage: python %s matrix.mm" % argv[0]
mm_filename = argv[1]

from matplotlib.pyplot import figure, show
import numpy
from scipy.io import mmread, mminfo

fig = figure()
ax = fig.add_subplot(111)

x = mmread(mm_filename)

ax.spy(x, markersize=1)
show()
