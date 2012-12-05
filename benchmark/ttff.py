#!/usr/bin/python2.7

import sys, os, shelve

from bench import *

NFRAMES = 100

def build_db(model):
    db = set()
    for k in activeKernels:
        for l in range(7):
            db.add( do_run(frames=NFRAMES, model=model, kernel=k, level=l+1) )
    return db

def gen_dat_file(ofile, db):
    kernel_set = { r.kernel for r in db if r.kernel }
    size_set = { r.nverts for r in db if r.nverts }
    kernel_list = sorted(kernel_set, key=lambda k: kernelNum[k])
    size_list = sorted(size_set)
    print >>ofile, "time", " ".join(kernel_list)

    for k in range(len(kernel_list)):
        kernel = kernel_list[k]
        for f in range(NFRAMES):


def main(argv):
    model = argv[1][5:-4]
    with open("ttff_%s.dat" % model, 'w') as ofile:
        gen_dat_file(ofile, build_db(model))


if __name__ == '__main__':
    main(sys.argv)
