#!/usr/bin/env python2.7

import sys, os, shelve

from bench import *

modelLevel = {
    "BigGuy":       4,
    "Bunny":        4,
    "MonsterFrog":  4,
    "Venus":        4,
    "Cube":         7,
    "Icosahedron":  7,
}

def build_db(model):
    db = set()
    for l in range( modelLevel[model] ):
        try:
            run = do_run(frames=2, model=model, kernel="MKL", level=l+1, regression=True)
            db.add(run)
        except ExecutionError as e:
            print "\tFailed with: %s" % e.message
    return db

def gen_dat_file(ofile, model, db):
    kernel_set = { r.kernel for r in db if r.kernel }
    size_set = { r.nverts for r in db if r.nverts }
    kernel_list = sorted(kernel_set, key=lambda k: kernelNum[k])
    size_list = sorted(size_set)
    print >>ofile, "nVerts", escape_latex(model)
    for size in size_list:
        print >>ofile, size,
        for kernel in kernel_list:
            run_list = filter(lambda r: r.nverts == size and r.kernel == kernel, db)
            if len(run_list) == 1:
                print >>ofile, " %e" % run_list[0].maxerror,
            if len(run_list) == 0:
                print >>ofile, " ?",
        print >>ofile

def main(argv):
    model = argv[1][10:-4]
    with open("stability_%s.dat" % model, 'w') as ofile:
        gen_dat_file(ofile, model, build_db(model))


if __name__ == '__main__':
    main(sys.argv)
