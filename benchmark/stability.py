#!/usr/bin/env python2.7

import sys, os, shelve

from bench import *

modelLevel = {
    "BigGuy":       4,
    "Bunny":        4,
    "MonsterFrog":  4,
    "Venus":        3,
    "Cube":         7,
    "Icosahedron":  7,
}

NSAMPLES = 10

def build_db(model):
    db = set()
    for k in activeKernels:
        for l in range( modelLevel[model] ):
            try:
                for s in range(NSAMPLES):
                    run = do_run(frames=2, model=model, kernel=k, level=l+1)
                    db.add(run)
            except ExecutionError as e:
                print "\tFailed with: %s" % e.message
    return db

def gen_dat_file(ofile, db):
    kernel_set = { r.kernel for r in db if r.kernel }
    size_set = { r.nverts for r in db if r.nverts }
    kernel_list = sorted(kernel_set, key=lambda k: kernelNum[k])
    size_list = sorted(size_set)
    print >>ofile, "nVerts", " ".join(["%s %s-std" % (name, name) for name in kernel_list])
    for size in size_list:
        print >>ofile, size,
        for kernel in kernel_list:
            run_list = filter(lambda r: r.nverts == size and r.kernel == kernel, db)
            if len(run_list) > 0:
                maxerrors = np.array([r.maxerror for r in run_list])
                print >>ofile, " %f %f" % (maxerrors.mean(), maxerrors.std()),
            if len(run_list) == 0:
                print >>ofile, " ?",
        print >>ofile

def main(argv):
    model = argv[1][5:-4]
    with open("stability_%s.dat" % model, 'w') as ofile:
        gen_dat_file(ofile, build_db(model))


if __name__ == '__main__':
    main(sys.argv)
