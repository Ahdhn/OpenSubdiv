#!/usr/bin/env python2.7

import sys, os, shelve

from bench import *

modelLevel = {
    "BigGuy":       4,
    "Bunny":        4,
    "MonsterFrog":  5,
    "Venus":        3,
    "Cube":         7,
    "Icosahedron":  7,
}

NSAMPLES = 10

def build_db(kernel):
    db = set()
    for model in modelNum.keys():
        for l in range( modelLevel[model] ):
            try:
                for s in range(NSAMPLES):
                    run = do_run(frames=2, model=model, kernel=kernel, level=l+1)
                    db.add(run)
            except ExecutionError as e:
                print "\tFailed with: %s" % e.message
    return db

def gen_dat_file(ofile, db):
    model_set = { r.model for r in db if r.model }
    size_set = { r.nverts for r in db if r.nverts }
    model_list = sorted(model_set, key=lambda m: modelNum[m])
    size_list = sorted(size_set)
    print >>ofile, "nVerts", " ".join(["%s %s-std" % (name, name) for name in model_list])
    for size in size_list:
        print >>ofile, size,
        for model in model_list:
            run_list = filter(lambda r: r.nverts == size and r.model == model, db)
            if len(run_list) > 0:
                times = np.array([r.matrixtime for r in run_list])
                print >>ofile, " %f %f" % (times.mean(), times.std()),
            if len(run_list) == 0:
                print >>ofile, " ? ?",
        print >>ofile

def main(argv):
    kernel = argv[1][len("matrixtime_"):-len(".dat")]
    with open("matrixtime_%s.dat" % kernel, 'w') as ofile:
        gen_dat_file(ofile, build_db(kernel))


if __name__ == '__main__':
    main(sys.argv)
