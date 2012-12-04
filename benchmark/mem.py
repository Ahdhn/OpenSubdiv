#!/usr/bin/python2.7

import sys, os, shelve

from bench import *

def build_db(model):
    db = set()
    for l in range(7):
        db.add( do_run(frames=3, model=model, kernel='MKL', level=l+1) )
    return db

def gen_dat_file(ofile, model, db):
    size_set = { r.nverts for r in db if r.nverts }
    size_list = sorted(size_set)
    print >>ofile, "nVerts", model
    for size in size_list:
        print >>ofile, size,
        run_list = filter(lambda r: r.nverts == size, db)
        if len(run_list) == 1:
            print >>ofile, " %f" % run_list[0].mem
        if len(run_list) == 0:
            print >>ofile, " ?"


def main(argv):
    for model in activeModels:
        db = build_db(model)
        with open("mem_%s.dat" % model, 'w') as ofile:
            gen_dat_file(ofile, model, db)


if __name__ == '__main__':
    main(sys.argv)
