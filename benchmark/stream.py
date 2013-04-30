#!/usr/bin/env python2.7

import sys, os, shelve

from bench import *

def build_db(model):
    db = set()
    for k in activeKernels:
        for l in range( modelMaxLevel[model] ):
            try:
                run = do_run(frames=10, model=model, kernel=k, level=l+1)
                db.add(run)
            except ExecutionError as e:
                print "\tFailed with: %s" % e.message
    return db

def gen_dat_file(ofile, db):
    kernel_set = { r.kernel for r in db if r.kernel }
    size_set = { r.nverts for r in db if r.nverts }
    kernel_list = sorted(kernel_set, key=lambda k: kernelNum[k])
    size_list = sorted(size_set)
    print >>ofile, "nVerts", " ".join(kernel_list)
    for size in size_list:
        print >>ofile, size,
        for kernel in kernel_list:
            run_list = filter(lambda r: r.nverts == size and r.kernel == kernel, db)
            if len(run_list) == 1:
                run = run_list[0]
                invtimes = run.mean()
                if kernel in ['CPU', 'Cuda', 'OpenCL', 'OpenMP']:
                    streambw = (run.tablemem + 6*4*run.nverts) * invtimes / run.nverts / 1e6
                else:
                    streambw = (run.mem + 6*4*run.nverts) * invtimes / run.nverts / 1e6
                print >>ofile, " %f" % streambw,
            elif len(run_list) == 0:
                print >>ofile, " ?",
        print >>ofile


def main(argv):
    model = argv[1][7:-4]
    with open("stream_%s.dat" % model, 'w') as ofile:
        gen_dat_file(ofile, build_db(model))


if __name__ == '__main__':
    main(sys.argv)
