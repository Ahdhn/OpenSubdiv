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
    level_set = { r.level for r in db if r.level }
    kernel_list = sorted(kernel_set, key=lambda k: kernelNum[k])
    level_list = sorted(level_set)
    print >>ofile, "Level", " ".join(kernel_list)
    for size in level_list:
        print >>ofile, '"Level %d"' % level,
        best_cpu = filter(lambda r: r.level == level and r.kernel == "OpenMP")[0].mean()
        best_gpu = filter(lambda r: r.level == level and r.kernel == "Cuda")[0].mean()
        for kernel in kernel_list:
            run_list = filter(lambda r: r.level == level and r.kernel == kernel, db)
            if r.kernel in ["CPU", "OpenMP", "MKL", "CustomCPU"]:
                best = best_cpu
            else:
                best = best_gpu
            if len(run_list) == 1:
                print >>ofile, " %f" % best / run_list[0].mean(),
            elif len(run_list) == 0:
                print >>ofile, " ?",
        print >>ofile


def main(argv):
    model = argv[1][9:-4]
    with open("relative_%s.dat" % model, 'w') as ofile:
        gen_dat_file(ofile, build_db(model))


if __name__ == '__main__':
    main(sys.argv)
