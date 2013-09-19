#!/usr/bin/env python2.7

import sys, os, shelve

from bench import *

activeKernels = [
    "CustomGPU",
    "CustomHYB",
]

paramRange = range(4, 30)

def columnName(k):
    names = {
      "CustomGPU": "Matrix-driven (GPU)",
      "CustomHYB": "Matrix-driven (CPU+GPU)",
    }
    try:
      return names[k]
    except KeyError:
      return k

def build_db(model):
    db = set()
    for k in activeKernels:
        for d in paramRange:
            try:
                level = modelMaxLevel[model]
                run = do_run(frames=100, model=model, kernel=k, level=level, divider=d)
                db.add(run)
            except ExecutionError as e:
                print "\tFailed with: %s" % e.message
    return db

def gen_dat_file(ofile, db):
    kernel_set = { r.kernel for r in db if r.kernel }
    kernel_list = sorted(kernel_set, key=lambda k: kernelNum[k])
    print >>ofile, "divider", " ".join(['"%s" "%s-std"' % (columnName(name), name) for name in kernel_list])
    for d in paramRange:
        print >>ofile, d,
        for kernel in kernel_list:
            run_list = filter(lambda r: r.divider == d and r.kernel == kernel, db)
            if len(run_list) == 1:
                print >>ofile, " %f %f" % (run_list[0].mean(), run_list[0].std()),
            elif len(run_list) == 0:
                print >>ofile, " ? ?",
        print >>ofile


def main(argv):
    model = argv[1][8:-4]
    with open("divider_%s.dat" % model, 'w') as ofile:
        gen_dat_file(ofile, build_db(model))


if __name__ == '__main__':
    main(sys.argv)
