#!/usr/bin/env python2.7

import sys, os, shelve

from bench import *

activeKernels = [
  "OpenMP",
  "Cuda",
  "CustomCPU",
  "CustomGPU",
]

best_k = {
  "BigGuy": 20,
  "Bunny": 17,
  "Cube": 8,
  "Icosahedron": 9,
}

def columnName(k):
    names = {
      "Cuda": "Table-driven (GPU)",
      "OpenMP": "Table-driven (CPU)",
      "CustomCPU": "Matrix-driven (CPU)",
      "CustomGPU": "Matrix-driven (GPU)",
    }
    try:
      return names[k]
    except KeyError:
      return k

def build_db(model):
    db = set()
    for k in activeKernels:
        for l in range( modelMaxLevel[model] ):
            try:
                run = do_run(frames=100, model=model, kernel=k, level=l+1, divider=best_k[model])
                db.add(run)
            except ExecutionError as e:
                print "\tFailed with: %s" % e.message
    return db

def gen_dat_file(ofile, db):
    kernel_set = { r.kernel for r in db if r.kernel }
    size_set = { r.nverts for r in db if r.nverts }
    kernel_list = sorted(kernel_set, key=lambda k: kernelNum[k])
    size_list = sorted(size_set)
    #print >>ofile, "nVerts", " ".join(['"%s" "%s stdev"' % (columnName(name), name) for name in kernel_list])
    print >>ofile, "nVerts", " ".join(['"%s"' % (columnName(name),) for name in kernel_list])
    for size in size_list:
        print >>ofile, size,
        for kernel in kernel_list:
            run_list = filter(lambda r: r.nverts == size and r.kernel == kernel, db)
            if len(run_list) == 1:
                #print >>ofile, " %f %f" % (run_list[0].mean(), run_list[0].std()),
                print >>ofile, " %f" % run_list[0].mean(),
            elif len(run_list) == 0:
                print >>ofile, " ? ?",
        print >>ofile


def main(argv):
    model = argv[1][5:-4]
    with open("perf_%s.dat" % model, 'w') as ofile:
        gen_dat_file(ofile, build_db(model))


if __name__ == '__main__':
    main(sys.argv)
