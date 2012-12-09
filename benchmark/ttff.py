#!/usr/bin/python2.7

import sys, os, shelve

from bench import *

NSAMPLES = 100
NFRAMESPERSAMPLE = 100

modelTtffLevel = {
    "BigGuy":       4,
    "Bunny":        5,
    "MonsterFrog":  5,
    "Venus":        4,
    "Cube":         7,
    "Icosahedron":  7,
}

def print_at_index(ofile, r0, ri, i, n):
    print >>ofile, r0,
    for k in range(n):
        if k == i:
            print >>ofile, ri,
        else:
            print >>ofile, "?",
    print >>ofile

def build_db(model):
    db = []
    for k in activeKernels:
        for sample in range(NSAMPLES):
            level=modelTtffLevel[model]
            db.append( do_run(frames=NFRAMESPERSAMPLE, model=model, kernel=k, level=level) )
    return db

def gen_dat_file(ofile, db):
    kernel_list = list({ r.kernel for r in db })
    size_set = { r.nverts for r in db if r.nverts }
    size_list = sorted(size_set)
    fxns = {}
    for k in range(len(kernel_list)):
        kernel = kernel_list[k]
        run_list = filter(lambda r: r.kernel == kernel, db)
        ttff_in_ms = np.mean([r.ttff for r in run_list])
        cumulative_verts = 0
        xs, ys = [], []
        for ms in range(1,NFRAMESPERSAMPLE):
            cumulative_verts += np.mean([ r.frame_times[ms] for r in run_list ])
            xs.append(ttff_in_ms+ms)
            ys.append(cumulative_verts)
        x = np.array(xs)
        y = np.array(ys)
        A = np.vstack([x, np.ones(len(x))]).T
        m, c = np.linalg.lstsq(A, y)[0]
        fxns[kernel] = lambda x: x*m + c
        print >>ofile, "%s(x) = %f * x + %f" % (kernel, m, c)

    xmax = max([r.ttff for r in db]) * 2
    ymax = max([fxns[k](xmax) for k in fxns]) * 1.05
    print >>ofile, "# xmax = %f, ymax = %f" % (xmax, ymax)

def main(argv):
    model = argv[1][5:-4]
    with open("ttff_%s.dat" % model, 'w') as ofile:
        gen_dat_file(ofile, build_db(model))


if __name__ == '__main__':
    main(sys.argv)
