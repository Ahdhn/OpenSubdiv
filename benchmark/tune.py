#!/usr/bin/env python2.7

import sys, os, shelve, subprocess

from bench import *

BASE = "/home/driscoll/OpenSubdiv"


def build_db():
    runno = 0
    db = set()
    for dist in range(0, 1026, 4):
        for dest in ["_MM_HINT_T0", "_MM_HINT_T1", "_MM_HINT_T2", "_MM_HINT_NTA"]:
            try:
                header = open("prefetch.h", 'w')
                print >>header, "#define SPMV_PREFETCH_DIST %d" % dist
                print >>header, "#define SPMV_PREFETCH_DEST %s" % dest
                header.close()

                os.utime(BASE+"/opensubdiv/osd/mklDispatcher.cpp", None)
                os.chdir(BASE + "/build")
                assert not subprocess.call(['make', '-j', '8'])

                print "#% 4d/1024 " % runno,
                runno += 1

                print "Testing %d %d" % (dest, dist)
                run = do_run(frames=100, model="BigGuy", kernel="CustomCPU", level=4)
                db.add((run, dist, dest))
                best = sorted(db, key=lambda run: run[0].mean())
                print "Best:", best[-1][0].mean(), best[-1][1], best[-1][2]
            except ExecutionError as e:
                print "\tFailed with: %s" % e.message
    return db

def gen_dat_file(db):
    best = sorted(db, key=lambda run: run[0].mean())
    for i in range(20):
        print best[i][0].mean(), best[i][1], best[i][2]


def main(argv):
    gen_dat_file(build_db())


if __name__ == '__main__':
    main(sys.argv)
