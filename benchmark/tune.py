#!/usr/bin/env python2.7

import sys, os, shelve, subprocess

from bench import *

BASE = "/Users/mbdriscoll/OpenSubdiv"

def build_db():
    db = set()
    #for dist in range(0, 1026, 4):
    for dist in range(4, 12, 4):
        #for dest in ["_MM_HINT_T0", "_MM_HINT_T1", "_MM_HINT_T2", "_MM_HINT_NTA"]:
        for dest in ["_MM_HINT_T2", "_MM_HINT_NTA"]:
            try:
                header = open("prefetch.h", 'w')
                print >>header, "#define SPMV_PREFETCH_DIST %d" % dist
                print >>header, "#define SPMV_PREFETCH_DEST %s" % dest
                header.close()

                os.utime(BASE+"/opensubdiv/osd/mklDispatcher.cpp", None)
                os.chdir(BASE + "/build")
                assert not subprocess.call(['make', '-j', '8'])

                run = do_run(frames=10, model="BigGuy", kernel="CustomCPU", level=2)
                db.add((run, dist, dest))
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
