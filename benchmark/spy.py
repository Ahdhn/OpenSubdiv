#!/usr/bin/env python2.7

import sys, os, shelve

from bench import *

spyplotLevel = {
    "BigGuy":           2,
    "Bunny":            1,
    "MonsterFrog":      2,
    "Venus":            2,
    "Cube":             5,
    "Icosahedron":      5,
}   


def main(argv):
    for model in argv[1:]:
        model = model[4:-3]
        spyfile = "spy_%s.mm" % model
        do_run(frames=1, model=model, kernel='MKL', level= spyplotLevel[model], spyfile=spyfile)

if __name__ == '__main__':
    main(sys.argv)

