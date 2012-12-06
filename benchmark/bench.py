#!/usr/bin/python2.7

import sys, re
from subprocess import *
import numpy as np

GLUT_VIEWER_PATH = "/home/driscoll/OpenSubdiv/build/bin/glutViewer"

kernelNum = {
	"CPU":        0,
	"OpenMP":     1,
	"Cuda":       2,
	"GLSL":       3,
	"OpenCL":     4,
	"MKL":        5,
	"ClSpMV":     6,
	"CuSPARSE":   7,
	"uBLAS":      8,
	"MAX":        9
}

activeKernels = [
    "CPU",
    "OpenMP",
    "Cuda",
    "GLSL",
    "OpenCL",
    "MKL",
    "CuSPARSE",
]

modelNum = {
    "catmark_cube":       8,
    "loop_icosahedron":     26,
    "catmark_dart_edgecorner": 9
}

class Run(object):
    def __init__(self, line):
        self.frame_times = []
        self.ttff = 0.0
        self.sparsity = 0.0
        self.model = ""
        self.kernel = ""
        self.nverts = ""

        tokens = line.split(' ')
        for token in tokens:
            if not token or re.match("\W+", token):
                continue
            elif '=' in token:
                key,val = token.split('=')
                try:
                    val = float(val)
                except ValueError:
                    pass
                setattr(self, key, val)
            elif token:
                self.frame_times.append( float(token) )
	    else:
		print "Bad token: <%s>" % token
        self.frame_times = np.array(self.frame_times)
	self.frame_times.sort()

    def timeofframe(frameid):
        return self.ttff + sum(self.frame_times[:frameid])

    def __repr__(self):
        return "Run(model=%s, kernel=%s, level=%s, nverts=%s, ttff=%f, sparsity=%f, mean=%f, std=%f, nframe=%d)" % \
            (self.model, self.kernel, self.level, self.nverts, self.ttff, self.sparsity, self.mean(), self.std(), self.nframes())

    def mean(self):
        return np.mean(self.frame_times)

    def std(self):
        return np.std(self.frame_times)

    def nframes(self):
        return len(self.frame_times)

    def kernel_id(self):
        return kernelNum[ self.kernel ]

def escape_latex(string):
    return string.replace('_', '\\\\_')

def do_run(model='cube', frames=1000, level=1, kernel='CPU'):
    assert 0 < level <= 7, "Must select positive subdiv level from 1 to 7."
    cmd_line = [
        GLUT_VIEWER_PATH,
        "--count", str(frames),
        "--kernel", str(kernelNum[kernel]),
        "--level", str(level),
        "--model", str(modelNum[model])
    ]
    print "Running: %s" % " ".join(cmd_line)
    osd = Popen(cmd_line, stdin=PIPE, stdout=PIPE, stderr=PIPE)
    stdout, stderr = osd.communicate()
    assert stderr == "", "OpenSubdiv must exit cleanly, got: %s" % stderr
    return Run(stdout)

def main(argv):
    print "Benchmarking ..."
    by_level = set()
    for k in activeKernels:
       for l in range(3):
           by_level.add( do_run(kernel=k, level=l+1) )
    print by_level

if __name__ == '__main__':
    main(sys.argv)