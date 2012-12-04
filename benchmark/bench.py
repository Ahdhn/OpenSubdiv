#!/usr/bin/python2.7

GLUT_VIEWER_PATH = "/Users/mbdriscoll/cs284/OpenSubdiv/build/bin/glutViewer"
import sys
from subprocess import *
import numpy as np

db = set()

class Run(object):
    def __init__(self, line):
        self.frame_times = []
        self.ttff = 0.0
        self.sparsity = 0.0
        self.model = ""
        self.kernel = ""

        tokens = line.split(' ')
        for token in tokens:
            if not token:
                continue
            elif '=' in token:
                key,val = token.split('=')
                try:
                    val = int(val)
                except ValueError:
                    try:
                        val = float(val)
                    except ValueError:
                        pass
                setattr(self, key, val)
            else:
                self.frame_times.append( float(token) )
        self.frame_times = np.array(self.frame_times)

    def __repr__(self):
        return "Run(model=%s, kernel=%s, ttff=%f, sparsity=%f, mean=%f, std=%f)" % \
            (self.model, self.kernel, self.ttff, self.sparsity, self.mean(), self.std())

    def mean(self):
        return np.mean(self.frame_times)

    def std(self):
        return np.std(self.frame_times)

def do_run(model=0, frames=10000, level=1, kernel=0):
    osd = Popen([GLUT_VIEWER_PATH, "-c", str(frames)], stdin=PIPE, stdout=PIPE, stderr=PIPE)
    stdout, stderr = osd.communicate()
    assert stderr == "", "OpenSubdiv must exit cleanly, got: %s" % stderr

    global db
    db = db.union({ Run(line) for line in stdout.split('\n') if line })
    print db

def main(argv):
    print "Benchmarking ..."
    for l in range(7):
        do_run(level=l)

if __name__ == '__main__':
    main(sys.argv)
