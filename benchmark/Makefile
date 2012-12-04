GLUT_VIEWER = /Users/mbdriscoll/cs284/OpenSubdiv/build/bin/glutViewer
MODELS = catmark_cube loop_icosahedron catmark_dart_edgecorner
MEM_DATS := $(addprefix mem_, $(addsuffix .dat, $(MODELS)))
INSTALL_PATH := /Users/mbdriscoll/cs284/hpg2013/graphs

TEXFILES = mem.tex

default:    all

all:  $(TEXFILES)

mem.tex: $(MEM_DATS)
	./mem.gpi $(INSTALL_PATH)/$@ $^

mem_%.dat: $(GLUT_VIEWER)
	./mem.py $@

clean:
	rm -rf *tex
