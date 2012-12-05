GLUT_VIEWER = /Users/mbdriscoll/cs284/OpenSubdiv/build/bin/glutViewer
MODELS = catmark_cube loop_icosahedron catmark_dart_edgecorner
MEM_DATS := $(addprefix mem_, $(addsuffix .dat, $(MODELS)))
PERF_DATS := $(addprefix perf_, $(addsuffix .dat, $(MODELS)))
PERF_TEXS := $(addprefix perf_, $(addsuffix .tex, $(MODELS)))
INSTALL_PATH := /Users/mbdriscoll/cs284/hpg2013/graphs

TEXFILES = mem.tex  $(PERF_TEXS)

default:    all

all:  $(TEXFILES)

perf_%.dat:	$(GLUT_VIEWER)
	./perf.py $@
perf_%.tex: perf_%.dat
	./perf.gpi $(INSTALL_PATH)/$@ $^

mem.tex: $(MEM_DATS)
	./mem.gpi $(INSTALL_PATH)/$@ $^
mem_%.dat: $(GLUT_VIEWER)
	./mem.py $@

clean:
	rm -rf *tex
