GLUT_VIEWER = /Users/driscoll/OpenSubdiv/build/bin/glutViewer
MODELS = catmark_cube loop_icosahedron catmark_dart_edgecorner
MEM_DATS := $(addprefix mem_, $(addsuffix .dat, $(MODELS)))
PERF_TEXS := $(addprefix perf_, $(addsuffix .tex, $(MODELS)))
INSTALL_PATH := /Users/driscoll/hpg2013/graphs

TEXFILES = mem.tex  $(PERF_TEXS)

default:    all

all:  $(TEXFILES)

perf_%.tex: perf_%.dat
	./perf.gpi $(INSTALL_PATH)/$@ $^

mem.tex: $(MEM_DATS)
	./mem.gpi $(INSTALL_PATH)/$@ $^

clean:
	rm -rf *tex

.SECONDARY:

perf_%.dat:	$(GLUT_VIEWER)
	./perf.py $@

mem_%.dat: $(GLUT_VIEWER)
	./mem.py $@

