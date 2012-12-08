GLUT_VIEWER = /home/driscoll/OpenSubdiv/build/bin/glutViewer
MODELS = BigGuy Bunny MonsterFrog Venus Cube Icosahedron
MEM_DATS := $(addprefix mem_, $(addsuffix .dat, $(MODELS)))
PERF_TEXS := $(addprefix perf_, $(addsuffix .tex, $(MODELS)))
INSTALL_PATH := /home/driscoll/hpg2013/graphs

TEXFILES = mem.tex  $(PERF_TEXS)
SPYFILES = $(addprefix spy_, $(addsuffix .pdf, $(MODELS)))

default:    all

all:  $(SPYFILES) $(TEXFILES)

perf_%.tex: perf_%.dat
	./perf.gpi $(INSTALL_PATH)/$@ $^ MODELNAME

mem.tex: $(MEM_DATS)
	./mem.gpi $(INSTALL_PATH)/$@ $^

spy_%.pdf: spy_%.mm
	./mmtospy.py $^
	cp $@ $(INSTALL_PATH)/$@

clean:
	rm -rf *tex

.SECONDARY:

perf_%.dat:	$(GLUT_VIEWER)
	./perf.py $@

mem_%.dat: $(GLUT_VIEWER)
	./mem.py $@

spy_%.mm: $(GLUT_VIEWER)
	./spy.py $@

