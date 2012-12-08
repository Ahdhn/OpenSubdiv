MODELS = BigGuy Bunny MonsterFrog Venus Cube Icosahedron
MEM_DATS := $(addprefix mem_, $(addsuffix .dat, $(MODELS)))
SPARSITY_DATS := $(addprefix sparsity_, $(addsuffix .dat, $(MODELS)))
PERF_TEXS := $(addprefix perf_, $(addsuffix .tex, $(MODELS)))
INSTALL_PATH := /home/driscoll/hpg2013/graphs

TEXFILES = mem.tex  sparsity.tex $(PERF_TEXS)
SPYFILES = $(addprefix spy_, $(addsuffix .pdf, $(MODELS)))

default:    all

all:  $(SPYFILES) $(TEXFILES)

perf_%.tex: perf_%.dat
	./perf.gpi $(INSTALL_PATH)/$@ $^ MODELNAME

mem.tex: $(MEM_DATS)
	./mem.gpi $(INSTALL_PATH)/$@ $^

sparsity.tex: $(SPARSITY_DATS)
	./sparsity.gpi $(INSTALL_PATH)/$@ $^

spy_%.pdf: spy_%.mm
	./mmtospy.py $^
	cp $@ $(INSTALL_PATH)/$@

clean:
	rm -rf *tex

.SECONDARY:

perf_%.dat:
	./perf.py $@

mem_%.dat:
	./mem.py $@

spy_%.mm:
	./spy.py $@

sparsity_%.dat:
	./sparsity.py $@

