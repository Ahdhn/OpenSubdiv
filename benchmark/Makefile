MODELS = BigGuy Bunny Cube Icosahedron
MEM_DATS := $(addprefix mem_, $(addsuffix .dat, $(MODELS)))
SPARSITY_DATS := $(addprefix sparsity_, $(addsuffix .dat, $(MODELS)))
STABILITY_DATS := $(addprefix stability_, $(addsuffix .dat, $(MODELS)))
PERF_TEXS := $(addprefix perf_, $(addsuffix .tex, $(MODELS)))
RELATIVE_TEXS := $(addprefix relative_, $(addsuffix .tex, $(MODELS)))
STREAM_TEXS := $(addprefix stream_, $(addsuffix .tex, $(MODELS)))
TTFF_TEXS := $(addprefix ttff_, $(addsuffix .tex, $(MODELS)))
FPS_TEXS := $(addprefix fps_, $(addsuffix .tex, $(MODELS)))
MATRIXTIME_TEXS := matrixtime_MKL.tex matrixtime_CuSPARSE.tex
BUILD_PATH := ./graphs
#INSTALL_PATH := /home/driscoll/sga2013/graphs
INSTALL_PATH := /Users/mbdriscoll/sga2013/graphs

TEXFILES = mem.tex sparsity.tex stability.tex $(PERF_TEXS) $(RELATIVE_TEXS) $(STREAM_TEXS) $(TTFF_TEXS) $(FPS_TEXS) $(MATRIXTIME_TEXS)
SPYFILES = $(addprefix spy_, $(addsuffix .pdf, $(MODELS)))

default:    all

all:  $(SPYFILES) $(TEXFILES)

perf_%.tex: perf_%.dat
	./perf.gpi $(BUILD_PATH)/$@ $^ $*
	cp $(BUILD_PATH)/$@ $(INSTALL_PATH)/$@
	cp $(BUILD_PATH)/$(patsubst %.tex,%.eps, $@) $(INSTALL_PATH)/

relative_%.tex: relative_%.dat
	./relative.gpi $(BUILD_PATH)/$@ $^ $*
	cp $(BUILD_PATH)/$@ $(INSTALL_PATH)/$@
	cp $(BUILD_PATH)/$(patsubst %.tex,%.eps, $@) $(INSTALL_PATH)/

stream_%.tex: stream_%.dat
	./stream.gpi $(BUILD_PATH)/$@ $^ $*
	cp $(BUILD_PATH)/$@ $(INSTALL_PATH)/$@
	cp $(BUILD_PATH)/$(patsubst %.tex,%.eps, $@) $(INSTALL_PATH)/

ttff_%.tex: ttff_%.dat
	./ttff.gpi $(BUILD_PATH)/$@ $^ $*
	cp $(BUILD_PATH)/$@ $(INSTALL_PATH)/$@
	cp $(BUILD_PATH)/$(patsubst %.tex,%.eps, $@) $(INSTALL_PATH)/

fps_%.tex: fps_%.dat
	./fps.gpi $(BUILD_PATH)/$@ $^ $*
	cp $(BUILD_PATH)/$@ $(INSTALL_PATH)/$@
	cp $(BUILD_PATH)/$(patsubst %.tex,%.eps, $@) $(INSTALL_PATH)/

matrixtime_%.tex: matrixtime_%.dat
	./matrixtime.gpi $(BUILD_PATH)/$@ $^ $*
	cp $(BUILD_PATH)/$@ $(INSTALL_PATH)/$@
	cp $(BUILD_PATH)/$(patsubst %.tex,%.eps, $@) $(INSTALL_PATH)/

mem.tex: $(MEM_DATS)
	./mem.gpi $(BUILD_PATH)/$@ $^
	cp $(BUILD_PATH)/$@ $(INSTALL_PATH)/$@
	cp $(BUILD_PATH)/$(patsubst %.tex,%.eps, $@) $(INSTALL_PATH)/

sparsity.tex: $(SPARSITY_DATS)
	./sparsity.gpi $(BUILD_PATH)/$@ $^
	cp $(BUILD_PATH)/$@ $(INSTALL_PATH)/$@
	cp $(BUILD_PATH)/$(patsubst %.tex,%.eps, $@) $(INSTALL_PATH)/

stability.tex: $(STABILITY_DATS)
	./stability.gpi $(BUILD_PATH)/$@ $^
	cp $(BUILD_PATH)/$@ $(INSTALL_PATH)/$@
	cp $(BUILD_PATH)/$(patsubst %.tex,%.eps, $@) $(INSTALL_PATH)/

spy_%.pdf: spy_%.mm
	./mmtospy.py $^
	cp $@ $(INSTALL_PATH)/$@

clean:
	rm -rf *tex

.SECONDARY:

perf_%.dat:
	./perf.py $@

relative_%.dat:
	./relative.py $@

stream_%.dat:
	./stream.py $@

ttff_%.dat:
	./ttff.py $@

fps_%.dat:
	./fps.py $@

matrixtime_%.dat:
	./matrixtime.py $@

mem_%.dat:
	./mem.py $@

spy_%.mm:
	./spy.py $@

sparsity_%.dat:
	./sparsity.py $@

stability_%.dat:
	./stability.py $@
