MODELS = BigGuy Bunny Cube Icosahedron
MEM_DATS := $(addprefix mem_, $(addsuffix .dat, $(MODELS)))
SPARSITY_DATS := $(addprefix sparsity_, $(addsuffix .dat, $(MODELS)))
STABILITY_DATS := $(addprefix stability_, $(addsuffix .dat, $(MODELS)))
PERF_TEXS := $(addprefix perf_, $(addsuffix .tex, $(MODELS)))
CSRMM_TEXS := $(addprefix csrmm_, $(addsuffix .tex, $(MODELS)))
RELATIVE_DATS := $(addprefix relative_, $(addsuffix .dat, $(MODELS)))
REORDER_DATS := $(addprefix reorder_, $(addsuffix .dat, $(MODELS)))
PERF_DATS := $(addprefix perf_, $(addsuffix .dat, $(MODELS)))
CSRMM_DATS := $(addprefix csrmm_, $(addsuffix .dat, $(MODELS)))
DIVIDER_DATS := $(addprefix divider_, $(addsuffix .dat, $(MODELS)))
STREAM_TEXS := $(addprefix stream_, $(addsuffix .tex, $(MODELS)))
TTFF_TEXS := $(addprefix ttff_, $(addsuffix .tex, $(MODELS)))
FPS_TEXS := $(addprefix fps_, $(addsuffix .tex, $(MODELS)))
MATRIXTIME_TEXS := matrixtime_MKL.tex matrixtime_CuSPARSE.tex
RELATIVE_TEX := relative.tex
BUILD_PATH := ./graphs
INSTALL_PATH := ./graphs/
INSTALL_PATH := /home/driscoll/hpg2014/graphs

TEXFILES = mem.tex sparsity.tex stability.tex perf.tex $(PERF_TEXS) $(RELATIVE_TEX) $(STREAM_TEXS) $(TTFF_TEXS) $(FPS_TEXS) $(MATRIXTIME_TEXS)
SPYFILES = $(addprefix spy_, $(addsuffix .pdf, $(MODELS)))

default:    all

all:  $(SPYFILES) $(TEXFILES)

csrmm_Bunny.tex: csrmm_Bunny.dat
	./csrmm.gpi $(BUILD_PATH)/$@ $^ $*
	cp $(BUILD_PATH)/$@ $(INSTALL_PATH)/$@
	cp $(BUILD_PATH)/$(patsubst %.tex,%.eps, $@) $(INSTALL_PATH)/

perf_%.tex: perf_%.dat
	./perf.gpi $(BUILD_PATH)/$@ $^ $*
	cp $(BUILD_PATH)/$@ $(INSTALL_PATH)/$@
	cp $(BUILD_PATH)/$(patsubst %.tex,%.eps, $@) $(INSTALL_PATH)/

perf.tex: perf.gpi $(PERF_DATS)
	./perf.gpi $(BUILD_PATH)/$@
	cp $(BUILD_PATH)/$@ $(INSTALL_PATH)/$@
	cp $(BUILD_PATH)/$(patsubst %.tex,%.eps, $@) $(INSTALL_PATH)/

divider.tex: divider.gpi $(DIVIDER_DATS)
	./divider.gpi $(BUILD_PATH)/$@
	cp $(BUILD_PATH)/$@ $(INSTALL_PATH)/$@
	cp $(BUILD_PATH)/$(patsubst %.tex,%.eps, $@) $(INSTALL_PATH)/

relative.tex: $(RELATIVE_DATS)
	./relative.gpi $(BUILD_PATH)/$@
	cp $(BUILD_PATH)/$@ $(INSTALL_PATH)/$@
	cp $(BUILD_PATH)/$(patsubst %.tex,%.eps, $@) $(INSTALL_PATH)/

reorder.tex: $(REORDER_DATS)
	./reorder.gpi $(BUILD_PATH)/$@
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

csrmm_%.dat:
	./csrmm.py $@

divider_%.dat:
	./divider.py $@

relative_%.dat:
	./relative.py $@

reorder_%.dat:
	./reorder.py $@

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
