#!/bin/bash
# 
# Download and process data from Kemmeren 2014
ORG_DIR="org_data"
if ! [ -d "$ORG_DIR" ] ; then
	mkdir $ORG_DIR
	cd $ORG_DIR
	wget http://deleteome.holstegelab.nl/data/downloads/causal_inference/limma_MARG_responsive_mutants.txt
	wget http://deleteome.holstegelab.nl/data/downloads/causal_inference/limma_MARG_non-responsive_mutants.txt
	wget http://deleteome.holstegelab.nl/data/downloads/causal_inference/limma_MARG_wt_pool_platereader.txt
	wget http://deleteome.holstegelab.nl/data/downloads/causal_inference/limma_MARG_wt_pool.txt
	cd ..
fi
Rscript processData.R