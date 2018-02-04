#!/bin/bash
# 
# Download and process data from MacIsaac 2005

ORG_DIR="org_data"
if ! [ -d "$ORG_DIR" ] ; then
	mkdir $ORG_DIR
	cd $ORG_DIR
	wget http://fraenkel.mit.edu/improved_map/orfs_by_factor.tar.gz .
	tar -xzvf orfs_by_factor.tar.gz
	cd ..
fi