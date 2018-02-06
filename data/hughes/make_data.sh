#!/bin/bash
# 
# Download and process data from Hughes 2000
ORG_DIR="org_data"
if ! [ -d "$ORG_DIR" ] ; then
	mkdir $ORG_DIR
	cd $ORG_DIR
	    wget "http://yfgdb.princeton.edu/cgi-bin/README.txt?db=study_id&id=179" -O 'README.txt'
	    wget "http://yfgdb.princeton.edu/DBXREF/yfgdb/10929718id179/Hughes00.pcl"
	    wget "http://yfgdb.princeton.edu/DBXREF/yfgdb/10929718id179/Hughes00.flt.knn.avg.pcl"
	cd ..
fi