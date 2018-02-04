#!/bin/bash

# install R requirements
R -q -e "install.packages(c('rhdf5','rjson','pcalg','InvariantCausalPrediction','glmnet','Matrix','parcor','doMC'), repos='https://cloud.r-project.org')"

# install python requirements
pip install -r "code/requirements.txt"