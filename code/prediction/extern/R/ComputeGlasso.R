##########
# Author: Philip Versteeg (2016)
# Email: pjjpversteeg@gmail.com
# Compute graphical LASSO estimation using the huge package for data X_i^n with regressor X_{i\j} and regressand X_j.
# input:
#		input 				hdf5 file with 'data' element as N X p matrix. (Expects normalized data!)
#		method 				{'glasso', 'mbapprox'} for full glasso or Meinshausen Buhlmann approximation
# return:
#		-					write to output hdf5 file 'data' 
#
# NOTE: for some reason method='mb' doens't work now.
##########
library(huge)
library(rhdf5)

ComputeGlasso <- function(input='__test_input.hdf5',
                          input.dataset='data', 
                          output='output_glasso.hdf5',
                          method='glasso',
                          round.size=6) {
  
	data <- h5read(input, input.dataset)
  result <- huge(data, method=method, verbose=TRUE) 

  # last matrix contains the inverse covariances (edge weights of UG)
  sparse.result <- round(as.matrix(tail(result$icov,n=1)[[1]]),round.size)

	# save result to disk
	if (!h5createFile(output)) {
	  unlink(output)
	  h5createFile(output)
	}
	h5write(sparse.result, output, "data", level=9)
  H5close()  
}

#############
# Wrapper code for executing by external bash/python script with arguments
# - need load and save data from disk.
# - use kwargs input= and output= to get input and output data location
# - if none are given, 1st argument is the input data location
# - rest is the keyword arguments
#############
func <- 'ComputeGlasso'
#############
commandlineargs <- commandArgs(trailingOnly=TRUE)
# (1.) get argument and default values!
functionargs <- as.list(formals(func))

# (2.) fill in positional args
args <- commandlineargs[grep('=',commandlineargs, invert=TRUE)]
if (length(args) > 0) {
  for (i in 1:length(args)) {
    functionargs[[i]] <- args[i]
  }
}

# (3.) fill in kwargs (if okay formatted)
kwargs <- commandlineargs[grep('=',commandlineargs)]
for (i in kwargs) {
  tmpvec <- unlist(strsplit(i, split='=', fixed=TRUE))
  if (length(tmpvec) < 2) {
    stop('** argument parse error ** invalid argument: ', i, '\n')
  } 
  if (!tmpvec[1] %in% names(functionargs)) {
    stop('** argument parse error ** argument not found: ', tmpvec[1], '\n')
  }
  functionargs[tmpvec[1]] <- paste(tmpvec[-1], collapse='=') 
}

# (4.) check if all arguments are filled and parse strings to numeric, int, bool or NULL if possible
if(!all(sapply(functionargs, function(x) !is.symbol(x)))) {
  stop('** argument parse error ** required argument(s) not filled: ', names(functionargs)[!sapply(functionargs, function(x) !is.symbol(x))], '\n')
}
for (i in names(functionargs)) {
  x <- functionargs[i]
  # the default NULL arguments are still here, skip these.
  if (!is.null(x)) {
    # check if argument value is numeric
    is.num <- FALSE
    try(is.num <- !is.na(suppressWarnings(as.numeric(x))), silent=TRUE)
    if (is.num) {
      functionargs[i] <- as.numeric(x)}
    # check if argument value is integer
    else {
      is.int <- FALSE
      try(is.int <- !is.na(suppressWarnings(as.integer(x))), silent=TRUE) 
      if (is.int) {functionargs[i] <- as.integer(x)}
    }
    # check if argument value is boolean
    if (x == 'TRUE') {functionargs[i] <- TRUE} 
    if (x == 'FALSE') {functionargs[i] <- FALSE}
    # check if argument value is 'NULL' using list syntax or it removes the element
    if (x == 'NULL') {functionargs[i] <- list(NULL)}
  }
}
 
# (5.) call function
cat('********************************\nCalling', func, 'with arguments:\n')
for (i in names(functionargs)) {
  if (is.null(functionargs[[i]])) {
    val <- 'NULL'
  } else {
    val <- functionargs[[i]]
  }
  cat('   ', i, '=', val, '\n')
}
cat('********************************\n')
do.call(func, functionargs)