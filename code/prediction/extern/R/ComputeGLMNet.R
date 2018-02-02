##########
# Author: Philip Versteeg (2017)
# Email: pjjpversteeg@gmail.com
# Compute graphical LASSO estimation using the huge package for data X_i^n with regressor X_{i\j} and regressand X_j.
# input:
#		input 				hdf5 file with 'data' element as n X p data (dataframe / matrix)
#		method 				{'glasso', 'mbapprox'} for full glasso or Meinshausen Buhlmann approximation
# return:
#		-					write to output hdf5 file 'data' 
##########
library(glmnet)
library(rhdf5)
suppressMessages(library(foreach))
suppressMessages(library(doMC))

### NOTES
# - The problem with running the constrained regression on a subset of selected causes,
# and that as such the set of regressors X_i is restricted, is that it will find another 
# set of beta_ij as predictive for Y_j than when running on all p regressors.
#
# - Standardized = FALSE, as we already standardize the training data a proiri.
ComputeGLMNet <- function(
  input='__test_input.hdf5',
  input.dataset='data',
  output='output_glmnet.hdf5',
  alpha=1,
  lambda=NULL,
  selectCauses=NULL,
  nfolds=4,
  processes=1
) {

  # alpha: parameter that controls the mixture of the ridge and lasso regression.
  #   alpha = 0 <==> ridge penalty only
  #   alpha = 1 <==> lasso penalty only

  # processes
  if(processes > 1) {
    registerDoMC(processes)
    parallel <- TRUE
  } else {
    parallel <- FALSE
  }

  # data
	data <- h5read(input, input.dataset) # N x p
  p <- ncol(data)

  print(class(selectCauses))

  # selected causes only
  if (is.null(selectCauses)) {
    these.causes.only <- 1:p
  } else {
    # parse the string list
    cat('Selected causes only!\n-->\tWARNING: USE R-ARRAY ENCODING OF 1...length!\n')
    these.causes.only <- sapply(strsplit(selectCauses,','), strtoi)[,1]
  }

  # IMPROVE THIS:
  # for (i in 1:p){
  #   # optimal glmnet object obtained from nfold cross-validation
  #   tmp.glmnet <- cv.glmnet(x=data[,-i], y=data[,i], lambda=lambda, nfolds=nfolds, parallel=parallel, alpha=alpha)$glmnet.fit
  #   # beta coef vector is the last column the beta matrix of nvars x length(lambda)
  #   x <- as.matrix(tmp.glmnet$beta)[,length(tmp.glmnet$lambda)]
  #   # add zeros at its own index and combine result
  #   if (i == 1) {
  #     result <- c(0., x)
  #   } else if (i < p) {
  #     result <- cbind(result, c(x[1:i-1], 0., x[i:length(x)]))
  #   } else {
  #     result <- cbind(result, c(x, 0.))
  #   }
  # }

  # construct result matrix
  result <- matrix(0., nrow=length(these.causes.only), ncol=p) # row = causes!
  for (i in 1:p){
    # if the response index is in one of the causes sets, we should remove it from the predictor variables
    if (i %in% these.causes.only) {
      j <- which(these.causes.only == i)
      print(i)
      # optimal glmnet object obtained from nfold cross-validation
      tmp.glmnet <- cv.glmnet(x=data[,these.causes.only[-j]], y=data[,i], lambda=lambda, nfolds=nfolds, parallel=parallel, alpha=alpha, standardize=FALSE)$glmnet.fit
      # beta coef vector is the last column the beta matrix of nvars x length(lambda)
      x <- as.matrix(tmp.glmnet$beta)[,length(tmp.glmnet$lambda)]
      result[-j,i] = x # fill in the whole column EXCEPT the row index of the cause i itself
    } else {
      # optimal glmnet object obtained from nfold cross-validation
      tmp.glmnet <- cv.glmnet(x=data[,these.causes.only], y=data[,i], lambda=lambda, nfolds=nfolds, parallel=parallel, alpha=alpha, standardize=FALSE)$glmnet.fit
      # beta coef vector is the last column the beta matrix of nvars x length(lambda)
      x <- as.matrix(tmp.glmnet$beta)[,length(tmp.glmnet$lambda)]
      result[,i] = x # fill in the whole column EXCEPT the row index of the cause i itself
    }
  }

	# save result to disk
	if (!h5createFile(output)) {
	  unlink(output)
	  h5createFile(output)
	}
	h5write(result, output, "data")
	H5close()
}

#############
# Wrapper code for executing by external bash/python script with arguments
# - need load and save data from disk.
# - use kwargs input= and output= to get input and output data location
# - if none are given, 1st argument is the input data location
# - rest is the keyword arguments
#############
func <- 'ComputeGLMNet'
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