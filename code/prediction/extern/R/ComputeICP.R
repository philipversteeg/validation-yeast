##########
# Author: Philip Versteeg (2017)
# Compute ICP prediction
# args:
#   input         Input MicroArrayData hdf5 format file
#   output        Output MicroArrayData hdf5 format file# return:
#       -                   write to output hdf5 file 'data' 
# return:
#   NULL          Write to output hdf5 file 'data' 
##########
suppressMessages(library(rhdf5))
suppressMessages(library(InvariantCausalPrediction))
suppressMessages(library(foreach))
suppressMessages(library(doMC))
source('../libs/LoadMicroArrayData.R')
source('../libs/LoadCausalArray.R')

ComputeICP <- function(input='testset/__input__icp_10.hdf5',
                       output='testset/__output_ICP.hdf5',
                       alpha=0.05,                 # conf level 
                       bootstraps=2,               # number of bootstrapsfor 'bagging' (more like stability selection)
                       bootstrapFraction=.5,       # fracton in [0, 1] to sample in each bootstrap
                       # selectCauses=NULL,          # indices of genes in data$genesnames that
                       # selectCauses = '1, 4, 6, 9, 10, 11, 12, 13, 15, 19, 23, 24, 26, 27, 30, 31, 32, 34, 40, 42, 43, 46, 48, 49, 50', 
                       processes=2,
                       verbose=TRUE) {
  # use ICP version > 0.6-0
  if( compareVersion(as.character(packageVersion("InvariantCausalPrediction")),"0.6") < 0){
    stop(" need to use version at least 0.6-0 of 'InvariantCausalPrediction'-- please update the package")
  }

  # load data
  data <- LoadMicroArrayData(file.in=input, verbose=TRUE)

  # FWER-control at level 0.05
  alpha <- alpha/data$p

  # # parse selectCauses to get indices.
  # if (!is.null(selectCauses)) {
  #   cat('Selected causes only!\n-->\tWARNING: USE R-ARRAY ENCODING OF 1...length!\n')
  #   selectCauses <- sapply(strsplit(selectCauses,','), strtoi)[,1]
  # } else{
  #   selectCauses <- 1:data$p
  # }
  selectCauses <- 1:data$p

  # No bootstraps, just single estimate
  if (bootstraps == 1) {

    # Joint data
    X <- rbind(data$int, data$obs)
    # experimental conditions vector, 1 for obs, 2 for int
    ExpInd <- c(rep(2, data$nInt), rep(1, data$nObs))
    # result matrix as c x ef
    result <- matrix(FALSE, nrow=length(selectCauses), ncol=data$p)

    # loop over all target genes
    for (target in 1:data$p){
      if (TRUE) cat("Computing ICP for target", target, "::",data$genenames[target],'\n')
      Ytarget <- X[, target]
      Xtarget <- X[, -target]
      # compute ICP
      icp <- ICP(Xtarget, Ytarget, ExpInd=ExpInd, alpha=alpha, showCompletion=FALSE,showAcceptedSets=FALSE, stopIfEmpty=TRUE)
      # look at results only if not whole model has been rejected
      if(! icp$modelReject){
        # check if significant results
        if(any(icp$pvalues <= alpha)){
          result[which(icp$pvalues<=alpha), target] <- TRUE
        }
      }
    }
  }

  if (bootstraps > 1) {
    cat('Sample', floor(data$nObs * bootstrapFraction), 'observations and', floor(data$nInt * bootstrapFraction), 'interventions in each bootstrap.\n')

    X <- rbind(data$int, data$obs)

    # For single bootstrap...
    registerDoMC(processes)

    foreachResult = foreach(j=1:bootstraps, .inorder=TRUE) %dopar% {
      # sample with replacements set of interventions
      cat('Performing bootstrap:', j, '..\n')
      ints <- sample(1:data$nInt, floor(data$nInt * bootstrapFraction), replace=FALSE)
      obs <- sample(1:data$nObs, floor(data$nObs * bootstrapFraction), replace=FALSE)

      # select this X (not that observations are shifted in X)
      tmpX <- X[c(ints, data$nInt + obs), ]
      ExpInd <- c(rep(2, times=length(ints)), rep(1, times=length(obs)))
      # result matrix as c x ef
      tmpResult <- matrix(FALSE, nrow=length(selectCauses), ncol=data$p)

      # loop over all target genes
      for (target in 1:data$p){ 
        Ytarget <- tmpX[, target]
        Xtarget <- tmpX[, -target]
        # compute ICP
        icp <- ICP(Xtarget, Ytarget, ExpInd=ExpInd, alpha=alpha, showCompletion=FALSE,showAcceptedSets=FALSE, stopIfEmpty=TRUE)
        # look at results only if not whole model has been rejected
        if(! icp$modelReject){
          # check if significant results
          if(any(icp$pvalues <= alpha)){
            tmpResult[which(icp$pvalues<=alpha), target] <- TRUE
          }
        }
      }
      cat('..found', sum(tmpResult), 'edges in bootstrap', j, '.\n')
      return(tmpResult)
    }
    # combine results
    result <- matrix(0, nrow=length(selectCauses), ncol=data$p)
    for (j in foreachResult) {
      result <- result + j
    }
  }

  # Done
  cat('ICP done, found', sum(result != 0), 'non-zero edges.\n')

  str(result)

  # saving results
  cat('Saving results..\n')
  if (!h5createFile(output)) {
    unlink(output)
    h5createFile(output)
  }
  # HAVE TO LOAD IT TRANSPOSED IN PYTHON TO GET THE CORRECT SHAPE WHEN LOADING IN!
  # --> fixed by using libs/misc.load_array(..., load_from_r=True).
  h5write(result, output, 'data', level=9)
  H5close()
}

#############
# Wrapper code for executing by external bash/python script with arguments
# - need load and save data from disk.
# - use kwargs input= and output= to get input and output data location
# - if none are given, 1st argument is the input data location
# - rest is the keyword arguments
#############
func <- 'ComputeICP'
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