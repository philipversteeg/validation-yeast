##########
# Author: Philip Versteeg (2017)
# Email: pjjpversteeg@gmail.com
# #######
# Compute Greedy Interventional Equivalence Search (GIES), callable from command line
# with arugments. Can be bootstrapped if and can use multiple processes if so.
#
# Args:
#   input         Input MicroArrayData hdf5 format file
#   output        Output MicroArrayData hdf5 format file
#   maxDegree     Parameter used to limit the vertex degree of the estimated graph. Possible values:
#                   1. Vector of length 0 (default): vertex degree is not limited.
#                   2. Real number r, 0 < r < 1: degree of vertex v is limited to r · nv, where nv
#                   denotes the number of data points where v was not intervened.
#                   3. Single integer: uniform bound of vertex degree for all vertices of the graph.
#                   4. Integer vector of length p: vector of individual bounds for the vertex degrees.
#   selectCauses  Integer vector of indices in 1:p that where causes will return
# Return:
#   NULL          Write to output hdf5 file 'data' 
# 
# 
#  Additional details on settings:
#   phase: Character vector listing the phases that should be used;
#          possible values: ‘forward’, ‘backward’, and ‘turning’ (cf.
#          details).
# 
# iterate: Logical indicating whether the phases listed in the argument
#          ‘phase’ should be iterated more than once (‘iterate = TRUE’)
#          or not.
# 
# turning: Setting ‘turning = TRUE’ is equivalent to setting ‘phases =
#          c("forward", "backward")’ and ‘iterate = FALSE’; the use of
#          the argument ‘turning’ is deprecated.
# 
# 
##########
library(pcalg)
library(Matrix)
library(rhdf5)
library(foreach)
library(doMC)
source('../libs/LoadMicroArrayData.R')
source('../libs/LoadCausalArray.R')

# ComputeGIES <- function(input='../data/kemmeren/Kemmeren.hdf5',
# ComputeGIES <- function(input='kemmeren_100ints_100obs/__input__gies.hdf5',
ComputeGIES <- function(input='testset/__input__gies_100_prescreened.hdf5',                      
                        output='testset/__output_gies_prescreened.hdf5', 
                        maxDegree=NULL,             
                        bootstraps=1,               # number of bootstrapsfor 'bagging' (more like stability selection)
                        bootstrapFraction=.5,       # fracton in [0, 1] to sample in each bootstrap
                        processes=1,                # number of simultaneous threads to compute with
                        selectCauses=NULL,          # list of indices of parents in 1:p that need to be considered.
                        prescreeningFile=NULL,      # the hdf5 file containing the binary CausalArray 
                        verbose=FALSE)              # detailed printout
{ 
  
  # load data
  data <- LoadMicroArrayData(file.in=input, verbose=TRUE)

  # set maxDegree default
  if (is.null(maxDegree)) {
    maxDegree <- integer(0)
  }

  # used for both selcetCauses and prescreening
  variableSelMat <- matrix(TRUE, nrow=data$p, ncol=data$p)
  rownames(variableSelMat) <- data$genenames
  colnames(variableSelMat) <- data$genenames

  # return only results for selectCauses.
  if (!is.null(selectCauses)) {
    cat('Selected causes only!\n-->\tWARNING: USE R-ARRAY ENCODING OF 1...length!\n')
    selectCauses <- sapply(strsplit(selectCauses,','), strtoi)[,1]
    # --> need to fill a logical matrix variableSelMat with dimension p x p with TRUE for entry (i,j) if 
    #     says that variable i should be considered as a potential parent for variable j and vice versa for false.
    # variableSelMat <- matrix(FALSE, nrow=data$p, ncol=data$p)
    variableSelMat[-selectCauses,] <- FALSE # put to false all the rows that are not in selectCauses
  }

  if (is.character(prescreeningFile)) {
    print(paste('Pre-screening file given:', prescreeningFile, '.'))
    beforeScreeningSum <- sum(variableSelMat == TRUE)
    selectCausalArray <- LoadCausalArray(prescreeningFile)
    # check for all causes and effects... 
    if(!is.null(selectCauses)) {
      causesGeneId <- intersect(data$genenames[selectCauses], selectCausalArray$causes)
    } else {
      causesGeneId <- intersect(data$genenames, selectCausalArray$causes)
    }
    effectsGeneId <- intersect(data$genenames, selectCausalArray$effects)
    for (i in causesGeneId) {
      for (j in effectsGeneId) {
        # look up if each i,j pair exists in causalarray
        if (selectCausalArray$array[i, j] == 0) { 
          variableSelMat[i, j] <- FALSE # put to zero terms
        }
      }
    }
    print(paste('Pre-screening removed', beforeScreeningSum - sum(variableSelMat == TRUE), 'pairs from computation.'))
  }
  
  # Result matrix
  result <- matrix(0, nrow=data$p, ncol=data$p)

  ## bootstrap sample interventions
  if (bootstraps > 1) {

    registerDoMC(processes)
    
    foreachResult = foreach(j=1:bootstraps, .inorder=FALSE) %dopar% {
      # sample with replacements set of interventions
      # intervention.bootstrap <- sample(data$intpos, floor(data$nInt * bootstrapFraction), replace=FALSE) # easier for now
      mutants <- sample(data$mutants, floor(data$nInt * bootstrapFraction), replace=FALSE)
      # and set of observations
      obs <- sample(1:data$nObs, floor(data$nObs * bootstrapFraction), replace=FALSE)
      intpos <- sapply(mutants,function(x) which(data$genenames == x))
      
      interventions <- lapply(intpos, as.integer)   # all interventions
      targets <- unique(interventions)              # unique interventions
      target.index <- match(interventions, targets) # indexin target list
      cat('Performing bootstrap:', j, '\n')
      cat(length(targets), 'unique interventions sampled out of', length(interventions), 'total.\n')

      score <- new("GaussL0penIntScore", data=rbind(data$obs, data$int), 
        targets=targets, target.index=target.index)
      if(is.null(selectCauses) & is.null(prescreeningFile)) {
        fixedGaps <- NULL
      } else {
        fixedGaps <- !variableSelMat
      }
      str(fixedGaps)
      tmp <- gies(score, 
                  fixedGaps=fixedGaps,
                  phase=c("forward", "backward"),
                  iterate=FALSE,
                  maxDegree=maxDegree,
                  verbose=verbose)
      tmp.result <- as(tmp$essgraph, "matrix")
      tmp.result <- tmp.result & ! t(tmp.result) # result per j
    }
    # combine results
    for (j in foreachResult) {
      result <- result + j
    }
  ## no bootstrap
  } else {    

    # get intervention targets for Gaussian score
    interventions <- lapply(data$intpos, as.integer)
    targets <- unique(interventions)
    target.index <- match(interventions, targets)

    # compute gies
    score <- new("GaussL0penIntScore", data=rbind(data$obs, data$int), 
      targets=targets, target.index=target.index)
    tmp <- gies(score, 
            fixedGaps=if (is.null(selectCauses)) NULL else (!variableSelMat),
            maxDegree=maxDegree,
            phase='turning',
            verbose=verbose)
    result <- as(tmp$essgraph, "matrix")
    result <- result & ! t(result)
  }

  # only take the selected causes as output
  if (!is.null(selectCauses)) {
    result <- result[selectCauses,]  
  }

  cat('Found', length(which(result != 0)), 'non-zero edges out of', ncol(result) * nrow(result), 'total.\n')

  #####
  # saving results
  ###
  cat('Saving results.\n')
  if (!h5createFile(output)) {
    unlink(output)
    h5createFile(output)
  }

  # HAVE TO LOAD IT TRANSPOSED IN PYTHON TO GET THE CORRECT SHAPE WHEN LOADING IN!
  # --> fixed by using libs/misc.load_array(..., load_from_r=True).
  h5write(result, output, 'data', level=9)
  # h5write(round(ida.rel.predictions, round.size), output, 'data', level=9)
  H5close()
}


#############
# Wrapper code for executing by external bash/python script with arguments
# - need load and save data from disk.
# - use kwargs input= and output= to get input and output data location
# - if none are given, 1st argument is the input data location
# - rest is the keyword arguments
#############
func <- 'ComputeGIES'
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