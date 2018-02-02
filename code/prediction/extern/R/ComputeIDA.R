##########
# Author: Philip Versteeg (2016)
# Email: pjjpversteeg@gmail.com
# R code with Python wrapper to simple run IDA on 1 method (PC or PCStable)
# input:
#   input         hdf5 file with MicroArrayData format
#   method        pc or pcstable
#   pcgraphfile   
#   alpha         conf level for (cond)independence test
#   selectCauses 
# return:
#   -         write to output hdf5 file 'data' 
##########
library(pcalg)
library(Matrix)
library(rhdf5)
library(foreach)
library(doMC)

ComputeIDA <- function(input='__test_input.hdf5',
                       input.dataset='data', 
                       output='output_ida.hdf5',
                       method='pcstable', 
                       pcgraphfile=NULL,
                       alpha=0.01,
                       processes=1,
                       selectCauses=NULL,    # vector of indices in 1:p that where causes will return
                       # round.size=6,
                       make.new.computation=FALSE,
                       verbose=FALSE) {
  # Alain Hauser + Jonas Peters + Joris Mooij + Philip Versteeg 
  # 2014-03-13 + 2014-06-16 + ... +  2016-10-20
  # method can be 
  #       PC, PCstablefast, PCstable
 
  if (!is.null(selectCauses)) {
    cat('Selected causes only!\n-->\tWARNING: USE R-ARRAY ENCODING OF 1...length!\n')
  }

  # process default parameters
  if (is.null(pcgraphfile)) {
    # the pc output will be similarly processed as the input file, same naming scheme.
    pcgraphfile <- sub('\\.hdf5','\\_graph.RData', input)
  }
  if (file.exists(pcgraphfile)) {
    print('WARNING: using existing pcgraphfile!')
    if (make.new.computation) {
      '...overriding, rerunning PC alg...'
    }
  }

  # register processes for parallel comp
  registerDoMC(processes)

  method <- tolower(method)

  ##
  ## load data
  ##
  data <- h5read(input, input.dataset)
  n <- dim(data)[1]
  p <- dim(data)[2]
  print(paste0('Data loaded, p=', p, ' n=', n))
  H5close()

  #####
  # computing graph 
  #####
  if(make.new.computation || !file.exists(pcgraphfile))
  {
    method <- tolower(method)
    if( method == 'pc' || method == 'pcstablefast' || method == 'pcstable' ) 
    {
      ## interventional training samples and handle them as observational ones?
      ## Same question for GES
      print(Sys.time())
      cat(' :: Starting pc with method', method, 'be patient!\n')
      skel.method <- switch(method,
                  pc = 'original',
                  pcstablefast = 'stable.fast',
                  pcstable = 'stable')
      pc.fit <- pc(suffStat=list(C=cor(data),n=n),
                   skel.method=skel.method,
                   indepTest=gaussCItest,
                   p=p,
                   alpha=alpha,
                   numCores=if (skel.method == 'stable.fast') processes else 1, # ONLY works for stablefast!
                   verbose=verbose)
      Adj <- pc.fit@graph
      save(Adj,file=pcgraphfile)
      # 2.7 h on hactar, 220 MB file
    } else if( method == 'empty' ) 
    {
      Adj <- matrix(0, p, p)
      Adj <- as(Adj, "graphNEL")
      save(Adj,file=pcgraphfile)
    } else {
      stop('Invalid method in ComputeIda.R')
    }
  }
  
  #####
  # loading graph
  #####
  load(pcgraphfile)
  
  #####
  # either run for predictions on all vars, or a subset 
  #####
  if (is.null(selectCauses)) {
    intervNodes <- 1:p
  } else {
    # parse the string list
    intervNodes <- sapply(strsplit(selectCauses,','), strtoi)[,1]
  }

  #####
  # applying ida
  #####
  print(Sys.time())
  cat(' :: Starting IDA, be patient!\n')
  tic <- proc.time()[3]
  ida.rel.predictions <- wrapIda(X=data,intervNodes=intervNodes,graphObject=Adj,processes=processes) # philip
  show(proc.time()[3] - tic)
  
  # DGEMatrix format.. Convert to matrix and save to hdf5.
  # print(paste('NROWS', nrow(ida.rel.predictions)))
  # print(paste('NCOLS', ncol(ida.rel.predictions)))

  #####
  # saving ida results
  #####
  print('Saving results')
  if (!h5createFile(output)) {
    unlink(output)
    h5createFile(output)
  }

  # HAVE TO LOAD IT TRANSPOSED IN PYTHON TO GET THE CORRECT SHAPE WHEN LOADING IN!
  # --> fixed by using libs/misc.load_array(..., load_from_r=True).
  h5write(ida.rel.predictions, output, 'data', level=9)
  # h5write(round(ida.rel.predictions, round.size), output, 'data', level=9)
  H5close()  
}

wrapIda <- function(X,intervNodes=1:dim(X)[2], graphObject, processes=1)
  # 2014, Jonas Peters, Joris Mooij
{
  p <- dim(X)[2]
  n <- dim(X)[1]

  # ida.rel.pred <- matrix(0,ncol=p,nrow=length(intervNodes)) # dirty fix..
  # ida.rel.pred <- matrix(NA,ncol=p,nrow=length(intervNodes))
  # covariance matrix
  ida.rel.pred <- matrix(0,nrow=length(intervNodes), ncol=p) # NEW: these.ints.only x all variables shape (i.e. cause x effect)
  mcov <- cov(X)
  
  if(processes == 1)
  {
    for(j in 1:length(intervNodes)) {
      int_node <- intervNodes[j] # index of the intervention in the covariance matrix
      cat("currently computing interventional effects from variable ", int_node, " on all others (", j, "out of", length(intervNodes), ")\n")
      idaCEs <- idaFast(x.pos=int_node,y.pos.set=(1:p)[-int_node],mcov=mcov,graphEst=graphObject)
      ida.rel.pred[j, -int_node] <- apply(X=abs(idaCEs),MARGIN=1,FUN=min)
    }
  } else
  {
    results = foreach(j=1:length(intervNodes), .inorder=TRUE) %dopar% {
      int_node <- intervNodes[j]
      cat("computing intervention effects from variable ", int_node, " on all others (", j, "out of", length(intervNodes), ")\n")
      idaCEs <- idaFast(x.pos=int_node,y.pos.set=(1:p)[-int_node],mcov=mcov,graphEst=graphObject)
      apply(X=abs(idaCEs),MARGIN=1,FUN=min)        
    }
    # combine results
    for(j in 1:length(intervNodes)) {
      int_node <- intervNodes[j]
      ida.rel.pred[j, -int_node] <- results[[j]]
    }
  }
  
  return(ida.rel.pred)
}

#############
# Wrapper code for executing by external bash/python script with arguments
# - need load and save data from disk.
# - use kwargs input= and output= to get input and output data location
# - if none are given, 1st argument is the input data location
# - rest is the keyword arguments
#############
func <- 'ComputeIDA'
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