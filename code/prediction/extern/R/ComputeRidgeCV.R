##########
# Author: Philip Versteeg (2016)
# Email: pjjpversteeg@gmail.com
# Compute all ridge regressions for data X_i^n with regressor X_{i\j} and regressand X_j.
# input:
#		input 				hdf5 file with 'data' element as n X p data (dataframe / matrix)
#		lambda 				grid for parameter search (lambda.low, lambda.high, lambda.step)
#		fix.lambda.after 	option to fix lambda after set amount of regressions (bool)
# return:
#		-					write to output hdf5 file 'data' 
##########
ComputeRidgeCV <- function(
	input='__test_input.hdf5',
	input.dataset='data', 
	output='output.hdf5', 
	lambda.low=1E0, 
	lambda.high=1E3, 
	lambda.step='log', 
	fix.lambda.after=-1,
	round.size=7,
	verbose=TRUE
) {
	library(parcor)
	library(MASS)
	library(rhdf5)

	# read data from disk
	data <- h5read(input, input.dataset)
	# data <- h5read(input, 'data')
	# data <- h5read(input, '/obser/data')

	# lambdas <- c(1:9 %o% 10^(0:4))
	if (lambda.step == 'log'){ 
		# log scale 
		lambdas <- c(((1:99)/100) %o% 10^(log10(lambda.low):log10(lambda.high)))
	}
	if (is.numeric(lambda.step)) {
		lambdas <- seq(from=lambda.low,to=lambda.high,by=lambda.step)
	}

	p <- ncol(data)
	colnames(data) <- 1:p # Need for lazy indices later...

	result <- matrix(data=0, nrow=p, ncol=p)

	# if fix lambda after > eta > 0, after eta iterations take the mean of lambda as cv setting.
	if (fix.lambda.after > 0) {
		tmp <- c()
		for (i in 1:fix.lambda.after){
			ridgereg <- ridge.cv(X=data[,-i],y=data[,i],lambda=lambdas,scale=TRUE,plot.it=FALSE)
			if (verbose){cat("var:", i,"lambda:", ridgereg$lambda.opt, "\n")}
			tmp <- c(tmp, ridgereg$lambda.opt)
		}
		lambda <- mean(tmp)
		if (verbose) {cat("Fixed lambda =", lambda, "after CV on", fix.lambda.after, "vars\n")}

		# run regression with found lambda TODO: fix..
		for (i in 1:p){
			X <- data[,-i]
			ridgereg <- lm.ridge(data[,i] ~ X,lambda=lambda,scale=TRUE)
			if (verbose){cat("var:", i, "\n")}
			id.cause <- strtoi(sub("X","",names(ridgereg$coef))) # j index
			for (j in 1:length(id.cause)) {
				result[id.cause[j],i] <- ridgereg$coef[j]
			}
		}
	}

	# run full CV
	else {
		for (i in 1:p){
			ridgereg <- ridge.cv(X=data[,-i],y=data[,i],lambda=lambdas,scale=TRUE,plot.it=FALSE)
			if (verbose) {cat("var:", i,"lambda:", ridgereg$lambda.opt, "\n")}
			id.cause <- strtoi(sub("X","",names(ridgereg$coefficients))) # j index
			for (j in 1:length(id.cause)) {
				result[id.cause[j],i] <- ridgereg$coefficients[j]
			}
		}
	}

	print('Saving results')
	if (!h5createFile(output)) {
	  unlink(output)
	  h5createFile(output)
	}
	h5write(round(result, round.size), output, 'data', level=9)
	H5close()
}

#############
# Wrapper code for executing by external bash/python script with arguments
# - need load and save data from disk.
# - use kwargs input= and output= to get input and output data location
# - if none are given, 1st argument is the input data location
# - rest is the keyword arguments
#############
func <- 'ComputeRidgeCV'
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