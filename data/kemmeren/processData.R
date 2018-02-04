################################################################################
# Summary: Pre-process gene-expression data from Kemmeren et al. (2014)
# Authors: Alain Hauser (original scripts), Philip Versteeg (Adapted)
# Input:
#   Default location of original data files:
#     org_data/limma_MARG_non-responsive_mutants.txt  
#     org_data/limma_MARG_responsive_mutants.txt
#     org_data/limma_MARG_wt_pool.txt
#     org_data/limma_MARG_wt_pool_platereader.txt
#   SGD naming file:
#     sgd_naming.json
# Output: (choose one)
#   Kemmeren.RData
#   Kemmeren.hdf5
################################################################################

# Settings
folder.org.data <- 'org_data'
file.out.type <- c('hdf5', 'RData')[1] # Choose one
file.out <- paste('kemmeren',file.out.type, sep='.')
file.sgd.naming <- 'sgd_naming.json'

ReadOrigFile <- function(file, data.types = c('M','extprot')) {
  # Read in one data file from Kemmeren (2014). 
  #
  # Args:
  #   file: Text file name to read in.
  #   data.types: Single variable or vector containing the requested 
  #     datatypes to read in from files in character format. Subset of 
  #     {'M', 'A', 'G', 'R', 'p_value', 'extprot'}.
  #
  # Returns:
  #   List of requested processed data, default = {'M', 'extprot'}

  # Read first 5 lines, which are header lines
  n.head <- 5
  data.head <- matrix(scan(file, what = character(), sep = '\t', nlines = n.head), nrow = n.head, byrow = TRUE)
  # [1] systematic name; [2] data type; [3] dye swap; [4] extraction protocol; [5] sample name;

  # First column are rownames
  rownames(data.head) <- data.head[,1]
  data.head <- data.head[,2:ncol(data.head)]

  # Get data types, each index and column (experiment) naming from header
  data.types.avail <- unique(data.head[2,])
  index.type <- list()
  for (i in data.types.avail) {
    index.type[[i]] <- which(data.head[2,] == i)
  }
  col.names <- unlist(sub(' vs. wt','-vs-wt',data.head[1,]))

  #    extraction protocol from header
  if (all(grepl('yeast HTP RNA isolation for robot v2.0',data.head[4,]) == TRUE)) {
    print('Extraction protocol: platereader') 
  } else if (all(grepl('yeast HTP RNA isolation for robot v2.0',data.head[4,]) == FALSE)) {
    print('Extraction protocol: erlenmeyer')
  } else {
    print('Extraction protocol: mixed')
  }

  # Read the bulk and scratch the first 5 header lines
  data.raw <- read.table(file, header = FALSE, sep = '\t', quote = '', stringsAsFactors = FALSE, skip = n.head)
    # colClasses = c(rep('character', 3), rep('numeric', ncol(first.lines) - 3)))
  stopifnot(sum(is.na(data.raw[,2:ncol(data.raw)])) == 0)

  # Check if gene names are unique and otherwise remove duplicates
  if (length(data.raw[,1]) != length(unique(data.raw[,1]))) {
    data.raw <- data.raw[-which(duplicated(data.raw[,1])),]
  } 
  stopifnot(length(data.raw[,1]) == length(unique(data.raw[,1])))

  # Process first column as row names, conver rest to matrix (more efficient later on)
  rownames(data.raw) <- data.raw[,1]
  data.raw <- data.matrix(data.raw[,2:ncol(data.raw)])
  colnames(data.raw) = col.names

  # Prepare data result, one matrix per data type
  data.result <- list()
  for (i in data.types) {
    # M, A, R, G, p_value
    if (i %in% data.types.avail) {
      # print(colnames(data.raw))
      # print(paste('-',i,sep = ''))
      data.result[[i]] <- data.raw[,index.type[[i]]]
      print(str(data.result[[i]]))
      # colnames(data.result[[i]]) <- 
      # which(grepl(paste('-',i),col.names(data.raw)]))
    # Extraction protocol
    } else if (i == 'logR') {   # from Alain, but is logR needed?
      if ('R' %in% data.types.avail) {
        data.result$logR <- log2(data.raw[,index.type[['R']]])
      } else if (all(c('A', 'M') %in% data.types.avail)) {
        data.mat$logR <- data.raw[,index.type[['A']]] + 0.5*data.raw[,index.type[['M']]]
      } else {
        stop('logR values cannot be calculated: R or A or M values missing.')
      }
    } else if (i == 'extprot') {
      data.result$extprot <- grepl('*yeast HTP RNA isolation for robot v2.0*',data.head[4,index.type[['M']]]) # Shouldn't matter
    # Error
    } else {
      print(paste(i,'not available!',sep = ' '))
    }
  }
  return(data.result)
}

### Read in all data ###
variable <- c('M','extprot') # Only need 'M' and 'extprot' variables. Possibly more in future.
mutants.nonprof <- ReadOrigFile(paste(folder.org.data, 'limma_MARG_non-responsive_mutants.txt', sep='/'),variable)
mutants.prof <- ReadOrigFile(paste(folder.org.data, 'limma_MARG_responsive_mutants.txt', sep='/'),variable)
wildtype.plate <- ReadOrigFile(paste(folder.org.data, 'limma_MARG_wt_pool_platereader.txt', sep='/'),variable)
wildtype.erlen <- ReadOrigFile(paste(folder.org.data, 'limma_MARG_wt_pool.txt', sep='/'),variable)
# Note that there is one plate observation in the txt-file for erlenmeyers. 

# Combine wildtype
wildtype.all <- list()
wildtype.all$M <- cbind(wildtype.erlen$M, wildtype.plate$M)
wildtype.all$extprot <- c(wildtype.erlen$extprot, wildtype.plate$extprot)
names(wildtype.all$extprot) <- colnames(wildtype.all$M)

# Combine mutants
mutants.all <- list()
mutants.all$M <- cbind(mutants.prof$M, mutants.nonprof$M)
mutants.all$extprot <- c(mutants.prof$extprot, mutants.nonprof$extprot)

### Match gene names to SGD second identifiers.
# sgd.genes.aliasses: names(sgd.genes.aliasses) = all aliasses |--> sgd.genes.aliasses 
library(rjson)
sgd.genes.aliasses <- fromJSON(file = file.sgd.naming)
stopifnot(all(!duplicated(names(sgd.genes.aliasses))))  # Check: map is onto

genes.names <- rownames(mutants.all$M)                  # Check: all rows have equal same unique gene naming
stopifnot(all(toupper(genes.names) == toupper(rownames(wildtype.plate$M))))
stopifnot(all(toupper(genes.names) == toupper(rownames(wildtype.erlen$M))))
genes.names <- sgd.genes.aliasses[toupper(genes.names)]
stopifnot(all(!duplicated(genes.names)))                # Check: unique naming
rownames(mutants.all$M) <- genes.names
rownames(wildtype.all$M) <- genes.names

# Match mutant interventions
mutants.names <- toupper(unlist(sub('-del-.*$','',colnames(mutants.all$M))))
matched.names.index <- charmatch(mutants.names,toupper(names(sgd.genes.aliasses)))
stopifnot(all(!is.na(matched.names.index)))             # Check: all matched
mutants.names <- toupper(sgd.genes.aliasses[mutants.names])
colnames(mutants.all$M) <- mutants.names
names(mutants.all$extprot) <- mutants.names

# Delete mutants not in wildtype observations
mutants.delete.log <- mutants.names %in% genes.names == FALSE
mutants.all$M <- mutants.all$M[,!(mutants.delete.log)]
mutants.all$extprot <- mutants.all$extprot[!(mutants.delete.log)]
print(paste('Deleted nonobserved mutant:',mutants.names[mutants.delete.log], sep = ' '))

# Order rows by gene names for both wt and mutant
wildtype.all$M <- wildtype.all$M[order(rownames(wildtype.all$M)),]
mutants.all$M <- mutants.all$M[order(rownames(mutants.all$M)),]

# Order columns by extraction protocol for wt
# wildtype.all$M <- wildtype.all$M[,order(wildtype.all$extprot)]
# wildtype.all$extprot <- wildtype.all$extprot[order(wildtype.all$extprot)]

# Order mutant columns by naming
colorder <- order(colnames(mutants.all$M))
mutants.all$M <- mutants.all$M[,colorder]
mutants.all$extprot <- mutants.all$extprot[colorder]

# colorder <- order(mutants.all$extprot)
# mutants.all$M <- mutants.all$M[,colorder]
# mutants.all$extprot <- mutants.all$extprot[colorder]

# Final sorted list
genes.names <- rownames(wildtype.all$M)                 # same as sort(genes.names)
mutants.names <- colnames(mutants.all$M)
intpos <- sapply(colnames(mutants.all$M),function(x) which(rownames(wildtype.all$M) == x))

### Write to RData ###
# Unfortunately some matrices are transposed w.r.t. legacy format..
if (file.out.type == 'RData') {
  # Write to new format
  data <- list(
    obs = t(wildtype.all$M),
    int = t(mutants.all$M), 
    plateobs = wildtype.all$extprot,
    plateint = mutants.all$extprot,
    intpos = intpos,
    p = nrow(wildtype.all$M),
    nObs = ncol(wildtype.all$M),
    nInt = ncol(mutants.all$M),
    genenames = rownames(wildtype.all$M),
    mutantnames = colnames(mutants.all$M) 
  )
  save(data, file=file.out)

  ## Write to legacy format
  # data <- list(
  #   obs = t(wildtype.all$M),
  #   int = t(mutants.all$M), 
  #   intpos = intpos,
  #   p = nrow(wildtype.all$M),
  #   nObs = ncol(wildtype.all$M),
  #   nInt = ncol(mutants.all$M),
  #   genenames = rownames(wildtype.all$M)
  # )
  # save(data, file='KemmerenOld.RData')
}

### Write to HDF5 
if (file.out.type == 'hdf5') {
  library(rhdf5)
  H5close()
  if (!h5createFile(file=file.out)) {
    unlink(file.out)
    h5createFile(file=file.out)
  }
  h5createGroup(file=file.out, group='inter')
  h5createGroup(file=file.out, group='obser')
  h5createGroup(file=file.out, group='ids')
  h5write(t(wildtype.all$M), file.out, 'obser/data')
  h5write(c(wildtype.all$extprot), file.out, 'obser/extplate')
  h5write(t(mutants.all$M), file.out, 'inter/data')
  h5write(c(mutants.all$extprot), file.out, 'inter/extplate')
  h5write(rownames(wildtype.all$M), file.out, 'ids/genes')
  h5write(colnames(mutants.all$M), file.out, 'ids/mutants')
  H5close()  
}