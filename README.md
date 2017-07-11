# CL-SciSumm2017

an example of passed parameters is

"" "PreProcessPipeline" "D:\Research\UPF\Projects\CL-SciSumm2017" "ALL_training_train" "GS" "DI" "BN"

Param 0 - always empty

Param 1 - Processing pipeline to use

Param 2 - Working directory which contains the property file of the configuration file of Dr Inventor as well as a folder nammed datasets which will have (training, development and testing folders)

Param 3 - three parts part1_part2_part3

  part1 - What to process - either a Cluter "ACLID" to process only that cluster or "ALL" to process all clusters.
  
  part2 - the target dataset
  
  part3 - train to indecate training processing otherwise it will be treated as testing
  
rest of the params are basically arguments to control the flow of the processing. In the example here:

GS : Annotate Gold Standard
DI : Annotate Dr Inventor
BN : Annotate Babelnet Synset IDs
CV : context vectors
GZ : Gazetteers
NG : N-grams
WE : for word embeddings

