OUTPUT_FILE="runMe"
NUM_LOCALES=4
ALPHA=2.0
N=30
BLOCK=4


all: clean compile run

clean:
	rm -f ${OUTPUT_FILE}*

compile: 
	chpl daxpy_block_cyclic.chpl -o ${OUTPUT_FILE}

run: 
	./${OUTPUT_FILE} -nl ${NUM_LOCALES} --numElements=${N} --blkSize=${BLOCK} --alpha=${ALPHA}


