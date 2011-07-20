OUTPUT_FILE="runMe"
NUM_LOCALES=4
ALPHA=2.0
N=10000
BLOCK=100


all: clean compile run

clean:
	rm -f ${OUTPUT_FILE}*

compile: 
	chpl --fast daxpy_block_cyclic.chpl -o ${OUTPUT_FILE} --print-commands

run: 
	./${OUTPUT_FILE} -nl ${NUM_LOCALES} --numElements=${N} --blkSize=${BLOCK} --alpha=${ALPHA}


