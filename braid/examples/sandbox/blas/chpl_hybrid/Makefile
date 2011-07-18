OUTPUT_FILE="runMe"
TEMP_C_DIR="temp"
NUM_LOCALES=4
ALPHA=2.0
N=30
BLOCK=3


all: clean compile run

clean:
	rm -f ${OUTPUT_FILE}*
	rm -rf ${TEMP_C_DIR}

compile: 
	chpl daxpy_support.h daxpy_hybird_block_cyclic.chpl -o ${OUTPUT_FILE} --savec ${TEMP_C_DIR}

run: 
	./${OUTPUT_FILE} -nl ${NUM_LOCALES} --numElements=${N} --blkSize=${BLOCK} --alpha=${ALPHA}


