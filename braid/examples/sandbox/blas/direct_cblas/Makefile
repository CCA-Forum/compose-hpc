
# gcc daxpy_main.c -I/System/Library/Frameworks/Accelerate.framework/Versions/Current/Frameworks/vecLib.framework/Versions/Current/Headers -lcblas


BLAS_LIB="cblas"
BLAS_INCLUDE_DIR="/System/Library/Frameworks/Accelerate.framework/Versions/Current/Frameworks/vecLib.framework/Versions/Current/Headers"
OUTPUT_FILE="runMe"

all: clean compile run

clean:
	rm -f ${OUTPUT_FILE}

compile: 
	gcc daxpy_main.c -I${BLAS_INCLUDE_DIR} -l${BLAS_LIB} -o ${OUTPUT_FILE}

run: 
	./${OUTPUT_FILE}


