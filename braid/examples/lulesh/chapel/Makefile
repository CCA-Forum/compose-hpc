luleshCHPL: luleshCHPL.dir/Makefile
	make -f $< CC='mpicc -march=native'

luleshCHPL.dir/Makefile: luleshCHPL.chpl
	chpl --devel --fast $< --savec luleshCHPL.dir --make true

clean:
	rm -rf luleshCHPL.dir luleshCHPL
