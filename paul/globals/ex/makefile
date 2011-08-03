
SP=/home/baallan/cokerc3/bin/spatch

DEBUG=-debug
DEBUG=

#Usage: spatch -sp_file <SP> <infile> [-o <outfile>] [-iso_file <iso>] [options]
arg:
	$(RM) stage2.cocci data.pkl
	$(SP) $(DEBUG) -sp_file argshift1 stub.c -o c_stub.c
	$(SP) $(DEBUG) -sp_file argshift1 stub.h -o c_stub.h
	python gen_stage2.py
	$(SP) $(DEBUG) -sp_file stage2 c_stub.c -o d_stub.c
	$(SP) $(DEBUG) -sp_file stage2 c_stub.h -o d_stub.h
	gcc -c d_stub.c
