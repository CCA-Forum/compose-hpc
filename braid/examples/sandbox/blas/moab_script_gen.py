#!/usr/bin/python

f = 1
blockSize = f * 100
numElems = 16 * blockSize
maxNodes = 16
numIter = 4

writeln("# echo Num Elements = ", numElems);
writeln("# echo Block Size = ", blockSize);

for t in range(1,5):
  n = t * numElems
  b = t * blockSize

  for r in range(1,numIter+1):
    for i in range(2, maxNodes, 2):
        fh = open('data_hybrid-{i}-{n}-{b}-{r}.moab'.format(i=i,n=n,b=b,r=r), 'w')
        fh.write('''
#MSUB -A comp
#MSUB -l nodes=4
#MSUB -l partition=ansel
#MSUB -l walltime=0:05:00
#MSUB -q pdebug
#MSUB -m be
#MSUB -N ptrans
#MSUB -V
#MSUB -o /p/lscratchb/prantl1/data_hybrid-{i}-{n}-{b}-{r}.txt

export MPIRUN_CMD="srun -N %%N %%P %%A"
chpl_hybrid/runChapel/runChapel2C -nl {i} -v --rowSize={n} --blkSize={b}
'''.format(i=i,n=n,b=b,r=r))
        fh.close()

        fh = open('data_pure-{i}-{n}-{b}-{r}.moab'.format(i=i,n=n,b=b,r=r), 'w')
        fh.write('''
#MSUB -A comp
#MSUB -l nodes=4
#MSUB -l partition=ansel
#MSUB -l walltime=0:05:00
#MSUB -q pdebug
#MSUB -m be
#MSUB -N ptrans
#MSUB -V
#MSUB -o /p/lscratchb/prantl1/data_pure-{i}-{n}-{b}-{r}.txt

export MPIRUN_CMD="srun -N %%N %%P %%A"
chpl_pure/runMe -nl {i} -v --rowSize={n} --blkSize={b} --maxThreadsPerLocale=12 --dataParTasksPerLocale=12
'''.format(i=i,n=n,b=b,r=r))
        fh.close()
