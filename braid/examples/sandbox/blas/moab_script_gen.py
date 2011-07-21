#!/usr/bin/python

p = "pbatch"

blockSize = 100
numElems = 16 * blockSize
maxNodes = 16
numRuns = 3

for t in range(1,100,50):
  n = t * numElems
  b = t * blockSize

  for r in range(1,numRuns+1):
    for i in range(2, maxNodes, 2):
        fh = open('data_hybrid-{i}-{n}-{b}-{r}.moab'.format(i=i,n=n,b=b,r=r), 'w')
        fh.write('''
#MSUB -A comp
#MSUB -l nodes={i}
#MSUB -l partition=ansel
#MSUB -l walltime=0:30:00
#MSUB -q {p}
#MSUB -m a
#MSUB -N hy_{i}-{n}-{b}-{r}
#MSUB -V
#MSUB -o /p/lscratchb/prantl1/data_hybrid-{i}-{n}-{b}-{r}.txt

export MPIRUN_CMD="srun -N %N %P %A"
chpl_hybrid/runChapel/runChapel2C -nl {i} -v --rowSize=32 --colSize={n} --blkSize={b} --maxThreadsPerLocale=12 --dataParTasksPerLocale=12
'''.format(i=i,n=n,b=b,r=r,p=p))
        fh.close()

        fh = open('data_pure-{i}-{n}-{b}-{r}.moab'.format(i=i,n=n,b=b,r=r), 'w')
        fh.write('''
#MSUB -A comp
#MSUB -l nodes={i}
#MSUB -l partition=ansel
#MSUB -l walltime=0:30:00
#MSUB -q {p}
#MSUB -m a
#MSUB -N pu_{i}-{n}-{b}-{r}        
#MSUB -V
#MSUB -o /p/lscratchb/prantl1/data_pure-{i}-{n}-{b}-{r}.txt

export MPIRUN_CMD="srun -N %N %P %A"
chpl_pure/runMe -nl {i} -v --rowSize=32 --colSize={n} --blkSize={b} --maxThreadsPerLocale=12 --dataParTasksPerLocale=12
'''.format(i=i,n=n,b=b,r=r,p=p))
        fh.close()
