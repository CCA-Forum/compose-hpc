#!/usr/bin/python

p = "pbatch"

blockSize = 512
numElems = 16 * blockSize
#maxNodes = 32
numRuns = 3
walltime = '0:07:00'

for t in range(1,200,50):
  n = t * numElems
  b = t * blockSize
  for r in range(1,numRuns+1):
      for nodes in [1,2,4,6,8,10,12,16,20,24,32,64,128]:
        fh = open('blas_hybrid-{nodes}-{n}-{b}-{r}.moab'.format(nodes=nodes,n=n,b=b,r=r), 'w')
        fh.write('''
#MSUB -A comp
#MSUB -l nodes={nodes}
#MSUB -l partition=ansel
#MSUB -l walltime={walltime}
#MSUB -q {p}
#MSUB -m a
#MSUB -N blhy{nodes}-{n}-{b}-{r}
#MSUB -V
#MSUB -o /p/lscratchb/prantl1/blas_hybrid-{nodes}-{n}-{b}-{r}.txt

export MPIRUN_CMD="srun -N %N %P %A"
chpl_hybrid/runChapel/runChapel2C -nl {nodes} -v --rowSize=32 --colSize={n} --blkSize={b} --maxThreadsPerLocale=12 --dataParTasksPerLocale=12
'''.format(nodes=nodes,n=n,b=b,r=r,p=p,walltime=walltime))
        fh.close()

        fh = open('data_pure-{nodes}-{n}-{b}-{r}.moab'.format(nodes=nodes,n=n,b=b,r=r), 'w')
        fh.write('''
#MSUB -A comp
#MSUB -l nodes={nodes}
#MSUB -l partition=ansel
#MSUB -l walltime={walltime}
#MSUB -q {p}
#MSUB -m a
#MSUB -N blpu{nodes}-{n}-{b}-{r}        
#MSUB -V
#MSUB -o /p/lscratchb/prantl1/blas_pure-{nodes}-{n}-{b}-{r}.txt

export MPIRUN_CMD="srun -N %N %P %A"
chpl_pure/runMe -nl {nodes} -v --rowSize=32 --colSize={n} --blkSize={b} --maxThreadsPerLocale=12 --dataParTasksPerLocale=12
'''.format(nodes=nodes,n=n,b=b,r=r,p=p,walltime=walltime))
        fh.close()
