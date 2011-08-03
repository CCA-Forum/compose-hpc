

config const f: int(32) = 1;
config const blockSize: int(64) = f * 100;
config const numElems: int(64) = 16 * blockSize;
config const maxNodes = 16;
config const numIter = 4;

writeln("#!/bin/sh");

writeln("# echo Num Elements = ", numElems);
writeln("# echo Block Size = ", blockSize);

for param t in 1 .. 5 do {
  var n = t * numElems;
  var b = t * blockSize;

  for r in [1 .. #numIter] do {
    for i in [2 .. maxNodes by 2] do {

      writeln("chpl_hybrid/runChapel/runChapel2C -nl ", i, " -v --rowSize=", n, " --blkSize=", b, 
        " &> data_hybrid", "-", i, "-", n, "-", b, "-", r, ".txt &" );

      writeln("chpl_pure/runMe -nl ", i, " -v --rowSize=", n, " --blkSize=", b, 
        " --maxThreadsPerLocale=8 --dataParTasksPerLocale=8 ",
        "&> data_pure", "-", i, "-", n, "-", b, "-", r, ".txt &" );

    }
  }
}

