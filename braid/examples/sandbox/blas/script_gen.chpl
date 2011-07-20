

config const f: int(32) = 1;
config const b: int(64) = f * 10000;
config const n: int(64) = 16 * b;

writeln("#!/bin/sh");

writeln("echo Num Elements = ", n);
writeln("echo Block Size = ", b);

for i in [2 .. 12 by 2] do {

  writeln("chpl_hybrid/runChapel/runChapel2C -nl ", i, " -v --numElements=", n, " --blkSize=", b, " &> data_hybrid-", i, "-", n, "-", b, ".txt &" );

  writeln("chpl_pure/runMe -nl ", i, " -v --numElements=", n, " --blkSize=", b, " --maxThreadsPerLocale=8 --dataParTasksPerLocale=8 &> data_pure-", i, "-", n, "-", b, ".txt &" );

}


