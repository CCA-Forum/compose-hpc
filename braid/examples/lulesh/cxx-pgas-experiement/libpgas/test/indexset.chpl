/* var n = stdin.read(int); */
/* var dim = 0..#n; */
/* var space = [dim, dim, dim]; */
/* var idxset = space dmapped Block(space); */
/* var test: [idxset] real; */
/* test[0,1,2] = 24; */

/* proc x(a:int) {return a;} */
/* x(test); */

config const edgeElems = 2 ;// 45; 
       const edgeNodes = edgeElems + 1;

/* Declare per-dimension ranges */

const ElemDim = 0..#edgeElems,
      NodeDim = 0..#edgeNodes;


/* Declare abstract problem domains */

const ElemSpace = [ElemDim, ElemDim, ElemDim],
      NodeSpace = [NodeDim, NodeDim, NodeDim],
      NodeFace  = [NodeDim, NodeDim];


/* Declare the (potentially distributed) problem domains */
use BlockDist;
param useBlockDist = true;

const Elems = if useBlockDist then ElemSpace dmapped Block(ElemSpace)
                              else ElemSpace,
      Nodes = if useBlockDist then NodeSpace dmapped Block(NodeSpace)
                              else NodeSpace;

iter elemToNodes(i: index(Elems)) {
  //(need to make sure these are in the right order, so hard-coded)
  yield (i[1]  , i[2]  , i[3]  );
  yield (i[1]  , i[2]  , i[3]+1);
  yield (i[1]  , i[2]+1, i[3]+1);
  yield (i[1]  , i[2]+1, i[3]  );
  yield (i[1]+1, i[2]  , i[3]  );
  yield (i[1]+1, i[2]  , i[3]+1);
  yield (i[1]+1, i[2]+1, i[3]+1);
  yield (i[1]+1, i[2]+1, i[3]  );
}

iter elemToNodesTuple(i: index(Elems)) {
  for (node,num) in (elemToNodes(i), 1..8) do
    yield (node, num);
}


/* Helper functions */

inline proc localizeNeighborNodes(eli: index(Elems),
                                  x: [] real, inout x_local: 8*real,
                                  y: [] real, inout y_local: 8*real,
                                  z: [] real, inout z_local: 8*real) {

  for (noi, t) in elemToNodesTuple(eli) {
    x_local[t] = x[noi];
    y_local[t] = y[noi];
    z_local[t] = z[noi];
  }
}


for eli in Elems {
  writeln("eli = ", eli);
  for (noi, t) in elemToNodesTuple(eli) {
    writeln("noi = ", noi, ", t = ", t);
  }
}