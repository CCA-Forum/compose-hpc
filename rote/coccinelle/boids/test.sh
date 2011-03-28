#!/usr/bin/env bash

./boids_orig > orig.out
./boids_new > new.out

# Diff the outputs of the original and transformed programs. They should be
# the same.
diff orig.out new.out

rm orig.out new.out
