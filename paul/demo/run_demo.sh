#!/usr/bin/env bash

echo "Running PAUL"
./paul boids/boids.c > boids.cocci

echo "Transforming boids.c"
cd boids
spatch -cocci_file ../boids.cocci boids.c -o boids_new.c > /dev/null

echo "Building both versions of boids"
gcc -Wall -o boids_orig vector.c boids.c -lm
gcc -Wall -o boids_new vector.c boids_new.c -lm

echo "Comparing versions"
./boids_orig > orig.out
./boids_new > new.out
diff orig.out new.out

echo "Cleanup"
rm -f orig.out new.out
rm -f boids_orig boids_new boids_new.c
cd ..
rm -f boids.cocci

echo "If you don't see any errors... probably success!"
