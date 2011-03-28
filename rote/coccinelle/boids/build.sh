#!/usr/bin/env bash

spatch -cocci_file ex.cocci boids.c -o boids_new.c
gcc -Wall -o boids_orig vector.c boids.c
gcc -Wall -o boids_new vector.c boids_new.c
