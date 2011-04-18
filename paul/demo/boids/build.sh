#!/usr/bin/env bash

spatch -cocci_file reference.cocci boids.c -o boids_new.c
gcc -Wall -o boids_orig vector.c boids.c
gcc -Wall -o boids_new vector.c boids_new.c
