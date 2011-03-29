#undef __BLOCKS__ // This is required for Mac OS X's stupid mangled stdlib.h
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "vector.h"

/*****************************************************************************
  Boid stuff
*****************************************************************************/

/*%%% ABSORB_STRUCT_ARRAY */
struct boid {
  Vec pos;
  Vec vel;
  Vec delta;
};

Vec pos_centroid(struct boid *boids, int n) {
  Vec v;
  int i;
  v.x = v.y = 0.0;
  for(i=0; i < n; i++) {
    v.x += boids[i].pos.x;
    v.y += boids[i].pos.y;
  }
  return vec_scale(v,1.0/n);
}

Vec vel_centroid(struct boid *boids, int n) {
  Vec v;
  int i;
  v.x = v.y = 0.0;
  for(i=0; i < n; i++) {
    v.x += boids[i].vel.x;
    v.y += boids[i].vel.y;
  }
  return vec_scale(v,1.0/n);
}

Vec cohesion(int b, struct boid *boids, int n, double a) {
  Vec c = pos_centroid(boids,n);
  Vec delta = vec_sub(c,boids[b].pos);
  return vec_scale(delta,a);
}

Vec alignment(int b, struct boid *boids, int n, double a) {
  Vec c = vel_centroid(boids,n);
  Vec delta = vec_sub(c,boids[b].vel);
  return vec_scale(delta,a);
}

Vec separation(int b, struct boid *boids, int n, double a) {
  Vec v;
  v.x = v.y = 0;
  int i;
  for(i=0; i < n; i++) {
    Vec dist = vec_sub(boids[b].pos,boids[i].pos);
    if(vec_length(dist) < a) {
      v.x -= boids[i].pos.x;
      v.y -= boids[i].pos.y;
    }
  }
  return vec_scale(v,1.25);
}


/*****************************************************************************
  Simulation
*****************************************************************************/

void init(struct boid *boids, int n) {
  int i;
  for(i=0; i < n; i++) {
    boids[i].pos = vec_rand();
    boids[i].vel = vec_rand();
  }
}

void step(struct boid *boids, int n) {
  int i;
  for(i=0; i < n; i++) {
    Vec c = cohesion(i,boids,n,0.0075);
    Vec s = separation(i,boids,n,0.1);
    Vec a = alignment(i,boids,n,1.0/1.8);
    boids[i].delta = vec_scale(vec_add(c,vec_add(s,a)),0.1);
  }
  for(i=0; i < n; i++) {
    boids[i].pos = vec_add(boids[i].vel,boids[i].pos);
    boids[i].vel = vec_limit(vec_add(boids[i].vel,boids[i].delta),100.0);
  }
}

void output(struct boid *boids, int n) {
  int i;
  for(i=0; i < n; i++) {
    printf("Boid ");
    printf("(%0.2f,%0.2f):",boids[i].pos.x,boids[i].pos.y);
    printf("(%0.2f,%0.2f)\n",boids[i].vel.x,boids[i].vel.y);
  }
}


/*****************************************************************************
  Main
*****************************************************************************/

int main() {
  long seed = 10231977L;
  int n = 100,i;
  struct boid *boids;
  printf("Using seed: %ld\n", seed);
  srand(seed);
  boids = malloc(n * sizeof(struct boid));
  init(boids,n);
  for(i=0;i < 100;i++) {
    step(boids,n);
  }
  output(boids,n);
  free(boids);
  return 0;
}
