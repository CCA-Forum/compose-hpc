#include <stdio.h>
#include <stdlib.h>
#include <math.h>

/*****************************************************************************
  Vector stuff
*****************************************************************************/

typedef struct {
  double x;
  double y;
} Vec;

double vec_length(Vec a) {
  return sqrt(a.x * a.x + a.y * a.y);
}

Vec vec_scale(Vec a, double scale) {
  Vec b;
  b.x = a.x * scale;
  b.y = a.y * scale;
  return b;
}

Vec vec_normalize(Vec a) {
  return vec_scale(a,1.0/vec_length(a));
}

Vec vec_add(Vec a, Vec b) {
  Vec c;
  c.x = a.x + b.x;
  c.y = a.y + b.y;
  return c;
}

Vec vec_sub(Vec a, Vec b) {
  Vec c;
  c.x = a.x - b.x;
  c.y = a.y - b.y;
  return c;
}

Vec vec_limit(Vec a, double limit) {
  if(vec_length(a) > limit) {
    return vec_scale(vec_normalize(a),limit);
  } else {
    return a;
  }
}

// Biased but it doesn't matter here
Vec vec_rand() {
  Vec a;
  a.x = (double)(rand() - RAND_MAX / 2);
  a.y = (double)(rand() - RAND_MAX / 2);
  return vec_normalize(a);
}


/*****************************************************************************
  Boid stuff
*****************************************************************************/

struct boid {
  Vec pos;
  Vec vel;
  Vec delta;
};

void print_boid(struct boid *boid) {
  printf("Boid ");
  printf("(%0.2f,%0.2f):",boid->pos.x,boid->pos.y);
  printf("(%0.2f,%0.2f)\n",boid->vel.x,boid->vel.y);
}

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

Vec cohesion(struct boid *b, struct boid *boids, int n, double a) {
  Vec c = pos_centroid(boids,n);
  Vec delta = vec_sub(c,b->pos);
  return vec_scale(delta,a);
}

Vec alignment(struct boid *b, struct boid *boids, int n, double a) {
  Vec c = vel_centroid(boids,n);
  Vec delta = vec_sub(c,b->vel);
  return vec_scale(delta,a);
}

Vec separation(struct boid *b, struct boid *boids, int n, double a) {
  Vec v;
  v.x = v.y = 0;
  int i;
  for(i=0; i < n; i++) {
    Vec dist = vec_sub(b->pos,boids[i].pos);
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
  struct boid *b;
  int i;
  for(i=0; i < n; i++) {
    b = &boids[i];
    b->pos = vec_rand();
    b->vel = vec_rand();
  }
}

void step(struct boid *boids, int n) {
  struct boid *b;
  int i;
  for(i=0; i < n; i++) {
    b = &boids[i];
    Vec c = cohesion(b,boids,n,0.0075);
    Vec s = separation(b,boids,n,0.1);
    Vec a = alignment(b,boids,n,1.0/1.8);
    b->delta = vec_scale(vec_add(c,vec_add(s,a)),0.1);
  }
  for(i=0; i < n; i++) {
    b = &boids[i];
    b->pos = vec_add(b->vel,b->pos);
    b->vel = vec_limit(vec_add(b->vel,b->delta),100.0);
  }
}

void output(struct boid *boids, int n) {
  struct boid *b;
  int i;
  for(i=0; i < n; i++) {
    b = &boids[i];
    print_boid(b);
  }
}


/*****************************************************************************
  Main
*****************************************************************************/

int main() {
  int n = 100,i;
  struct boid *boids;
  srand(10231977L);
  boids = (struct boid *)malloc(n * sizeof(struct boid));
  init(boids,n);
  for(i=0;i < 100;i++) {
    step(boids,n);
  }
  output(boids,n);
  free(boids);
  return 0;
}
