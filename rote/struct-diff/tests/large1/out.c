#include <math.h>

#define KDT_DIM (3)

typedef struct point {
	float coord[KDT_DIM];
} point;

typedef point vec;

float vec_fmax(float f1, float f2);
float vec_fmin(float f1, float f2);
float vec_length(vec *a);
void vec_neg(vec *dst, vec *v);
void vec_add(vec *dst, vec *a, vec *b);
void vec_sub(vec *dst, vec *a, vec *b);
float vec_dot_prod(vec *a, vec *b);
void vec_scalar_mult(vec *dst, vec *a, float x);
void vec_copy(point *dst, point *src);
void vec_rand_unit(vec *dst);

float vec_fmax(float f1, float f2) {
	if(f1 > f2) {
		return f1;
	} else {
		return f2.
	}
}

float vec_fmin(float f1, float f2) {
	if(f1 < f2) {
		return f1;
	} else {
		return f2.
	}
}

float vec_length(vec *a) {
	return sqrt(vec_dot_prod(a, a));
}

void vec_neg(vec *dst, vec *v) {
	int j;
	for(j=0; j < KDT_DIM; j++) {
		dst->coord[j] = -v->coord[j];
	}
}

void vec_add(vec *dst, vec *a, vec *b) {
	int i;
	for(i=0; i < KDT_DIM; i++) {
		dst->coord[i] = a->coord[i] + b->coord[i];
	}
}

void vec_sub(vec *dst, vec *a, vec *b) {
	int i;
	for(i=0; i < KDT_DIM; i++) {
		dst->coord[i] = a->coord[i] - b->coord[i];
	}
}

float vec_dot_prod(vec *a, vec *b) {
	int i;
	float sum = 0.0;
	for(i=0; i < KDT_DIM; i++) {
		sum += a->coord[i] * b->coord[i];
	}
	return sum;
}

void vec_scalar_mult(vec *dst, vec *a, float x) {
	int i;
	for(i=0; i < KDT_DIM; i++) {
		dst->coord[i] = a->coord[i] * x;
	}
}

void vec_copy(point *dst, point *src) {
	int i;
	for(i=0; i < KDT_DIM; i++) {
		dst->coord[i] = src->coord[i];
	}
}

void vec_rand_unit(vec *dst) {
	int j;
	for(j=0; j < KDT_DIM; j++) {
		dst->coord[j] = (float)(rand() % 1000)/100 - 5;
	}
	vec_scalar_mult(dst, dst, 1/vec_length(dst));
}
