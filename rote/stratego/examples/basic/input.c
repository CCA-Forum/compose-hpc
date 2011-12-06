#include <stdio.h>

int main() {
    int w,x,y,z; 
    w = 1;
    x = 3;
    y = 5;
    z = w * (x + y);
    // rewrite to:
    // z = w * x + w * y;
    printf("z=%d\n",z);
    return 0;
}
