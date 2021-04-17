#include "kernel/ocl/common.cl"


__kernel void life_ocl (__global unsigned *in, __global unsigned *out)
{
    unsigned x = get_global_id (0);
    unsigned y = get_global_id (1);
    __local unsigned n;

    if (x > 0 && x < DIM - 1 && y > 0 && y < DIM - 1){
        n = (  in[(y-1)*DIM + x-1]  + in[(y-1)*DIM + x] + in[(y-1)*DIM + x + 1] +
               in[y*DIM + x -1]     + in[y*DIM + x]     + in[y*DIM + x + 1 ]    +
               in[(y+1)*DIM + x-1]  + in[(y+1)*DIM + x] + in[(y+1)*DIM + x + 1]);

        n = (n == 3 + in[y*DIM + x]) | (n == 3);
        out[y*DIM + x] = n;
    }
}

__kernel void life_ocl_finish (__global unsigned *in, __global unsigned *out, __global unsigned *change){
    unsigned x = get_global_id (0);
    unsigned y = get_global_id (1);
    __local unsigned n;

    if (x > 0 && x < DIM - 1 && y > 0 && y < DIM - 1){
        n = (  in[(y-1)*DIM + x-1]  + in[(y-1)*DIM + x] + in[(y-1)*DIM + x + 1] +
               in[y*DIM + x -1]    + in[y*DIM + x]      + in[y*DIM + x + 1 ]    +
               in[(y+1)*DIM + x-1] + in[(y+1)*DIM + x]  + in[(y+1)*DIM + x + 1]);

        n = (n == 3 + in[y*DIM + x]) | (n == 3);

        out[y*DIM + x] = n;

        if(out[y*DIM + x] != in[y*DIM + x]){
            change[0]=1;
        }
    }
}