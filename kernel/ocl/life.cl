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
	__local unsigned me;
	me = in[y*DIM + x];

	if (x > 0 && x < DIM - 1 && y > 0 && y < DIM - 1){
		n = (  in[(y-1)*DIM + x-1]  + in[(y-1)*DIM + x] + in[(y-1)*DIM + x + 1] +
		       in[y*DIM + x -1]    + in[y*DIM + x]      + in[y*DIM + x + 1 ]    +
		       in[(y+1)*DIM + x-1] + in[(y+1)*DIM + x]  + in[(y+1)*DIM + x + 1]);
		n = (n == 3 + in[y*DIM + x]) | (n == 3);

		out[y*DIM + x] = n;
		if(n != me){
			change[0] = 1;
		}
	}
}

//__kernel void life_ocl_hybrid (__global unsigned *in, __global unsigned *out, unsigned offset){
//	unsigned x = get_global_id (0);
//	unsigned y = get_global_id (1);
//	__local unsigned n;
//	__local unsigned me;
//	me = in[y*DIM + x];
//
//	if(y < offset){
//		out[y*DIM + x] = in[y*DIM + x];
//	}
//
//	if (x > 0 && x < DIM - 1 && y >= 0 && y+offset < DIM - 1){
//		n = (  in[(y+offset-1)*DIM + x-1]  + in[(y+offset-1)*DIM + x] + in[(y+offset-1)*DIM + x + 1] +
//		       in[(y+offset)*DIM   + x-1]  + in[(y+offset)*DIM   + x] + in[(y+offset)*DIM   + x + 1] +
//		       in[(y+1+offset)*DIM + x-1]  + in[(y+1+offset)*DIM + x] + in[(y+1+offset)*DIM + x + 1]);
//
//		n = (n == 3 + in[(y+offset)*DIM + x]) | (n == 3);
//		out[(y+offset)*DIM + x] = n;
//	}
//}

__kernel void life_ocl_hybrid (__global unsigned *in, __global unsigned *out,__global unsigned *frontier, unsigned offset){
    unsigned x = get_global_id (0);
    unsigned y = get_global_id (1) + offset;
    __local unsigned n;

    if (x > 0 && x < DIM - 1 && y > 0 && y < DIM - 1){
        n = (  in[(y-1)*DIM + x-1] + in[(y-1)*DIM + x] + in[(y-1)*DIM + x + 1] +
               in[y    *DIM + x-1] + in[y    *DIM + x] + in[y    *DIM + x + 1] +
               in[(y+1)*DIM + x-1] + in[(y+1)*DIM + x] + in[(y+1)*DIM + x + 1]);
        n = (n == 3 + in[y*DIM + x]) | (n == 3);

        out[y*DIM + x] = n;
        if(y == offset)
            frontier[x] == n;
    }
}

__kernel void life_update_texture (__global unsigned *cur, __write_only image2d_t tex)
{
	int y = get_global_id (1);
	int x = get_global_id (0);
	write_imagef (tex, (int2)(x, y), color_scatter (cur [y * DIM + x] * 0xFFFF00FF));
}