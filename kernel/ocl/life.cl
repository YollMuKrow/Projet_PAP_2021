#include "kernel/ocl/common.cl"


__kernel void life_ocl (__global unsigned *in, __global unsigned *out)
{
	unsigned x = get_global_id (0);
	unsigned y = get_global_id (1);
	unsigned xloc = get_local_id (0);
	unsigned yloc = get_local_id (1);

	__local unsigned n;

	if (x > 0 && x < DIM - 1 && y > 0 && y < DIM - 1){
		n = (  in[(y-1)*DIM + x-1] + in[(y-1)*DIM + x] + in[(y-1)*DIM + x + 1] +
				in[y*DIM + x -1]    + in[y*DIM + x]     + in[y*DIM + x + 1 ]    +
				in[(y+1)*DIM + x-1] + in[(y+1)*DIM + x] + in[(y+1)*DIM + x + 1]);
		//printf("n = %u, value = %u, x = %u, y = %u\n", n, in[y*DIM + x], x, y);
		n = (n == 3 + in[y*DIM + x]) | (n == 3);
		//printf("n new = %u\n", n);
		out[y*DIM + x] = n ;
	}
//	if (x%10 == xloc && y %10 == yloc){
//		//printf();
//		out[y*DIM + x] = in[y*DIM + x] ;
//	}
}

__kernel void life_update_texture (__global unsigned *cur, __write_only image2d_t tex)
{
	int y = get_global_id (1);
	int x = get_global_id (0);

	write_imagef (tex, (int2)(x, y), color_scatter (cur [y * DIM + x] * 0xFFFF00FF));
}