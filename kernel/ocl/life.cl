#include "kernel/ocl/common.cl"


__kernel void life_ocl (__global unsigned *in, __global unsigned *out)
{
  // Nothing
}

__kernel void update_texture (__global unsigned *cur, __write_only image2d_t tex)
{
	int y = get_global_id (1);
	int x = get_global_id (0);

	write_imagef (tex, (int2)(x, y), color_scatter (cur [y * DIM + x] * 0xFFFF00FF));
}