#include "kernel/ocl/common.cl"


__kernel void life_ocl (__global unsigned *in, __global unsigned *out)
{
    unsigned x = get_global_id (0);
    unsigned y = get_global_id (1);

    __local unsigned n;

    if (x > 0 && x < DIM - 1 && y > 0 && y < DIM - 1){
        n = (  in[(y-1)*DIM + x-1]  + in[(y-1)*DIM + x] + in[(y-1)*DIM + x + 1] +
               in[y*DIM + x -1]    + in[y*DIM + x]     + in[y*DIM + x + 1 ]    +
               in[(y+1)*DIM + x-1] + in[(y+1)*DIM + x] + in[(y+1)*DIM + x + 1]);

        n = (n == 3 + in[y*DIM + x]) | (n == 3);
        out[y*DIM + x] = n;
    }
}

__kernel void life_ocl_finish (__global unsigned *in, __global unsigned *out, __global unsigned *change){
    unsigned x = get_global_id (0);
    unsigned y = get_global_id (1);
    __local unsigned n;
    __local unsigned tmp;
    change[0]=0;
    if (x > 0 && x < DIM - 1 && y > 0 && y < DIM - 1){
        n = (  in[(y-1)*DIM + x-1]  + in[(y-1)*DIM + x] + in[(y-1)*DIM + x + 1] +
               in[y*DIM + x -1]    + in[y*DIM + x]     + in[y*DIM + x + 1 ]    +
               in[(y+1)*DIM + x-1] + in[(y+1)*DIM + x] + in[(y+1)*DIM + x + 1]);

        n = (n == 3 + in[y*DIM + x]) | (n == 3);

        out[y*DIM + x] = n;

    }
    if(out[y*DIM + x] != in[y*DIM + x]){
        printf("changement");
        change[0]=1;
    }
    //else{
        //printf("changement");
    //}
}


//__kernel void life_ocl_tile (__global unsigned *in, __global unsigned *out)
//{
//	unsigned x = get_global_id (0);
//	unsigned y = get_global_id (1);
//	unsigned xloc = get_local_id (0);
//	unsigned yloc = get_local_id (1);
//	unsigned xgroup = get_group_id (0);
//	unsigned ygroup = get_group_id (1);
//    __local unsigned tile [GPU_TILE_H][GPU_TILE_W];
//
//    printf("local_size x = %u,local_size y = %u, group_size y = %u, group_size x= %u\n",get_local_size(0),get_local_size(1),GPU_TILE_H, GPU_TILE_W);
//
//    tile[yloc][xloc] = in[y*DIM+x];
//
//    barrier (CLK_LOCAL_MEM_FENCE);
//	__local unsigned n;
//	if (x > 0 && x < DIM - 1 && y > 0 && y < DIM - 1){
//		for (int i = y-1; i < y+1; i++)
//            for (int j = x-1; j < x+1; j++){
//                n+=in[i][j];
//            }
//        n = (n == 3 + ) | (n == 3);
//        if (n != me)
//            change |= 1;
//
//
//                //printf("n = %u, value = %u, x = %u, y = %u\n", n, in[y*DIM + x], x, y);
//		n = (n == 3 + in[y*DIM + x]) | (n == 3);
//        barrier (CLK_LOCAL_MEM_FENCE);
////		//printf("groupx = %u groupy = %u\n", xgroup, ygroup);
//		out[y*DIM + x] = tile[yloc][xloc] ;
////	}
//}
//
//__kernel void life_update_texture (__global unsigned *cur, __write_only image2d_t tex)
//{
//	int y = get_global_id (1);
//	int x = get_global_id (0);
//
//	write_imagef (tex, (int2)(x, y), color_scatter (cur [y * DIM + x] * 0xFFFF00FF));
//}