
#include "easypap.h"
#include "rle_lexer.h"

#include <omp.h>
#include <stdbool.h>
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>

static unsigned color = 0xFFFF00FF; // Living cells have the yellow color

typedef unsigned cell_t;

static cell_t *restrict _table = NULL, *restrict _alternate_table = NULL;

static inline cell_t *table_cell (cell_t *restrict i, int y, int x)
{
return i + y * DIM + x;
}
////////////
static cell_t *restrict _change_table = NULL, *restrict _alternate_change_table = NULL; //storing change variables
static unsigned tile_w_power, tile_h_power;

static inline cell_t *table_change (cell_t *restrict i, int y, int x)
{
return i + (y+1) * NB_TILES_Y + (x+1); // +1 to each argument to account for the extra borders
}
/////////////

// This kernel does not directly work on cur_img/next_img.
// Instead, we use 2D arrays of boolean values, not colors
#define cur_table(y, x) (*table_cell (_table, (y), (x)))
#define next_table(y, x) (*table_cell (_alternate_table, (y), (x)))
////////////////
#define cur_change_table(y, x) (*table_change (_change_table, (y), (x)))
#define next_change_table(y, x) (*table_change (_alternate_change_table, (y), (x)))
///////////////

/*void life_init (void)
{
	// life_init may be (indirectly) called several times so we check if data were
	// already allocated
	if (_table == NULL) {
		const unsigned size = DIM * DIM * sizeof (cell_t);

		PRINT_DEBUG ('u', "Memory footprint = 2 x %d bytes\n", size);

		_table = mmap (NULL, size, PROT_READ | PROT_WRITE,
		               MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);

		_alternate_table = mmap (NULL, size, PROT_READ | PROT_WRITE,
		                         MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
	}
}*/
void life_init (void)
{
    // life_init may be (indirectly) called several times so we check if data were
    // already allocated
    if (_table == NULL) {
        const unsigned size = DIM * DIM * sizeof (cell_t);
        const unsigned tiles = (NB_TILES_X+2)*(NB_TILES_Y+2) * sizeof(cell_t); // We allocate extra borders
        PRINT_DEBUG ('u', "Memory footprint = 2 x %d bytes\n", size);

        _table = mmap (NULL, size, PROT_READ | PROT_WRITE,
                       MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);

        _alternate_table = mmap (NULL, size, PROT_READ | PROT_WRITE,
                                 MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);

        _change_table = mmap (NULL, tiles, PROT_READ | PROT_WRITE,
                              MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);

        _alternate_change_table = mmap (NULL, tiles, PROT_READ | PROT_WRITE,
                                        MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);

        for(unsigned tile_x = -1; tile_x <=NB_TILES_X; tile_x++){
            for(unsigned tile_y = -1; tile_y <=NB_TILES_Y; tile_y++){
                cur_change_table(tile_x, tile_y) = 0;
                next_change_table(tile_x, tile_y) = 0;
            }
        }
        for(unsigned tile_x = 0; tile_x <NB_TILES_X; tile_x++){
            for(unsigned tile_y = 0; tile_y <NB_TILES_Y; tile_y++){
                cur_change_table(tile_x, tile_y) = 1;
            }
        }

        //Tile_w_power and tile_h_power hold log2(TILE_W) and log2(TILE_H)
        //they are used to quickly divide when we want to know in which tile a cell is
        tile_w_power = 0;
        while((0x1<<tile_w_power) != TILE_W)
            tile_w_power++;

        tile_h_power = 0;
        while((0x1<<tile_h_power) != TILE_H)
            tile_h_power++;

        printf("Tiles = 2^%u×2^%u\n", tile_w_power, tile_h_power);
    }
}
/*
void life_finalize (void)
{
	const unsigned size = DIM * DIM * sizeof (cell_t);

	munmap (_table, size);
	munmap (_alternate_table, size);
}
*/
void life_finalize (void)
{
    const unsigned size = DIM * DIM * sizeof (cell_t);

    munmap (_table, size);
    munmap (_alternate_table, size);
    munmap (_change_table, NB_TILES_Y*NB_TILES_X);
    munmap (_alternate_change_table, NB_TILES_Y*NB_TILES_X);

}

// This function is called whenever the graphical window needs to be refreshed
void life_refresh_img (void)
{
    for (int i = 0; i < DIM; i++)
        for (int j = 0; j < DIM; j++)
            cur_img (i, j) = cur_table (i, j) * color;
}
static inline void swap_tables (void)
{
    cell_t *tmp = _table;

    _table           = _alternate_table;
    _alternate_table = tmp;

    tmp = _change_table;

    _change_table = _alternate_change_table;
    _alternate_change_table = tmp;
}
/*
static inline void swap_tables (void)
{
	cell_t *tmp = _table;

	_table           = _alternate_table;
	_alternate_table = tmp;
}
*/

////////////////// TOUCH TILE FUNCTION
void do_touch_tile(int x, int y, int width, int height)
{
    for (int i = y; i < y + height; i++)
        next_table (i, x) = cur_table (i, x);
}

void life_ft(void){
#pragma omp parallel
    {
#pragma omp for collapse(2) schedule(static)
        for(int i = 0; i < DIM; i+=TILE_W)
            for (int j = 0; j < DIM; j+=TILE_W)
                do_touch_tile(i, j, TILE_W, TILE_H);
    }
}

///////////////////////////// Sequential version (seq)

static int compute_new_state (int y, int x)
{
    unsigned n      = 0;
    unsigned me     = cur_table (y, x) != 0;
    unsigned change = 0;

    if (x > 0 && x < DIM - 1 && y > 0 && y < DIM - 1) {

        for (int i = y - 1; i <= y + 1; i++)
            for (int j = x - 1; j <= x + 1; j++)
                n += cur_table (i, j);

        n = (n == 3 + me) | (n == 3);
        if (n != me)
            change |= 1;

        next_table (y, x) = n;
    }

    return change;
}

unsigned life_compute_seq (unsigned nb_iter)
{
    for (unsigned it = 1; it <= nb_iter; it++) {
        int change = 0;

        monitoring_start_tile (0);

        for (int i = 0; i < DIM; i++)
            for (int j = 0; j < DIM; j++)
                change |= compute_new_state (i, j);

        monitoring_end_tile (0, 0, DIM, DIM, 0);

        swap_tables ();

        if (!change)
            return it;
    }

    return 0;
}

///////////////////////////// Tiled sequential version (tiled)
// Tile inner computation
static int do_tile_reg (int x, int y, int width, int height)
{
    int change = 0;

    for (int i = y; i < y + height; i++)
        for (int j = x; j < x + width; j++)
            change |= compute_new_state (i, j);

    return change;
}

static int do_tile (int x, int y, int width, int height, int who)
{
    int r;

    monitoring_start_tile (who);

    r = do_tile_reg (x, y, width, height);

    monitoring_end_tile (x, y, width, height, who);

    return r;
}
//test OMP_NUM_THREAD=46 OMP_PLACES=cores ./run -k life -a random -s 2048 -n -v tiled -i 100 -> 413.442
unsigned life_compute_tiled (unsigned nb_iter)
{
    unsigned res = 0;

    for (unsigned it = 1; it <= nb_iter; it++) {
        unsigned change = 0;

        for (int y = 0; y < DIM; y += TILE_H)
            for (int x = 0; x < DIM; x += TILE_W)
                change |= do_tile (x, y, TILE_W, TILE_H, 0);

        swap_tables ();

        if (!change) { // we stop when all cells are stable
            res = it;
            break;
        }
    }

    return res;
}

/////////////////////////// TEST FIRST TOUCH 32x32 for_inner_c + for_inner
static int compute_new_state_nocheck (int y, int x)
{
    unsigned n      = 0;
    unsigned me     = cur_table (y, x) != 0;
    unsigned change = 0;

    for (int i = y - 1; i <= y + 1; i++)
        for (int j = x - 1; j <= x + 1; j++)
            n += cur_table (i, j);

    n = (n == 3 + me) | (n == 3);
    if (n != me)
        change |= 1;

    next_table (y, x) = n;

    return change;
}

//// Tile inner computation
static int do_tile_nocheck (int x, int y, int width, int height)
{
    int change = 0;

    for (int i = y; i < y + height; i++)
        for (int j = x; j < x + width; j++)
            change |= compute_new_state_nocheck(i, j);

    return change;
}
static int do_inner_tile (int x, int y, int width, int height, int who)
{
    int r;

    monitoring_start_tile (who);

    r = do_tile_nocheck(x, y, width, height);

    monitoring_end_tile (x, y, width, height, who);

    return r;
}

/*
////Simple multithreaded version (collapsed for, static scheduling)
////test OMP_NUM_THREADS=46 OMP_PLACES=cores ./run -k life -n -i 100 -a random -s 2048 -v tiled_omp_for_cs1 -th 16 -tw 16 -> 263.057
unsigned life_compute_tiled_omp_for(unsigned nb_iter)
{
    unsigned res = 0;

    for (unsigned it = 1; it <= nb_iter; it++) {
        unsigned change = 0;
#pragma omp parallel
        {
#pragma omp for collapse(2) schedule(static, 1)
            for (int y = 0; y < DIM; y += TILE_H)
                for (int x = 0; x < DIM; x += TILE_W)
                    change |= do_tile (x, y, TILE_W, TILE_H, omp_get_thread_num());
        }
        swap_tables ();
        if (!change) { // we stop when all cells are stable
            res = it;
            break;
        }
    }

    return res;
}

unsigned life_compute_tiled_omp_for_inner_opt (unsigned nb_iter)
{
    unsigned res = 0;

    for (unsigned it = 1; it <= nb_iter; it++) {
        unsigned change = 0;
#pragma omp parallel
        {
            //Outer loops
            //0xxxxxxx0
            //y       y
            //y       y
            //y       y
            //y       y
            //0xxxxxxx0

#pragma omp for nowait schedule(static)
            for(int y = TILE_H; y<(DIM-TILE_H); y+=TILE_H){
                change |= do_inner_tile(         1, y, TILE_W-1, TILE_H, omp_get_thread_num());
                change |= do_inner_tile(DIM-TILE_W, y, TILE_W-1, TILE_H, omp_get_thread_num());
            }

#pragma omp for nowait schedule(static)
            for(int x = TILE_W; x<(DIM-TILE_W); x+=TILE_W){
                change |= do_inner_tile(x,          1, TILE_W, TILE_H-1, omp_get_thread_num());
                change |= do_inner_tile(x, DIM-TILE_H, TILE_W, TILE_H-1, omp_get_thread_num());
            }

            //Top left corner
#pragma omp single
            change |= do_inner_tile(         1,          1, TILE_W-1, TILE_H-1, omp_get_thread_num());

            //Bottom left corner
#pragma omp single
            change |= do_inner_tile(         1, DIM-TILE_H, TILE_W-1, TILE_H-1, omp_get_thread_num());

            //Top right corner
#pragma omp single
            change |= do_inner_tile(DIM-TILE_W,          1, TILE_W-1, TILE_H-1, omp_get_thread_num());

            //Bottom right corner
#pragma omp single
            change |= do_inner_tile(DIM-TILE_W, DIM-TILE_H, TILE_W-1, TILE_H-1, omp_get_thread_num());

            //Inner loop
            //000000000
            //0xxxxxxx0
            //0xxxxxxx0
            //0xxxxxxx0
            //0xxxxxxx0
            //000000000
#pragma omp for collapse(2) schedule(static)
            for(int y=TILE_H; y<(DIM-TILE_H); y+=TILE_H){
                for(int x=TILE_W; x<(DIM-TILE_W); x+=TILE_W){
                    change |= do_inner_tile(x, y, TILE_W, TILE_H, omp_get_thread_num());
                }
            }
        } // omp parallel
        swap_tables ();

        if (!change) { // we stop when all cells are stable
            res = it;
            break;
        }
    }

    return res;
}
*/

/* LIFE COMPUTE LAZY
 * Update all tiles that changed during the last iteration, or
 * those that are adjacent (corners included) to tiles that have changed
 * Based on inner_opt to minimize branching
 */


bool tile_needs_update(unsigned tile_x, unsigned tile_y){
    bool update = 	cur_change_table(tile_x	 , tile_y	)
                     ||	cur_change_table(tile_x+1, tile_y	)
                     ||	cur_change_table(tile_x-1, tile_y	)
                     ||	cur_change_table(tile_x	 , tile_y+1	)
                     ||	cur_change_table(tile_x+1, tile_y+1	)
                     ||	cur_change_table(tile_x-1, tile_y+1	)
                     ||	cur_change_table(tile_x	 , tile_y-1	)
                     ||	cur_change_table(tile_x+1, tile_y-1	)
                     ||	cur_change_table(tile_x-1, tile_y-1	);
    if(!update){
        //PRINT_MASTER("Tile at %2u, %2u doesn't need updating\n", tile_x, tile_y);
    }
    return 	update;
}

void print_change_table(){
    for(unsigned y = 0; y<NB_TILES_Y; y++){
        for(unsigned x = 0; x<NB_TILES_X; x++){
            printf("%c",  cur_change_table(y, x)?'#':' ');
        }
        printf("\n");
    }
}

//test OMP_NUM_THREADS=46 OMP_PLACES=cores ./run -k life -n -i 100 -a random -s 2048 -v lazy -th 16 -tw 16 -> 134.776
unsigned life_compute_lazy(unsigned nb_iter)
{
    unsigned res = 0;
    for (unsigned it = 1; it <= nb_iter; it++) {
        //unsigned change = 0;
#pragma omp parallel
        {
            //Outer loops
            //0xxxxxxx0
            //y       y
            //y       y
            //y       y
            //y       y
            //0xxxxxxx0

#pragma omp for nowait schedule(static)
            for(int y = TILE_H; y < (DIM-TILE_H); y += TILE_H){
                unsigned tile_y = y>>tile_h_power;
                next_change_table(0, tile_y) 			= tile_needs_update(0, tile_y) && do_inner_tile(         1, y, TILE_W-1, TILE_H, omp_get_thread_num());
                next_change_table(NB_TILES_X-1, tile_y) = tile_needs_update(NB_TILES_X-1, tile_y) && do_inner_tile(DIM-TILE_W, y, TILE_W-1, TILE_H, omp_get_thread_num());
            }

#pragma omp for nowait schedule(static)
            for(int x = TILE_W; x<(DIM-TILE_W); x+=TILE_W){
                unsigned tile_x = x>>tile_w_power;
                next_change_table(tile_x, 0) 			= tile_needs_update(tile_x, 0) && do_inner_tile(x,          1, TILE_W, TILE_H-1, omp_get_thread_num());
                next_change_table(tile_x, NB_TILES_Y-1) = tile_needs_update(tile_x, NB_TILES_Y-1) && do_inner_tile(x, DIM-TILE_H, TILE_W, TILE_H-1, omp_get_thread_num());
            }

            //Top left corner
#pragma omp single
            next_change_table(0, 0) = tile_needs_update(0, 0) && do_inner_tile( 1, 1, TILE_W-1, TILE_H-1, omp_get_thread_num());

            //Bottom left corner
#pragma omp single
            next_change_table(0, NB_TILES_Y-1) = tile_needs_update(0, NB_TILES_Y-1) && do_inner_tile( 1, DIM-TILE_H, TILE_W-1, TILE_H-1, omp_get_thread_num());

            //Top right corner
#pragma omp single
            next_change_table(NB_TILES_X-1, 0) = tile_needs_update(NB_TILES_X-1, 0) && do_inner_tile(DIM-TILE_W, 1, TILE_W-1, TILE_H-1, omp_get_thread_num());

            //Bottom right corner
#pragma omp single
            next_change_table(NB_TILES_X-1, NB_TILES_Y-1) = tile_needs_update(NB_TILES_X-1, NB_TILES_Y-1) && do_inner_tile(DIM-TILE_W, DIM-TILE_H, TILE_W-1, TILE_H-1, omp_get_thread_num());

            //Inner loop
            //000000000
            //0xxxxxxx0
            //0xxxxxxx0
            //0xxxxxxx0
            //0xxxxxxx0
            //000000000
#pragma omp for collapse(2) schedule(static)
            for(int y=TILE_H; y<(DIM-TILE_H); y+=TILE_H){
                for(int x=TILE_W; x<(DIM-TILE_W); x+=TILE_W){
                    unsigned tile_y = y>>tile_h_power;
                    unsigned tile_x = x>>tile_w_power;
                    next_change_table(tile_x, tile_y) = tile_needs_update(tile_x, tile_y) && do_inner_tile(x, y, TILE_W, TILE_H, omp_get_thread_num());
                }
            }
        } // omp parallel
        swap_tables ();
    }

    return res;
}


///////////////////////////OCL version
void life_refresh_img_ocl (void){
    cl_int err;
    err = clEnqueueReadBuffer(queue, cur_buffer, CL_TRUE, 0,
                              sizeof (unsigned) * DIM * DIM, _table, 0, NULL,
                              NULL);
    check(err, "Failed to read cur buffer from GPU");
    life_refresh_img ();
}

///////////////////////////OCL version finish

static cl_mem change_buffer, reset_buffer = 0;

void life_refresh_img_ocl_finish (void){
    cl_int err;

    err = clEnqueueReadBuffer(queue, cur_buffer, CL_TRUE, 0,
                              sizeof (unsigned) * DIM * DIM, _table, 0, NULL,
                              NULL);
    check(err, "Failed to read cur buffer from GPU");
    life_refresh_img ();
}

void life_init_ocl_finish (void)
{
    change_buffer = clCreateBuffer (context, CL_MEM_READ_WRITE, sizeof (unsigned), NULL, NULL);
    if (!change_buffer)
        exit_with_error ("Failed to allocate change buffer");
    reset_buffer = clCreateBuffer (context, CL_MEM_READ_WRITE, sizeof (unsigned), NULL, NULL);
    if (!reset_buffer)
        exit_with_error ("Failed to allocate reset buffer");
    life_init();
}

unsigned life_invoke_ocl_finish (unsigned nb_iter)
{
    size_t global[2] = {GPU_SIZE_X, GPU_SIZE_Y};
    size_t local[2]  = {GPU_TILE_W, GPU_TILE_H};
    cl_int err;

    unsigned *change_buffer_value = malloc(sizeof (unsigned));
    if (change_buffer_value == NULL){
        printf("Echec de l'initialisation du malloc !\n");
        return EXIT_FAILURE;
    }

    monitoring_start_tile (easypap_gpu_lane (TASK_TYPE_COMPUTE));

    for (unsigned it = 1; it <= nb_iter; it++) {
        printf("unsigned it = %d\n",it);
        change_buffer_value[0] = 0;
        //change_buffer = reset_buffer;

        err = 0;
        err |= clSetKernelArg(compute_kernel, 0, sizeof(cl_mem), &cur_buffer);
        err |= clSetKernelArg(compute_kernel, 1, sizeof(cl_mem), &next_buffer);
        err |= clSetKernelArg(compute_kernel, 2, sizeof(cl_mem), &change_buffer);
        check(err, "Failed to set kernel arguments");

        err = clEnqueueNDRangeKernel(queue, compute_kernel, 2, NULL, global, local,
                                     0, NULL, NULL);
        check(err, "Failed to execute kernel");

        {
            cl_mem tmp = cur_buffer;
            cur_buffer = next_buffer;
            next_buffer = tmp;
        }

        printf("buffer init = %u\n", change_buffer_value[0]);
        err = clEnqueueReadBuffer(queue, change_buffer, CL_TRUE, 0,
                                  sizeof(unsigned), change_buffer_value, 0, NULL,
                                  NULL);
        check(err, "Failed to read change buffer from GPU");
        printf("buffer = %u\n", change_buffer_value[0]);
        if (change_buffer_value[0] == 0){
            break;
        }
    }

    clFinish (queue);
    monitoring_end_tile (0, 0, DIM, DIM, easypap_gpu_lane (TASK_TYPE_COMPUTE));
    return 0;
}

///////////////////////////OCL version hybrid
static unsigned cpu_y_part;
static unsigned gpu_y_part;
static long gpu_duration = 0, cpu_duration = 0;

void life_refresh_img_ocl_hybrid (void){
	cl_int err;
	// Façon de faire non optimisé mais en jouant avec l'offset de readbuffer, j'avais que des problèmes

	// on écrit la partie du CPU (cur_table) dans le GPU (cur_buffer)
	err = clEnqueueWriteBuffer (queue, cur_buffer, CL_TRUE, 0,
	                            DIM * cpu_y_part * sizeof (unsigned), _table, 0,
	                            NULL, NULL);
	check (err, "Failed to write to buffer");
	// Et on récupère l'image complète dans cur_buffer du GPU pour la stocké dans le CPU
	err = clEnqueueReadBuffer(queue, cur_buffer, CL_TRUE, 0,
	                          sizeof (unsigned) * DIM * DIM, _table, 0, NULL,
	                          NULL);
	check(err, "Failed to read cur buffer from GPU");
	life_refresh_img ();
}

//static int much_greater_than (long t1, long t2)
//{
//	return (t1 > t2) && ((t1 - t2) * 100 / t1 > THRESHOLD);
//}

void life_init_ocl_hybrid (void)
{
	if (GPU_TILE_H != TILE_H)
		exit_with_error ("CPU and GPU Tiles should have the same height (%d != %d)",
		                 GPU_TILE_H, TILE_H);

	cpu_y_part = (NB_TILES_Y / 2) * GPU_TILE_H; // Start with fifty-fifty
	gpu_y_part = DIM - cpu_y_part;

	life_init();
}

unsigned life_invoke_ocl_hybrid (unsigned nb_iter)
{
	size_t global[2] = {DIM, gpu_y_part};
	size_t local[2]  = {GPU_TILE_W, GPU_TILE_H};
	cl_int err;
	cl_event kernel_event;
	long t1, t2;
	int gpu_accumulated_lines = 0;

	for (unsigned it = 1; it <= nb_iter; it++) {
/*		// Load balancing
		if (gpu_duration) {
			if (much_greater_than (gpu_duration, cpu_duration) &&
			    gpu_y_part > GPU_TILE_H) {
				gpu_y_part -= GPU_TILE_H;
				cpu_y_part += GPU_TILE_H;
				global[1] = gpu_y_part;
			} else if (much_greater_than (cpu_duration, gpu_duration) &&
			           cpu_y_part > GPU_TILE_H) {
				cpu_y_part -= GPU_TILE_H;
				gpu_y_part += GPU_TILE_H;
				global[1] = gpu_y_part;
			}
		}*/

		err = 0;
		err |= clSetKernelArg(compute_kernel, 0, sizeof(cl_mem), &cur_buffer);
		err |= clSetKernelArg(compute_kernel, 1, sizeof(cl_mem), &next_buffer);
		err |= clSetKernelArg(compute_kernel, 2, sizeof (unsigned), &cpu_y_part);
		check(err, "Failed to set kernel arguments");

		err = clEnqueueNDRangeKernel(queue, compute_kernel, 2, NULL, global, local,
		                             0, NULL, &kernel_event); // le résultat va dans next_buffer

		check(err, "Failed to execute kernel");

		/////// début du temps de calcul pour le CPU
		t1 = what_time_is_it ();
		//On effectue la partie du CPU
		////////////////// CODE OPENCL
#pragma omp parallel
		{
			//Outer loops
			//0xxxxxxx0
			//y       y
			//y       y
			//y       y
			//y       y
			//0xxxxxxx0

#pragma omp for nowait schedule(static)
			for(int y = TILE_H; y < (DIM-TILE_H); y += TILE_H){
				unsigned tile_y = y>>tile_h_power;
				next_change_table(0, tile_y) 			= tile_needs_update(0, tile_y) && do_inner_tile(         1, y, TILE_W-1, TILE_H, omp_get_thread_num());
				next_change_table(NB_TILES_X-1, tile_y) = tile_needs_update(NB_TILES_X-1, tile_y) && do_inner_tile(DIM-TILE_W, y, TILE_W-1, TILE_H, omp_get_thread_num());
			}

#pragma omp for nowait schedule(static)
			for(int x = TILE_W; x<(DIM-TILE_W); x+=TILE_W){
				unsigned tile_x = x>>tile_w_power;
				next_change_table(tile_x, 0) 			= tile_needs_update(tile_x, 0) && do_inner_tile(x,          1, TILE_W, TILE_H-1, omp_get_thread_num());
				next_change_table(tile_x, NB_TILES_Y-1) = tile_needs_update(tile_x, NB_TILES_Y-1) && do_inner_tile(x, DIM-TILE_H, TILE_W, TILE_H-1, omp_get_thread_num());
			}
			//Top left corner
#pragma omp single
			next_change_table(0, 0) = tile_needs_update(0, 0) && do_inner_tile( 1, 1, TILE_W-1, TILE_H-1, omp_get_thread_num());

			//Bottom left corner
#pragma omp single
			next_change_table(0, NB_TILES_Y-1) = tile_needs_update(0, NB_TILES_Y-1) && do_inner_tile( 1, DIM-TILE_H, TILE_W-1, TILE_H-1, omp_get_thread_num());

			//Top right corner
#pragma omp single
			next_change_table(NB_TILES_X-1, 0) = tile_needs_update(NB_TILES_X-1, 0) && do_inner_tile(DIM-TILE_W, 1, TILE_W-1, TILE_H-1, omp_get_thread_num());

			//Bottom right corner
#pragma omp single
			next_change_table(NB_TILES_X-1, NB_TILES_Y-1) = tile_needs_update(NB_TILES_X-1, NB_TILES_Y-1) && do_inner_tile(DIM-TILE_W, DIM-TILE_H, TILE_W-1, TILE_H-1, omp_get_thread_num());

			//Inner loop
			//000000000
			//0xxxxxxx0
			//0xxxxxxx0
			//0xxxxxxx0
			//0xxxxxxx0
			//000000000
#pragma omp for collapse(2) schedule(static)
			for(int y=TILE_H; y<(DIM-TILE_H); y+=TILE_H){
				for(int x=TILE_W; x<(DIM-TILE_W); x+=TILE_W){
					unsigned tile_y = y>>tile_h_power;
					unsigned tile_x = x>>tile_w_power;
					next_change_table(tile_x, tile_y) = tile_needs_update(tile_x, tile_y) && do_inner_tile(x, y, TILE_W, TILE_H, omp_get_thread_num());
				}
			}
		} // omp parallel
		///////////////// FIN CODE OPENCL
		t2 = what_time_is_it ();
		////// Fin du temps de calcul pour le cpu

		err = clEnqueueWriteBuffer (queue, next_buffer,CL_TRUE, 0,
		                            DIM * cpu_y_part * sizeof (unsigned), _alternate_table, 0,
		                            NULL, NULL);  // CPU -> GPU
		check(err, "Failed to write next buffer");

		err = clEnqueueReadBuffer(queue, next_buffer, CL_TRUE, cpu_y_part*DIM*sizeof (unsigned ),
		                          sizeof (unsigned) * gpu_y_part * DIM, _alternate_table + DIM*cpu_y_part, 0, NULL,
		                          NULL);    // GPU -> CPU
		check(err, "Failed to read next_buffer");

		cpu_duration = t2 - t1;
		gpu_duration = ocl_monitor (kernel_event, 0, cpu_y_part, global[0],
		                            global[1], TASK_TYPE_COMPUTE);
		clReleaseEvent (kernel_event);
		gpu_accumulated_lines += gpu_y_part;

		{
			cl_mem tmp = cur_buffer;
			cur_buffer = next_buffer;
			next_buffer = tmp;
		}

		gpu_accumulated_lines += gpu_y_part;
		swap_tables ();
	}
	if(do_display){
		// On renvoit le résultat du CPU dans next_table
		err = clEnqueueWriteBuffer (queue, cur_buffer, CL_TRUE, 0,
		                            DIM * cpu_y_part * sizeof (unsigned), _table, 0,
		                            NULL, NULL);
		check (err, "Failed to write to buffer");
	}
	else
		PRINT_DEBUG ('u', "In average, GPU took %.1f%% of the lines\n",
		             (float)gpu_accumulated_lines * 100 / (DIM * nb_iter));
	clFinish (queue);
	return 0;
}

////////////////////////// OCL version hybrid
void life_refresh_img_ocl_hybrid_opti (void){
    cl_int err;
    // Façon de faire non optimisé mais en jouant avec l'offset de readbuffer, j'avais que des problèmes

    // on écrit la partie du CPU (cur_table) dans le GPU (cur_buffer)
    err = clEnqueueWriteBuffer (queue, cur_buffer, CL_TRUE, 0,
                                DIM * cpu_y_part * sizeof (unsigned), _table, 0,
                                NULL, NULL);
    check (err, "Failed to write to buffer");
    // Et on récupère l'image complète dans cur_buffer du GPU pour la stocké dans le CPU
    err = clEnqueueReadBuffer(queue, cur_buffer, CL_TRUE, 0,
                              sizeof (unsigned) * DIM * DIM, _table, 0, NULL,
                              NULL);
    check(err, "Failed to read cur buffer from GPU");
    life_refresh_img ();
}

//static int much_greater_than (long t1, long t2)
//{
//	return (t1 > t2) && ((t1 - t2) * 100 / t1 > THRESHOLD);
//}

void life_init_ocl_hybrid_opti (void)
{
    if (GPU_TILE_H != TILE_H)
        exit_with_error ("CPU and GPU Tiles should have the same height (%d != %d)",
                         GPU_TILE_H, TILE_H);

    cpu_y_part = (NB_TILES_Y / 2) * GPU_TILE_H; // Start with fifty-fifty
    gpu_y_part = DIM - cpu_y_part;

    life_init();
}

unsigned life_invoke_ocl_hybrid_opti (unsigned nb_iter)
{
    size_t global[2] = {DIM, gpu_y_part};
    size_t local[2]  = {GPU_TILE_W, GPU_TILE_H};
    cl_int err;
    cl_event kernel_event;
    long t1, t2;
    int gpu_accumulated_lines = 0;

    for (unsigned it = 1; it <= nb_iter; it++) {
/*		// Load balancing
		if (gpu_duration) {
			if (much_greater_than (gpu_duration, cpu_duration) &&
			    gpu_y_part > GPU_TILE_H) {
				gpu_y_part -= GPU_TILE_H;
				cpu_y_part += GPU_TILE_H;
				global[1] = gpu_y_part;
			} else if (much_greater_than (cpu_duration, gpu_duration) &&
			           cpu_y_part > GPU_TILE_H) {
				cpu_y_part -= GPU_TILE_H;
				gpu_y_part += GPU_TILE_H;
				global[1] = gpu_y_part;
			}
		}*/

        err = 0;
        err |= clSetKernelArg(compute_kernel, 0, sizeof(cl_mem), &cur_buffer);
        err |= clSetKernelArg(compute_kernel, 1, sizeof(cl_mem), &next_buffer);
        err |= clSetKernelArg(compute_kernel, 2, sizeof (unsigned), &cpu_y_part);
        check(err, "Failed to set kernel arguments");

        err = clEnqueueNDRangeKernel(queue, compute_kernel, 2, NULL, global, local,
                                     0, NULL, &kernel_event); // le résultat va dans next_buffer

        check(err, "Failed to execute kernel");

        /////// début du temps de calcul pour le CPU
        t1 = what_time_is_it ();
        //On effectue la partie du CPU
        ////////////////// CODE OPENCL
#pragma omp parallel
        {
            //Outer loops
            //0xxxxxxx0
            //y       y
            //y       y
            //y       y
            //y       y
            //0xxxxxxx0

#pragma omp for nowait schedule(static)
            for(int y = TILE_H; y < (DIM-TILE_H); y += TILE_H){
                unsigned tile_y = y>>tile_h_power;
                next_change_table(0, tile_y) 			= tile_needs_update(0, tile_y) && do_inner_tile(         1, y, TILE_W-1, TILE_H, omp_get_thread_num());
                next_change_table(NB_TILES_X-1, tile_y) = tile_needs_update(NB_TILES_X-1, tile_y) && do_inner_tile(DIM-TILE_W, y, TILE_W-1, TILE_H, omp_get_thread_num());
            }

#pragma omp for nowait schedule(static)
            for(int x = TILE_W; x<(DIM-TILE_W); x+=TILE_W){
                unsigned tile_x = x>>tile_w_power;
                next_change_table(tile_x, 0) 			= tile_needs_update(tile_x, 0) && do_inner_tile(x,          1, TILE_W, TILE_H-1, omp_get_thread_num());
                next_change_table(tile_x, NB_TILES_Y-1) = tile_needs_update(tile_x, NB_TILES_Y-1) && do_inner_tile(x, DIM-TILE_H, TILE_W, TILE_H-1, omp_get_thread_num());
            }
            //Top left corner
#pragma omp single
            next_change_table(0, 0) = tile_needs_update(0, 0) && do_inner_tile( 1, 1, TILE_W-1, TILE_H-1, omp_get_thread_num());

            //Bottom left corner
#pragma omp single
            next_change_table(0, NB_TILES_Y-1) = tile_needs_update(0, NB_TILES_Y-1) && do_inner_tile( 1, DIM-TILE_H, TILE_W-1, TILE_H-1, omp_get_thread_num());

            //Top right corner
#pragma omp single
            next_change_table(NB_TILES_X-1, 0) = tile_needs_update(NB_TILES_X-1, 0) && do_inner_tile(DIM-TILE_W, 1, TILE_W-1, TILE_H-1, omp_get_thread_num());

            //Bottom right corner
#pragma omp single
            next_change_table(NB_TILES_X-1, NB_TILES_Y-1) = tile_needs_update(NB_TILES_X-1, NB_TILES_Y-1) && do_inner_tile(DIM-TILE_W, DIM-TILE_H, TILE_W-1, TILE_H-1, omp_get_thread_num());

            //Inner loop
            //000000000
            //0xxxxxxx0
            //0xxxxxxx0
            //0xxxxxxx0
            //0xxxxxxx0
            //000000000
#pragma omp for collapse(2) schedule(static)
            for(int y=TILE_H; y<(DIM-TILE_H); y+=TILE_H){
                for(int x=TILE_W; x<(DIM-TILE_W); x+=TILE_W){
                    unsigned tile_y = y>>tile_h_power;
                    unsigned tile_x = x>>tile_w_power;
                    next_change_table(tile_x, tile_y) = tile_needs_update(tile_x, tile_y) && do_inner_tile(x, y, TILE_W, TILE_H, omp_get_thread_num());
                }
            }
        } // omp parallel
        ///////////////// FIN CODE OPENCL
        t2 = what_time_is_it ();
        ////// Fin du temps de calcul pour le cpu

        err = clEnqueueWriteBuffer (queue, next_buffer,CL_TRUE, (cpu_y_part-1)*DIM*sizeof (unsigned ),
                                    DIM * sizeof (unsigned), _alternate_table + DIM*(cpu_y_part-1) , 0,
                                    NULL, NULL);  // CPU -> GPU
        check(err, "Failed to write next buffer");

        err = clEnqueueReadBuffer(queue, next_buffer, CL_TRUE, cpu_y_part*DIM*sizeof (unsigned ),
                                  sizeof (unsigned) * DIM, _alternate_table + DIM*cpu_y_part, 0, NULL,
                                  NULL);    // GPU -> CPU
        check(err, "Failed to read next_buffer");

        cpu_duration = t2 - t1;
        gpu_duration = ocl_monitor (kernel_event, 0, cpu_y_part, global[0],
                                    global[1], TASK_TYPE_COMPUTE);
        clReleaseEvent (kernel_event);
        gpu_accumulated_lines += gpu_y_part;

        {
            cl_mem tmp = cur_buffer;
            cur_buffer = next_buffer;
            next_buffer = tmp;
        }

        gpu_accumulated_lines += gpu_y_part;
        swap_tables ();
    }
    if(do_display){
        // On renvoit le résultat du CPU dans next_table
        err = clEnqueueWriteBuffer (queue, cur_buffer, CL_TRUE, 0,
                                    DIM * cpu_y_part * sizeof (unsigned), _table, 0,
                                    NULL, NULL);
        check (err, "Failed to write to buffer");
    }
    else
        PRINT_DEBUG ('u', "In average, GPU took %.1f%% of the lines\n",
                     (float)gpu_accumulated_lines * 100 / (DIM * nb_iter));
    clFinish (queue);
    return 0;
}


///////////////////////////// Initial configs

void life_draw_guns (void);

static inline void set_cell (int y, int x)
{
    cur_table (y, x) = 1;
    if (opencl_used)
        cur_img (y, x) = 1;
}

static inline int get_cell (int y, int x)
{
    return cur_table (y, x);
}

static void inline life_rle_parse (char *filename, int x, int y,
                                   int orientation)
{
    rle_lexer_parse (filename, x, y, set_cell, orientation);
}

static void inline life_rle_generate (char *filename, int x, int y, int width,
                                      int height)
{
    rle_generate (x, y, width, height, get_cell, filename);
}

void life_draw (char *param)
{
    if (param && (access (param, R_OK) != -1)) {
        // The parameter is a filename, so we guess it's a RLE-encoded file
        life_rle_parse (param, 1, 1, RLE_ORIENTATION_NORMAL);
    } else
        // Call function ${kernel}_draw_${param}, or default function (second
        // parameter) if symbol not found
        hooks_draw_helper (param, life_draw_guns);
}

static void otca_autoswitch (char *name, int x, int y)
{
    life_rle_parse (name, x, y, RLE_ORIENTATION_NORMAL);
    life_rle_parse ("data/rle/autoswitch-ctrl.rle", x + 123, y + 1396,
                    RLE_ORIENTATION_NORMAL);
}

static void otca_life (char *name, int x, int y)
{
    life_rle_parse (name, x, y, RLE_ORIENTATION_NORMAL);
    life_rle_parse ("data/rle/b3-s23-ctrl.rle", x + 123, y + 1396,
                    RLE_ORIENTATION_NORMAL);
}

static void at_the_four_corners (char *filename, int distance)
{
    life_rle_parse (filename, distance, distance, RLE_ORIENTATION_NORMAL);
    life_rle_parse (filename, distance, distance, RLE_ORIENTATION_HINVERT);
    life_rle_parse (filename, distance, distance, RLE_ORIENTATION_VINVERT);
    life_rle_parse (filename, distance, distance,
                    RLE_ORIENTATION_HINVERT | RLE_ORIENTATION_VINVERT);
}

// Suggested cmdline: ./run -k life -s 2176 -a otca_off -ts 64 -r 10 -si
void life_draw_otca_off (void)
{
    if (DIM < 2176)
        exit_with_error ("DIM should be at least %d", 2176);

    otca_autoswitch ("data/rle/otca-off.rle", 1, 1);
}

// Suggested cmdline: ./run -k life -s 2176 -a otca_on -ts 64 -r 10 -si
void life_draw_otca_on (void)
{
    if (DIM < 2176)
        exit_with_error ("DIM should be at least %d", 2176);

    otca_autoswitch ("data/rle/otca-on.rle", 1, 1);
}

// Suggested cmdline: ./run -k life -s 6208 -a meta3x3 -ts 64 -r 50 -si
void life_draw_meta3x3 (void)
{
    if (DIM < 6208)
        exit_with_error ("DIM should be at least %d", 6208);

    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
            otca_life (j == 1 ? "data/rle/otca-on.rle" : "data/rle/otca-off.rle",
                       1 + j * (2058 - 10), 1 + i * (2058 - 10));
}

// Suggested cmdline: ./run -k life -a bugs -ts 64
void life_draw_bugs (void)
{
    for (int y = 16; y < DIM / 2; y += 32) {
        life_rle_parse ("data/rle/tagalong.rle", y + 1, y + 8,
                        RLE_ORIENTATION_NORMAL);
        life_rle_parse ("data/rle/tagalong.rle", y + 1, (DIM - 32 - y) + 8,
                        RLE_ORIENTATION_NORMAL);
    }
}

// Suggested cmdline: ./run -k life -v omp -a ship -s 512 -m -ts 16
void life_draw_ship (void)
{
    for (int y = 16; y < DIM / 2; y += 32) {
        life_rle_parse ("data/rle/tagalong.rle", y + 1, y + 8,
                        RLE_ORIENTATION_NORMAL);
        life_rle_parse ("data/rle/tagalong.rle", y + 1, (DIM - 32 - y) + 8,
                        RLE_ORIENTATION_NORMAL);
    }

    for (int y = 43; y < DIM - 134; y += 148) {
        life_rle_parse ("data/rle/greyship.rle", DIM - 100, y,
                        RLE_ORIENTATION_NORMAL);
    }
}

void life_draw_stable (void)
{
    for (int i = 1; i < DIM - 2; i += 4)
        for (int j = 1; j < DIM - 2; j += 4) {
            set_cell (i, j);
            set_cell (i, j + 1);
            set_cell (i + 1, j);
            set_cell (i + 1, j + 1);
        }
}

void life_draw_guns (void)
{
    at_the_four_corners ("data/rle/gun.rle", 1);
}

void life_draw_random (void)
{
    for (int i = 1; i < DIM - 1; i++)
        for (int j = 1; j < DIM - 1; j++)
            if (random () & 1)
                set_cell (i, j);
}

// Suggested cmdline: ./run -k life -a clown -s 256 -i 110
void life_draw_clown (void)
{
    life_rle_parse ("data/rle/clown-seed.rle", DIM / 2, DIM / 2,
                    RLE_ORIENTATION_NORMAL);
}

void life_draw_diehard (void)
{
    life_rle_parse ("data/rle/diehard.rle", DIM / 2, DIM / 2,
                    RLE_ORIENTATION_NORMAL);
}