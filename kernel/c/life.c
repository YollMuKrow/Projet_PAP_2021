
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

// This kernel does not directly work on cur_img/next_img.
// Instead, we use 2D arrays of boolean values, not colors
#define cur_table(y, x) (*table_cell (_table, (y), (x)))
#define next_table(y, x) (*table_cell (_alternate_table, (y), (x)))

void life_init (void)
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
}

void life_finalize (void)
{
    const unsigned size = DIM * DIM * sizeof (cell_t);

    munmap (_table, size);
    munmap (_alternate_table, size);
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
}

///////////////////////////OCL version

///////////////////////////OCL version
static cl_mem change_buffer = 0;

void life_refresh_img_ocl_finish (void){
    cl_int err;

    err = clEnqueueReadBuffer(queue, cur_buffer, CL_TRUE, 0,
                              sizeof (unsigned) * DIM * DIM, _table, 0, NULL,
                              NULL);
    check(err, "Failed to read cur buffer from GPU");
    life_refresh_img ();
}

void life_refresh_img_ocl (void){
    cl_int err;

    err = clEnqueueReadBuffer(queue, cur_buffer, CL_TRUE, 0,
                              sizeof (unsigned) * DIM * DIM, _table, 0, NULL,
                              NULL);
    check(err, "Failed to read cur buffer from GPU");
    life_refresh_img ();
}

void life_init_ocl_finish (void)
{
    life_init();
    change_buffer = clCreateBuffer (context, CL_MEM_WRITE_ONLY, sizeof (unsigned), NULL, NULL);
    if (!change_buffer)
        exit_with_error ("Failed to allocate change buffer");
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
        change_buffer_value[0]=1;
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

        err = clEnqueueReadBuffer(queue, change_buffer, CL_TRUE, 0,
                                  sizeof(unsigned), change_buffer_value, 0, NULL,
                                  NULL);
        check(err, "Failed to read change buffer from GPU");

        if (change_buffer_value[0] != 1){
            printf("on s'arrete à %u iteration\n", it);
            break;
        }
    }

    clFinish (queue);
    monitoring_end_tile (0, 0, DIM, DIM, easypap_gpu_lane (TASK_TYPE_COMPUTE));
    return 0;
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

///////////////////////////// Tiled parallel version

///////////////////////// TEST FIRST TOUCH 32x32 for_inner_c + for_inner
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

// Tile inner computation
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

//////////////// TOUCH TILE FUNCTION
static void do_touch_tile(int x, int y, int width, int height)
{
//un com random
    for (int i = y; i < y + height; i++)
        next_table (i, x) = cur_table (i, x);
}

static void life_ft_omp(void){
#pragma omp parallel
    {
#pragma omp for collapse(2) schedule(runtime)
        for(int i = 0; i < DIM; i+=TILE_W)
            for (int j = 0; j < DIM; j+=TILE_W)
                do_touch_tile(i, j, TILE_W, TILE_H);
    }
}
//test OMP_NUM_THREADS=46 OMP_PLACES=cores ./run -k life -n -i 50 -a random -s 2048 -v tiled_omp_for_inner_first_touch -th 32 -tw 32
unsigned life_compute_tiled_omp_for_c_inner_first_touch (unsigned nb_iter)
{
    unsigned res = 0;
    life_ft_omp();
    for (unsigned it = 1; it <= nb_iter; it++) {
        unsigned change = 0;
#pragma omp parallel
        {
#pragma omp for collapse(2)
            //On analyse la partie interne
            for (int y = 0; y < DIM; y += TILE_H)
                for (int x = 0; x < DIM; x += TILE_W){
                    if (x == 0 || x == DIM-TILE_W || y == 0 || y == DIM-TILE_H)
                        change |= do_tile (x, y, TILE_W, TILE_H, omp_get_thread_num());
                    else
                        change |= do_inner_tile (x, y, TILE_W, TILE_H, omp_get_thread_num());
                }
        }
        swap_tables ();

        if (!change) { // we stop when all cells are stable
            res = it;
            break;
        }
    }

    return res;
}

//Inner multithreaded version (collapsed for)
//test OMP_NUM_THREADS=46 OMP_PLACES=cores ./run -k life -n -i 50 -a random -s 2048 -v tiled_omp_for_inner_c_first_touch -th 32 -tw 32
unsigned life_compute_tiled_omp_for_inner_first_touch (unsigned nb_iter)
{
    unsigned res = 0;
    life_ft_omp();
    for (unsigned it = 1; it <= nb_iter; it++) {
        unsigned change = 0;
#pragma omp parallel
        {
#pragma omp for
            for (int y = 0; y < DIM; y += TILE_H)
                for (int x = 0; x < DIM; x += TILE_W)
                    if (x == 0 || x == DIM-TILE_W || y == 0 || y == DIM-TILE_H)
                        change |= do_tile (x, y, TILE_W, TILE_H, omp_get_thread_num());
                    else
                        change |= do_inner_tile (x, y, TILE_W, TILE_H, omp_get_thread_num());
        }
        swap_tables ();

        if (!change) { // we stop when all cells are stable
            res = it;
            break;
        }
    }

    return res;
}

///////////////////////// TEST INNER_TILED OMP VERSION
//test OMP_NUM_THREAD=46 OMP_PLACES=cores ./run -k life -n -i 100 -a random -s 2048 -v tiled_omp_for_inner -th 16 -tw 16-> 118.774
unsigned life_compute_tiled_omp_for_inner (unsigned nb_iter)
{
    unsigned res = 0;

    for (unsigned it = 1; it <= nb_iter; it++) {
        unsigned change = 0;
#pragma omp parallel
        {
#pragma omp for
            //On analyse la partie interne
            for (int y = 0; y < DIM; y += TILE_H)
                for (int x = 0; x < DIM; x += TILE_W){
                    if (x == 0 || x == DIM-TILE_W || y == 0 || y == DIM-TILE_H)
                        change |= do_tile (x, y, TILE_W, TILE_H, omp_get_thread_num());
                    else
                        change |= do_inner_tile (x, y, TILE_W, TILE_H, omp_get_thread_num());
                }

        }
        swap_tables ();

        if (!change) { // we stop when all cells are stable
            res = it;
            break;
        }
    }

    return res;
}

//Inner multithreaded version (collapsed for)
//test OMP_NUM_THREADS=46 OMP_PLACES=cores ./run -k life -n -i 100 -a random -s 2048 -v tiled_omp_for_inner_c -th 16 -tw 16 -> 104.705
unsigned life_compute_tiled_omp_for_inner_c (unsigned nb_iter)
{
    unsigned res = 0;

    for (unsigned it = 1; it <= nb_iter; it++) {
        unsigned change = 0;
#pragma omp parallel
        {
#pragma omp for collapse(2)
            for (int y = 0; y < DIM; y += TILE_H)
                for (int x = 0; x < DIM; x += TILE_W)
                    if (x == 0 || x == DIM-TILE_W || y == 0 || y == DIM-TILE_H)
                        change |= do_tile (x, y, TILE_W, TILE_H, omp_get_thread_num());
                    else
                        change |= do_inner_tile (x, y, TILE_W, TILE_H, omp_get_thread_num());
        }
        swap_tables ();

        if (!change) { // we stop when all cells are stable
            res = it;
            break;
        }
    }

    return res;
}

//Inner multithreaded version (collapsed for, dynamic scheduling)
//test OMP_NUM_THREADS=46 OMP_PLACES=cores ./run -k life -n -i 100 -a random -s 2048 -v tiled_omp_for_inner_cd -th 16 -tw 16 -> 282.471
unsigned life_compute_tiled_omp_for_inner_cd(unsigned nb_iter)
{
    unsigned res = 0;

    for (unsigned it = 1; it <= nb_iter; it++) {
        unsigned change = 0;
#pragma omp parallel
        {
#pragma omp for collapse(2) schedule(dynamic)
            for (int y = 0; y < DIM; y += TILE_H)
                for (int x = 0; x < DIM; x += TILE_W)
                    if (x == 0 || x == DIM-TILE_W || y == 0 || y == DIM-TILE_H)
                        change |= do_tile (x, y, TILE_W, TILE_H, omp_get_thread_num());
                    else
                        change |= do_inner_tile (x, y, TILE_W, TILE_H, omp_get_thread_num());
        }
        swap_tables ();

        if (!change) { // we stop when all cells are stable
            res = it;
            break;
        }
    }

    return res;
}

//Inner multithreaded version (collapsed for, static scheduling)
//test OMP_NUM_THREADS=46 OMP_PLACES=cores ./run -k life -n -i 100 -a random -s 2048 -v tiled_omp_for_inner_cs -th 16 -tw 16 -> 106.707
unsigned life_compute_tiled_omp_for_inner_cs(unsigned nb_iter)
{
    unsigned res = 0;

    for (unsigned it = 1; it <= nb_iter; it++) {
        unsigned change = 0;
#pragma omp parallel
        {
#pragma omp for collapse(2) schedule(static)
            for (int y = 0; y < DIM; y += TILE_H)
                for (int x = 0; x < DIM; x += TILE_W)
                    if (x == 0 || x == DIM-TILE_W || y == 0 || y == DIM-TILE_H)
                        change |= do_tile (x, y, TILE_W, TILE_H, omp_get_thread_num());
                    else
                        change |= do_inner_tile (x, y, TILE_W, TILE_H, omp_get_thread_num());
        }
        swap_tables ();

        if (!change) { // we stop when all cells are stable
            res = it;
            break;
        }
    }

    return res;
}

//Inner multithreaded version (collapsed for, static scheduling)
//test OMP_NUM_THREADS=46 OMP_PLACES=cores ./run -k life -n -i 100 -a random -s 2048 -v tiled_omp_for_inner_cs1 -th 16 -tw 16 -> 194.762
unsigned life_compute_tiled_omp_for_inner_cs1(unsigned nb_iter)
{
    unsigned res = 0;

    for (unsigned it = 1; it <= nb_iter; it++) {
        unsigned change = 0;
#pragma omp parallel
        {
#pragma omp for collapse(2) schedule(static, 1)
            for (int y = 0; y < DIM; y += TILE_H)
                for (int x = 0; x < DIM; x += TILE_W)
                    if (x == 0 || x == DIM-TILE_W || y == 0 || y == DIM-TILE_H)
                        change |= do_tile (x, y, TILE_W, TILE_H, omp_get_thread_num());
                    else
                        change |= do_inner_tile (x, y, TILE_W, TILE_H, omp_get_thread_num());
        }
        swap_tables ();

        if (!change) { // we stop when all cells are stable
            res = it;
            break;
        }
    }

    return res;
}


///////////////////////// TEST TILED OMP
//test OMP_NUM_THREAD=46 OMP_PLACES=cores ./run -k life -n -i 100 -a random -s 2048 -v tiled_omp_for -th 16 -tw 16 -> 177.348
unsigned life_compute_tiled_omp_for (unsigned nb_iter)
{
    unsigned res = 0;

    for (unsigned it = 1; it <= nb_iter; it++) {
        unsigned change = 0;
#pragma omp parallel
        {
#pragma omp for
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

//Simple multithreaded version (collapsed for)
//test OMP_NUM_THREADS=46 OMP_PLACES=cores ./run -k life -n -i 100 -a random -s 2048 -v tiled_omp_for_c -th 16 -tw 16 -> 134.776
unsigned life_compute_tiled_omp_for_c (unsigned nb_iter)
{
    unsigned res = 0;

    for (unsigned it = 1; it <= nb_iter; it++) {
        unsigned change = 0;
#pragma omp parallel
        {
#pragma omp for collapse(2)
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

//Simple multithreaded version (collapsed for, dynamic scheduling)
//test OMP_NUM_THREADS=46 OMP_PLACES=cores ./run -k life -n -i 100 -a random -s 2048 -v tiled_omp_for_cd -th 16 -tw 16 -> 294.129
unsigned life_compute_tiled_omp_for_cd(unsigned nb_iter)
{
    unsigned res = 0;

    for (unsigned it = 1; it <= nb_iter; it++) {
        unsigned change = 0;
#pragma omp parallel
        {
#pragma omp for collapse(2) schedule(dynamic)
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

//Simple multithreaded version (collapsed for, static scheduling)
//test OMP_NUM_THREADS=46 OMP_PLACES=cores ./run -k life -n -i 100 -a random -s 2048 -v tiled_omp_for_cs -th 16 -tw 16 -> 134.175
unsigned life_compute_tiled_omp_for_cs(unsigned nb_iter)
{
    unsigned res = 0;

    for (unsigned it = 1; it <= nb_iter; it++) {
        unsigned change = 0;
#pragma omp parallel
        {
#pragma omp for collapse(2) schedule(static)
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

//Simple multithreaded version (collapsed for, static scheduling)
//test OMP_NUM_THREADS=46 OMP_PLACES=cores ./run -k life -n -i 100 -a random -s 2048 -v tiled_omp_for_cs1 -th 16 -tw 16 -> 263.057
unsigned life_compute_tiled_omp_for_cs1(unsigned nb_iter)
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
