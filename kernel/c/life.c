
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
static cell_t *restrict _change_table = NULL, *restrict _alternate_change_table = NULL; //storing change variables

static unsigned tile_w_power, tile_h_power;

static inline cell_t *table_cell (cell_t *restrict i, int y, int x)
{
    return i + (y+1) * DIM + (x+1); // +1 to each argument to account for the extra borders
}

static inline cell_t *table_change (cell_t *restrict i, int y, int x)
{
    return i + (y+1) * NB_TILES_Y + (x+1); // +1 to each argument to account for the extra borders
}

// This kernel does not directly work on cur_img/next_img.
// Instead, we use 2D arrays of boolean values, not colors
#define cur_table(y, x) (*table_cell (_table, (y), (x)))
#define next_table(y, x) (*table_cell (_alternate_table, (y), (x)))
#define cur_change_table(y, x) (*table_change (_change_table, (y), (x)))
#define next_change_table(y, x) (*table_change (_alternate_change_table, (y), (x)))

void life_init (void)
{
    // life_init may be (indirectly) called several times so we check if data were
    // already allocated
    if (_table == NULL) {
        const unsigned size = (DIM+2) * (DIM+2) * sizeof (cell_t);  //Allocating extra borders for vec version
        const unsigned tiles = (NB_TILES_X+2)*(NB_TILES_Y+2) * sizeof(cell_t); // We allocate extra borders
        PRINT_DEBUG ('u', "Memory footprint = 2 x (%d+%d)= %d bytes\n", size,tiles,(size+tiles)*2);

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

void life_finalize (void)
{
    const unsigned size = DIM * DIM * sizeof (cell_t);
    const unsigned tiles = (NB_TILES_X+2)*(NB_TILES_Y+2) * sizeof(cell_t); // We allocate extra borders

    munmap (_table, size);
    munmap (_alternate_table, size);
    munmap (_change_table, tiles);
    munmap (_alternate_change_table, tiles);

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


bool tile_needs_update(unsigned tile_x, unsigned tile_y){
    return  cur_change_table(tile_x	 , tile_y	)
        ||	cur_change_table(tile_x+1, tile_y	)
        ||	cur_change_table(tile_x-1, tile_y	)
        ||	cur_change_table(tile_x	 , tile_y+1	)
        ||	cur_change_table(tile_x+1, tile_y+1	)
        ||	cur_change_table(tile_x-1, tile_y+1	)
        ||	cur_change_table(tile_x	 , tile_y-1	)
        ||	cur_change_table(tile_x+1, tile_y-1	)
        ||	cur_change_table(tile_x-1, tile_y-1	);
}

// VECTORIZED VERSION
#if defined(ENABLE_VECTO) && (VEC_SIZE_INT == 8) || true
#include <immintrin.h>

static int do_tile_vec(unsigned x, unsigned y, unsigned w, unsigned h, unsigned cpu);
int compute_multiple_cells(int x, int y);

unsigned life_compute_vec(unsigned nb_iter){
    unsigned res = 0;
	for (unsigned it = 1; it <= nb_iter; it++) {
		unsigned change = 0;
        for (int y = 0; y < DIM; y += TILE_H)
            for (int x = 0; x < DIM; x += TILE_W){
                change |= do_tile_vec (x, y, TILE_W, TILE_H, omp_get_thread_num());
            }
		swap_tables ();

		if (!change) { // we stop when all cells are stable
			res = it;
			break;
		}
	}

	return res;
}

static int do_tile_vec(unsigned x, unsigned y, unsigned w, unsigned h, unsigned cpu){
    int change = 0;
    monitoring_start_tile (cpu);
    for(int j = 0; j<w; j+=VEC_SIZE_INT){ //Loop columns
        for(int i = 0; i<h; i++){ //Loop lines
            change |= compute_multiple_cells(x+j, y+i);
        }
    }
    monitoring_end_tile (x, y, w, h, cpu);
    return change;
}

int compute_multiple_cells(int x, int y){
    __m256i one    = _mm256_set1_epi32(1);
    
    __m256i cells   = _mm256_lddqu_si256((__m256i*)&cur_table(y,x)); //Current cells value
    __m256i top     = _mm256_lddqu_si256((__m256i*)&cur_table(y-1, x));
    __m256i bot     = _mm256_lddqu_si256((__m256i*)&cur_table(y+1, x));
    __m256i right   = _mm256_lddqu_si256((__m256i*)&cur_table(y, x+1));
    __m256i left    = _mm256_lddqu_si256((__m256i*)&cur_table(y, x-1));

    __m256i topright    = _mm256_lddqu_si256((__m256i*)&cur_table(y-1, x+1));
    __m256i botright    = _mm256_lddqu_si256((__m256i*)&cur_table(y+1, x+1));
    __m256i topleft     = _mm256_lddqu_si256((__m256i*)&cur_table(y-1, x-1));
    __m256i botleft     = _mm256_lddqu_si256((__m256i*)&cur_table(y+1, x-1));

    __m256i sum = _mm256_add_epi32(
                    _mm256_add_epi32(
                        _mm256_add_epi32(top,bot),
                        _mm256_add_epi32(right,left)),
                    _mm256_add_epi32(
                        _mm256_add_epi32(topright,topleft),
                        _mm256_add_epi32(botright,botleft))
                    ); //Number of alive neighbours of each cell

    //Contains cells that are born (even if already alive) this iteration
    __m256i alive = _mm256_and_si256(_mm256_cmpeq_epi32(_mm256_add_epi32(sum, cells), _mm256_set1_epi32(3)), one);

    //Contain cells that didn't change this iteration
    __m256i same  = _mm256_and_si256(_mm256_cmpeq_epi32(_mm256_add_epi32(sum, cells), _mm256_set1_epi32(4)), one);
    
    //cell = alive | same&cell
    __m256i result = _mm256_or_si256(alive, _mm256_and_si256(same, cells));
    _mm256_storeu_si256((__m256i_u*)&next_table(y,x),result); //Copy the result vector into memory

    return _mm256_movemask_epi8(_mm256_cmpeq_epi32(cells, result)) != 0xFFFFffffU; //Test for equality, flip the output
}

unsigned life_compute_tiled_omp_for_vec (unsigned nb_iter)
{
    unsigned res = 0;

    for (unsigned it = 1; it <= nb_iter; it++) {
        unsigned change = 0;
        #pragma omp parallel shared(change)
        {
            #pragma omp for collapse(2) schedule(static)
            for(int y=0; y<(DIM-TILE_H); y+=TILE_H){
                for(int x=0; x<(DIM-TILE_W); x+=TILE_W){
                    #pragma omp atomic
                    change |= do_tile_vec(x, y, TILE_W, TILE_H, omp_get_thread_num());
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

unsigned life_compute_omp_for_lazy_vec(unsigned nb_iter)
{
    unsigned res = 0;

    for (unsigned it = 1; it <= nb_iter; it++) {
        #pragma omp parallel
        {
            #pragma omp for collapse(2) schedule(static,1)
            for(int y=0; y<DIM; y+=TILE_H){
                for(int x=0; x<DIM; x+=TILE_W){
                    unsigned tile_x = x >> tile_w_power;
                    unsigned tile_y = y >> tile_h_power;
                    next_change_table(tile_x, tile_y) = tile_needs_update(tile_x, tile_y) && do_tile_vec(x, y, TILE_W, TILE_H, omp_get_thread_num());
                }
            }
        } // omp parallel
        swap_tables ();

        /*if (!change) { // we stop when all cells are stable
            res = it;
            break;
        }*/
    }

    return res;
}



int compute_multiple_cells_opt(int x, int y){
    // Tried to minimize useless memory access
    // Didn't do much
    
    __m256i one    = _mm256_set1_epi32(1);
    __m256i shiftl = _mm256_set_epi32(0,7,6,5,4,3,2,1);
    __m256i shiftr = _mm256_set_epi32(6,5,4,3,2,1,0,7);
     
    __m256i cells   = _mm256_lddqu_si256((__m256i*)&cur_table(y,x)); //Current cells value
    __m256i right   = _mm256_insert_epi32(_mm256_permutevar8x32_epi32(cells, shiftl), cur_table(y,x+8), 7);
    __m256i left    = _mm256_insert_epi32(_mm256_permutevar8x32_epi32(cells, shiftr), cur_table(y,x-1), 0);

    __m256i top        = _mm256_lddqu_si256((__m256i*)&cur_table(y-1, x));
    __m256i topright   = _mm256_insert_epi32(_mm256_permutevar8x32_epi32(top, shiftl), cur_table(y-1,x+8), 7);
    __m256i topleft    = _mm256_insert_epi32(_mm256_permutevar8x32_epi32(top, shiftr), cur_table(y-1,x-1), 0);
    
    __m256i bot        = _mm256_lddqu_si256((__m256i*)&cur_table(y+1, x));
    __m256i botright   = _mm256_insert_epi32(_mm256_permutevar8x32_epi32(bot, shiftl), cur_table(y+1,x+8), 7);
    __m256i botleft    = _mm256_insert_epi32(_mm256_permutevar8x32_epi32(bot, shiftr), cur_table(y+1,x-1), 0);
    
    __m256i sum = _mm256_add_epi32(
                    _mm256_add_epi32(
                        _mm256_add_epi32(top,bot),
                        _mm256_add_epi32(right,left)),
                    _mm256_add_epi32(
                        _mm256_add_epi32(topright,topleft),
                        _mm256_add_epi32(botright,botleft))
                    ); //Number of alive neighbours of each cell

    //Contains cells that are born (even if already alive) this iteration
    __m256i alive = _mm256_and_si256(_mm256_cmpeq_epi32(_mm256_add_epi32(sum, cells), _mm256_set1_epi32(3)), one);

    //Contain cells that didn't change this iteration
    __m256i same  = _mm256_and_si256(_mm256_cmpeq_epi32(_mm256_add_epi32(sum, cells), _mm256_set1_epi32(4)), one);
    
    //cell = alive | same&cell
    __m256i result = _mm256_or_si256(alive, _mm256_and_si256(same, cells));

    _mm256_storeu_si256((__m256i_u*)&next_table(y,x), result); //Write the result vector into memory

    return _mm256_movemask_epi8(_mm256_cmpeq_epi32(cells, result)) != 0xFFFFffffU; //Test for equality, flip the output
}

static int do_tile_vec_opt(unsigned x, unsigned y, unsigned w, unsigned h, unsigned cpu){
    int change = 0;
    monitoring_start_tile (cpu);
    for(int j = 0; j<w; j+=VEC_SIZE_INT){ //Loop columns
        for(int i = 0; i<h; i++){ //Loop lines
            change |= compute_multiple_cells_opt(x+j, y+i);
        }
    }
    monitoring_end_tile (x, y, w, h, cpu);
    return change;
}

unsigned life_compute_vec_opt(unsigned nb_iter){
    unsigned res = 0;
	for (unsigned it = 1; it <= nb_iter; it++) {
		unsigned change = 0;
        for (int y = 0; y < DIM; y += TILE_H)
            for (int x = 0; x < DIM; x += TILE_W){
                change |= do_tile_vec_opt(x, y, TILE_W, TILE_H, omp_get_thread_num());
            }
		swap_tables ();

		if (!change) { // we stop when all cells are stable
			res = it;
			break;
		}
	}

	return res;
}

unsigned life_compute_omp_for_lazy_vec_opt(unsigned nb_iter)
{
    unsigned res = 0;

    for (unsigned it = 1; it <= nb_iter; it++) {
        #pragma omp parallel
        {
            #pragma omp for collapse(2) schedule(static,1)
            for(int y=0; y<DIM; y+=TILE_H){
                for(int x=0; x<DIM; x+=TILE_W){
                    unsigned tile_x = x >> tile_w_power;
                    unsigned tile_y = y >> tile_h_power;
                    next_change_table(tile_x, tile_y) = tile_needs_update(tile_x, tile_y) && do_tile_vec_opt(x, y, TILE_W, TILE_H, omp_get_thread_num());
                }
            }
        } // omp parallel
        swap_tables ();

        /*if (!change) { // we stop when all cells are stable
            res = it;
            break;
        }*/
    }

    return res;
}

#endif
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
