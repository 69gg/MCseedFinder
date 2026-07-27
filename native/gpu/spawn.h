#ifndef MCSEED_GPU_SPAWN_H
#define MCSEED_GPU_SPAWN_H

#include "abi.h"

#include <math.h>
#include <stdint.h>

#if !defined(__CUDACC__) && !defined(__HIPCC__)
#error "GPU spawn estimation must be compiled by CUDA or HIP"
#endif

#define MCSEED_GPU_DEVICE_INLINE __device__ __forceinline__

enum {
    MCSEED_GPU_SPAWN_BLOCK_THREADS = 256,
    MCSEED_GPU_SPAWN_OUTER_THREADS = 64,
};

typedef struct McSeedGpuSpawnOffset {
    int32_t x;
    int32_t z;
} McSeedGpuSpawnOffset;

typedef struct McSeedGpuXoroshiro {
    uint64_t lo;
    uint64_t hi;
} McSeedGpuXoroshiro;

typedef struct McSeedGpuPerlinNoise {
    uint8_t permutation[257];
    uint8_t y_section;
    double x_offset;
    double y_offset;
    double z_offset;
    double amplitude;
    double lacunarity;
    double y_fraction;
    double y_smooth;
} McSeedGpuPerlinNoise;

static_assert(sizeof(McSeedGpuPerlinNoise) == 320, "GPU Perlin layout changed");

MCSEED_GPU_DEVICE_INLINE uint64_t mcseed_gpu_rotate_left_u64(
    uint64_t value,
    uint32_t bits
)
{
    return (value << bits) | (value >> (64 - bits));
}

MCSEED_GPU_DEVICE_INLINE void mcseed_gpu_xoroshiro_set_seed(
    McSeedGpuXoroshiro *random,
    uint64_t seed
)
{
    const uint64_t golden_ratio = UINT64_C(0x9e3779b97f4a7c15);
    const uint64_t silver_ratio = UINT64_C(0x6a09e667f3bcc909);
    const uint64_t mix_a = UINT64_C(0xbf58476d1ce4e5b9);
    const uint64_t mix_b = UINT64_C(0x94d049bb133111eb);
    uint64_t lo = seed ^ silver_ratio;
    uint64_t hi = lo + golden_ratio;
    lo = (lo ^ (lo >> 30)) * mix_a;
    hi = (hi ^ (hi >> 30)) * mix_a;
    lo = (lo ^ (lo >> 27)) * mix_b;
    hi = (hi ^ (hi >> 27)) * mix_b;
    random->lo = lo ^ (lo >> 31);
    random->hi = hi ^ (hi >> 31);
}

MCSEED_GPU_DEVICE_INLINE uint64_t mcseed_gpu_xoroshiro_next_long(
    McSeedGpuXoroshiro *random
)
{
    uint64_t lo = random->lo;
    uint64_t hi = random->hi;
    uint64_t result = mcseed_gpu_rotate_left_u64(lo + hi, 17) + lo;
    hi ^= lo;
    random->lo = mcseed_gpu_rotate_left_u64(lo, 49) ^ hi ^ (hi << 21);
    random->hi = mcseed_gpu_rotate_left_u64(hi, 28);
    return result;
}

MCSEED_GPU_DEVICE_INLINE uint32_t mcseed_gpu_xoroshiro_next_int(
    McSeedGpuXoroshiro *random,
    uint32_t bound
)
{
    uint64_t product =
        (mcseed_gpu_xoroshiro_next_long(random) & UINT64_C(0xffffffff)) * bound;
    if ((uint32_t)product < bound) {
        const uint32_t threshold = (uint32_t)(~bound + 1U) % bound;
        while ((uint32_t)product < threshold) {
            product =
                (mcseed_gpu_xoroshiro_next_long(random) & UINT64_C(0xffffffff)) * bound;
        }
    }
    return (uint32_t)(product >> 32);
}

MCSEED_GPU_DEVICE_INLINE double mcseed_gpu_xoroshiro_next_double(
    McSeedGpuXoroshiro *random
)
{
    return (mcseed_gpu_xoroshiro_next_long(random) >> 11) *
        1.1102230246251565E-16;
}

MCSEED_GPU_DEVICE_INLINE void mcseed_gpu_initialize_perlin(
    McSeedGpuPerlinNoise *noise,
    const McSeedGpuSpawnPerlin *specification,
    uint64_t world_seed
)
{
    McSeedGpuXoroshiro world_random;
    McSeedGpuXoroshiro parameter_random;
    McSeedGpuXoroshiro perlin_random;
    uint64_t world_lo;
    uint64_t world_hi;
    uint64_t perlin_lo;
    uint64_t perlin_hi;
    uint32_t index;

    mcseed_gpu_xoroshiro_set_seed(&world_random, world_seed);
    world_lo = mcseed_gpu_xoroshiro_next_long(&world_random);
    world_hi = mcseed_gpu_xoroshiro_next_long(&world_random);
    parameter_random.lo = world_lo ^ specification->parameter_seed_xor_lo;
    parameter_random.hi = world_hi ^ specification->parameter_seed_xor_hi;
    perlin_lo = mcseed_gpu_xoroshiro_next_long(&parameter_random);
    perlin_hi = mcseed_gpu_xoroshiro_next_long(&parameter_random);
    if (specification->half != 0) {
        perlin_lo = mcseed_gpu_xoroshiro_next_long(&parameter_random);
        perlin_hi = mcseed_gpu_xoroshiro_next_long(&parameter_random);
    }
    perlin_random.lo = perlin_lo ^ specification->octave_seed_xor_lo;
    perlin_random.hi = perlin_hi ^ specification->octave_seed_xor_hi;

    noise->x_offset = mcseed_gpu_xoroshiro_next_double(&perlin_random) * 256.0;
    noise->y_offset = mcseed_gpu_xoroshiro_next_double(&perlin_random) * 256.0;
    noise->z_offset = mcseed_gpu_xoroshiro_next_double(&perlin_random) * 256.0;
    noise->amplitude = specification->amplitude;
    noise->lacunarity = specification->lacunarity;
    for (index = 0; index < 256; index++)
        noise->permutation[index] = (uint8_t)index;
    for (index = 0; index < 256; index++) {
        const uint32_t swap_index = index +
            mcseed_gpu_xoroshiro_next_int(&perlin_random, 256 - index);
        const uint8_t value = noise->permutation[index];
        noise->permutation[index] = noise->permutation[swap_index];
        noise->permutation[swap_index] = value;
    }
    noise->permutation[256] = noise->permutation[0];
    {
        const double section = floor(noise->y_offset);
        const double fraction = noise->y_offset - section;
        noise->y_section = (uint8_t)(int32_t)section;
        noise->y_fraction = fraction;
        noise->y_smooth = fraction * fraction * fraction *
            (fraction * (fraction * 6.0 - 15.0) + 10.0);
    }
}

MCSEED_GPU_DEVICE_INLINE double mcseed_gpu_gradient(
    uint8_t index,
    double x,
    double y,
    double z
)
{
    switch (index & 15) {
    case 0: return x + y;
    case 1: return -x + y;
    case 2: return x - y;
    case 3: return -x - y;
    case 4: return x + z;
    case 5: return -x + z;
    case 6: return x - z;
    case 7: return -x - z;
    case 8: return y + z;
    case 9: return -y + z;
    case 10: return y - z;
    case 11: return -y - z;
    case 12: return x + y;
    case 13: return -y + z;
    case 14: return -x + y;
    default: return -y - z;
    }
}

MCSEED_GPU_DEVICE_INLINE double mcseed_gpu_lerp(
    double part,
    double from,
    double to
)
{
    return from + part * (to - from);
}

MCSEED_GPU_DEVICE_INLINE double mcseed_gpu_sample_perlin(
    const McSeedGpuPerlinNoise *noise,
    double x,
    double y,
    double z
)
{
    uint8_t x_section;
    uint8_t y_section;
    uint8_t z_section;
    double x_smooth;
    double y_smooth;
    double z_smooth;
    const uint8_t *permutation = noise->permutation;
    uint8_t x0;
    uint8_t x1;
    uint8_t xy00;
    uint8_t xy01;
    uint8_t xy10;
    uint8_t xy11;
    double l1;
    double l2;
    double l3;
    double l4;
    double l5;
    double l6;
    double l7;
    double l8;

    if (y == 0.0) {
        y = noise->y_fraction;
        y_section = noise->y_section;
        y_smooth = noise->y_smooth;
    } else {
        const double section = floor(y + noise->y_offset);
        y = y + noise->y_offset - section;
        y_section = (uint8_t)(int32_t)section;
        y_smooth = y * y * y * (y * (y * 6.0 - 15.0) + 10.0);
    }

    x += noise->x_offset;
    z += noise->z_offset;
    {
        const double section = floor(x);
        x -= section;
        x_section = (uint8_t)(int32_t)section;
    }
    {
        const double section = floor(z);
        z -= section;
        z_section = (uint8_t)(int32_t)section;
    }
    x_smooth = x * x * x * (x * (x * 6.0 - 15.0) + 10.0);
    z_smooth = z * z * z * (z * (z * 6.0 - 15.0) + 10.0);

    x0 = permutation[x_section];
    x1 = permutation[(uint32_t)x_section + 1];
    x0 = (uint8_t)(x0 + y_section);
    x1 = (uint8_t)(x1 + y_section);
    xy00 = permutation[x0];
    xy01 = permutation[(uint32_t)x0 + 1];
    xy10 = permutation[x1];
    xy11 = permutation[(uint32_t)x1 + 1];
    xy00 = (uint8_t)(xy00 + z_section);
    xy01 = (uint8_t)(xy01 + z_section);
    xy10 = (uint8_t)(xy10 + z_section);
    xy11 = (uint8_t)(xy11 + z_section);

    l1 = mcseed_gpu_gradient(permutation[xy00], x, y, z);
    l5 = mcseed_gpu_gradient(permutation[(uint32_t)xy00 + 1], x, y, z - 1.0);
    l2 = mcseed_gpu_gradient(permutation[xy10], x - 1.0, y, z);
    l6 = mcseed_gpu_gradient(
        permutation[(uint32_t)xy10 + 1], x - 1.0, y, z - 1.0
    );
    l3 = mcseed_gpu_gradient(permutation[xy01], x, y - 1.0, z);
    l7 = mcseed_gpu_gradient(
        permutation[(uint32_t)xy01 + 1], x, y - 1.0, z - 1.0
    );
    l4 = mcseed_gpu_gradient(permutation[xy11], x - 1.0, y - 1.0, z);
    l8 = mcseed_gpu_gradient(
        permutation[(uint32_t)xy11 + 1], x - 1.0, y - 1.0, z - 1.0
    );
    l1 = mcseed_gpu_lerp(x_smooth, l1, l2);
    l3 = mcseed_gpu_lerp(x_smooth, l3, l4);
    l5 = mcseed_gpu_lerp(x_smooth, l5, l6);
    l7 = mcseed_gpu_lerp(x_smooth, l7, l8);
    l1 = mcseed_gpu_lerp(y_smooth, l1, l3);
    l5 = mcseed_gpu_lerp(y_smooth, l5, l7);
    return mcseed_gpu_lerp(z_smooth, l1, l5);
}

MCSEED_GPU_DEVICE_INLINE double mcseed_gpu_sample_octaves(
    const McSeedGpuPerlinNoise *perlins,
    uint32_t offset,
    uint32_t count,
    double x,
    double y,
    double z
)
{
    double value = 0.0;
    uint32_t index;
    for (index = 0; index < count; index++) {
        const McSeedGpuPerlinNoise *perlin = &perlins[offset + index];
        const double lacunarity = perlin->lacunarity;
        const double sampled = mcseed_gpu_sample_perlin(
            perlin,
            x * lacunarity,
            y * lacunarity,
            z * lacunarity
        );
        value += perlin->amplitude * sampled;
    }
    return value;
}

MCSEED_GPU_DEVICE_INLINE double mcseed_gpu_sample_double_perlin(
    const McSeedGpuSpawnNoise *noise,
    const McSeedGpuPerlinNoise *perlins,
    double x,
    double y,
    double z
)
{
    const double second_scale = 337.0 / 331.0;
    double value = mcseed_gpu_sample_octaves(
        perlins,
        noise->octave_a_offset,
        noise->octave_a_count,
        x,
        y,
        z
    );
    value += mcseed_gpu_sample_octaves(
        perlins,
        noise->octave_b_offset,
        noise->octave_b_count,
        x * second_scale,
        y * second_scale,
        z * second_scale
    );
    return value * noise->amplitude;
}

MCSEED_GPU_DEVICE_INLINE uint64_t mcseed_gpu_spawn_parameter_distance(
    int64_t value,
    const int64_t range[2]
)
{
    if (value > range[1])
        return (uint64_t)(value - range[1]);
    if (value < range[0])
        return (uint64_t)(range[0] - value);
    return 0;
}

MCSEED_GPU_DEVICE_INLINE uint64_t mcseed_gpu_spawn_origin_bias(
    const McSeedGpuSpawnConfig *config,
    uint64_t distance,
    int32_t x,
    int32_t z
)
{
    const uint64_t x_squared = (uint64_t)((int64_t)x * x);
    const uint64_t z_squared = (uint64_t)((int64_t)z * z);
    return distance * config->fitness_scale + x_squared + z_squared;
}

typedef struct McSeedGpuSpawnSample {
    double shifted_x;
    double shifted_z;
    int64_t continentalness;
    int64_t weirdness;
} McSeedGpuSpawnSample;

MCSEED_GPU_DEVICE_INLINE void mcseed_gpu_sample_spawn_shift(
    const McSeedGpuSpawnConfig *config,
    const McSeedGpuPerlinNoise *perlins,
    int32_t quart_x,
    int32_t quart_z,
    McSeedGpuSpawnSample *sample
)
{
    const McSeedGpuSpawnNoise *shift = &config->noises[0];
    sample->shifted_x = quart_x +
        mcseed_gpu_sample_double_perlin(shift, perlins, quart_x, 0.0, quart_z) * 4.0;
    sample->shifted_z = quart_z +
        mcseed_gpu_sample_double_perlin(shift, perlins, quart_z, quart_x, 0.0) * 4.0;
}

MCSEED_GPU_DEVICE_INLINE int64_t mcseed_gpu_scaled_spawn_noise(
    const McSeedGpuSpawnNoise *noise,
    const McSeedGpuPerlinNoise *perlins,
    double x,
    double z
)
{
    const float value = (float)mcseed_gpu_sample_double_perlin(
        noise, perlins, x, 0.0, z
    );
    return (int64_t)(10000.0F * value);
}

MCSEED_GPU_DEVICE_INLINE uint64_t mcseed_gpu_spawn_weirdness_distance(
    const McSeedGpuSpawnConfig *config,
    int64_t weirdness
)
{
    uint64_t first = mcseed_gpu_spawn_parameter_distance(
        weirdness, config->targets[5]
    );
    uint64_t second = mcseed_gpu_spawn_parameter_distance(
        weirdness, config->targets[6]
    );
    first *= first;
    second *= second;
    return first < second ? first : second;
}

MCSEED_GPU_DEVICE_INLINE uint64_t mcseed_gpu_spawn_fitness_from_sample(
    const McSeedGpuSpawnConfig *config,
    const McSeedGpuPerlinNoise *perlins,
    int32_t x,
    int32_t z,
    const McSeedGpuSpawnSample *sample
)
{
    int64_t parameters[5];
    uint64_t distance = 0;
    uint32_t index;
    parameters[0] = mcseed_gpu_scaled_spawn_noise(
        &config->noises[1], perlins, sample->shifted_x, sample->shifted_z
    );
    parameters[1] = mcseed_gpu_scaled_spawn_noise(
        &config->noises[2], perlins, sample->shifted_x, sample->shifted_z
    );
    parameters[2] = sample->continentalness;
    parameters[3] = mcseed_gpu_scaled_spawn_noise(
        &config->noises[4], perlins, sample->shifted_x, sample->shifted_z
    );
    parameters[4] = 0;
    for (index = 0; index < 5; index++) {
        const uint64_t delta = mcseed_gpu_spawn_parameter_distance(
            parameters[index], config->targets[index]
        );
        distance += delta * delta;
    }
    distance += mcseed_gpu_spawn_weirdness_distance(config, sample->weirdness);
    return mcseed_gpu_spawn_origin_bias(config, distance, x, z);
}

MCSEED_GPU_DEVICE_INLINE uint64_t mcseed_gpu_spawn_fitness(
    const McSeedGpuSpawnConfig *config,
    const McSeedGpuPerlinNoise *perlins,
    int32_t x,
    int32_t z,
    uint64_t threshold
)
{
    McSeedGpuSpawnSample sample;
    uint64_t lower_bound = mcseed_gpu_spawn_origin_bias(config, 0, x, z);
    uint64_t distance;
    uint64_t delta;
    if (lower_bound >= threshold)
        return UINT64_MAX;

    mcseed_gpu_sample_spawn_shift(config, perlins, x >> 2, z >> 2, &sample);
    sample.weirdness = mcseed_gpu_scaled_spawn_noise(
        &config->noises[5], perlins, sample.shifted_x, sample.shifted_z
    );
    distance = mcseed_gpu_spawn_weirdness_distance(config, sample.weirdness);
    lower_bound = mcseed_gpu_spawn_origin_bias(config, distance, x, z);
    if (lower_bound >= threshold)
        return UINT64_MAX;

    sample.continentalness = mcseed_gpu_scaled_spawn_noise(
        &config->noises[3], perlins, sample.shifted_x, sample.shifted_z
    );
    delta = mcseed_gpu_spawn_parameter_distance(
        sample.continentalness, config->targets[2]
    );
    distance += delta * delta;
    lower_bound = mcseed_gpu_spawn_origin_bias(config, distance, x, z);
    if (lower_bound >= threshold)
        return UINT64_MAX;
    return mcseed_gpu_spawn_fitness_from_sample(config, perlins, x, z, &sample);
}

MCSEED_GPU_DEVICE_INLINE int mcseed_gpu_spawn_result_is_better(
    uint64_t candidate_fitness,
    uint32_t candidate_rank,
    uint64_t current_fitness,
    uint32_t current_rank
)
{
    return candidate_fitness < current_fitness ||
        (candidate_fitness == current_fitness && candidate_rank < current_rank);
}

MCSEED_GPU_DEVICE_INLINE void mcseed_gpu_search_spawn_offsets(
    const McSeedGpuSpawnConfig *config,
    const McSeedGpuPerlinNoise *perlins,
    const McSeedGpuSpawnOffset *offsets,
    uint32_t offset_count,
    int32_t origin_x,
    int32_t origin_z,
    uint64_t base_fitness,
    uint64_t *fitness_by_thread,
    uint32_t *rank_by_thread
)
{
    const uint32_t thread = threadIdx.x;
    uint64_t best_fitness = base_fitness;
    uint32_t best_rank = 0;
    uint32_t tile;
    for (tile = 0; tile < offset_count; tile += blockDim.x) {
        const uint32_t index = tile + thread;
        if (index < offset_count) {
            const int32_t x = origin_x + offsets[index].x;
            const int32_t z = origin_z + offsets[index].z;
            const uint64_t fitness = mcseed_gpu_spawn_fitness(
                config, perlins, x, z, best_fitness
            );
            const uint32_t rank = index + 1;
            if (mcseed_gpu_spawn_result_is_better(
                    fitness, rank, best_fitness, best_rank
                )) {
                best_fitness = fitness;
                best_rank = rank;
            }
        }
        fitness_by_thread[thread] = best_fitness;
        rank_by_thread[thread] = best_rank;
        __syncthreads();
        for (uint32_t stride = blockDim.x >> 1; stride != 0; stride >>= 1) {
            if (thread < stride && mcseed_gpu_spawn_result_is_better(
                    fitness_by_thread[thread + stride],
                    rank_by_thread[thread + stride],
                    fitness_by_thread[thread],
                    rank_by_thread[thread]
                )) {
                fitness_by_thread[thread] = fitness_by_thread[thread + stride];
                rank_by_thread[thread] = rank_by_thread[thread + stride];
            }
            __syncthreads();
        }
        best_fitness = fitness_by_thread[0];
        best_rank = rank_by_thread[0];
        __syncthreads();
    }
}

enum {
    MCSEED_GPU_SPAWN_STAGE_FULL = 0,
    MCSEED_GPU_SPAWN_STAGE_OUTER = 1,
    MCSEED_GPU_SPAWN_STAGE_INNER = 2,
};

__global__ static __launch_bounds__(MCSEED_GPU_SPAWN_BLOCK_THREADS)
void mcseed_gpu_estimate_spawn_kernel(
    McSeedGpuCandidate *candidates,
    size_t candidate_count,
    const McSeedGpuSpawnConfig *config,
    const McSeedGpuSpawnOffset *outer_offsets,
    uint32_t outer_count,
    const McSeedGpuSpawnOffset *inner_offsets,
    uint32_t inner_count,
    uint32_t stage,
    const uint32_t *state_indices,
    McSeedGpuPerlinNoise *spawn_states,
    size_t spawn_state_count
)
{
    __shared__ McSeedGpuPerlinNoise perlins[MCSEED_GPU_SPAWN_MAX_PERLINS];
    __shared__ uint64_t fitness_by_thread[MCSEED_GPU_SPAWN_BLOCK_THREADS];
    __shared__ uint32_t rank_by_thread[MCSEED_GPU_SPAWN_BLOCK_THREADS];
    __shared__ uint64_t best_fitness;
    __shared__ int32_t origin_x;
    __shared__ int32_t origin_z;
    __shared__ uint32_t cached_state;
    const size_t candidate_index = blockIdx.x;
    const uint32_t thread = threadIdx.x;
    McSeedGpuCandidate *candidate;

    if (candidate_index >= candidate_count)
        return;
    candidate = &candidates[candidate_index];
    if (thread == 0) {
        cached_state = stage == MCSEED_GPU_SPAWN_STAGE_INNER &&
            state_indices && spawn_states &&
            state_indices[candidate_index] < spawn_state_count;
    }
    __syncthreads();
    if (cached_state) {
        const uint32_t state_index = state_indices[candidate_index];
        const uint64_t *source = (const uint64_t *)(
            spawn_states + (size_t)state_index * config->perlin_count
        );
        uint64_t *destination = (uint64_t *)perlins;
        const size_t words =
            (size_t)config->perlin_count * sizeof(*perlins) / sizeof(*source);
        size_t index;
        for (index = thread; index < words; index += blockDim.x)
            destination[index] = source[index];
    } else {
        uint32_t index;
        for (index = thread; index < config->perlin_count; index += blockDim.x) {
            mcseed_gpu_initialize_perlin(
                &perlins[index], &config->perlins[index], candidate->seed
            );
        }
    }
    __syncthreads();

    if (thread == 0) {
        if (stage == MCSEED_GPU_SPAWN_STAGE_INNER) {
            origin_x = candidate->spawn_x;
            origin_z = candidate->spawn_z;
        } else {
            origin_x = 0;
            origin_z = 0;
        }
        best_fitness = mcseed_gpu_spawn_fitness(
            config, perlins, origin_x, origin_z, UINT64_MAX
        );
    }
    __syncthreads();

    if (stage != MCSEED_GPU_SPAWN_STAGE_INNER) {
        mcseed_gpu_search_spawn_offsets(
            config,
            perlins,
            outer_offsets,
            outer_count,
            origin_x,
            origin_z,
            best_fitness,
            fitness_by_thread,
            rank_by_thread
        );
        if (thread == 0) {
            const uint32_t rank = rank_by_thread[0];
            best_fitness = fitness_by_thread[0];
            if (rank != 0) {
                origin_x = outer_offsets[rank - 1].x;
                origin_z = outer_offsets[rank - 1].z;
            }
        }
        __syncthreads();
    }

    if (stage == MCSEED_GPU_SPAWN_STAGE_OUTER && spawn_states &&
        candidate_index < spawn_state_count) {
        const uint64_t *source = (const uint64_t *)perlins;
        uint64_t *destination = (uint64_t *)(
            spawn_states + candidate_index * config->perlin_count
        );
        const size_t words =
            (size_t)config->perlin_count * sizeof(*perlins) / sizeof(*source);
        size_t index;
        for (index = thread; index < words; index += blockDim.x)
            destination[index] = source[index];
        __syncthreads();
    }

    if (stage != MCSEED_GPU_SPAWN_STAGE_OUTER) {
        mcseed_gpu_search_spawn_offsets(
            config,
            perlins,
            inner_offsets,
            inner_count,
            origin_x,
            origin_z,
            best_fitness,
            fitness_by_thread,
            rank_by_thread
        );
        if (thread == 0) {
            const uint32_t rank = rank_by_thread[0];
            if (rank != 0) {
                origin_x += inner_offsets[rank - 1].x;
                origin_z += inner_offsets[rank - 1].z;
            }
            candidate->spawn_x = (origin_x & ~15) + 8;
            candidate->spawn_z = (origin_z & ~15) + 8;
        }
    } else if (thread == 0) {
        /* Preserve the exact outer winner so the inner stage can resume without
         * repeating the outer scan. It is also a valid conservative anchor. */
        candidate->spawn_x = origin_x;
        candidate->spawn_z = origin_z;
    }
}

__global__ static __launch_bounds__(MCSEED_GPU_SPAWN_OUTER_THREADS)
void mcseed_gpu_estimate_spawn_outer_kernel(
    McSeedGpuCandidate *candidates,
    size_t candidate_count,
    const McSeedGpuSpawnConfig *config,
    const McSeedGpuSpawnOffset *outer_offsets,
    uint32_t outer_count,
    McSeedGpuPerlinNoise *spawn_states,
    size_t spawn_state_count
)
{
    __shared__ McSeedGpuPerlinNoise perlins[MCSEED_GPU_SPAWN_MAX_PERLINS];
    __shared__ uint64_t fitness_by_thread[MCSEED_GPU_SPAWN_OUTER_THREADS];
    __shared__ uint32_t rank_by_thread[MCSEED_GPU_SPAWN_OUTER_THREADS];
    __shared__ uint64_t best_fitness;
    __shared__ int32_t origin_x;
    __shared__ int32_t origin_z;
    const size_t candidate_index = blockIdx.x;
    const uint32_t thread = threadIdx.x;
    McSeedGpuCandidate *candidate;
    uint32_t index;

    if (candidate_index >= candidate_count)
        return;
    candidate = &candidates[candidate_index];
    for (index = thread; index < config->perlin_count; index += blockDim.x) {
        mcseed_gpu_initialize_perlin(
            &perlins[index], &config->perlins[index], candidate->seed
        );
    }
    __syncthreads();

    if (thread == 0) {
        origin_x = 0;
        origin_z = 0;
        best_fitness = mcseed_gpu_spawn_fitness(
            config, perlins, 0, 0, UINT64_MAX
        );
    }
    __syncthreads();
    mcseed_gpu_search_spawn_offsets(
        config,
        perlins,
        outer_offsets,
        outer_count,
        origin_x,
        origin_z,
        best_fitness,
        fitness_by_thread,
        rank_by_thread
    );
    if (thread == 0) {
        const uint32_t rank = rank_by_thread[0];
        if (rank != 0) {
            origin_x = outer_offsets[rank - 1].x;
            origin_z = outer_offsets[rank - 1].z;
        }
    }
    __syncthreads();

    if (spawn_states && candidate_index < spawn_state_count) {
        const uint64_t *source = (const uint64_t *)perlins;
        uint64_t *destination = (uint64_t *)(
            spawn_states + candidate_index * config->perlin_count
        );
        const size_t words =
            (size_t)config->perlin_count * sizeof(*perlins) / sizeof(*source);
        size_t word;
        for (word = thread; word < words; word += blockDim.x)
            destination[word] = source[word];
    }
    if (thread == 0) {
        candidate->spawn_x = origin_x;
        candidate->spawn_z = origin_z;
    }
}

#endif
