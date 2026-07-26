#ifndef MCSEED_GPU_ABI_H
#define MCSEED_GPU_ABI_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

enum {
    MCSEED_GPU_PLACEMENT_UNSUPPORTED = 0,
    MCSEED_GPU_PLACEMENT_FEATURE = 1,
    MCSEED_GPU_PLACEMENT_LARGE = 2,
    MCSEED_GPU_PLACEMENT_OUTPOST = 3,
    MCSEED_GPU_PLACEMENT_TREASURE = 4,
    MCSEED_GPU_PLACEMENT_FORTRESS = 5,
    MCSEED_GPU_PLACEMENT_BASTION = 6,
    MCSEED_GPU_PLACEMENT_STRONGHOLD = 7,
};

enum {
    MCSEED_GPU_PLACEMENT_END_DISTANCE = 1,
};

enum {
    MCSEED_GPU_SPAWN_UNSUPPORTED = 0,
    MCSEED_GPU_SPAWN_MULTI_NOISE_ORIGIN_BIAS = 1,
};

enum {
    MCSEED_GPU_SPAWN_NOISE_COUNT = 6,
    MCSEED_GPU_SPAWN_TARGET_COUNT = 7,
    MCSEED_GPU_SPAWN_MAX_PERLINS = 48,
    MCSEED_GPU_SPAWN_MAX_OFFSETS = 1024,
};

enum {
    MCSEED_GPU_ANCHOR_ORIGIN = 0,
    MCSEED_GPU_ANCHOR_SPAWN = 1,
    MCSEED_GPU_ANCHOR_NETHER_SPAWN = 2,
    MCSEED_GPU_ANCHOR_COORDINATES = 3,
};

typedef struct McSeedGpuStructureConfig {
    int32_t kind;
    /*
     * Random-spread kinds use salt/region_size/chunk_range normally.
     * STRONGHOLD uses count/distance/spread in the same three fields and
     * stores its conservative per-axis block margin in reserved.
     */
    int32_t salt;
    int32_t region_size;
    int32_t chunk_range;
    int32_t flags;
    int32_t reserved;
} McSeedGpuStructureConfig;

typedef struct McSeedGpuPredicate {
    uint32_t config_offset;
    uint32_t config_count;
    uint32_t radius;
    int32_t anchor_kind;
    int32_t anchor_x;
    int32_t anchor_z;
    uint64_t minimum;
} McSeedGpuPredicate;

typedef struct McSeedGpuPairPredicate {
    uint32_t left_config_offset;
    uint32_t left_config_count;
    uint32_t right_config_offset;
    uint32_t right_config_count;
    uint32_t left_radius;
    uint32_t right_radius;
    uint32_t anchor_radius;
    uint32_t reserved;
} McSeedGpuPairPredicate;

typedef struct McSeedGpuCandidate {
    uint64_t seed;
    int32_t spawn_x;
    int32_t spawn_z;
} McSeedGpuCandidate;

typedef struct McSeedGpuSpawnNoise {
    uint32_t octave_a_offset;
    uint32_t octave_a_count;
    uint32_t octave_b_offset;
    uint32_t octave_b_count;
    double amplitude;
} McSeedGpuSpawnNoise;

typedef struct McSeedGpuSpawnPerlin {
    uint64_t parameter_seed_xor_lo;
    uint64_t parameter_seed_xor_hi;
    uint64_t octave_seed_xor_lo;
    uint64_t octave_seed_xor_hi;
    double amplitude;
    double lacunarity;
    uint32_t half;
    uint32_t reserved;
} McSeedGpuSpawnPerlin;

/**
 * Version-profiled description of the modern overworld spawn estimator.
 * Keeping the data outside the kernel lets future compatible profiles reuse
 * the implementation without matching on a version string in the CLI.
 */
typedef struct McSeedGpuSpawnConfig {
    uint32_t algorithm;
    uint32_t noise_count;
    uint32_t perlin_count;
    uint32_t outer_radius;
    uint32_t outer_step;
    uint32_t inner_radius;
    uint32_t inner_step;
    uint32_t reserved;
    uint64_t fitness_scale;
    McSeedGpuSpawnNoise noises[MCSEED_GPU_SPAWN_NOISE_COUNT];
    McSeedGpuSpawnPerlin perlins[MCSEED_GPU_SPAWN_MAX_PERLINS];
    int64_t targets[MCSEED_GPU_SPAWN_TARGET_COUNT][2];
} McSeedGpuSpawnConfig;

#ifdef __cplusplus
static_assert(sizeof(McSeedGpuStructureConfig) == 24, "GPU config ABI changed");
static_assert(sizeof(McSeedGpuPredicate) == 32, "GPU predicate ABI changed");
static_assert(sizeof(McSeedGpuPairPredicate) == 32, "GPU pair predicate ABI changed");
static_assert(sizeof(McSeedGpuCandidate) == 16, "GPU candidate ABI changed");
static_assert(sizeof(McSeedGpuSpawnNoise) == 24, "GPU spawn noise ABI changed");
static_assert(sizeof(McSeedGpuSpawnPerlin) == 56, "GPU spawn Perlin ABI changed");
static_assert(sizeof(McSeedGpuSpawnConfig) == 2984, "GPU spawn config ABI changed");
#else
_Static_assert(sizeof(McSeedGpuStructureConfig) == 24, "GPU config ABI changed");
_Static_assert(sizeof(McSeedGpuPredicate) == 32, "GPU predicate ABI changed");
_Static_assert(sizeof(McSeedGpuPairPredicate) == 32, "GPU pair predicate ABI changed");
_Static_assert(sizeof(McSeedGpuCandidate) == 16, "GPU candidate ABI changed");
_Static_assert(sizeof(McSeedGpuSpawnNoise) == 24, "GPU spawn noise ABI changed");
_Static_assert(sizeof(McSeedGpuSpawnPerlin) == 56, "GPU spawn Perlin ABI changed");
_Static_assert(sizeof(McSeedGpuSpawnConfig) == 2984, "GPU spawn config ABI changed");
#endif

/** Return 1 and fill config when the active version supports GPU spawn estimation. */
int32_t mcseed_gpu_spawn_config(McSeedGpuSpawnConfig *config);

/** CPU implementation of the exact same conservative placement prefilter. */
void mcseed_gpu_reference_filter(
    const McSeedGpuCandidate *candidates,
    size_t candidate_count,
    const McSeedGpuStructureConfig *configs,
    size_t config_count,
    const McSeedGpuPredicate *predicates,
    size_t predicate_count,
    uint8_t *matches
);

/** CPU implementation of the pre-spawn pairwise necessary-condition filter. */
void mcseed_gpu_reference_pair_filter(
    const McSeedGpuCandidate *candidates,
    size_t candidate_count,
    const McSeedGpuStructureConfig *configs,
    size_t config_count,
    const McSeedGpuPairPredicate *predicates,
    size_t predicate_count,
    uint8_t *matches
);

#ifdef __cplusplus
}
#endif

#endif
