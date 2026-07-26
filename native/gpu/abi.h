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
};

enum {
    MCSEED_GPU_PLACEMENT_END_DISTANCE = 1,
};

enum {
    MCSEED_GPU_ANCHOR_ORIGIN = 0,
    MCSEED_GPU_ANCHOR_SPAWN = 1,
    MCSEED_GPU_ANCHOR_NETHER_SPAWN = 2,
    MCSEED_GPU_ANCHOR_COORDINATES = 3,
};

typedef struct McSeedGpuStructureConfig {
    int32_t kind;
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

typedef struct McSeedGpuCandidate {
    uint64_t seed;
    int32_t spawn_x;
    int32_t spawn_z;
} McSeedGpuCandidate;

#ifdef __cplusplus
static_assert(sizeof(McSeedGpuStructureConfig) == 24, "GPU config ABI changed");
static_assert(sizeof(McSeedGpuPredicate) == 32, "GPU predicate ABI changed");
static_assert(sizeof(McSeedGpuCandidate) == 16, "GPU candidate ABI changed");
#else
_Static_assert(sizeof(McSeedGpuStructureConfig) == 24, "GPU config ABI changed");
_Static_assert(sizeof(McSeedGpuPredicate) == 32, "GPU predicate ABI changed");
_Static_assert(sizeof(McSeedGpuCandidate) == 16, "GPU candidate ABI changed");
#endif

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

#ifdef __cplusplus
}
#endif

#endif
