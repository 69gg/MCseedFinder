#ifndef MCSEED_GPU_PLACEMENT_H
#define MCSEED_GPU_PLACEMENT_H

#include "abi.h"

#include <limits.h>
#include <stdint.h>

#if defined(__CUDACC__) || defined(__HIPCC__)
#define MCSEED_GPU_INLINE __host__ __device__ __forceinline__
#else
#define MCSEED_GPU_INLINE static inline
#endif

typedef struct McSeedGpuPosition {
    int32_t x;
    int32_t z;
} McSeedGpuPosition;

MCSEED_GPU_INLINE int64_t mcseed_gpu_floor_div_i64(int64_t value, int64_t divisor)
{
    int64_t quotient = value / divisor;
    int64_t remainder = value % divisor;
    return quotient - (remainder < 0);
}

MCSEED_GPU_INLINE uint64_t mcseed_gpu_java_set_seed(uint64_t value)
{
    return (value ^ UINT64_C(0x5deece66d)) & ((UINT64_C(1) << 48) - 1);
}

MCSEED_GPU_INLINE int32_t mcseed_gpu_java_next(uint64_t *seed, int bits)
{
    *seed = (*seed * UINT64_C(0x5deece66d) + UINT64_C(0xb)) &
        ((UINT64_C(1) << 48) - 1);
    return (int32_t)(*seed >> (48 - bits));
}

MCSEED_GPU_INLINE int32_t mcseed_gpu_java_next_int(uint64_t *seed, int32_t bound)
{
    int32_t bits;
    int32_t value;
    int32_t mask = bound - 1;
    if ((mask & bound) == 0) {
        uint64_t product = (uint64_t)(uint32_t)bound *
            (uint64_t)(uint32_t)mcseed_gpu_java_next(seed, 31);
        return (int32_t)(product >> 31);
    }
    do {
        bits = mcseed_gpu_java_next(seed, 31);
        value = bits % bound;
    } while ((int32_t)((uint32_t)bits - (uint32_t)value + (uint32_t)mask) < 0);
    return value;
}

MCSEED_GPU_INLINE uint64_t mcseed_gpu_java_next_long(uint64_t *seed)
{
    uint64_t high = (uint64_t)(int64_t)mcseed_gpu_java_next(seed, 32);
    uint64_t low = (uint64_t)(int64_t)mcseed_gpu_java_next(seed, 32);
    return (high << 32) + low;
}

MCSEED_GPU_INLINE McSeedGpuPosition mcseed_gpu_feature_position(
    const McSeedGpuStructureConfig *config,
    uint64_t world_seed,
    int32_t region_x,
    int32_t region_z
)
{
    const uint64_t multiplier = UINT64_C(0x5deece66d);
    const uint64_t mask = (UINT64_C(1) << 48) - 1;
    uint64_t seed = world_seed + (uint64_t)(int64_t)region_x * UINT64_C(341873128712) +
        (uint64_t)(int64_t)region_z * UINT64_C(132897987541) +
        (uint64_t)(int64_t)config->salt;
    uint64_t range = (uint64_t)(uint32_t)config->chunk_range;
    uint64_t offset_x;
    uint64_t offset_z;
    McSeedGpuPosition position;

    seed ^= multiplier;
    seed = (seed * multiplier + UINT64_C(0xb)) & mask;
    if ((range & (range - 1)) != 0) {
        offset_x = (seed >> 17) % range;
        seed = (seed * multiplier + UINT64_C(0xb)) & mask;
        offset_z = (seed >> 17) % range;
    } else {
        offset_x = (range * (seed >> 17)) >> 31;
        seed = (seed * multiplier + UINT64_C(0xb)) & mask;
        offset_z = (range * (seed >> 17)) >> 31;
    }

    position.x = (int32_t)(((uint64_t)(int64_t)region_x *
        (uint64_t)(uint32_t)config->region_size + offset_x) << 4);
    position.z = (int32_t)(((uint64_t)(int64_t)region_z *
        (uint64_t)(uint32_t)config->region_size + offset_z) << 4);
    return position;
}

MCSEED_GPU_INLINE McSeedGpuPosition mcseed_gpu_large_position(
    const McSeedGpuStructureConfig *config,
    uint64_t world_seed,
    int32_t region_x,
    int32_t region_z
)
{
    const uint64_t multiplier = UINT64_C(0x5deece66d);
    const uint64_t mask = (UINT64_C(1) << 48) - 1;
    uint64_t seed = world_seed + (uint64_t)(int64_t)region_x * UINT64_C(341873128712) +
        (uint64_t)(int64_t)region_z * UINT64_C(132897987541) +
        (uint64_t)(int64_t)config->salt;
    uint64_t range = (uint64_t)(uint32_t)config->chunk_range;
    uint64_t offset_x;
    uint64_t offset_z;
    McSeedGpuPosition position;

    seed ^= multiplier;
    seed = (seed * multiplier + UINT64_C(0xb)) & mask;
    offset_x = (seed >> 17) % range;
    seed = (seed * multiplier + UINT64_C(0xb)) & mask;
    offset_x += (seed >> 17) % range;
    seed = (seed * multiplier + UINT64_C(0xb)) & mask;
    offset_z = (seed >> 17) % range;
    seed = (seed * multiplier + UINT64_C(0xb)) & mask;
    offset_z += (seed >> 17) % range;
    offset_x >>= 1;
    offset_z >>= 1;

    position.x = (int32_t)(((uint64_t)(int64_t)region_x *
        (uint64_t)(uint32_t)config->region_size + offset_x) << 4);
    position.z = (int32_t)(((uint64_t)(int64_t)region_z *
        (uint64_t)(uint32_t)config->region_size + offset_z) << 4);
    return position;
}

MCSEED_GPU_INLINE int mcseed_gpu_outpost_selected(
    uint64_t world_seed,
    McSeedGpuPosition position
)
{
    int32_t chunk_x = position.x >> 4;
    int32_t chunk_z = position.z >> 4;
    world_seed ^= (uint64_t)(int64_t)(chunk_x >> 4) ^
        ((uint64_t)(int64_t)(chunk_z >> 4) << 4);
    world_seed = mcseed_gpu_java_set_seed(world_seed);
    (void)mcseed_gpu_java_next(&world_seed, 31);
    return mcseed_gpu_java_next_int(&world_seed, 5) == 0;
}

MCSEED_GPU_INLINE int mcseed_gpu_bastion_selected(
    uint64_t world_seed,
    McSeedGpuPosition position
)
{
    uint64_t random = mcseed_gpu_java_set_seed(world_seed);
    uint64_t a = mcseed_gpu_java_next_long(&random);
    uint64_t b = mcseed_gpu_java_next_long(&random);
    uint64_t mixed = (a * (uint64_t)(int64_t)(position.x >> 4)) ^
        (b * (uint64_t)(int64_t)(position.z >> 4)) ^ world_seed;
    random = mcseed_gpu_java_set_seed(mixed);
    return mcseed_gpu_java_next_int(&random, 5) >= 2;
}

MCSEED_GPU_INLINE int mcseed_gpu_treasure_position(
    const McSeedGpuStructureConfig *config,
    uint64_t world_seed,
    int32_t region_x,
    int32_t region_z,
    McSeedGpuPosition *position
)
{
    uint64_t seed = (uint64_t)(int64_t)region_x * UINT64_C(341873128712) +
        (uint64_t)(int64_t)region_z * UINT64_C(132897987541) + world_seed +
        (uint64_t)(int64_t)config->salt;
    seed = mcseed_gpu_java_set_seed(seed);
    position->x = (int32_t)((int64_t)region_x * 16 + 9);
    position->z = (int32_t)((int64_t)region_z * 16 + 9);
    return mcseed_gpu_java_next(&seed, 24) <= 167772;
}

MCSEED_GPU_INLINE int mcseed_gpu_position_for_region(
    const McSeedGpuStructureConfig *config,
    uint64_t world_seed,
    int32_t region_x,
    int32_t region_z,
    McSeedGpuPosition *position
)
{
    switch (config->kind) {
    case MCSEED_GPU_PLACEMENT_FEATURE:
    case MCSEED_GPU_PLACEMENT_FORTRESS:
        *position = mcseed_gpu_feature_position(config, world_seed, region_x, region_z);
        return 1;
    case MCSEED_GPU_PLACEMENT_LARGE:
        *position = mcseed_gpu_large_position(config, world_seed, region_x, region_z);
        if ((config->flags & MCSEED_GPU_PLACEMENT_END_DISTANCE) != 0 &&
            position->x > -1008 && position->x < 1008 &&
            position->z > -1008 && position->z < 1008) {
            int64_t x = position->x;
            int64_t z = position->z;
            if (x * x + z * z < INT64_C(1008) * 1008)
                return 0;
        }
        return 1;
    case MCSEED_GPU_PLACEMENT_OUTPOST:
        *position = mcseed_gpu_feature_position(config, world_seed, region_x, region_z);
        return mcseed_gpu_outpost_selected(world_seed, *position);
    case MCSEED_GPU_PLACEMENT_TREASURE:
        return mcseed_gpu_treasure_position(
            config, world_seed, region_x, region_z, position
        );
    case MCSEED_GPU_PLACEMENT_BASTION:
        *position = mcseed_gpu_feature_position(config, world_seed, region_x, region_z);
        return mcseed_gpu_bastion_selected(world_seed, *position);
    default:
        return 0;
    }
}

MCSEED_GPU_INLINE int mcseed_gpu_within_radius(
    McSeedGpuPosition position,
    int32_t anchor_x,
    int32_t anchor_z,
    uint32_t radius
)
{
    int64_t dx = (int64_t)position.x - anchor_x;
    int64_t dz = (int64_t)position.z - anchor_z;
    uint64_t absolute_x = dx < 0 ? (uint64_t)(-dx) : (uint64_t)dx;
    uint64_t absolute_z = dz < 0 ? (uint64_t)(-dz) : (uint64_t)dz;
    uint64_t radius_squared = (uint64_t)radius * radius;
    if (absolute_x > radius || absolute_z > radius)
        return 0;
    return absolute_z * absolute_z <= radius_squared - absolute_x * absolute_x;
}

MCSEED_GPU_INLINE void mcseed_gpu_resolve_anchor(
    const McSeedGpuCandidate *candidate,
    const McSeedGpuPredicate *predicate,
    int32_t *anchor_x,
    int32_t *anchor_z
)
{
    switch (predicate->anchor_kind) {
    case MCSEED_GPU_ANCHOR_SPAWN:
        *anchor_x = candidate->spawn_x;
        *anchor_z = candidate->spawn_z;
        break;
    case MCSEED_GPU_ANCHOR_NETHER_SPAWN:
        *anchor_x = (int32_t)mcseed_gpu_floor_div_i64(candidate->spawn_x, 8);
        *anchor_z = (int32_t)mcseed_gpu_floor_div_i64(candidate->spawn_z, 8);
        break;
    case MCSEED_GPU_ANCHOR_COORDINATES:
        *anchor_x = predicate->anchor_x;
        *anchor_z = predicate->anchor_z;
        break;
    default:
        *anchor_x = 0;
        *anchor_z = 0;
        break;
    }
}

MCSEED_GPU_INLINE int mcseed_gpu_predicate_matches(
    const McSeedGpuCandidate *candidate,
    const McSeedGpuStructureConfig *configs,
    const McSeedGpuPredicate *predicate
)
{
    int32_t anchor_x;
    int32_t anchor_z;
    uint64_t found = 0;
    uint32_t config_index;
    mcseed_gpu_resolve_anchor(candidate, predicate, &anchor_x, &anchor_z);

    for (config_index = 0; config_index < predicate->config_count; config_index++) {
        const McSeedGpuStructureConfig *config =
            &configs[predicate->config_offset + config_index];
        int64_t span = (int64_t)config->region_size * 16;
        int64_t region_x_min = mcseed_gpu_floor_div_i64(
            (int64_t)anchor_x - predicate->radius, span
        ) - 1;
        int64_t region_x_max = mcseed_gpu_floor_div_i64(
            (int64_t)anchor_x + predicate->radius, span
        ) + 1;
        int64_t region_z_min = mcseed_gpu_floor_div_i64(
            (int64_t)anchor_z - predicate->radius, span
        ) - 1;
        int64_t region_z_max = mcseed_gpu_floor_div_i64(
            (int64_t)anchor_z + predicate->radius, span
        ) + 1;
        int64_t region_z;

        for (region_z = region_z_min; region_z <= region_z_max; region_z++) {
            int64_t region_x;
            if (region_z < INT32_MIN || region_z > INT32_MAX)
                continue;
            for (region_x = region_x_min; region_x <= region_x_max; region_x++) {
                McSeedGpuPosition position;
                if (region_x < INT32_MIN || region_x > INT32_MAX)
                    continue;
                if (!mcseed_gpu_position_for_region(
                        config,
                        candidate->seed,
                        (int32_t)region_x,
                        (int32_t)region_z,
                        &position
                    ))
                    continue;
                if (!mcseed_gpu_within_radius(
                        position, anchor_x, anchor_z, predicate->radius
                    ))
                    continue;
                found++;
                if (found >= predicate->minimum)
                    return 1;
            }
        }
    }
    return 0;
}

MCSEED_GPU_INLINE int mcseed_gpu_candidate_matches(
    const McSeedGpuCandidate *candidate,
    const McSeedGpuStructureConfig *configs,
    const McSeedGpuPredicate *predicates,
    size_t predicate_count
)
{
    size_t index;
    for (index = 0; index < predicate_count; index++) {
        if (!mcseed_gpu_predicate_matches(candidate, configs, &predicates[index]))
            return 0;
    }
    return 1;
}

#undef MCSEED_GPU_INLINE

#endif
