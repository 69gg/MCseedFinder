#ifndef MCSEED_GPU_PLACEMENT_H
#define MCSEED_GPU_PLACEMENT_H

#include "abi.h"

#include <limits.h>
#include <math.h>
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

MCSEED_GPU_INLINE double mcseed_gpu_java_next_double(uint64_t *seed)
{
    uint64_t value = (uint64_t)(uint32_t)mcseed_gpu_java_next(seed, 26);
    value <<= 27;
    value += (uint64_t)(uint32_t)mcseed_gpu_java_next(seed, 27);
    return (double)(int64_t)value / (double)(UINT64_C(1) << 53);
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

MCSEED_GPU_INLINE int mcseed_gpu_within_radius_envelope(
    McSeedGpuPosition position,
    int32_t anchor_x,
    int32_t anchor_z,
    uint32_t radius,
    uint32_t per_axis_margin
)
{
    int64_t dx = (int64_t)position.x - anchor_x;
    int64_t dz = (int64_t)position.z - anchor_z;
    uint64_t absolute_x = dx < 0 ? (uint64_t)(-dx) : (uint64_t)dx;
    uint64_t absolute_z = dz < 0 ? (uint64_t)(-dz) : (uint64_t)dz;
    uint64_t bounded_x = absolute_x > per_axis_margin
        ? absolute_x - per_axis_margin
        : 0;
    uint64_t bounded_z = absolute_z > per_axis_margin
        ? absolute_z - per_axis_margin
        : 0;
    uint64_t radius_squared = (uint64_t)radius * radius;
    if (bounded_x > radius || bounded_z > radius)
        return 0;
    return bounded_z * bounded_z <= radius_squared - bounded_x * bounded_x;
}

MCSEED_GPU_INLINE uint64_t mcseed_gpu_stronghold_potential_count(
    const McSeedGpuStructureConfig *config,
    uint64_t world_seed,
    int32_t anchor_x,
    int32_t anchor_z,
    uint32_t radius,
    uint64_t limit
)
{
    const double pi = 3.14159265358979323846;
    uint64_t random;
    double angle;
    double distance;
    int32_t ring_number = 0;
    int32_t ring_index = 0;
    int32_t ring_size;
    int32_t index;
    uint64_t found = 0;

    if (config->salt <= 0 || config->region_size <= 0 ||
        config->chunk_range <= 0 || config->reserved < 0 || limit == 0)
        return 0;

    random = mcseed_gpu_java_set_seed(world_seed);
    angle = 2.0 * pi * mcseed_gpu_java_next_double(&random);
    distance = (4.0 * config->region_size) +
        (mcseed_gpu_java_next_double(&random) - 0.5) *
        config->region_size * 2.5;
    ring_size = config->chunk_range;

    for (index = 0; index < config->salt; index++) {
        McSeedGpuPosition position;
        position.x = (int32_t)round(cos(angle) * distance) * 16 + 4;
        position.z = (int32_t)round(sin(angle) * distance) * 16 + 4;
        if (mcseed_gpu_within_radius_envelope(
                position,
                anchor_x,
                anchor_z,
                radius,
                (uint32_t)config->reserved
            )) {
            found++;
            if (found >= limit)
                return found;
        }

        (void)mcseed_gpu_java_next_long(&random);
        ring_index++;
        angle += 2.0 * pi / ring_size;
        if (ring_index == ring_size) {
            ring_number++;
            ring_index = 0;
            ring_size += 2 * ring_size / (ring_number + 1);
            if (ring_size > config->salt - index)
                ring_size = config->salt - index;
            angle += mcseed_gpu_java_next_double(&random) * pi * 2.0;
        }
        distance = (4.0 * config->region_size) +
            (6.0 * ring_number * config->region_size) +
            (mcseed_gpu_java_next_double(&random) - 0.5) *
            config->region_size * 2.5;
    }
    return found;
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
        if (config->kind == MCSEED_GPU_PLACEMENT_STRONGHOLD) {
            found += mcseed_gpu_stronghold_potential_count(
                config,
                candidate->seed,
                anchor_x,
                anchor_z,
                predicate->radius,
                predicate->minimum - found
            );
            if (found >= predicate->minimum)
                return 1;
            continue;
        }
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

MCSEED_GPU_INLINE int mcseed_gpu_pair_predicate_matches(
    const McSeedGpuCandidate *candidate,
    const McSeedGpuStructureConfig *configs,
    const McSeedGpuPairPredicate *predicate
)
{
    uint64_t left_envelope_u64 = (uint64_t)predicate->anchor_radius +
        predicate->left_radius;
    uint64_t right_envelope_u64 = (uint64_t)predicate->anchor_radius +
        predicate->right_radius;
    uint64_t pair_radius_u64 = (uint64_t)predicate->left_radius +
        predicate->right_radius;
    uint32_t left_envelope;
    uint32_t right_envelope;
    uint32_t pair_radius;
    uint32_t left_config_index;
    McSeedGpuPosition origin = {0, 0};

    /* Concentric rings do not have independent random-spread regions. */
    for (left_config_index = 0;
         left_config_index < predicate->left_config_count;
         left_config_index++) {
        if (configs[predicate->left_config_offset + left_config_index].kind ==
            MCSEED_GPU_PLACEMENT_STRONGHOLD)
            return 1;
    }
    for (left_config_index = 0;
         left_config_index < predicate->right_config_count;
         left_config_index++) {
        if (configs[predicate->right_config_offset + left_config_index].kind ==
            MCSEED_GPU_PLACEMENT_STRONGHOLD)
            return 1;
    }

    if (left_envelope_u64 > UINT32_MAX || right_envelope_u64 > UINT32_MAX ||
        pair_radius_u64 > UINT32_MAX)
        return 1;
    left_envelope = (uint32_t)left_envelope_u64;
    right_envelope = (uint32_t)right_envelope_u64;
    pair_radius = (uint32_t)pair_radius_u64;

    for (left_config_index = 0;
         left_config_index < predicate->left_config_count;
         left_config_index++) {
        const McSeedGpuStructureConfig *left_config =
            &configs[predicate->left_config_offset + left_config_index];
        int64_t left_span = (int64_t)left_config->region_size * 16;
        int64_t left_region_min = mcseed_gpu_floor_div_i64(
            -(int64_t)left_envelope, left_span
        ) - 1;
        int64_t left_region_max = mcseed_gpu_floor_div_i64(
            (int64_t)left_envelope, left_span
        ) + 1;
        int64_t left_region_z;

        for (left_region_z = left_region_min;
             left_region_z <= left_region_max;
             left_region_z++) {
            int64_t left_region_x;
            if (left_region_z < INT32_MIN || left_region_z > INT32_MAX)
                continue;
            for (left_region_x = left_region_min;
                 left_region_x <= left_region_max;
                 left_region_x++) {
                McSeedGpuPosition left_position;
                uint32_t right_config_index;
                if (left_region_x < INT32_MIN || left_region_x > INT32_MAX)
                    continue;
                if (!mcseed_gpu_position_for_region(
                        left_config,
                        candidate->seed,
                        (int32_t)left_region_x,
                        (int32_t)left_region_z,
                        &left_position
                    ) ||
                    !mcseed_gpu_within_radius(
                        left_position, origin.x, origin.z, left_envelope
                    ))
                    continue;

                for (right_config_index = 0;
                     right_config_index < predicate->right_config_count;
                     right_config_index++) {
                    const McSeedGpuStructureConfig *right_config =
                        &configs[predicate->right_config_offset + right_config_index];
                    int64_t right_span = (int64_t)right_config->region_size * 16;
                    int64_t right_region_x_min = mcseed_gpu_floor_div_i64(
                        (int64_t)left_position.x - pair_radius, right_span
                    ) - 1;
                    int64_t right_region_x_max = mcseed_gpu_floor_div_i64(
                        (int64_t)left_position.x + pair_radius, right_span
                    ) + 1;
                    int64_t right_region_z_min = mcseed_gpu_floor_div_i64(
                        (int64_t)left_position.z - pair_radius, right_span
                    ) - 1;
                    int64_t right_region_z_max = mcseed_gpu_floor_div_i64(
                        (int64_t)left_position.z + pair_radius, right_span
                    ) + 1;
                    int64_t right_region_z;

                    for (right_region_z = right_region_z_min;
                         right_region_z <= right_region_z_max;
                         right_region_z++) {
                        int64_t right_region_x;
                        if (right_region_z < INT32_MIN || right_region_z > INT32_MAX)
                            continue;
                        for (right_region_x = right_region_x_min;
                             right_region_x <= right_region_x_max;
                             right_region_x++) {
                            McSeedGpuPosition right_position;
                            if (right_region_x < INT32_MIN ||
                                right_region_x > INT32_MAX)
                                continue;
                            if (!mcseed_gpu_position_for_region(
                                    right_config,
                                    candidate->seed,
                                    (int32_t)right_region_x,
                                    (int32_t)right_region_z,
                                    &right_position
                                ) ||
                                !mcseed_gpu_within_radius(
                                    right_position,
                                    origin.x,
                                    origin.z,
                                    right_envelope
                                ) ||
                                !mcseed_gpu_within_radius(
                                    right_position,
                                    left_position.x,
                                    left_position.z,
                                    pair_radius
                                ))
                                continue;
                            return 1;
                        }
                    }
                }
            }
        }
    }
    return 0;
}

MCSEED_GPU_INLINE int mcseed_gpu_candidate_matches_pairs(
    const McSeedGpuCandidate *candidate,
    const McSeedGpuStructureConfig *configs,
    const McSeedGpuPairPredicate *predicates,
    size_t predicate_count
)
{
    size_t index;
    for (index = 0; index < predicate_count; index++) {
        if (!mcseed_gpu_pair_predicate_matches(
                candidate, configs, &predicates[index]
            ))
            return 0;
    }
    return 1;
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
