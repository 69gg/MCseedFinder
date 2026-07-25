#include "bridge.h"

#include "biomes.h"
#include "finders.h"
#include "util.h"

#include <limits.h>
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

enum {
    MCSEED_STRUCTURE_NETHER_FOSSIL = 1000,
};

typedef struct McSeedStructureInfo {
    int32_t id;
    int32_t cubiomes_type;
    int32_t dimension;
    int32_t accuracy;
    const char *name;
} McSeedStructureInfo;

/*
 * Canonical names follow Minecraft's built-in structure resource locations.
 * Entries that share one structure set (villages, ocean ruins, etc.) are
 * intentionally exposed as one searchable family.
 */
static const McSeedStructureInfo STRUCTURES[] = {
    { 1, Outpost,          DIM_OVERWORLD, 0, "pillager_outpost" },
    { 2, Mineshaft,        DIM_OVERWORLD, 0, "mineshaft" },
    { 3, Mansion,          DIM_OVERWORLD, 1, "mansion" },
    { 4, Jungle_Pyramid,   DIM_OVERWORLD, 1, "jungle_pyramid" },
    { 5, Desert_Pyramid,   DIM_OVERWORLD, 1, "desert_pyramid" },
    { 6, Igloo,            DIM_OVERWORLD, 0, "igloo" },
    { 7, Shipwreck,        DIM_OVERWORLD, 0, "shipwreck" },
    { 8, Swamp_Hut,        DIM_OVERWORLD, 0, "swamp_hut" },
    { 9, Stronghold,       DIM_OVERWORLD, 0, "stronghold" },
    {10, Monument,         DIM_OVERWORLD, 0, "monument" },
    {11, Ocean_Ruin,       DIM_OVERWORLD, 0, "ocean_ruin" },
    {12, Fortress,         DIM_NETHER,    0, "fortress" },
    {13, MCSEED_STRUCTURE_NETHER_FOSSIL, DIM_NETHER, 2, "nether_fossil" },
    {14, End_City,         DIM_END,       0, "end_city" },
    {15, Treasure,         DIM_OVERWORLD, 0, "buried_treasure" },
    {16, Bastion,          DIM_NETHER,    0, "bastion_remnant" },
    {17, Village,          DIM_OVERWORLD, 0, "village" },
    {18, Ruined_Portal,    DIM_OVERWORLD, 2, "ruined_portal" },
    {19, Ancient_City,     DIM_OVERWORLD, 0, "ancient_city" },
    {20, Trail_Ruins,      DIM_OVERWORLD, 0, "trail_ruins" },
    {21, Trial_Chambers,   DIM_OVERWORLD, 0, "trial_chambers" },
    {22, Ruined_Portal_N,  DIM_NETHER,    2, "ruined_portal_nether" },
};

struct McSeedContext {
    Generator overworld;
    Generator nether;
    Generator end;
    uint64_t seed;
    uint8_t ready_mask;
    uint8_t spawn_ready;
    Pos spawn;
    int32_t spawn_y;
    int32_t spawn_biome;
};

static const size_t STRUCTURE_COUNT = sizeof(STRUCTURES) / sizeof(STRUCTURES[0]);

static int64_t floor_div_i64(int64_t value, int64_t divisor)
{
    int64_t quotient = value / divisor;
    int64_t remainder = value % divisor;
    if (remainder < 0)
        quotient--;
    return quotient;
}

static int32_t clamp_i64_to_i32(int64_t value)
{
    if (value < INT32_MIN)
        return INT32_MIN;
    if (value > INT32_MAX)
        return INT32_MAX;
    return (int32_t)value;
}

static int64_t clamp_i64(int64_t value, int64_t minimum, int64_t maximum)
{
    if (value < minimum)
        return minimum;
    if (value > maximum)
        return maximum;
    return value;
}

static int within_radius(
    int32_t x,
    int32_t z,
    int32_t anchor_x,
    int32_t anchor_z,
    uint32_t radius
)
{
    int64_t dx = (int64_t)x - anchor_x;
    int64_t dz = (int64_t)z - anchor_z;
    uint64_t distance_squared = (uint64_t)(dx * dx) + (uint64_t)(dz * dz);
    uint64_t radius_squared = (uint64_t)radius * radius;
    return distance_squared <= radius_squared;
}

static Generator *generator_for_dimension(McSeedContext *context, int32_t dimension)
{
    uint8_t mask;
    Generator *generator;
    if (!context)
        return NULL;

    switch (dimension) {
    case DIM_OVERWORLD:
        mask = 1;
        generator = &context->overworld;
        break;
    case DIM_NETHER:
        mask = 2;
        generator = &context->nether;
        break;
    case DIM_END:
        mask = 4;
        generator = &context->end;
        break;
    default:
        return NULL;
    }

    if (!(context->ready_mask & mask)) {
        applySeed(generator, dimension, context->seed);
        context->ready_mask |= mask;
    }
    return generator;
}

static const McSeedStructureInfo *structure_by_id(int32_t id)
{
    size_t index;
    for (index = 0; index < STRUCTURE_COUNT; index++) {
        if (STRUCTURES[index].id == id)
            return &STRUCTURES[index];
    }
    return NULL;
}

static const char *without_namespace(const char *name)
{
    static const char prefix[] = "minecraft:";
    if (name && strncmp(name, prefix, sizeof(prefix) - 1) == 0)
        return name + sizeof(prefix) - 1;
    return name;
}

static int name_is(const char *name, const char *canonical)
{
    name = without_namespace(name);
    return name && strcmp(name, canonical) == 0;
}

McSeedContext *mcseed_context_create(void)
{
    McSeedContext *context = (McSeedContext *)calloc(1, sizeof(*context));
    if (!context)
        return NULL;
    setupGenerator(&context->overworld, MC_26_2, 0);
    setupGenerator(&context->nether, MC_26_2, 0);
    setupGenerator(&context->end, MC_26_2, 0);
    return context;
}

void mcseed_context_destroy(McSeedContext *context)
{
    free(context);
}

void mcseed_context_set_seed(McSeedContext *context, uint64_t seed)
{
    if (!context)
        return;
    context->seed = seed;
    context->ready_mask = 0;
    context->spawn_ready = 0;
}

int32_t mcseed_spawn(McSeedContext *context, McSeedHit *spawn, int32_t *biome_id)
{
    Generator *generator;
    if (!context || !spawn || !biome_id)
        return -1;
    generator = generator_for_dimension(context, DIM_OVERWORLD);
    if (!generator)
        return -2;

    if (!context->spawn_ready) {
        SurfaceNoise surface_noise;
        float height = 64.0f;
        int height_sample_biome = none;
        int biome;
        context->spawn = getSpawn(generator);
        initSurfaceNoise(&surface_noise, DIM_OVERWORLD, context->seed);
        if (mapApproxHeight(
                &height,
                &height_sample_biome,
                generator,
                &surface_noise,
                context->spawn.x >> 2,
                context->spawn.z >> 2,
                1,
                1
            ) != 0)
            height = 64.0f;
        context->spawn_y = (int32_t)lroundf(height);
        biome = getBiomeAt(
            generator,
            1,
            context->spawn.x,
            context->spawn_y,
            context->spawn.z
        );
        if (biome == none)
            biome = height_sample_biome;
        context->spawn_biome = biome;
        context->spawn_ready = 1;
    }

    spawn->x = context->spawn.x;
    spawn->y = context->spawn_y;
    spawn->z = context->spawn.z;
    spawn->id = 0;
    *biome_id = context->spawn_biome;
    return 0;
}

int32_t mcseed_biome_count(void)
{
    int32_t id;
    int32_t count = 0;
    for (id = 0; id <= UINT8_MAX; id++) {
        if (biomeExists(MC_26_2, id) && biome2str(MC_26_2, id))
            count++;
    }
    return count;
}

static int32_t biome_id_for_index(int32_t index)
{
    int32_t id;
    int32_t current = 0;
    if (index < 0)
        return none;
    for (id = 0; id <= UINT8_MAX; id++) {
        if (!biomeExists(MC_26_2, id) || !biome2str(MC_26_2, id))
            continue;
        if (current == index)
            return id;
        current++;
    }
    return none;
}

const char *mcseed_biome_name_at(int32_t index)
{
    int32_t id = biome_id_for_index(index);
    return id == none ? NULL : biome2str(MC_26_2, id);
}

int32_t mcseed_biome_id_at(int32_t index)
{
    return biome_id_for_index(index);
}

int32_t mcseed_biome_dimension_at(int32_t index)
{
    int32_t id = biome_id_for_index(index);
    return id == none ? DIM_UNDEF : getDimension(id);
}

int32_t mcseed_biome_id_from_name(const char *name)
{
    int32_t id;
    name = without_namespace(name);
    if (!name)
        return none;
    for (id = 0; id <= UINT8_MAX; id++) {
        const char *candidate;
        if (!biomeExists(MC_26_2, id))
            continue;
        candidate = biome2str(MC_26_2, id);
        if (candidate && strcmp(name, candidate) == 0)
            return id;
    }
    return none;
}

const char *mcseed_biome_name_from_id(int32_t id)
{
    if (!biomeExists(MC_26_2, id))
        return NULL;
    return biome2str(MC_26_2, id);
}

int32_t mcseed_structure_count(void)
{
    return (int32_t)STRUCTURE_COUNT;
}

const char *mcseed_structure_name_at(int32_t index)
{
    if (index < 0 || (size_t)index >= STRUCTURE_COUNT)
        return NULL;
    return STRUCTURES[index].name;
}

int32_t mcseed_structure_id_at(int32_t index)
{
    if (index < 0 || (size_t)index >= STRUCTURE_COUNT)
        return -1;
    return STRUCTURES[index].id;
}

int32_t mcseed_structure_dimension_at(int32_t index)
{
    if (index < 0 || (size_t)index >= STRUCTURE_COUNT)
        return DIM_UNDEF;
    return STRUCTURES[index].dimension;
}

int32_t mcseed_structure_accuracy_at(int32_t index)
{
    if (index < 0 || (size_t)index >= STRUCTURE_COUNT)
        return -1;
    return STRUCTURES[index].accuracy;
}

int32_t mcseed_structure_id_from_name(const char *name)
{
    size_t index;
    name = without_namespace(name);
    if (!name)
        return -1;
    for (index = 0; index < STRUCTURE_COUNT; index++) {
        if (strcmp(name, STRUCTURES[index].name) == 0)
            return STRUCTURES[index].id;
    }

    if (name_is(name, "outpost"))
        return 1;
    if (name_is(name, "woodland_mansion"))
        return 3;
    if (name_is(name, "jungle_temple"))
        return 4;
    if (name_is(name, "ocean_monument"))
        return 10;
    if (name_is(name, "bastion"))
        return 16;
    if (name_is(name, "treasure"))
        return 15;
    if (name_is(name, "ruined_portal_n"))
        return 22;
    return -1;
}

static int biome_is_target(int32_t biome, const int32_t *targets, size_t count)
{
    size_t index;
    for (index = 0; index < count; index++) {
        if (targets[index] == biome)
            return 1;
    }
    return 0;
}

int32_t mcseed_find_biomes(
    McSeedContext *context,
    int32_t dimension,
    const int32_t *biome_ids,
    size_t biome_count,
    int32_t anchor_x,
    int32_t anchor_z,
    uint32_t radius,
    int32_t y_min,
    int32_t y_max,
    uint64_t limit,
    McSeedHit *hits,
    size_t hit_capacity,
    uint64_t *found,
    int32_t *limit_reached
)
{
    Generator *generator;
    int64_t qx_min, qx_max, qy_min, qy_max, qz_min, qz_max;
    int64_t qx, qy, qz;
    uint64_t total = 0;
    if (!context || !biome_ids || biome_count == 0 || !found || !limit_reached)
        return -1;
    if (y_min > y_max)
        return -2;
    *found = 0;
    *limit_reached = 0;
    if (limit == 0)
        return 0;
    generator = generator_for_dimension(context, dimension);
    if (!generator)
        return -3;

    qx_min = floor_div_i64((int64_t)anchor_x - radius, 4);
    qx_max = floor_div_i64((int64_t)anchor_x + radius, 4);
    qz_min = floor_div_i64((int64_t)anchor_z - radius, 4);
    qz_max = floor_div_i64((int64_t)anchor_z + radius, 4);
    qy_min = floor_div_i64(y_min, 4);
    qy_max = floor_div_i64(y_max, 4);

    for (qy = qy_min; qy <= qy_max; qy++) {
        for (qz = qz_min; qz <= qz_max; qz++) {
            int64_t cell_z_min = qz * 4;
            int64_t near_z = clamp_i64(anchor_z, cell_z_min, cell_z_min + 3);
            for (qx = qx_min; qx <= qx_max; qx++) {
                int64_t cell_x_min = qx * 4;
                int64_t near_x = clamp_i64(anchor_x, cell_x_min, cell_x_min + 3);
                int32_t biome;
                if (!within_radius(
                        clamp_i64_to_i32(near_x),
                        clamp_i64_to_i32(near_z),
                        anchor_x,
                        anchor_z,
                        radius
                    ))
                    continue;
                biome = getBiomeAt(
                    generator,
                    4,
                    clamp_i64_to_i32(qx),
                    clamp_i64_to_i32(qy),
                    clamp_i64_to_i32(qz)
                );
                if (!biome_is_target(biome, biome_ids, biome_count))
                    continue;
                if (total < hit_capacity && hits) {
                    int64_t cell_y_min = qy * 4;
                    hits[total].x = clamp_i64_to_i32(near_x);
                    hits[total].y = clamp_i64_to_i32(
                        clamp_i64((cell_y_min + 2), y_min, y_max)
                    );
                    hits[total].z = clamp_i64_to_i32(near_z);
                    hits[total].id = biome;
                }
                total++;
                if (total >= limit) {
                    *found = total;
                    *limit_reached = 1;
                    return 0;
                }
            }
        }
    }

    *found = total;
    return 0;
}

static int store_structure_hit(
    int32_t structure_id,
    Pos position,
    McSeedHit *hits,
    size_t hit_capacity,
    uint64_t total
)
{
    if (total < hit_capacity && hits) {
        hits[total].x = position.x;
        hits[total].y = INT32_MIN;
        hits[total].z = position.z;
        hits[total].id = structure_id;
    }
    return 1;
}

static int32_t find_strongholds(
    McSeedContext *context,
    const McSeedStructureInfo *info,
    int32_t anchor_x,
    int32_t anchor_z,
    uint32_t radius,
    uint64_t limit,
    McSeedHit *hits,
    size_t hit_capacity,
    uint64_t *found,
    int32_t *limit_reached
)
{
    Generator *generator = generator_for_dimension(context, DIM_OVERWORLD);
    StrongholdIter iterator;
    uint64_t total = 0;
    int index;
    if (!generator)
        return -3;
    initFirstStronghold(&iterator, MC_26_2, context->seed);
    for (index = 0; index < 128; index++) {
        if (nextStronghold(&iterator, generator) <= 0)
            break;
        if (!within_radius(iterator.pos.x, iterator.pos.z, anchor_x, anchor_z, radius))
            continue;
        store_structure_hit(info->id, iterator.pos, hits, hit_capacity, total);
        total++;
        if (total >= limit) {
            *found = total;
            *limit_reached = 1;
            return 0;
        }
    }
    *found = total;
    return 0;
}

static int32_t find_nether_fossils(
    McSeedContext *context,
    const McSeedStructureInfo *info,
    int32_t anchor_x,
    int32_t anchor_z,
    uint32_t radius,
    uint64_t limit,
    McSeedHit *hits,
    size_t hit_capacity,
    uint64_t *found,
    int32_t *limit_reached
)
{
    const StructureConfig config = {
        14357921, 2, 1, 0, DIM_NETHER, 0.0f
    };
    Generator *generator = generator_for_dimension(context, DIM_NETHER);
    int64_t region_span = (int64_t)config.regionSize * 16;
    int64_t rx_min = floor_div_i64((int64_t)anchor_x - radius, region_span) - 1;
    int64_t rx_max = floor_div_i64((int64_t)anchor_x + radius, region_span) + 1;
    int64_t rz_min = floor_div_i64((int64_t)anchor_z - radius, region_span) - 1;
    int64_t rz_max = floor_div_i64((int64_t)anchor_z + radius, region_span) + 1;
    int64_t rx, rz;
    uint64_t total = 0;
    if (!generator)
        return -3;

    for (rz = rz_min; rz <= rz_max; rz++) {
        for (rx = rx_min; rx <= rx_max; rx++) {
            Pos position;
            int32_t biome;
            if (rx < INT32_MIN || rx > INT32_MAX || rz < INT32_MIN || rz > INT32_MAX)
                continue;
            position = getFeaturePos(config, context->seed, (int32_t)rx, (int32_t)rz);
            if (!within_radius(position.x, position.z, anchor_x, anchor_z, radius))
                continue;
            biome = getBiomeAt(
                generator,
                4,
                (position.x >> 2) + 2,
                33 >> 2,
                (position.z >> 2) + 2
            );
            if (biome != soul_sand_valley)
                continue;
            store_structure_hit(info->id, position, hits, hit_capacity, total);
            total++;
            if (total >= limit) {
                *found = total;
                *limit_reached = 1;
                return 0;
            }
        }
    }
    *found = total;
    return 0;
}

int32_t mcseed_find_structure(
    McSeedContext *context,
    int32_t structure_id,
    int32_t anchor_x,
    int32_t anchor_z,
    uint32_t radius,
    uint64_t limit,
    McSeedHit *hits,
    size_t hit_capacity,
    uint64_t *found,
    int32_t *limit_reached
)
{
    const McSeedStructureInfo *info = structure_by_id(structure_id);
    StructureConfig config;
    Generator *generator;
    int64_t region_span, rx_min, rx_max, rz_min, rz_max, rx, rz;
    uint64_t total = 0;
    SurfaceNoise end_surface_noise;
    int check_end_terrain = 0;
    if (!context || !info || !found || !limit_reached)
        return -1;
    *found = 0;
    *limit_reached = 0;
    if (limit == 0)
        return 0;
    if (info->cubiomes_type == Stronghold)
        return find_strongholds(
            context, info, anchor_x, anchor_z, radius, limit,
            hits, hit_capacity, found, limit_reached
        );
    if (info->cubiomes_type == MCSEED_STRUCTURE_NETHER_FOSSIL)
        return find_nether_fossils(
            context, info, anchor_x, anchor_z, radius, limit,
            hits, hit_capacity, found, limit_reached
        );
    if (!getStructureConfig(info->cubiomes_type, MC_26_2, &config))
        return -2;
    generator = generator_for_dimension(context, info->dimension);
    if (!generator)
        return -3;
    if (info->cubiomes_type == End_City) {
        initSurfaceNoise(&end_surface_noise, DIM_END, context->seed);
        check_end_terrain = 1;
    }

    region_span = (int64_t)config.regionSize * 16;
    rx_min = floor_div_i64((int64_t)anchor_x - radius, region_span) - 1;
    rx_max = floor_div_i64((int64_t)anchor_x + radius, region_span) + 1;
    rz_min = floor_div_i64((int64_t)anchor_z - radius, region_span) - 1;
    rz_max = floor_div_i64((int64_t)anchor_z + radius, region_span) + 1;

    for (rz = rz_min; rz <= rz_max; rz++) {
        for (rx = rx_min; rx <= rx_max; rx++) {
            Pos position;
            if (rx < INT32_MIN || rx > INT32_MAX || rz < INT32_MIN || rz > INT32_MAX)
                continue;
            if (!getStructurePos(
                    info->cubiomes_type,
                    MC_26_2,
                    context->seed,
                    (int32_t)rx,
                    (int32_t)rz,
                    &position
                ))
                continue;
            if (!within_radius(position.x, position.z, anchor_x, anchor_z, radius))
                continue;
            if (!isViableStructurePos(
                    info->cubiomes_type,
                    generator,
                    position.x,
                    position.z,
                    0
                ))
                continue;
            if (info->dimension == DIM_OVERWORLD &&
                !isViableStructureTerrain(
                    info->cubiomes_type,
                    generator,
                    position.x,
                    position.z
                ))
                continue;
            if (check_end_terrain && !isViableEndCityTerrain(
                    generator,
                    &end_surface_noise,
                    position.x,
                    position.z
                ))
                continue;
            store_structure_hit(info->id, position, hits, hit_capacity, total);
            total++;
            if (total >= limit) {
                *found = total;
                *limit_reached = 1;
                return 0;
            }
        }
    }

    *found = total;
    return 0;
}
