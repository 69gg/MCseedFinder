#ifndef MCSEED_BRIDGE_H
#define MCSEED_BRIDGE_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct McSeedContext McSeedContext;

typedef struct McSeedHit {
    int32_t x;
    int32_t y;
    int32_t z;
    int32_t id;
} McSeedHit;

McSeedContext *mcseed_context_create(void);
void mcseed_context_destroy(McSeedContext *context);
void mcseed_context_set_seed(McSeedContext *context, uint64_t seed);

int32_t mcseed_spawn(
    McSeedContext *context,
    McSeedHit *spawn,
    int32_t *biome_id
);

int32_t mcseed_biome_count(void);
const char *mcseed_biome_name_at(int32_t index);
int32_t mcseed_biome_id_at(int32_t index);
int32_t mcseed_biome_dimension_at(int32_t index);
int32_t mcseed_biome_id_from_name(const char *name);
const char *mcseed_biome_name_from_id(int32_t id);

int32_t mcseed_structure_count(void);
const char *mcseed_structure_name_at(int32_t index);
int32_t mcseed_structure_id_at(int32_t index);
int32_t mcseed_structure_dimension_at(int32_t index);
int32_t mcseed_structure_accuracy_at(int32_t index);
int32_t mcseed_structure_id_from_name(const char *name);

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
);

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
);

#ifdef __cplusplus
}
#endif

#endif
