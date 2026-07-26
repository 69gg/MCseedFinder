#ifndef MCSEED_BRIDGE_H
#define MCSEED_BRIDGE_H

#include <stddef.h>
#include <stdint.h>

#include "gpu/abi.h"

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

typedef struct McSeedPieceHit {
    int32_t x;
    int32_t y;
    int32_t z;
    int32_t parent_x;
    int32_t parent_z;
    int32_t eye_mask;
    const char *name;
} McSeedPieceHit;

McSeedContext *mcseed_context_create(void);
void mcseed_context_destroy(McSeedContext *context);
void mcseed_context_set_seed(McSeedContext *context, uint64_t seed);

int32_t mcseed_spawn(
    McSeedContext *context,
    McSeedHit *spawn,
    int32_t *biome_id
);
/** Refine a caller-provided modern spawn estimate without estimating it again. */
int32_t mcseed_spawn_from_estimate(
    McSeedContext *context,
    int32_t estimate_x,
    int32_t estimate_z,
    McSeedHit *spawn,
    int32_t *biome_id
);

/** Fast spawn estimate used by the conservative two-stage GPU prefilter. */
int32_t mcseed_estimated_spawn(McSeedContext *context, McSeedHit *spawn);
/** Slow reference path used to verify optimized spawn estimation. */
int32_t mcseed_estimated_spawn_reference(McSeedContext *context, McSeedHit *spawn);
/** Return 1 and the strict estimate-to-final horizontal bound when available. */
int32_t mcseed_spawn_refinement_radius(uint32_t *radius);
/** Return 1 and a strict origin-to-final-spawn horizontal bound when available. */
int32_t mcseed_spawn_origin_radius(uint32_t *radius);

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
/** Return 1 when the structure has a conservative GPU placement prefilter. */
int32_t mcseed_structure_gpu_config(
    int32_t structure_id,
    McSeedGpuStructureConfig *config
);

int32_t mcseed_piece_count(void);
const char *mcseed_piece_name_at(int32_t index);
int32_t mcseed_piece_structure_id_at(int32_t index);
int32_t mcseed_piece_accuracy_at(int32_t index);
int32_t mcseed_piece_is_group_at(int32_t index);
int32_t mcseed_piece_selector_valid(int32_t structure_id, const char *name);

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

/**
 * Count matching pieces in structures whose start position lies within radius.
 * Selectors are validated with mcseed_piece_selector_valid before the scan.
 */
int32_t mcseed_find_structure_pieces(
    McSeedContext *context,
    int32_t structure_id,
    const char *const *selectors,
    size_t selector_count,
    int32_t anchor_x,
    int32_t anchor_z,
    uint32_t radius,
    uint64_t limit,
    McSeedPieceHit *hits,
    size_t hit_capacity,
    uint64_t *found,
    int32_t *limit_reached
);

/**
 * Find the stronghold selected by Eye of Ender locate logic from an anchor.
 * The returned portal-room hit includes a 12-bit End Portal eye mask.
 */
int32_t mcseed_nearest_stronghold_portal(
    McSeedContext *context,
    int32_t anchor_x,
    int32_t anchor_z,
    McSeedPieceHit *hit
);

#ifdef __cplusplus
}
#endif

#endif
