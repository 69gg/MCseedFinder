#ifndef MCSEED_JIGSAW_H
#define MCSEED_JIGSAW_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

enum {
    MCJIGSAW_PIECE_CAPACITY = 512,
};

enum {
    MCJIGSAW_DIR_DOWN,
    MCJIGSAW_DIR_UP,
    MCJIGSAW_DIR_NORTH,
    MCJIGSAW_DIR_SOUTH,
    MCJIGSAW_DIR_WEST,
    MCJIGSAW_DIR_EAST,
};

enum {
    MCJIGSAW_JOINT_ALIGNED,
    MCJIGSAW_JOINT_ROLLABLE,
};

enum {
    MCJIGSAW_ELEMENT_EMPTY,
    MCJIGSAW_ELEMENT_TEMPLATE,
    MCJIGSAW_ELEMENT_FEATURE,
};

enum {
    MCJIGSAW_PROJECTION_RIGID,
    MCJIGSAW_PROJECTION_TERRAIN_MATCHING,
};

enum {
    MCJIGSAW_STYLE_PLAINS,
    MCJIGSAW_STYLE_DESERT,
    MCJIGSAW_STYLE_SAVANNA,
    MCJIGSAW_STYLE_SNOWY,
    MCJIGSAW_STYLE_TAIGA,
    MCJIGSAW_STYLE_COUNT,
};

typedef struct McJigsawConnectorData {
    int16_t x;
    int16_t y;
    int16_t z;
    uint8_t front;
    uint8_t top;
    uint8_t joint;
    uint16_t pool;
    int32_t placement_priority;
    int32_t selection_priority;
    const char *name;
    const char *target;
} McJigsawConnectorData;

typedef struct McJigsawElementData {
    const char *name;
    int16_t size_x;
    int16_t size_y;
    int16_t size_z;
    uint32_t connector_offset;
    uint16_t connector_count;
    uint8_t kind;
    uint8_t projection;
} McJigsawElementData;

typedef struct McJigsawPoolEntryData {
    uint16_t element;
    uint16_t weight;
} McJigsawPoolEntryData;

typedef struct McJigsawPoolData {
    const char *name;
    uint16_t fallback;
    uint32_t entry_offset;
    uint16_t entry_count;
    uint16_t total_weight;
    uint16_t max_height;
} McJigsawPoolData;

typedef struct McJigsawData {
    const char *version;
    const McJigsawConnectorData *connectors;
    size_t connector_count;
    const McJigsawElementData *elements;
    size_t element_count;
    const McJigsawPoolEntryData *pool_entries;
    size_t pool_entry_count;
    const McJigsawPoolData *pools;
    size_t pool_count;
    uint16_t start_pools[MCJIGSAW_STYLE_COUNT];
} McJigsawData;

typedef struct McJigsawBox {
    int32_t min_x;
    int32_t min_y;
    int32_t min_z;
    int32_t max_x;
    int32_t max_y;
    int32_t max_z;
} McJigsawBox;

typedef struct McJigsawPiece {
    const char *name;
    int32_t x;
    int32_t y;
    int32_t z;
    McJigsawBox box;
    uint16_t element;
    uint8_t rotation;
    uint8_t depth;
} McJigsawPiece;

typedef int32_t (*McJigsawHeightFunction)(void *context, int32_t x, int32_t z);
typedef struct McJigsawWorkspace McJigsawWorkspace;

/** Return the checked-in village dataset for a version, or NULL if unavailable. */
const McJigsawData *mcjigsaw_village_data(const char *version);

/** Allocate reusable, thread-confined scratch storage for Jigsaw generation. */
McJigsawWorkspace *mcjigsaw_workspace_create(void);
void mcjigsaw_workspace_destroy(McJigsawWorkspace *workspace);

/**
 * Generate a village's Jigsaw pieces using the same legacy RNG and queue order
 * as Java Edition. The callback must return WORLD_SURFACE_WG's first free Y.
 *
 * Returns a piece count, or a negative error code. A return value equal to
 * capacity is valid; internal overflow is reported as -4 instead of truncating.
 */
int32_t mcjigsaw_generate(
    McJigsawWorkspace *workspace,
    const McJigsawData *data,
    uint64_t world_seed,
    int32_t start_x,
    int32_t start_z,
    int32_t style,
    McJigsawHeightFunction height_function,
    void *height_context,
    McJigsawPiece *pieces,
    size_t capacity
);

#ifdef __cplusplus
}
#endif

#endif
