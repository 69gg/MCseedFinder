#include "bridge.h"

#include "jigsaw.h"
#include "version.h"

#include "biomes.h"
#include "features/stronghold.h"
#include "finders.h"
#include "terrainnoise.h"
#include "util.h"

#include <limits.h>
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

enum {
    MCSEED_STRUCTURE_NETHER_FOSSIL = 1000,
    MCSEED_PIECE_BUFFER_CAPACITY = 1024,
    MCSEED_JIGSAW_BUFFER_CAPACITY = MCJIGSAW_PIECE_CAPACITY,
    MCSEED_TERRAIN_COLUMN_CACHE_SIZE = 256,
    MCSEED_STRONGHOLD_COUNT = 128,
    MCSEED_STRONGHOLD_LOCATE_MARGIN = 128,
};

enum {
    MCSEED_ID_OUTPOST = 1,
    MCSEED_ID_JUNGLE_PYRAMID = 4,
    MCSEED_ID_DESERT_PYRAMID = 5,
    MCSEED_ID_IGLOO = 6,
    MCSEED_ID_SHIPWRECK = 7,
    MCSEED_ID_SWAMP_HUT = 8,
    MCSEED_ID_STRONGHOLD = 9,
    MCSEED_ID_FORTRESS = 12,
    MCSEED_ID_END_CITY = 14,
    MCSEED_ID_TREASURE = 15,
    MCSEED_ID_BASTION = 16,
    MCSEED_ID_VILLAGE = 17,
    MCSEED_ID_RUINED_PORTAL = 18,
    MCSEED_ID_RUINED_PORTAL_NETHER = 22,
};

enum {
    MCSEED_PIECE_EXACT = 0,
    MCSEED_PIECE_APPROXIMATE_TERRAIN = 1,
    MCSEED_PIECE_PARTIAL = 2,
};

typedef struct McSeedStructureInfo {
    int32_t id;
    int32_t cubiomes_type;
    int32_t dimension;
    int32_t accuracy;
    const char *name;
} McSeedStructureInfo;

typedef struct McSeedPieceInfo {
    int32_t structure_id;
    int32_t accuracy;
    int32_t is_group;
    const char *name;
} McSeedPieceInfo;

typedef struct McSeedTerrainColumn {
    int32_t cell_x;
    int32_t cell_z;
    uint8_t valid;
    double values[49];
} McSeedTerrainColumn;

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

#define MCSEED_PIECE(structure, accuracy, group, piece_name) \
    {structure, accuracy, group, piece_name}

static const McSeedPieceInfo STATIC_PIECES[] = {
    /* Village groups. Exact village template names come from generated data. */
    MCSEED_PIECE(MCSEED_ID_VILLAGE, MCSEED_PIECE_APPROXIMATE_TERRAIN, 1, "blacksmith"),
    MCSEED_PIECE(MCSEED_ID_VILLAGE, MCSEED_PIECE_APPROXIMATE_TERRAIN, 1, "smith"),
    MCSEED_PIECE(MCSEED_ID_VILLAGE, MCSEED_PIECE_APPROXIMATE_TERRAIN, 1, "house"),
    MCSEED_PIECE(MCSEED_ID_VILLAGE, MCSEED_PIECE_APPROXIMATE_TERRAIN, 1, "profession_house"),
    MCSEED_PIECE(MCSEED_ID_VILLAGE, MCSEED_PIECE_APPROXIMATE_TERRAIN, 1, "residential"),
    MCSEED_PIECE(MCSEED_ID_VILLAGE, MCSEED_PIECE_APPROXIMATE_TERRAIN, 1, "armorer"),
    MCSEED_PIECE(MCSEED_ID_VILLAGE, MCSEED_PIECE_APPROXIMATE_TERRAIN, 1, "weaponsmith"),
    MCSEED_PIECE(MCSEED_ID_VILLAGE, MCSEED_PIECE_APPROXIMATE_TERRAIN, 1, "toolsmith"),
    MCSEED_PIECE(MCSEED_ID_VILLAGE, MCSEED_PIECE_APPROXIMATE_TERRAIN, 1, "butcher"),
    MCSEED_PIECE(MCSEED_ID_VILLAGE, MCSEED_PIECE_APPROXIMATE_TERRAIN, 1, "cartographer"),
    MCSEED_PIECE(MCSEED_ID_VILLAGE, MCSEED_PIECE_APPROXIMATE_TERRAIN, 1, "fisher"),
    MCSEED_PIECE(MCSEED_ID_VILLAGE, MCSEED_PIECE_APPROXIMATE_TERRAIN, 1, "fisherman"),
    MCSEED_PIECE(MCSEED_ID_VILLAGE, MCSEED_PIECE_APPROXIMATE_TERRAIN, 1, "fletcher"),
    MCSEED_PIECE(MCSEED_ID_VILLAGE, MCSEED_PIECE_APPROXIMATE_TERRAIN, 1, "library"),
    MCSEED_PIECE(MCSEED_ID_VILLAGE, MCSEED_PIECE_APPROXIMATE_TERRAIN, 1, "librarian"),
    MCSEED_PIECE(MCSEED_ID_VILLAGE, MCSEED_PIECE_APPROXIMATE_TERRAIN, 1, "mason"),
    MCSEED_PIECE(MCSEED_ID_VILLAGE, MCSEED_PIECE_APPROXIMATE_TERRAIN, 1, "shepherd"),
    MCSEED_PIECE(MCSEED_ID_VILLAGE, MCSEED_PIECE_APPROXIMATE_TERRAIN, 1, "tannery"),
    MCSEED_PIECE(MCSEED_ID_VILLAGE, MCSEED_PIECE_APPROXIMATE_TERRAIN, 1, "leatherworker"),
    MCSEED_PIECE(MCSEED_ID_VILLAGE, MCSEED_PIECE_APPROXIMATE_TERRAIN, 1, "temple"),
    MCSEED_PIECE(MCSEED_ID_VILLAGE, MCSEED_PIECE_APPROXIMATE_TERRAIN, 1, "cleric"),
    MCSEED_PIECE(MCSEED_ID_VILLAGE, MCSEED_PIECE_APPROXIMATE_TERRAIN, 1, "farm"),
    MCSEED_PIECE(MCSEED_ID_VILLAGE, MCSEED_PIECE_APPROXIMATE_TERRAIN, 1, "farmer"),
    MCSEED_PIECE(MCSEED_ID_VILLAGE, MCSEED_PIECE_APPROXIMATE_TERRAIN, 1, "animal_pen"),
    MCSEED_PIECE(MCSEED_ID_VILLAGE, MCSEED_PIECE_APPROXIMATE_TERRAIN, 1, "stable"),
    MCSEED_PIECE(MCSEED_ID_VILLAGE, MCSEED_PIECE_APPROXIMATE_TERRAIN, 1, "small_house"),
    MCSEED_PIECE(MCSEED_ID_VILLAGE, MCSEED_PIECE_APPROXIMATE_TERRAIN, 1, "medium_house"),
    MCSEED_PIECE(MCSEED_ID_VILLAGE, MCSEED_PIECE_APPROXIMATE_TERRAIN, 1, "large_house"),
    MCSEED_PIECE(MCSEED_ID_VILLAGE, MCSEED_PIECE_APPROXIMATE_TERRAIN, 1, "meeting_point"),
    MCSEED_PIECE(MCSEED_ID_VILLAGE, MCSEED_PIECE_APPROXIMATE_TERRAIN, 1, "town_center"),
    MCSEED_PIECE(MCSEED_ID_VILLAGE, MCSEED_PIECE_APPROXIMATE_TERRAIN, 1, "street"),
    MCSEED_PIECE(MCSEED_ID_VILLAGE, MCSEED_PIECE_APPROXIMATE_TERRAIN, 1, "terminator"),
    MCSEED_PIECE(MCSEED_ID_VILLAGE, MCSEED_PIECE_APPROXIMATE_TERRAIN, 1, "decor"),
    MCSEED_PIECE(MCSEED_ID_VILLAGE, MCSEED_PIECE_APPROXIMATE_TERRAIN, 1, "feature"),
    MCSEED_PIECE(MCSEED_ID_VILLAGE, MCSEED_PIECE_APPROXIMATE_TERRAIN, 1, "zombie"),

    MCSEED_PIECE(MCSEED_ID_DESERT_PYRAMID, MCSEED_PIECE_EXACT, 0, "desert_pyramid/main"),
    MCSEED_PIECE(MCSEED_ID_IGLOO, MCSEED_PIECE_EXACT, 0, "igloo/top"),
    MCSEED_PIECE(MCSEED_ID_IGLOO, MCSEED_PIECE_EXACT, 0, "igloo/middle"),
    MCSEED_PIECE(MCSEED_ID_IGLOO, MCSEED_PIECE_EXACT, 0, "igloo/bottom"),
    MCSEED_PIECE(MCSEED_ID_JUNGLE_PYRAMID, MCSEED_PIECE_EXACT, 0, "jungle_pyramid/main"),
    MCSEED_PIECE(MCSEED_ID_OUTPOST, MCSEED_PIECE_PARTIAL, 0, "pillager_outpost/watchtower"),
    MCSEED_PIECE(MCSEED_ID_SWAMP_HUT, MCSEED_PIECE_EXACT, 0, "swamp_hut/main"),
    MCSEED_PIECE(MCSEED_ID_TREASURE, MCSEED_PIECE_EXACT, 0, "buried_treasure/chest"),

    MCSEED_PIECE(MCSEED_ID_SHIPWRECK, MCSEED_PIECE_EXACT, 1, "full"),
    MCSEED_PIECE(MCSEED_ID_SHIPWRECK, MCSEED_PIECE_EXACT, 1, "front_half"),
    MCSEED_PIECE(MCSEED_ID_SHIPWRECK, MCSEED_PIECE_EXACT, 1, "back_half"),
    MCSEED_PIECE(MCSEED_ID_SHIPWRECK, MCSEED_PIECE_EXACT, 1, "mast"),
    MCSEED_PIECE(MCSEED_ID_SHIPWRECK, MCSEED_PIECE_EXACT, 1, "degraded"),
    MCSEED_PIECE(MCSEED_ID_SHIPWRECK, MCSEED_PIECE_EXACT, 1, "intact"),
    MCSEED_PIECE(MCSEED_ID_SHIPWRECK, MCSEED_PIECE_EXACT, 0, "shipwreck/with_mast"),
    MCSEED_PIECE(MCSEED_ID_SHIPWRECK, MCSEED_PIECE_EXACT, 0, "shipwreck/upsidedown_full"),
    MCSEED_PIECE(MCSEED_ID_SHIPWRECK, MCSEED_PIECE_EXACT, 0, "shipwreck/upsidedown_fronthalf"),
    MCSEED_PIECE(MCSEED_ID_SHIPWRECK, MCSEED_PIECE_EXACT, 0, "shipwreck/upsidedown_backhalf"),
    MCSEED_PIECE(MCSEED_ID_SHIPWRECK, MCSEED_PIECE_EXACT, 0, "shipwreck/sideways_full"),
    MCSEED_PIECE(MCSEED_ID_SHIPWRECK, MCSEED_PIECE_EXACT, 0, "shipwreck/sideways_fronthalf"),
    MCSEED_PIECE(MCSEED_ID_SHIPWRECK, MCSEED_PIECE_EXACT, 0, "shipwreck/sideways_backhalf"),
    MCSEED_PIECE(MCSEED_ID_SHIPWRECK, MCSEED_PIECE_EXACT, 0, "shipwreck/rightsideup_full"),
    MCSEED_PIECE(MCSEED_ID_SHIPWRECK, MCSEED_PIECE_EXACT, 0, "shipwreck/rightsideup_fronthalf"),
    MCSEED_PIECE(MCSEED_ID_SHIPWRECK, MCSEED_PIECE_EXACT, 0, "shipwreck/rightsideup_backhalf"),
    MCSEED_PIECE(MCSEED_ID_SHIPWRECK, MCSEED_PIECE_EXACT, 0, "shipwreck/with_mast_degraded"),
    MCSEED_PIECE(MCSEED_ID_SHIPWRECK, MCSEED_PIECE_EXACT, 0, "shipwreck/upsidedown_full_degraded"),
    MCSEED_PIECE(MCSEED_ID_SHIPWRECK, MCSEED_PIECE_EXACT, 0, "shipwreck/upsidedown_fronthalf_degraded"),
    MCSEED_PIECE(MCSEED_ID_SHIPWRECK, MCSEED_PIECE_EXACT, 0, "shipwreck/upsidedown_backhalf_degraded"),
    MCSEED_PIECE(MCSEED_ID_SHIPWRECK, MCSEED_PIECE_EXACT, 0, "shipwreck/sideways_full_degraded"),
    MCSEED_PIECE(MCSEED_ID_SHIPWRECK, MCSEED_PIECE_EXACT, 0, "shipwreck/sideways_fronthalf_degraded"),
    MCSEED_PIECE(MCSEED_ID_SHIPWRECK, MCSEED_PIECE_EXACT, 0, "shipwreck/sideways_backhalf_degraded"),
    MCSEED_PIECE(MCSEED_ID_SHIPWRECK, MCSEED_PIECE_EXACT, 0, "shipwreck/rightsideup_full_degraded"),
    MCSEED_PIECE(MCSEED_ID_SHIPWRECK, MCSEED_PIECE_EXACT, 0, "shipwreck/rightsideup_fronthalf_degraded"),
    MCSEED_PIECE(MCSEED_ID_SHIPWRECK, MCSEED_PIECE_EXACT, 0, "shipwreck/rightsideup_backhalf_degraded"),

    MCSEED_PIECE(MCSEED_ID_FORTRESS, MCSEED_PIECE_EXACT, 1, "blaze_spawner"),
    MCSEED_PIECE(MCSEED_ID_FORTRESS, MCSEED_PIECE_EXACT, 1, "nether_wart_room"),
    MCSEED_PIECE(MCSEED_ID_FORTRESS, MCSEED_PIECE_EXACT, 0, "fortress/start"),
    MCSEED_PIECE(MCSEED_ID_FORTRESS, MCSEED_PIECE_EXACT, 0, "fortress/bridge_straight"),
    MCSEED_PIECE(MCSEED_ID_FORTRESS, MCSEED_PIECE_EXACT, 0, "fortress/bridge_crossing"),
    MCSEED_PIECE(MCSEED_ID_FORTRESS, MCSEED_PIECE_EXACT, 0, "fortress/bridge_fortified_crossing"),
    MCSEED_PIECE(MCSEED_ID_FORTRESS, MCSEED_PIECE_EXACT, 0, "fortress/bridge_stairs"),
    MCSEED_PIECE(MCSEED_ID_FORTRESS, MCSEED_PIECE_EXACT, 0, "fortress/blaze_spawner"),
    MCSEED_PIECE(MCSEED_ID_FORTRESS, MCSEED_PIECE_EXACT, 0, "fortress/corridor_entrance"),
    MCSEED_PIECE(MCSEED_ID_FORTRESS, MCSEED_PIECE_EXACT, 0, "fortress/corridor_straight"),
    MCSEED_PIECE(MCSEED_ID_FORTRESS, MCSEED_PIECE_EXACT, 0, "fortress/corridor_crossing"),
    MCSEED_PIECE(MCSEED_ID_FORTRESS, MCSEED_PIECE_EXACT, 0, "fortress/corridor_turn_right"),
    MCSEED_PIECE(MCSEED_ID_FORTRESS, MCSEED_PIECE_EXACT, 0, "fortress/corridor_turn_left"),
    MCSEED_PIECE(MCSEED_ID_FORTRESS, MCSEED_PIECE_EXACT, 0, "fortress/corridor_stairs"),
    MCSEED_PIECE(MCSEED_ID_FORTRESS, MCSEED_PIECE_EXACT, 0, "fortress/corridor_t_crossing"),
    MCSEED_PIECE(MCSEED_ID_FORTRESS, MCSEED_PIECE_EXACT, 0, "fortress/nether_wart_room"),
    MCSEED_PIECE(MCSEED_ID_FORTRESS, MCSEED_PIECE_EXACT, 0, "fortress/end"),

    MCSEED_PIECE(MCSEED_ID_BASTION, MCSEED_PIECE_PARTIAL, 0, "bastion/units/walls/wall_base"),
    MCSEED_PIECE(MCSEED_ID_BASTION, MCSEED_PIECE_PARTIAL, 0, "bastion/hoglin_stable/ramparts/ramparts_3"),
    MCSEED_PIECE(MCSEED_ID_BASTION, MCSEED_PIECE_PARTIAL, 0, "bastion/treasure/ramparts/mid_wall_main"),
    MCSEED_PIECE(MCSEED_ID_BASTION, MCSEED_PIECE_PARTIAL, 0, "bastion/bridge/starting_pieces/entrance"),

    MCSEED_PIECE(MCSEED_ID_END_CITY, MCSEED_PIECE_EXACT, 1, "ship"),
    MCSEED_PIECE(MCSEED_ID_END_CITY, MCSEED_PIECE_EXACT, 0, "end_city/base_floor"),
    MCSEED_PIECE(MCSEED_ID_END_CITY, MCSEED_PIECE_EXACT, 0, "end_city/base_roof"),
    MCSEED_PIECE(MCSEED_ID_END_CITY, MCSEED_PIECE_EXACT, 0, "end_city/bridge_end"),
    MCSEED_PIECE(MCSEED_ID_END_CITY, MCSEED_PIECE_EXACT, 0, "end_city/bridge_gentle_stairs"),
    MCSEED_PIECE(MCSEED_ID_END_CITY, MCSEED_PIECE_EXACT, 0, "end_city/bridge_piece"),
    MCSEED_PIECE(MCSEED_ID_END_CITY, MCSEED_PIECE_EXACT, 0, "end_city/bridge_steep_stairs"),
    MCSEED_PIECE(MCSEED_ID_END_CITY, MCSEED_PIECE_EXACT, 0, "end_city/fat_tower_base"),
    MCSEED_PIECE(MCSEED_ID_END_CITY, MCSEED_PIECE_EXACT, 0, "end_city/fat_tower_middle"),
    MCSEED_PIECE(MCSEED_ID_END_CITY, MCSEED_PIECE_EXACT, 0, "end_city/fat_tower_top"),
    MCSEED_PIECE(MCSEED_ID_END_CITY, MCSEED_PIECE_EXACT, 0, "end_city/second_floor_1"),
    MCSEED_PIECE(MCSEED_ID_END_CITY, MCSEED_PIECE_EXACT, 0, "end_city/second_floor_2"),
    MCSEED_PIECE(MCSEED_ID_END_CITY, MCSEED_PIECE_EXACT, 0, "end_city/second_roof"),
    MCSEED_PIECE(MCSEED_ID_END_CITY, MCSEED_PIECE_EXACT, 0, "end_city/ship"),
    MCSEED_PIECE(MCSEED_ID_END_CITY, MCSEED_PIECE_EXACT, 0, "end_city/third_floor_1"),
    MCSEED_PIECE(MCSEED_ID_END_CITY, MCSEED_PIECE_EXACT, 0, "end_city/third_floor_2"),
    MCSEED_PIECE(MCSEED_ID_END_CITY, MCSEED_PIECE_EXACT, 0, "end_city/third_roof"),
    MCSEED_PIECE(MCSEED_ID_END_CITY, MCSEED_PIECE_EXACT, 0, "end_city/tower_base"),
    MCSEED_PIECE(MCSEED_ID_END_CITY, MCSEED_PIECE_EXACT, 0, "end_city/tower_floor"),
    MCSEED_PIECE(MCSEED_ID_END_CITY, MCSEED_PIECE_EXACT, 0, "end_city/tower_piece"),
    MCSEED_PIECE(MCSEED_ID_END_CITY, MCSEED_PIECE_EXACT, 0, "end_city/tower_top"),

    MCSEED_PIECE(MCSEED_ID_STRONGHOLD, MCSEED_PIECE_EXACT, 1, "portal_room"),
    MCSEED_PIECE(MCSEED_ID_STRONGHOLD, MCSEED_PIECE_EXACT, 1, "library"),
    MCSEED_PIECE(MCSEED_ID_STRONGHOLD, MCSEED_PIECE_EXACT, 0, "stronghold/straight"),
    MCSEED_PIECE(MCSEED_ID_STRONGHOLD, MCSEED_PIECE_EXACT, 0, "stronghold/prison_hall"),
    MCSEED_PIECE(MCSEED_ID_STRONGHOLD, MCSEED_PIECE_EXACT, 0, "stronghold/left_turn"),
    MCSEED_PIECE(MCSEED_ID_STRONGHOLD, MCSEED_PIECE_EXACT, 0, "stronghold/right_turn"),
    MCSEED_PIECE(MCSEED_ID_STRONGHOLD, MCSEED_PIECE_EXACT, 0, "stronghold/room_crossing"),
    MCSEED_PIECE(MCSEED_ID_STRONGHOLD, MCSEED_PIECE_EXACT, 0, "stronghold/straight_stairs_down"),
    MCSEED_PIECE(MCSEED_ID_STRONGHOLD, MCSEED_PIECE_EXACT, 0, "stronghold/stairs_down"),
    MCSEED_PIECE(MCSEED_ID_STRONGHOLD, MCSEED_PIECE_EXACT, 0, "stronghold/five_crossing"),
    MCSEED_PIECE(MCSEED_ID_STRONGHOLD, MCSEED_PIECE_EXACT, 0, "stronghold/chest_corridor"),
    MCSEED_PIECE(MCSEED_ID_STRONGHOLD, MCSEED_PIECE_EXACT, 0, "stronghold/library"),
    MCSEED_PIECE(MCSEED_ID_STRONGHOLD, MCSEED_PIECE_EXACT, 0, "stronghold/portal_room"),
    MCSEED_PIECE(MCSEED_ID_STRONGHOLD, MCSEED_PIECE_EXACT, 0, "stronghold/filler_corridor"),

    MCSEED_PIECE(MCSEED_ID_RUINED_PORTAL, MCSEED_PIECE_EXACT, 1, "normal"),
    MCSEED_PIECE(MCSEED_ID_RUINED_PORTAL, MCSEED_PIECE_EXACT, 1, "giant"),
    MCSEED_PIECE(MCSEED_ID_RUINED_PORTAL, MCSEED_PIECE_EXACT, 0, "ruined_portal/portal_1"),
    MCSEED_PIECE(MCSEED_ID_RUINED_PORTAL, MCSEED_PIECE_EXACT, 0, "ruined_portal/portal_2"),
    MCSEED_PIECE(MCSEED_ID_RUINED_PORTAL, MCSEED_PIECE_EXACT, 0, "ruined_portal/portal_3"),
    MCSEED_PIECE(MCSEED_ID_RUINED_PORTAL, MCSEED_PIECE_EXACT, 0, "ruined_portal/portal_4"),
    MCSEED_PIECE(MCSEED_ID_RUINED_PORTAL, MCSEED_PIECE_EXACT, 0, "ruined_portal/portal_5"),
    MCSEED_PIECE(MCSEED_ID_RUINED_PORTAL, MCSEED_PIECE_EXACT, 0, "ruined_portal/portal_6"),
    MCSEED_PIECE(MCSEED_ID_RUINED_PORTAL, MCSEED_PIECE_EXACT, 0, "ruined_portal/portal_7"),
    MCSEED_PIECE(MCSEED_ID_RUINED_PORTAL, MCSEED_PIECE_EXACT, 0, "ruined_portal/portal_8"),
    MCSEED_PIECE(MCSEED_ID_RUINED_PORTAL, MCSEED_PIECE_EXACT, 0, "ruined_portal/portal_9"),
    MCSEED_PIECE(MCSEED_ID_RUINED_PORTAL, MCSEED_PIECE_EXACT, 0, "ruined_portal/portal_10"),
    MCSEED_PIECE(MCSEED_ID_RUINED_PORTAL, MCSEED_PIECE_EXACT, 0, "ruined_portal/giant_portal_1"),
    MCSEED_PIECE(MCSEED_ID_RUINED_PORTAL, MCSEED_PIECE_EXACT, 0, "ruined_portal/giant_portal_2"),
    MCSEED_PIECE(MCSEED_ID_RUINED_PORTAL, MCSEED_PIECE_EXACT, 0, "ruined_portal/giant_portal_3"),
    MCSEED_PIECE(MCSEED_ID_RUINED_PORTAL_NETHER, MCSEED_PIECE_EXACT, 1, "normal"),
    MCSEED_PIECE(MCSEED_ID_RUINED_PORTAL_NETHER, MCSEED_PIECE_EXACT, 1, "giant"),
    MCSEED_PIECE(MCSEED_ID_RUINED_PORTAL_NETHER, MCSEED_PIECE_EXACT, 0, "ruined_portal/portal_1"),
    MCSEED_PIECE(MCSEED_ID_RUINED_PORTAL_NETHER, MCSEED_PIECE_EXACT, 0, "ruined_portal/portal_2"),
    MCSEED_PIECE(MCSEED_ID_RUINED_PORTAL_NETHER, MCSEED_PIECE_EXACT, 0, "ruined_portal/portal_3"),
    MCSEED_PIECE(MCSEED_ID_RUINED_PORTAL_NETHER, MCSEED_PIECE_EXACT, 0, "ruined_portal/portal_4"),
    MCSEED_PIECE(MCSEED_ID_RUINED_PORTAL_NETHER, MCSEED_PIECE_EXACT, 0, "ruined_portal/portal_5"),
    MCSEED_PIECE(MCSEED_ID_RUINED_PORTAL_NETHER, MCSEED_PIECE_EXACT, 0, "ruined_portal/portal_6"),
    MCSEED_PIECE(MCSEED_ID_RUINED_PORTAL_NETHER, MCSEED_PIECE_EXACT, 0, "ruined_portal/portal_7"),
    MCSEED_PIECE(MCSEED_ID_RUINED_PORTAL_NETHER, MCSEED_PIECE_EXACT, 0, "ruined_portal/portal_8"),
    MCSEED_PIECE(MCSEED_ID_RUINED_PORTAL_NETHER, MCSEED_PIECE_EXACT, 0, "ruined_portal/portal_9"),
    MCSEED_PIECE(MCSEED_ID_RUINED_PORTAL_NETHER, MCSEED_PIECE_EXACT, 0, "ruined_portal/portal_10"),
    MCSEED_PIECE(MCSEED_ID_RUINED_PORTAL_NETHER, MCSEED_PIECE_EXACT, 0, "ruined_portal/giant_portal_1"),
    MCSEED_PIECE(MCSEED_ID_RUINED_PORTAL_NETHER, MCSEED_PIECE_EXACT, 0, "ruined_portal/giant_portal_2"),
    MCSEED_PIECE(MCSEED_ID_RUINED_PORTAL_NETHER, MCSEED_PIECE_EXACT, 0, "ruined_portal/giant_portal_3"),
};

#undef MCSEED_PIECE

struct McSeedContext {
    Generator overworld;
    Generator nether;
    Generator end;
    uint64_t seed;
    uint8_t ready_mask;
    uint8_t spawn_ready;
    uint8_t estimated_spawn_ready;
    Pos spawn;
    Pos estimated_spawn;
    uint64_t estimated_spawn_rng;
    int32_t spawn_y;
    int32_t spawn_biome;
    Piece *piece_buffer;
    McJigsawPiece *jigsaw_buffer;
    McJigsawWorkspace *jigsaw_workspace;
    SpawnSearchWorkspace *spawn_search_workspace;
    TerrainNoise terrain;
    McSeedTerrainColumn terrain_columns[MCSEED_TERRAIN_COLUMN_CACHE_SIZE];
    uint8_t terrain_setup;
    uint8_t terrain_ready;
};

static const size_t STRUCTURE_COUNT = sizeof(STRUCTURES) / sizeof(STRUCTURES[0]);
static const size_t STATIC_PIECE_COUNT = sizeof(STATIC_PIECES) / sizeof(STATIC_PIECES[0]);

static const StructureConfig NETHER_FOSSIL_CONFIG = {
    14357921, 2, 1, 0, DIM_NETHER, 0.0f
};

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
    uint64_t absolute_dx = dx < 0 ? (uint64_t)(-dx) : (uint64_t)dx;
    uint64_t absolute_dz = dz < 0 ? (uint64_t)(-dz) : (uint64_t)dz;
    uint64_t radius_squared = (uint64_t)radius * radius;
    uint64_t dx_squared;
    uint64_t dz_squared;
    if (absolute_dx > radius || absolute_dz > radius)
        return 0;
    dx_squared = absolute_dx * absolute_dx;
    dz_squared = absolute_dz * absolute_dz;
    return dz_squared <= radius_squared - dx_squared;
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
    context->piece_buffer = (Piece *)calloc(
        MCSEED_PIECE_BUFFER_CAPACITY,
        sizeof(*context->piece_buffer)
    );
    context->jigsaw_buffer = (McJigsawPiece *)calloc(
        MCSEED_JIGSAW_BUFFER_CAPACITY,
        sizeof(*context->jigsaw_buffer)
    );
    context->jigsaw_workspace = mcjigsaw_workspace_create();
    context->spawn_search_workspace = createSpawnSearchWorkspace();
    if (!context->piece_buffer || !context->jigsaw_buffer ||
        !context->jigsaw_workspace || !context->spawn_search_workspace) {
        free(context->piece_buffer);
        free(context->jigsaw_buffer);
        mcjigsaw_workspace_destroy(context->jigsaw_workspace);
        freeSpawnSearchWorkspace(context->spawn_search_workspace);
        free(context);
        return NULL;
    }
    setupGenerator(&context->overworld, MCSEED_CUBIOMES_VERSION, 0);
    setupGenerator(&context->nether, MCSEED_CUBIOMES_VERSION, 0);
    setupGenerator(&context->end, MCSEED_CUBIOMES_VERSION, 0);
    context->terrain_setup = (uint8_t)setupTerrainNoise(
        &context->terrain,
        MCSEED_CUBIOMES_VERSION,
        0
    );
    return context;
}

void mcseed_context_destroy(McSeedContext *context)
{
    if (!context)
        return;
    free(context->piece_buffer);
    free(context->jigsaw_buffer);
    mcjigsaw_workspace_destroy(context->jigsaw_workspace);
    freeSpawnSearchWorkspace(context->spawn_search_workspace);
    free(context);
}

void mcseed_context_set_seed(McSeedContext *context, uint64_t seed)
{
    if (!context)
        return;
    context->seed = seed;
    context->ready_mask = 0;
    context->spawn_ready = 0;
    context->estimated_spawn_ready = 0;
    context->terrain_ready = 0;
}

static void prepare_estimated_spawn(
    McSeedContext *context,
    const Generator *generator
)
{
    if (context->estimated_spawn_ready)
        return;
    context->estimated_spawn_rng = 0;
    context->estimated_spawn = estimateSpawnWithWorkspace(
        generator,
        &context->estimated_spawn_rng,
        context->spawn_search_workspace
    );
    context->estimated_spawn_ready = 1;
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
        prepare_estimated_spawn(context, generator);
        context->spawn = getSpawnForEstimate(
            generator,
            context->estimated_spawn,
            context->estimated_spawn_rng
        );
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

int32_t mcseed_spawn_from_estimate(
    McSeedContext *context,
    int32_t estimate_x,
    int32_t estimate_z,
    McSeedHit *spawn,
    int32_t *biome_id
)
{
    if (!context || !spawn || !biome_id)
        return -1;
    if (MCSEED_CUBIOMES_VERSION < MC_1_18)
        return -2;
    context->estimated_spawn.x = estimate_x;
    context->estimated_spawn.z = estimate_z;
    context->estimated_spawn_rng = 0;
    context->estimated_spawn_ready = 1;
    context->spawn_ready = 0;
    return mcseed_spawn(context, spawn, biome_id);
}

int32_t mcseed_estimated_spawn(McSeedContext *context, McSeedHit *spawn)
{
    Generator *generator;
    if (!context || !spawn)
        return -1;
    generator = generator_for_dimension(context, DIM_OVERWORLD);
    if (!generator)
        return -2;
    prepare_estimated_spawn(context, generator);
    spawn->x = context->estimated_spawn.x;
    spawn->y = INT32_MIN;
    spawn->z = context->estimated_spawn.z;
    spawn->id = 0;
    return 0;
}

int32_t mcseed_estimated_spawn_reference(McSeedContext *context, McSeedHit *spawn)
{
    Generator *generator;
    Pos position;
    if (!context || !spawn)
        return -1;
    generator = generator_for_dimension(context, DIM_OVERWORLD);
    if (!generator)
        return -2;
    position = estimateSpawnReference(generator, NULL);
    spawn->x = position.x;
    spawn->y = INT32_MIN;
    spawn->z = position.z;
    spawn->id = 0;
    return 0;
}

int32_t mcseed_spawn_refinement_radius(uint32_t *radius)
{
    uint32_t axis_bound;
    uint32_t squared_bound;
    uint32_t horizontal_bound = 0;
    if (!radius)
        return -1;
    if (MCSEED_CUBIOMES_VERSION < MC_1_18)
        return 0;

    /*
     * Modern getSpawn scans offsets j,k in [-5,5]. Relative to the estimated
     * chunk centre, a sampled block coordinate is j*16 - 8 + {0,4,8,12}, so
     * the maximum absolute offset on either axis is 5*16+8 = 88 blocks.
     */
    axis_bound = 5 * 16 + 8;
    squared_bound = 2 * axis_bound * axis_bound;
    while (horizontal_bound * horizontal_bound < squared_bound)
        horizontal_bound++;
    *radius = horizontal_bound;
    return 1;
}

int32_t mcseed_spawn_origin_radius(uint32_t *radius)
{
    uint32_t refinement_radius;
    int estimate_radius;
    int32_t status;
    if (!radius)
        return -1;
    estimate_radius = getSpawnEstimateOriginRadius(MCSEED_CUBIOMES_VERSION);
    if (estimate_radius <= 0)
        return 0;
    status = mcseed_spawn_refinement_radius(&refinement_radius);
    if (status != 1)
        return status;
    if ((uint32_t)estimate_radius > UINT32_MAX - refinement_radius)
        return -2;
    *radius = (uint32_t)estimate_radius + refinement_radius;
    return 1;
}

int32_t mcseed_biome_count(void)
{
    int32_t id;
    int32_t count = 0;
    for (id = 0; id <= UINT8_MAX; id++) {
        if (biomeExists(MCSEED_CUBIOMES_VERSION, id) &&
            biome2str(MCSEED_CUBIOMES_VERSION, id))
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
        if (!biomeExists(MCSEED_CUBIOMES_VERSION, id) ||
            !biome2str(MCSEED_CUBIOMES_VERSION, id))
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
    return id == none ? NULL : biome2str(MCSEED_CUBIOMES_VERSION, id);
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
        if (!biomeExists(MCSEED_CUBIOMES_VERSION, id))
            continue;
        candidate = biome2str(MCSEED_CUBIOMES_VERSION, id);
        if (candidate && strcmp(name, candidate) == 0)
            return id;
    }
    return none;
}

const char *mcseed_biome_name_from_id(int32_t id)
{
    if (!biomeExists(MCSEED_CUBIOMES_VERSION, id))
        return NULL;
    return biome2str(MCSEED_CUBIOMES_VERSION, id);
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

int32_t mcseed_structure_gpu_config(
    int32_t structure_id,
    McSeedGpuStructureConfig *output
)
{
    const McSeedStructureInfo *info = structure_by_id(structure_id);
    StructureConfig config;
    int32_t kind;
    int32_t flags = 0;
    if (!info || !output)
        return -1;
    memset(output, 0, sizeof(*output));

    if (info->cubiomes_type == MCSEED_STRUCTURE_NETHER_FOSSIL) {
        config = NETHER_FOSSIL_CONFIG;
        kind = MCSEED_GPU_PLACEMENT_FEATURE;
    } else {
        if (!getStructureConfig(
                info->cubiomes_type,
                MCSEED_CUBIOMES_VERSION,
                &config
            ))
            return 0;
        switch (info->cubiomes_type) {
        case Monument:
        case Mansion:
            kind = MCSEED_GPU_PLACEMENT_LARGE;
            break;
        case End_City:
            kind = MCSEED_GPU_PLACEMENT_LARGE;
            flags |= MCSEED_GPU_PLACEMENT_END_DISTANCE;
            break;
        case Outpost:
            kind = MCSEED_GPU_PLACEMENT_OUTPOST;
            break;
        case Treasure:
            kind = MCSEED_GPU_PLACEMENT_TREASURE;
            break;
        case Fortress:
            if (MCSEED_CUBIOMES_VERSION < MC_1_18)
                return 0;
            kind = MCSEED_GPU_PLACEMENT_FORTRESS;
            break;
        case Bastion:
            if (MCSEED_CUBIOMES_VERSION < MC_1_18)
                return 0;
            kind = MCSEED_GPU_PLACEMENT_BASTION;
            break;
        case Desert_Pyramid:
        case Jungle_Pyramid:
        case Swamp_Hut:
        case Igloo:
        case Village:
        case Ocean_Ruin:
        case Shipwreck:
        case Ruined_Portal:
        case Ruined_Portal_N:
        case Ancient_City:
        case Trail_Ruins:
        case Trial_Chambers:
            kind = MCSEED_GPU_PLACEMENT_FEATURE;
            break;
        case Mineshaft:
        case Stronghold:
        default:
            return 0;
        }
    }

    output->kind = kind;
    output->salt = config.salt;
    output->region_size = config.regionSize;
    output->chunk_range = config.chunkRange;
    output->flags = flags;
    return 1;
}

static const McJigsawElementData *village_element_for_list_index(int32_t requested)
{
    const McJigsawData *data = mcjigsaw_village_data(MCSEED_VERSION_NAME);
    int32_t current = 0;
    size_t index;
    if (!data || requested < 0)
        return NULL;
    for (index = 0; index < data->element_count; index++) {
        const McJigsawElementData *element = &data->elements[index];
        if (element->kind == MCJIGSAW_ELEMENT_EMPTY)
            continue;
        if (current == requested)
            return element;
        current++;
    }
    return NULL;
}

static int32_t village_element_count(void)
{
    int32_t count = 0;
    while (village_element_for_list_index(count))
        count++;
    return count;
}

int32_t mcseed_piece_count(void)
{
    return (int32_t)STATIC_PIECE_COUNT + village_element_count();
}

const char *mcseed_piece_name_at(int32_t index)
{
    const McJigsawElementData *element;
    if (index < 0)
        return NULL;
    if ((size_t)index < STATIC_PIECE_COUNT)
        return STATIC_PIECES[index].name;
    element = village_element_for_list_index(index - (int32_t)STATIC_PIECE_COUNT);
    return element ? element->name : NULL;
}

int32_t mcseed_piece_structure_id_at(int32_t index)
{
    if (index < 0)
        return -1;
    if ((size_t)index < STATIC_PIECE_COUNT)
        return STATIC_PIECES[index].structure_id;
    return village_element_for_list_index(index - (int32_t)STATIC_PIECE_COUNT)
        ? MCSEED_ID_VILLAGE
        : -1;
}

int32_t mcseed_piece_accuracy_at(int32_t index)
{
    if (index < 0)
        return -1;
    if ((size_t)index < STATIC_PIECE_COUNT)
        return STATIC_PIECES[index].accuracy;
    return village_element_for_list_index(index - (int32_t)STATIC_PIECE_COUNT)
        ? MCSEED_PIECE_APPROXIMATE_TERRAIN
        : -1;
}

int32_t mcseed_piece_is_group_at(int32_t index)
{
    if (index < 0)
        return -1;
    if ((size_t)index < STATIC_PIECE_COUNT)
        return STATIC_PIECES[index].is_group;
    return village_element_for_list_index(index - (int32_t)STATIC_PIECE_COUNT) ? 0 : -1;
}

int32_t mcseed_piece_selector_valid(int32_t structure_id, const char *name)
{
    const McJigsawData *data;
    size_t index;
    name = without_namespace(name);
    if (!name)
        return 0;
    for (index = 0; index < STATIC_PIECE_COUNT; index++) {
        if (STATIC_PIECES[index].structure_id == structure_id &&
            strcmp(STATIC_PIECES[index].name, name) == 0)
            return 1;
    }
    if (structure_id != MCSEED_ID_VILLAGE)
        return 0;
    data = mcjigsaw_village_data(MCSEED_VERSION_NAME);
    if (!data)
        return 0;
    for (index = 0; index < data->element_count; index++) {
        if (data->elements[index].kind != MCJIGSAW_ELEMENT_EMPTY &&
            strcmp(data->elements[index].name, name) == 0)
            return 1;
    }
    return 0;
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
    initFirstStronghold(&iterator, MCSEED_CUBIOMES_VERSION, context->seed);
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
    const StructureConfig config = NETHER_FOSSIL_CONFIG;
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
    if (!getStructureConfig(info->cubiomes_type, MCSEED_CUBIOMES_VERSION, &config))
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
                    MCSEED_CUBIOMES_VERSION,
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

static const char *const FORTRESS_PIECE_NAMES[PIECE_COUNT] = {
    "fortress/start",
    "fortress/bridge_straight",
    "fortress/bridge_crossing",
    "fortress/bridge_fortified_crossing",
    "fortress/bridge_stairs",
    "fortress/blaze_spawner",
    "fortress/corridor_entrance",
    "fortress/corridor_straight",
    "fortress/corridor_crossing",
    "fortress/corridor_turn_right",
    "fortress/corridor_turn_left",
    "fortress/corridor_stairs",
    "fortress/corridor_t_crossing",
    "fortress/nether_wart_room",
    "fortress/end",
};

static const char *const END_CITY_PIECE_NAMES[TOWER_TOP + 1] = {
    "end_city/base_floor",
    "end_city/base_roof",
    "end_city/bridge_end",
    "end_city/bridge_gentle_stairs",
    "end_city/bridge_piece",
    "end_city/bridge_steep_stairs",
    "end_city/fat_tower_base",
    "end_city/fat_tower_middle",
    "end_city/fat_tower_top",
    "end_city/second_floor_1",
    "end_city/second_floor_2",
    "end_city/second_roof",
    "end_city/ship",
    "end_city/third_floor_1",
    "end_city/third_floor_2",
    "end_city/third_roof",
    "end_city/tower_base",
    "end_city/tower_floor",
    "end_city/tower_piece",
    "end_city/tower_top",
};

static const char *const STRONGHOLD_PIECE_NAMES[SH_PIECE_COUNT] = {
    "stronghold/straight",
    "stronghold/prison_hall",
    "stronghold/left_turn",
    "stronghold/right_turn",
    "stronghold/room_crossing",
    "stronghold/straight_stairs_down",
    "stronghold/stairs_down",
    "stronghold/five_crossing",
    "stronghold/chest_corridor",
    "stronghold/library",
    "stronghold/portal_room",
    "stronghold/filler_corridor",
};

static const char *const NORMAL_PORTAL_NAMES[10] = {
    "ruined_portal/portal_1",
    "ruined_portal/portal_2",
    "ruined_portal/portal_3",
    "ruined_portal/portal_4",
    "ruined_portal/portal_5",
    "ruined_portal/portal_6",
    "ruined_portal/portal_7",
    "ruined_portal/portal_8",
    "ruined_portal/portal_9",
    "ruined_portal/portal_10",
};

static const char *const GIANT_PORTAL_NAMES[3] = {
    "ruined_portal/giant_portal_1",
    "ruined_portal/giant_portal_2",
    "ruined_portal/giant_portal_3",
};

static uint32_t terrain_column_hash(int32_t cell_x, int32_t cell_z)
{
    uint32_t x = (uint32_t)cell_x;
    uint32_t z = (uint32_t)cell_z;
    x ^= x >> 16;
    x *= UINT32_C(0x7feb352d);
    x ^= x >> 15;
    z ^= z >> 16;
    z *= UINT32_C(0x846ca68b);
    z ^= z >> 16;
    return x ^ z;
}

static const double *terrain_noise_column(
    McSeedContext *context,
    int32_t cell_x,
    int32_t cell_z
)
{
    size_t slot = terrain_column_hash(cell_x, cell_z) &
        (MCSEED_TERRAIN_COLUMN_CACHE_SIZE - 1);
    McSeedTerrainColumn *column = &context->terrain_columns[slot];
    if (!column->valid || column->cell_x != cell_x || column->cell_z != cell_z) {
        sampleNoiseColumn(&context->terrain, cell_x, cell_z, column->values);
        column->cell_x = cell_x;
        column->cell_z = cell_z;
        column->valid = 1;
    }
    return column->values;
}

static int32_t village_surface_height(void *opaque, int32_t x, int32_t z)
{
    McSeedContext *context = (McSeedContext *)opaque;
    int32_t cell_x;
    int32_t cell_z;
    int32_t height;
    const double *column_00;
    const double *column_01;
    const double *column_10;
    const double *column_11;
    if (!context || !context->terrain_setup)
        return 64;
    if (!context->terrain_ready) {
        if (!initTerrainNoise(&context->terrain, context->seed, DIM_OVERWORLD))
            return 64;
        memset(context->terrain_columns, 0, sizeof(context->terrain_columns));
        context->terrain_ready = 1;
    }
    cell_x = (int32_t)floor_div_i64(x, 4);
    cell_z = (int32_t)floor_div_i64(z, 4);
    column_00 = terrain_noise_column(context, cell_x, cell_z);
    column_01 = terrain_noise_column(context, cell_x, cell_z + 1);
    column_10 = terrain_noise_column(context, cell_x + 1, cell_z);
    column_11 = terrain_noise_column(context, cell_x + 1, cell_z + 1);
    /* generateColumn returns an index in the 384-block array whose origin is Y=-64. */
    height = generateColumn(
        x,
        z,
        NULL,
        column_00,
        column_01,
        column_10,
        column_11,
        1
    ) - 64;
    /* Sea level is 63; water occupies Y < 63, so the first free Y is 63. */
    return height < 63 ? 63 : height;
}

static int contains_text(const char *value, const char *needle)
{
    return value && needle && strstr(value, needle) != NULL;
}

static int village_profession_piece(const char *name)
{
    return contains_text(name, "armorer") ||
           contains_text(name, "weaponsmith") ||
           contains_text(name, "weapon_smith") ||
           contains_text(name, "tool_smith") ||
           contains_text(name, "toolsmith") ||
           contains_text(name, "butcher") ||
           contains_text(name, "cartographer") ||
           contains_text(name, "fisher") ||
           contains_text(name, "fletcher") ||
           contains_text(name, "library") ||
           contains_text(name, "mason") ||
           contains_text(name, "shepherd") ||
           contains_text(name, "tannery") ||
           contains_text(name, "temple");
}

static int village_piece_group_matches(
    const char *selector,
    const char *piece_name
)
{
    int is_house = contains_text(piece_name, "/houses/");
    if (name_is(selector, "blacksmith") || name_is(selector, "smith"))
        return contains_text(piece_name, "armorer") ||
               contains_text(piece_name, "weaponsmith") ||
               contains_text(piece_name, "weapon_smith") ||
               contains_text(piece_name, "tool_smith") ||
               contains_text(piece_name, "toolsmith");
    if (name_is(selector, "house"))
        return is_house;
    if (name_is(selector, "profession_house"))
        return is_house && village_profession_piece(piece_name);
    if (name_is(selector, "residential"))
        return is_house && (contains_text(piece_name, "small_house") ||
                            contains_text(piece_name, "medium_house") ||
                            contains_text(piece_name, "big_house") ||
                            contains_text(piece_name, "large_house"));
    if (name_is(selector, "armorer"))
        return contains_text(piece_name, "armorer");
    if (name_is(selector, "weaponsmith"))
        return contains_text(piece_name, "weaponsmith") ||
               contains_text(piece_name, "weapon_smith");
    if (name_is(selector, "toolsmith"))
        return contains_text(piece_name, "tool_smith") || contains_text(piece_name, "toolsmith");
    if (name_is(selector, "butcher"))
        return contains_text(piece_name, "butcher");
    if (name_is(selector, "cartographer"))
        return contains_text(piece_name, "cartographer");
    if (name_is(selector, "fisher") || name_is(selector, "fisherman"))
        return contains_text(piece_name, "fisher");
    if (name_is(selector, "fletcher"))
        return contains_text(piece_name, "fletcher");
    if (name_is(selector, "library") || name_is(selector, "librarian"))
        return contains_text(piece_name, "library");
    if (name_is(selector, "mason"))
        return contains_text(piece_name, "mason");
    if (name_is(selector, "shepherd"))
        return contains_text(piece_name, "shepherd");
    if (name_is(selector, "tannery") || name_is(selector, "leatherworker"))
        return contains_text(piece_name, "tannery");
    if (name_is(selector, "temple") || name_is(selector, "cleric"))
        return contains_text(piece_name, "temple");
    if (name_is(selector, "farm") || name_is(selector, "farmer"))
        return is_house && contains_text(piece_name, "farm");
    if (name_is(selector, "animal_pen"))
        return contains_text(piece_name, "animal_pen");
    if (name_is(selector, "stable"))
        return contains_text(piece_name, "stable");
    if (name_is(selector, "small_house"))
        return contains_text(piece_name, "small_house");
    if (name_is(selector, "medium_house"))
        return contains_text(piece_name, "medium_house");
    if (name_is(selector, "large_house"))
        return contains_text(piece_name, "large_house") || contains_text(piece_name, "big_house");
    if (name_is(selector, "meeting_point"))
        return contains_text(piece_name, "meeting_point") || contains_text(piece_name, "fountain");
    if (name_is(selector, "town_center"))
        return contains_text(piece_name, "/town_centers/");
    if (name_is(selector, "street"))
        return contains_text(piece_name, "/streets/");
    if (name_is(selector, "terminator"))
        return contains_text(piece_name, "/terminators/");
    if (name_is(selector, "decor"))
        return contains_text(piece_name, "_lamp_") ||
               contains_text(piece_name, "_decoration_") ||
               contains_text(piece_name, "/decor/") ||
               contains_text(piece_name, "feature/");
    if (name_is(selector, "feature"))
        return strncmp(piece_name, "feature/", 8) == 0;
    if (name_is(selector, "zombie"))
        return contains_text(piece_name, "/zombie/");
    return 0;
}

static int piece_selector_matches(
    int32_t structure_id,
    const char *selector,
    const char *piece_name
)
{
    selector = without_namespace(selector);
    if (!selector || !piece_name)
        return 0;
    if (strcmp(selector, piece_name) == 0)
        return 1;
    if (structure_id == MCSEED_ID_VILLAGE)
        return village_piece_group_matches(selector, piece_name);
    if (structure_id == MCSEED_ID_SHIPWRECK) {
        if (name_is(selector, "full"))
            return contains_text(piece_name, "_full") ||
                   contains_text(piece_name, "with_mast");
        if (name_is(selector, "front_half"))
            return contains_text(piece_name, "fronthalf");
        if (name_is(selector, "back_half"))
            return contains_text(piece_name, "backhalf");
        if (name_is(selector, "mast"))
            return contains_text(piece_name, "with_mast");
        if (name_is(selector, "degraded"))
            return contains_text(piece_name, "_degraded");
        if (name_is(selector, "intact"))
            return !contains_text(piece_name, "_degraded");
    }
    if (structure_id == MCSEED_ID_FORTRESS) {
        if (name_is(selector, "blaze_spawner"))
            return strcmp(piece_name, "fortress/blaze_spawner") == 0;
        if (name_is(selector, "nether_wart_room"))
            return strcmp(piece_name, "fortress/nether_wart_room") == 0;
    }
    if (structure_id == MCSEED_ID_END_CITY && name_is(selector, "ship"))
        return strcmp(piece_name, "end_city/ship") == 0;
    if (structure_id == MCSEED_ID_STRONGHOLD) {
        if (name_is(selector, "portal_room"))
            return strcmp(piece_name, "stronghold/portal_room") == 0;
        if (name_is(selector, "library"))
            return strcmp(piece_name, "stronghold/library") == 0;
    }
    if (structure_id == MCSEED_ID_RUINED_PORTAL ||
        structure_id == MCSEED_ID_RUINED_PORTAL_NETHER) {
        if (name_is(selector, "normal"))
            return contains_text(piece_name, "/portal_");
        if (name_is(selector, "giant"))
            return contains_text(piece_name, "/giant_portal_");
    }
    return 0;
}

static int any_selector_matches(
    int32_t structure_id,
    const char *const *selectors,
    size_t selector_count,
    const char *piece_name
)
{
    size_t index;
    for (index = 0; index < selector_count; index++) {
        if (piece_selector_matches(structure_id, selectors[index], piece_name))
            return 1;
    }
    return 0;
}

static const char *canonical_piece_name(
    const McSeedStructureInfo *info,
    const Piece *piece,
    const StructureVariant *variant
)
{
    if (!info || !piece)
        return NULL;
    switch (info->cubiomes_type) {
    case Desert_Pyramid:
        return "desert_pyramid/main";
    case Jungle_Pyramid:
        return "jungle_pyramid/main";
    case Swamp_Hut:
        return "swamp_hut/main";
    case Fortress:
        return piece->type >= 0 && piece->type < PIECE_COUNT
            ? FORTRESS_PIECE_NAMES[(int)piece->type]
            : NULL;
    case End_City:
        return piece->type >= 0 && piece->type <= TOWER_TOP
            ? END_CITY_PIECE_NAMES[(int)piece->type]
            : NULL;
    case Stronghold:
        return piece->type >= 0 && piece->type < SH_PIECE_COUNT
            ? STRONGHOLD_PIECE_NAMES[(int)piece->type]
            : NULL;
    case Treasure:
        return "buried_treasure/chest";
    case Ruined_Portal:
    case Ruined_Portal_N:
        if (!variant || variant->start == 0)
            return NULL;
        if (variant->giant)
            return variant->start <= 3 ? GIANT_PORTAL_NAMES[variant->start - 1] : NULL;
        return variant->start <= 10 ? NORMAL_PORTAL_NAMES[variant->start - 1] : NULL;
    default:
        return piece->name;
    }
}

static int village_style_from_biome(int biome)
{
    switch (biome) {
    case plains:
    case meadow:
        return MCJIGSAW_STYLE_PLAINS;
    case desert:
        return MCJIGSAW_STYLE_DESERT;
    case savanna:
        return MCJIGSAW_STYLE_SAVANNA;
    case snowy_tundra:
        return MCJIGSAW_STYLE_SNOWY;
    case taiga:
        return MCJIGSAW_STYLE_TAIGA;
    default:
        return -1;
    }
}

static int store_piece_hit(
    McSeedPieceHit *hits,
    size_t hit_capacity,
    uint64_t total,
    const char *name,
    int32_t x,
    int32_t y,
    int32_t z,
    Pos parent,
    int32_t eye_mask
)
{
    if (total < hit_capacity && hits) {
        hits[total].x = x;
        hits[total].y = y;
        hits[total].z = z;
        hits[total].parent_x = parent.x;
        hits[total].parent_z = parent.z;
        hits[total].eye_mask = eye_mask;
        hits[total].name = name;
    }
    return 1;
}

static int evaluate_piece_candidate(
    McSeedContext *context,
    Generator *generator,
    const McSeedStructureInfo *info,
    Pos parent,
    int viable_biome,
    const char *const *selectors,
    size_t selector_count,
    uint64_t limit,
    McSeedPieceHit *hits,
    size_t hit_capacity,
    uint64_t *total
)
{
    size_t index;
    if (info->id == MCSEED_ID_VILLAGE) {
        const McJigsawData *data = mcjigsaw_village_data(MCSEED_VERSION_NAME);
        int style = village_style_from_biome(viable_biome);
        int32_t count;
        if (!data || style < 0)
            return -1;
        count = mcjigsaw_generate(
            context->jigsaw_workspace,
            data,
            context->seed,
            parent.x,
            parent.z,
            style,
            village_surface_height,
            context,
            context->jigsaw_buffer,
            MCSEED_JIGSAW_BUFFER_CAPACITY
        );
        if (count < 0)
            return -2;
        for (index = 0; index < (size_t)count; index++) {
            const McJigsawPiece *piece = &context->jigsaw_buffer[index];
            if (!any_selector_matches(info->id, selectors, selector_count, piece->name))
                continue;
            store_piece_hit(
                hits,
                hit_capacity,
                *total,
                piece->name,
                piece->x,
                piece->y,
                piece->z,
                parent,
                -1
            );
            (*total)++;
            if (*total >= limit)
                return 1;
        }
        return 0;
    }

    {
        StructureVariant variant;
        StructureSaltConfig salt_config;
        int variant_biome = viable_biome;
        int count;
        if (info->dimension == DIM_OVERWORLD) {
            variant_biome = getBiomeAt(
                generator,
                4,
                (parent.x >> 2) + 2,
                319 >> 2,
                (parent.z >> 2) + 2
            );
        } else if (info->dimension == DIM_NETHER) {
            variant_biome = getBiomeAt(
                generator,
                4,
                (parent.x >> 2) + 2,
                33 >> 2,
                (parent.z >> 2) + 2
            );
        }
        memset(&variant, 0, sizeof(variant));
        memset(&salt_config, 0, sizeof(salt_config));
        getVariant(
            &variant,
            info->cubiomes_type,
            MCSEED_CUBIOMES_VERSION,
            context->seed,
            parent.x,
            parent.z,
            variant_biome
        );
        getStructureSaltConfig(
            info->cubiomes_type,
            MCSEED_CUBIOMES_VERSION,
            variant.biome,
            &salt_config
        );
        memset(
            context->piece_buffer,
            0,
            MCSEED_PIECE_BUFFER_CAPACITY * sizeof(*context->piece_buffer)
        );
        count = getStructurePieces(
            context->piece_buffer,
            MCSEED_PIECE_BUFFER_CAPACITY,
            info->cubiomes_type,
            salt_config,
            &variant,
            MCSEED_CUBIOMES_VERSION,
            context->seed,
            parent.x,
            parent.z
        );
        if (count < 0)
            return -3;
        for (index = 0; index < (size_t)count; index++) {
            const Piece *piece = &context->piece_buffer[index];
            const char *name = canonical_piece_name(info, piece, &variant);
            int32_t piece_x = piece->pos.x;
            int32_t piece_z = piece->pos.z;
            int32_t eye_mask = -1;
            if (!name || !any_selector_matches(info->id, selectors, selector_count, name))
                continue;
            if (info->cubiomes_type == Bastion) {
                piece_x = parent.x;
                piece_z = parent.z;
            }
            if (info->cubiomes_type == Stronghold &&
                piece->type == SH_PORTAL_ROOM) {
                eye_mask = piece->additionalData & 0x0fff;
            }
            store_piece_hit(
                hits,
                hit_capacity,
                *total,
                name,
                piece_x,
                INT32_MIN,
                piece_z,
                parent,
                eye_mask
            );
            (*total)++;
            if (*total >= limit)
                return 1;
        }
    }
    return 0;
}

static uint64_t stronghold_locate_distance_squared(
    Pos position,
    int32_t anchor_x,
    int32_t anchor_z
)
{
    int64_t center_x = floor_div_i64(position.x, 16) * 16 + 8;
    int64_t center_z = floor_div_i64(position.z, 16) * 16 + 8;
    int64_t dx = center_x - anchor_x;
    int64_t dz = center_z - anchor_z;
    return (uint64_t)(dx * dx) + (uint64_t)(dz * dz);
}

static uint64_t stronghold_distance_bound_squared(
    Pos position,
    int32_t anchor_x,
    int32_t anchor_z,
    int upper_bound
)
{
    int64_t center_x = floor_div_i64(position.x, 16) * 16 + 8;
    int64_t center_z = floor_div_i64(position.z, 16) * 16 + 8;
    int64_t dx = center_x - anchor_x;
    int64_t dz = center_z - anchor_z;
    uint64_t absolute_dx = dx < 0 ? (uint64_t)(-dx) : (uint64_t)dx;
    uint64_t absolute_dz = dz < 0 ? (uint64_t)(-dz) : (uint64_t)dz;
    uint64_t bounded_dx;
    uint64_t bounded_dz;
    if (upper_bound) {
        bounded_dx = absolute_dx + MCSEED_STRONGHOLD_LOCATE_MARGIN;
        bounded_dz = absolute_dz + MCSEED_STRONGHOLD_LOCATE_MARGIN;
    } else {
        bounded_dx = absolute_dx > MCSEED_STRONGHOLD_LOCATE_MARGIN
            ? absolute_dx - MCSEED_STRONGHOLD_LOCATE_MARGIN
            : 0;
        bounded_dz = absolute_dz > MCSEED_STRONGHOLD_LOCATE_MARGIN
            ? absolute_dz - MCSEED_STRONGHOLD_LOCATE_MARGIN
            : 0;
    }
    return bounded_dx * bounded_dx + bounded_dz * bounded_dz;
}

static int nearest_stronghold_position(
    McSeedContext *context,
    Generator *generator,
    int32_t anchor_x,
    int32_t anchor_z,
    Pos *nearest
)
{
    Pos approximate[MCSEED_STRONGHOLD_COUNT];
    StrongholdIter iterator;
    uint64_t best_upper_bound = UINT64_MAX;
    uint64_t best_distance = UINT64_MAX;
    int approximate_count = 0;
    int found = 0;
    int index;

    if (MCSEED_CUBIOMES_VERSION <= MC_1_19_2) {
        initFirstStronghold(&iterator, MCSEED_CUBIOMES_VERSION, context->seed);
        for (index = 0; index < MCSEED_STRONGHOLD_COUNT; index++) {
            uint64_t distance;
            if (nextStronghold(&iterator, generator) <= 0)
                break;
            distance = stronghold_locate_distance_squared(
                iterator.pos,
                anchor_x,
                anchor_z
            );
            if (!found || distance < best_distance) {
                *nearest = iterator.pos;
                best_distance = distance;
                found = 1;
            }
        }
        return found;
    }

    /*
     * Biome adjustment can move a stronghold by at most the locate search
     * radius, plus chunk snapping. First collect cheap approximate positions
     * so exact biome searches are limited to candidates whose bounding square
     * can still beat the best possible upper bound.
     */
    initFirstStronghold(&iterator, MCSEED_CUBIOMES_VERSION, context->seed);
    for (index = 0; index < MCSEED_STRONGHOLD_COUNT; index++) {
        uint64_t upper_bound;
        if (nextStronghold(&iterator, NULL) <= 0)
            break;
        approximate[approximate_count++] = iterator.pos;
        upper_bound = stronghold_distance_bound_squared(
            iterator.pos,
            anchor_x,
            anchor_z,
            1
        );
        if (upper_bound < best_upper_bound)
            best_upper_bound = upper_bound;
    }

    initFirstStronghold(&iterator, MCSEED_CUBIOMES_VERSION, context->seed);
    for (index = 0; index < approximate_count; index++) {
        uint64_t lower_bound = stronghold_distance_bound_squared(
            approximate[index],
            anchor_x,
            anchor_z,
            0
        );
        const Generator *candidate_generator =
            lower_bound <= best_upper_bound && lower_bound <= best_distance
            ? generator
            : NULL;
        uint64_t distance;
        if (nextStronghold(&iterator, candidate_generator) <= 0)
            break;
        if (!candidate_generator)
            continue;
        distance = stronghold_locate_distance_squared(
            iterator.pos,
            anchor_x,
            anchor_z
        );
        if (!found || distance < best_distance) {
            *nearest = iterator.pos;
            best_distance = distance;
            found = 1;
        }
    }
    return found;
}

int32_t mcseed_nearest_stronghold_portal(
    McSeedContext *context,
    int32_t anchor_x,
    int32_t anchor_z,
    McSeedPieceHit *hit
)
{
    Generator *generator;
    StructureSaltConfig salt_config;
    StructureVariant variant;
    Pos nearest;
    int count;
    int index;
    if (!context || !hit)
        return -1;
    memset(hit, 0, sizeof(*hit));
    hit->y = INT32_MIN;
    hit->eye_mask = -1;
    generator = generator_for_dimension(context, DIM_OVERWORLD);
    if (!generator)
        return -2;
    if (!nearest_stronghold_position(
            context,
            generator,
            anchor_x,
            anchor_z,
            &nearest)) {
        return -3;
    }
    memset(&salt_config, 0, sizeof(salt_config));
    memset(&variant, 0, sizeof(variant));
    variant.biome = -1;
    if (!getStructureSaltConfig(
            Stronghold,
            MCSEED_CUBIOMES_VERSION,
            variant.biome,
            &salt_config)) {
        return -4;
    }
    memset(
        context->piece_buffer,
        0,
        MCSEED_PIECE_BUFFER_CAPACITY * sizeof(*context->piece_buffer)
    );
    count = getStructurePieces(
        context->piece_buffer,
        MCSEED_PIECE_BUFFER_CAPACITY,
        Stronghold,
        salt_config,
        &variant,
        MCSEED_CUBIOMES_VERSION,
        context->seed,
        nearest.x,
        nearest.z
    );
    if (count < 0)
        return -5;
    for (index = 0; index < count; index++) {
        const Piece *piece = &context->piece_buffer[index];
        if (piece->type != SH_PORTAL_ROOM)
            continue;
        hit->x = piece->pos.x;
        hit->z = piece->pos.z;
        hit->parent_x = nearest.x;
        hit->parent_z = nearest.z;
        hit->eye_mask = piece->additionalData & 0x0fff;
        hit->name = "stronghold/portal_room";
        return 0;
    }
    return -6;
}

static int32_t find_stronghold_pieces(
    McSeedContext *context,
    const McSeedStructureInfo *info,
    Generator *generator,
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
)
{
    StrongholdIter iterator;
    uint64_t total = 0;
    int index;
    initFirstStronghold(&iterator, MCSEED_CUBIOMES_VERSION, context->seed);
    for (index = 0; index < 128; index++) {
        int status;
        if (nextStronghold(&iterator, generator) <= 0)
            break;
        if (!within_radius(iterator.pos.x, iterator.pos.z, anchor_x, anchor_z, radius))
            continue;
        status = evaluate_piece_candidate(
            context,
            generator,
            info,
            iterator.pos,
            none,
            selectors,
            selector_count,
            limit,
            hits,
            hit_capacity,
            &total
        );
        if (status < 0)
            return status - 10;
        if (status > 0) {
            *found = total;
            *limit_reached = 1;
            return 0;
        }
    }
    *found = total;
    return 0;
}

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
)
{
    const McSeedStructureInfo *info = structure_by_id(structure_id);
    StructureConfig config;
    Generator *generator;
    int64_t region_span;
    int64_t rx_min;
    int64_t rx_max;
    int64_t rz_min;
    int64_t rz_max;
    int64_t rx;
    int64_t rz;
    uint64_t total = 0;
    SurfaceNoise end_surface_noise;
    int check_end_terrain = 0;
    size_t selector_index;
    if (!context || !info || !selectors || selector_count == 0 ||
        !found || !limit_reached)
        return -1;
    *found = 0;
    *limit_reached = 0;
    if (limit == 0)
        return 0;
    for (selector_index = 0; selector_index < selector_count; selector_index++) {
        if (!mcseed_piece_selector_valid(structure_id, selectors[selector_index]))
            return -2;
    }
    generator = generator_for_dimension(context, info->dimension);
    if (!generator)
        return -3;
    if (info->cubiomes_type == Stronghold)
        return find_stronghold_pieces(
            context,
            info,
            generator,
            selectors,
            selector_count,
            anchor_x,
            anchor_z,
            radius,
            limit,
            hits,
            hit_capacity,
            found,
            limit_reached
        );
    if (!getStructureConfig(info->cubiomes_type, MCSEED_CUBIOMES_VERSION, &config))
        return -4;
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
            int viable;
            int status;
            if (rx < INT32_MIN || rx > INT32_MAX ||
                rz < INT32_MIN || rz > INT32_MAX)
                continue;
            if (!getStructurePos(
                    info->cubiomes_type,
                    MCSEED_CUBIOMES_VERSION,
                    context->seed,
                    (int32_t)rx,
                    (int32_t)rz,
                    &position))
                continue;
            if (!within_radius(position.x, position.z, anchor_x, anchor_z, radius))
                continue;
            viable = isViableStructurePos(
                info->cubiomes_type,
                generator,
                position.x,
                position.z,
                0
            );
            if (!viable)
                continue;
            if (info->dimension == DIM_OVERWORLD &&
                !isViableStructureTerrain(
                    info->cubiomes_type,
                    generator,
                    position.x,
                    position.z))
                continue;
            if (check_end_terrain && !isViableEndCityTerrain(
                    generator,
                    &end_surface_noise,
                    position.x,
                    position.z))
                continue;
            status = evaluate_piece_candidate(
                context,
                generator,
                info,
                position,
                viable,
                selectors,
                selector_count,
                limit,
                hits,
                hit_capacity,
                &total
            );
            if (status < 0)
                return status - 20;
            if (status > 0) {
                *found = total;
                *limit_reached = 1;
                return 0;
            }
        }
    }
    *found = total;
    return 0;
}
