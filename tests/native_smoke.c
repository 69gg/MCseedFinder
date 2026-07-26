#include "bridge.h"
#include "finders.h"
#include "gpu/placement.h"
#include "jigsaw.h"
#include "tables/btree262.h"
#include "version.h"

#include <assert.h>
#include <limits.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

static int32_t flat_surface_height(void *context, int32_t x, int32_t z)
{
    (void)context;
    (void)x;
    (void)z;
    return 64;
}

typedef struct PlacementCase {
    const char *name;
    int32_t cubiomes_type;
} PlacementCase;

enum {
    MCSEED_TEST_CUSTOM_PLACEMENT = -1,
};

static void assert_btree262_indices_are_in_bounds(void)
{
    const size_t node_count = sizeof(btree262_nodes) / sizeof(btree262_nodes[0]);
    const size_t parameter_count = sizeof(btree262_param) / sizeof(btree262_param[0]);
    size_t node_index;
    assert(btree262_order == 6);
    assert(sizeof(btree262_steps) / sizeof(btree262_steps[0]) == 6);
    assert(btree262_steps[5] == 0);
    assert(parameter_count <= UINT8_MAX);
    for (node_index = 0; node_index < node_count; node_index++) {
        uint64_t node = btree262_nodes[node_index];
        int dimension;
        for (dimension = 0; dimension < 6; dimension++) {
            uint8_t parameter_index = (uint8_t)(node >> (dimension * 8));
            assert(parameter_index < parameter_count);
        }
        if ((node >> 56) != UINT8_MAX)
            assert((node >> 48) < node_count);
    }
}

static void assert_gpu_placements_match_cubiomes(void)
{
    static const PlacementCase CASES[] = {
        {"pillager_outpost", Outpost},
        {"mansion", Mansion},
        {"jungle_pyramid", Jungle_Pyramid},
        {"desert_pyramid", Desert_Pyramid},
        {"igloo", Igloo},
        {"shipwreck", Shipwreck},
        {"swamp_hut", Swamp_Hut},
        {"monument", Monument},
        {"ocean_ruin", Ocean_Ruin},
        {"fortress", Fortress},
        {"nether_fossil", MCSEED_TEST_CUSTOM_PLACEMENT},
        {"end_city", End_City},
        {"buried_treasure", Treasure},
        {"bastion_remnant", Bastion},
        {"village", Village},
        {"ruined_portal", Ruined_Portal},
        {"ancient_city", Ancient_City},
        {"trail_ruins", Trail_Ruins},
        {"trial_chambers", Trial_Chambers},
        {"ruined_portal_nether", Ruined_Portal_N},
    };
    static const uint64_t SEEDS[] = {
        UINT64_C(0),
        UINT64_MAX,
        UINT64_C(1),
        UINT64_C(0x8000000000000000),
        UINT64_C(0x0123456789abcdef),
        UINT64_C(0xfedcba9876543210),
        UINT64_C(0x9e3779b97f4a7c15),
        UINT64_C(0x5deece66d),
    };
    size_t case_index;
    size_t accelerated_count = 0;
    int32_t registry_index;
    for (registry_index = 0; registry_index < mcseed_structure_count(); registry_index++) {
        McSeedGpuStructureConfig config;
        int32_t structure_id = mcseed_structure_id_at(registry_index);
        assert(structure_id >= 0);
        if (mcseed_structure_gpu_config(structure_id, &config) == 1 &&
            config.kind != MCSEED_GPU_PLACEMENT_STRONGHOLD)
            accelerated_count++;
    }
    assert(sizeof(CASES) / sizeof(CASES[0]) == accelerated_count);

    for (case_index = 0; case_index < sizeof(CASES) / sizeof(CASES[0]); case_index++) {
        const PlacementCase *placement_case = &CASES[case_index];
        McSeedGpuStructureConfig gpu_config;
        StructureConfig feature_config = {0};
        int32_t structure_id = mcseed_structure_id_from_name(placement_case->name);
        int saw_valid = 0;
        int saw_invalid = 0;
        size_t seed_index;
        assert(structure_id >= 0);
        assert(mcseed_structure_gpu_config(structure_id, &gpu_config) == 1);
        feature_config.salt = gpu_config.salt;
        feature_config.regionSize = (int8_t)gpu_config.region_size;
        feature_config.chunkRange = (int8_t)gpu_config.chunk_range;

        for (seed_index = 0; seed_index < sizeof(SEEDS) / sizeof(SEEDS[0]); seed_index++) {
            int32_t region_z;
            for (region_z = -8; region_z <= 8; region_z++) {
                int32_t region_x;
                for (region_x = -8; region_x <= 8; region_x++) {
                    McSeedGpuPosition gpu_position;
                    Pos cubiomes_position;
                    int gpu_valid = mcseed_gpu_position_for_region(
                        &gpu_config,
                        SEEDS[seed_index],
                        region_x,
                        region_z,
                        &gpu_position
                    );
                    int cubiomes_valid;
                    if (placement_case->cubiomes_type == MCSEED_TEST_CUSTOM_PLACEMENT) {
                        cubiomes_position = getFeaturePos(
                            feature_config,
                            SEEDS[seed_index],
                            region_x,
                            region_z
                        );
                        cubiomes_valid = 1;
                    } else {
                        cubiomes_valid = getStructurePos(
                            placement_case->cubiomes_type,
                            MCSEED_CUBIOMES_VERSION,
                            SEEDS[seed_index],
                            region_x,
                            region_z,
                            &cubiomes_position
                        );
                    }
                    assert(gpu_valid == cubiomes_valid);
                    if (gpu_valid) {
                        saw_valid = 1;
                        assert(gpu_position.x == cubiomes_position.x);
                        assert(gpu_position.z == cubiomes_position.z);
                    } else {
                        saw_invalid = 1;
                    }
                }
            }
        }
        assert(saw_valid);
        if (gpu_config.kind == MCSEED_GPU_PLACEMENT_OUTPOST ||
            gpu_config.kind == MCSEED_GPU_PLACEMENT_TREASURE ||
            gpu_config.kind == MCSEED_GPU_PLACEMENT_BASTION ||
            (gpu_config.flags & MCSEED_GPU_PLACEMENT_END_DISTANCE) != 0)
            assert(saw_invalid);
    }
}

static int test_within_radius(
    int32_t x,
    int32_t z,
    int32_t anchor_x,
    int32_t anchor_z,
    uint32_t radius
)
{
    int64_t dx = (int64_t)x - anchor_x;
    int64_t dz = (int64_t)z - anchor_z;
    uint64_t absolute_x = dx < 0 ? (uint64_t)(-dx) : (uint64_t)dx;
    uint64_t absolute_z = dz < 0 ? (uint64_t)(-dz) : (uint64_t)dz;
    uint64_t radius_squared = (uint64_t)radius * radius;
    if (absolute_x > radius || absolute_z > radius)
        return 0;
    return absolute_z * absolute_z <= radius_squared - absolute_x * absolute_x;
}

static uint64_t brute_force_strongholds(
    uint64_t seed,
    int32_t anchor_x,
    int32_t anchor_z,
    uint32_t radius,
    McSeedHit *hits
)
{
    Generator generator;
    StrongholdIter iterator;
    uint64_t found = 0;
    int index;
    setupGenerator(&generator, MCSEED_CUBIOMES_VERSION, 0);
    applySeed(&generator, DIM_OVERWORLD, seed);
    initFirstStronghold(&iterator, MCSEED_CUBIOMES_VERSION, seed);
    for (index = 0; index < MCSEED_STRONGHOLD_COUNT; index++) {
        if (nextStronghold(&iterator, &generator) <= 0)
            break;
        if (!test_within_radius(
                iterator.pos.x,
                iterator.pos.z,
                anchor_x,
                anchor_z,
                radius
            ))
            continue;
        hits[found].x = iterator.pos.x;
        hits[found].z = iterator.pos.z;
        found++;
    }
    return found;
}

static void assert_pruned_stronghold_scan_matches_brute_force(int32_t stronghold_id)
{
    static const struct {
        int64_t seed;
        int32_t anchor_x;
        int32_t anchor_z;
        uint32_t radius;
    } CASES[] = {
        {0, 0, 0, 500},
        {0, -32, 0, 2000},
        {1, 2000, 0, 500},
        {-1, -1800, 800, 750},
        {INT64_C(0x0123456789abcdef), 0, 0, 3000},
        {-INT64_C(0x0123456789abcdef), 4800, -1600, 900},
    };
    McSeedContext *context = mcseed_context_create();
    size_t case_index;
    assert(context != NULL);
    for (case_index = 0; case_index < sizeof(CASES) / sizeof(CASES[0]); case_index++) {
        McSeedHit expected[MCSEED_STRONGHOLD_COUNT] = {{0}};
        McSeedHit actual[MCSEED_STRONGHOLD_COUNT] = {{0}};
        uint64_t expected_count = brute_force_strongholds(
            (uint64_t)CASES[case_index].seed,
            CASES[case_index].anchor_x,
            CASES[case_index].anchor_z,
            CASES[case_index].radius,
            expected
        );
        uint64_t actual_count = 0;
        int32_t limit_reached = 0;
        uint64_t hit_index;
        mcseed_context_set_seed(context, CASES[case_index].seed);
        assert(mcseed_find_structure(
            context,
            stronghold_id,
            CASES[case_index].anchor_x,
            CASES[case_index].anchor_z,
            CASES[case_index].radius,
            UINT64_MAX,
            actual,
            MCSEED_STRONGHOLD_COUNT,
            &actual_count,
            &limit_reached
        ) == 0);
        assert(limit_reached == 0);
        assert(actual_count == expected_count);
        for (hit_index = 0; hit_index < expected_count; hit_index++) {
            assert(actual[hit_index].x == expected[hit_index].x);
            assert(actual[hit_index].z == expected[hit_index].z);
        }
    }
    mcseed_context_destroy(context);
}

static void assert_village_centers_match_cubiomes_variant(void)
{
    static const int BIOMES[MCJIGSAW_STYLE_COUNT] = {
        plains,
        desert,
        savanna,
        snowy_tundra,
        taiga,
    };
    const McJigsawData *data = mcjigsaw_village_data(MCSEED_VERSION_NAME);
    McJigsawWorkspace *workspace = mcjigsaw_workspace_create();
    McJigsawPiece pieces[MCJIGSAW_PIECE_CAPACITY];
    int style;
    int sample;
    assert(data != NULL);
    assert(workspace != NULL);

    for (style = 0; style < MCJIGSAW_STYLE_COUNT; style++) {
        for (sample = 0; sample < 8; sample++) {
            uint64_t seed = UINT64_C(0x9e3779b97f4a7c15) * (uint64_t)(sample + 1) +
                (uint64_t)style;
            int32_t x = (sample - 4) * 16;
            int32_t z = (style * 7 - sample * 3) * 16;
            StructureVariant variant;
            int32_t count;
            assert(getVariant(
                &variant,
                Village,
                MCSEED_CUBIOMES_VERSION,
                seed,
                x,
                z,
                BIOMES[style]
            ));
            count = mcjigsaw_generate(
                workspace,
                data,
                seed,
                x,
                z,
                style,
                flat_surface_height,
                NULL,
                pieces,
                MCJIGSAW_PIECE_CAPACITY
            );
            assert(count > 0);
            assert(pieces[0].rotation == variant.rotation);
            assert(pieces[0].box.min_x == x + variant.x);
            assert(pieces[0].box.min_z == z + variant.z);
            assert(pieces[0].box.max_x - pieces[0].box.min_x + 1 == variant.sx);
            assert(pieces[0].box.max_y - pieces[0].box.min_y + 1 == variant.sy);
            assert(pieces[0].box.max_z - pieces[0].box.min_z + 1 == variant.sz);
            assert((strstr(pieces[0].name, "/zombie/") != NULL) == variant.abandoned);
        }
    }
    mcjigsaw_workspace_destroy(workspace);
}

int main(void)
{
    McSeedContext *context;
    McSeedContext *direct_spawn_context;
    McSeedHit spawn;
    McSeedHit hits[8];
    McSeedPieceHit piece_hits[8];
    int32_t biome_id;
    int32_t forest_id;
    int32_t village_id;
    int32_t shipwreck_id;
    int32_t stronghold_id;
    int32_t treasure_id;
    int32_t fossil_id;
    int32_t limit_reached;
    uint64_t found;
    uint32_t spawn_refinement_radius;
    uint32_t spawn_origin_radius;

    assert_btree262_indices_are_in_bounds();
    assert_gpu_placements_match_cubiomes();
    assert_village_centers_match_cubiomes_variant();
    assert(mcseed_biome_count() > 50);
    assert(mcseed_structure_count() == 22);
    assert(mcseed_piece_count() > 500);
    assert(mcseed_biome_id_from_name("sulfur_caves") >= 0);
    forest_id = mcseed_biome_id_from_name("forest");
    village_id = mcseed_structure_id_from_name("village");
    shipwreck_id = mcseed_structure_id_from_name("shipwreck");
    stronghold_id = mcseed_structure_id_from_name("stronghold");
    treasure_id = mcseed_structure_id_from_name("buried_treasure");
    fossil_id = mcseed_structure_id_from_name("nether_fossil");
    assert(forest_id >= 0);
    assert(village_id >= 0);
    assert(shipwreck_id >= 0);
    assert(stronghold_id >= 0);
    assert(treasure_id >= 0);
    assert(fossil_id >= 0);
    assert_pruned_stronghold_scan_matches_brute_force(stronghold_id);
    assert(mcseed_piece_selector_valid(village_id, "blacksmith"));
    assert(mcseed_piece_selector_valid(village_id, "village/plains/houses/plains_weaponsmith_1"));
    assert(mcseed_piece_selector_valid(shipwreck_id, "full"));
    assert(mcseed_piece_selector_valid(treasure_id, "buried_treasure/chest"));

    context = mcseed_context_create();
    assert(context != NULL);
    mcseed_context_set_seed(context, 0);

    {
        McSeedHit estimated;
        McSeedHit reference_estimated;
        McSeedHit direct_spawn;
        McSeedHit supplied_spawn;
        int32_t direct_biome_id;
        int32_t supplied_biome_id;
        int64_t dx;
        int64_t dz;
        assert(mcseed_estimated_spawn_reference(context, &reference_estimated) == 0);
        assert(mcseed_estimated_spawn(context, &estimated) == 0);
        assert(estimated.x == reference_estimated.x &&
            estimated.z == reference_estimated.z);
        assert(mcseed_spawn_refinement_radius(&spawn_refinement_radius) == 1);
        assert(spawn_refinement_radius == 125);
        assert(mcseed_spawn_origin_radius(&spawn_origin_radius) == 1);
        assert(spawn_origin_radius == 2697);
        assert(mcseed_spawn(context, &spawn, &biome_id) == 0);
        dx = (int64_t)spawn.x - estimated.x;
        dz = (int64_t)spawn.z - estimated.z;
        assert(dx * dx + dz * dz <=
            (int64_t)spawn_refinement_radius * spawn_refinement_radius);

        direct_spawn_context = mcseed_context_create();
        assert(direct_spawn_context != NULL);
        mcseed_context_set_seed(direct_spawn_context, 0);
        assert(mcseed_spawn(
            direct_spawn_context,
            &direct_spawn,
            &direct_biome_id
        ) == 0);
        assert(direct_spawn.x == spawn.x && direct_spawn.y == spawn.y &&
            direct_spawn.z == spawn.z && direct_biome_id == biome_id);
        mcseed_context_set_seed(direct_spawn_context, 0);
        assert(mcseed_spawn_from_estimate(
            direct_spawn_context,
            estimated.x,
            estimated.z,
            &supplied_spawn,
            &supplied_biome_id
        ) == 0);
        assert(supplied_spawn.x == spawn.x && supplied_spawn.y == spawn.y &&
            supplied_spawn.z == spawn.z && supplied_biome_id == biome_id);
        mcseed_context_destroy(direct_spawn_context);
    }

    assert(spawn.x == -32 && spawn.y == 65 && spawn.z == 0);
    assert(biome_id == forest_id);

    {
        McSeedGpuStructureConfig config;
        McSeedGpuPredicate predicate = {0, 1, 1024, MCSEED_GPU_ANCHOR_SPAWN, 0, 0, 1};
        McSeedGpuCandidate candidate = {0, spawn.x, spawn.z};
        uint8_t placement_match = 0;
        assert(mcseed_structure_gpu_config(village_id, &config) == 1);
        mcseed_gpu_reference_filter(
            &candidate, 1, &config, 1, &predicate, 1, &placement_match
        );
        assert(placement_match == 1);

        {
            McSeedGpuPairPredicate pair = {0, 1, 0, 1, 1024, 1024, 2697, 0};
            placement_match = 0;
            mcseed_gpu_reference_pair_filter(
                &candidate, 1, &config, 1, &pair, 1, &placement_match
            );
            assert(placement_match == 1);
        }
    }

    assert(mcseed_find_biomes(
        context, 0, &forest_id, 1, spawn.x, spawn.z, 64, 64, 64,
        1, hits, 8, &found, &limit_reached
    ) == 0);
    assert(found == 1 && limit_reached == 1);

    assert(mcseed_find_structure(
        context, village_id, spawn.x, spawn.z, 1024,
        1, hits, 8, &found, &limit_reached
    ) == 0);
    assert(found == 1);
    assert(hits[0].x == 272 && hits[0].z == 944);

    {
        McSeedPieceHit portal;
        assert(mcseed_nearest_stronghold_portal(
            context, spawn.x, spawn.z, &portal
        ) == 0);
        assert(strcmp(portal.name, "stronghold/portal_room") == 0);
        assert(portal.parent_x == -204 && portal.parent_z == -1692);
        assert(portal.x == -196 && portal.y == INT32_MIN && portal.z == -1728);
        assert(portal.eye_mask == 0x030);
    }

    {
        const char *selectors[] = {"blacksmith"};
        assert(mcseed_find_structure_pieces(
            context, village_id, selectors, 1, spawn.x, spawn.z, 1024,
            1, piece_hits, 8, &found, &limit_reached
        ) == 0);
        assert(found == 1 && limit_reached == 1);
        assert(strcmp(piece_hits[0].name, "village/plains/houses/plains_weaponsmith_1") == 0);
        assert(piece_hits[0].parent_x == 272 && piece_hits[0].parent_z == 944);
        assert(piece_hits[0].x == 267 && piece_hits[0].y == 71 && piece_hits[0].z == 960);
    }
    {
        const char *selectors[] = {"house"};
        assert(mcseed_find_structure_pieces(
            context, village_id, selectors, 1, spawn.x, spawn.z, 1024,
            100, piece_hits, 8, &found, &limit_reached
        ) == 0);
        assert(found == 32 && limit_reached == 0);
        assert(strcmp(piece_hits[0].name, "village/plains/houses/plains_cartographer_1") == 0);
    }
    {
        const char *selectors[] = {"town_center"};
        assert(mcseed_find_structure_pieces(
            context, village_id, selectors, 1, spawn.x, spawn.z, 1024,
            10, piece_hits, 8, &found, &limit_reached
        ) == 0);
        assert(found == 2 && limit_reached == 0);
        assert(strcmp(piece_hits[0].name, "village/plains/town_centers/plains_meeting_point_3") == 0);
    }
    {
        const char *selectors[] = {"full"};
        assert(mcseed_find_structure_pieces(
            context, shipwreck_id, selectors, 1, spawn.x, spawn.z, 4096,
            1, piece_hits, 8, &found, &limit_reached
        ) == 0);
        assert(found == 1 && limit_reached == 1);
        assert(strcmp(piece_hits[0].name, "shipwreck/rightsideup_full_degraded") == 0);
        assert(piece_hits[0].x == -2160 && piece_hits[0].y == INT32_MIN && piece_hits[0].z == -3376);
    }
    {
        const char *selectors[] = {"shipwreck/upsidedown_backhalf_degraded"};
        assert(mcseed_find_structure_pieces(
            context, shipwreck_id, selectors, 1, spawn.x, spawn.z, 4096,
            1, piece_hits, 8, &found, &limit_reached
        ) == 0);
        assert(found == 1 && limit_reached == 1);
        assert(strcmp(piece_hits[0].name, "shipwreck/upsidedown_backhalf_degraded") == 0);
        assert(piece_hits[0].parent_x == -1360 && piece_hits[0].parent_z == -3328);
        assert(piece_hits[0].x == -1352 && piece_hits[0].z == -3298);
    }
    {
        const char *selectors[] = {"buried_treasure/chest"};
        assert(mcseed_find_structure_pieces(
            context, treasure_id, selectors, 1, spawn.x, spawn.z, 4096,
            1, piece_hits, 8, &found, &limit_reached
        ) == 0);
        assert(found == 1 && limit_reached == 1);
        assert(strcmp(piece_hits[0].name, "buried_treasure/chest") == 0);
        assert(piece_hits[0].x == -1383 && piece_hits[0].z == -3415);
    }

    assert(mcseed_find_structure(
        context, fossil_id, -4, 0, 512,
        1, hits, 8, &found, &limit_reached
    ) == 0);
    assert(found == 1);
    assert(hits[0].x == -32 && hits[0].z == -480);

    mcseed_context_destroy(context);
    puts("native ASan/UBSan smoke test passed");
    return 0;
}
