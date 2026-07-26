#include "bridge.h"
#include "finders.h"
#include "jigsaw.h"
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
    McSeedHit spawn;
    McSeedHit hits[8];
    McSeedPieceHit piece_hits[8];
    int32_t biome_id;
    int32_t forest_id;
    int32_t village_id;
    int32_t shipwreck_id;
    int32_t treasure_id;
    int32_t fossil_id;
    int32_t limit_reached;
    uint64_t found;

    assert_village_centers_match_cubiomes_variant();
    assert(mcseed_biome_count() > 50);
    assert(mcseed_structure_count() == 22);
    assert(mcseed_piece_count() > 500);
    assert(mcseed_biome_id_from_name("sulfur_caves") >= 0);
    forest_id = mcseed_biome_id_from_name("forest");
    village_id = mcseed_structure_id_from_name("village");
    shipwreck_id = mcseed_structure_id_from_name("shipwreck");
    treasure_id = mcseed_structure_id_from_name("buried_treasure");
    fossil_id = mcseed_structure_id_from_name("nether_fossil");
    assert(forest_id >= 0);
    assert(village_id >= 0);
    assert(shipwreck_id >= 0);
    assert(treasure_id >= 0);
    assert(fossil_id >= 0);
    assert(mcseed_piece_selector_valid(village_id, "blacksmith"));
    assert(mcseed_piece_selector_valid(village_id, "village/plains/houses/plains_weaponsmith_1"));
    assert(mcseed_piece_selector_valid(shipwreck_id, "full"));
    assert(mcseed_piece_selector_valid(treasure_id, "buried_treasure/chest"));

    context = mcseed_context_create();
    assert(context != NULL);
    mcseed_context_set_seed(context, 0);

    assert(mcseed_spawn(context, &spawn, &biome_id) == 0);
    assert(spawn.x == -32 && spawn.y == 65 && spawn.z == 0);
    assert(biome_id == forest_id);

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
