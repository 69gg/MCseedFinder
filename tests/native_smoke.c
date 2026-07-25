#include "bridge.h"

#include <assert.h>
#include <stdint.h>
#include <stdio.h>

int main(void)
{
    McSeedContext *context;
    McSeedHit spawn;
    McSeedHit hits[8];
    int32_t biome_id;
    int32_t forest_id;
    int32_t village_id;
    int32_t fossil_id;
    int32_t limit_reached;
    uint64_t found;

    assert(mcseed_biome_count() > 50);
    assert(mcseed_structure_count() == 22);
    assert(mcseed_biome_id_from_name("sulfur_caves") >= 0);
    forest_id = mcseed_biome_id_from_name("forest");
    village_id = mcseed_structure_id_from_name("village");
    fossil_id = mcseed_structure_id_from_name("nether_fossil");
    assert(forest_id >= 0);
    assert(village_id >= 0);
    assert(fossil_id >= 0);

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

    assert(mcseed_find_structure(
        context, fossil_id, -4, 0, 512,
        1, hits, 8, &found, &limit_reached
    ) == 0);
    assert(found == 1);
    assert(hits[0].x == -32 && hits[0].z == -480);

    mcseed_context_destroy(context);
    puts("native ASan smoke test passed");
    return 0;
}
