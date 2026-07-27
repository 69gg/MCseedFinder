#ifndef MCSEED_VERSION_H
#define MCSEED_VERSION_H

/*
 * Active version profile. Keep version-specific generated data in native/generated/
 * and add its registry branch in jigsaw.c when enabling another profile.
 */
#define MCSEED_VERSION_NAME "26.2"
#define MCSEED_CUBIOMES_VERSION MC_26_2
/* Capability profile consumed by native/gpu/spawn_config.c; no version-string checks. */
#define MCSEED_GPU_SPAWN_ALGORITHM 1

/*
 * Modern concentric-rings stronghold profile. The GPU only evaluates a
 * conservative envelope around these approximate positions; Cubiomes still
 * performs the biome-adjusted final verification on the CPU.
 */
#define MCSEED_GPU_STRONGHOLD_ALGORITHM 1
#define MCSEED_STRONGHOLD_DISTANCE 32
#define MCSEED_STRONGHOLD_SPREAD 3
#define MCSEED_STRONGHOLD_COUNT 128
#define MCSEED_STRONGHOLD_LOCATE_MARGIN 128
#define MCSEED_STRONGHOLD_GPU_MATH_MARGIN 16

/*
 * First post-dragon End gateway profile. The exact CPU path follows terrain
 * to the linked outer-island gateway. The GPU uses the seed-selected gateway
 * direction at the nominal 1024-block distance. The 320-block margin covers
 * 16 ray steps (256), a chunk corner plus the 33x33 height scan (<44), and
 * 20 blocks of cross-device trigonometric/flooring slack.
 */
#define MCSEED_GPU_END_GATEWAY_ALGORITHM 1
#define MCSEED_END_GATEWAY_COUNT 20
#define MCSEED_END_GATEWAY_RING_RADIUS 96
#define MCSEED_END_GATEWAY_LINK_DISTANCE 1024
#define MCSEED_END_GATEWAY_GPU_MARGIN 320

#endif
