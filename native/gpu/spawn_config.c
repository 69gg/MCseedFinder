#include "abi.h"

#include <stddef.h>
#include <string.h>

#include "../version.h"

typedef struct McSeedSpawnNoiseProfile {
    uint64_t seed_xor_lo;
    uint64_t seed_xor_hi;
    int32_t first_octave;
    uint32_t amplitude_count;
    double amplitudes[9];
} McSeedSpawnNoiseProfile;

static const uint64_t MCSEED_OCTAVE_SEED_XORS[][2] = {
    {0xc613bf766619f992ULL, 0x954753f86691b86aULL},
    {0x7eee475a921c6cf5ULL, 0xf2bd39426f8da413ULL},
    {0xfc0027cef9683114ULL, 0xb758d3954dcbfdd3ULL},
    {0xd1fc8a05be565ecaULL, 0xdc2a3915cbdda25bULL},
    {0xb198de63a8012672ULL, 0x7b84cad43ef7b5a8ULL},
    {0x0fd787bfbc403ec3ULL, 0x74a4a31ca21b48b8ULL},
    {0x36d326eed40efeb2ULL, 0x5be9ce18223c636aULL},
    {0x082fe255f8be6631ULL, 0x4e96119e22dedc81ULL},
    {0x0ef68ec68504005eULL, 0x48b6bf93a2789640ULL},
    {0xf11268128982754fULL, 0x257a1d670430b0aaULL},
    {0xe51c98ce7d1de664ULL, 0x5f9478a733040c45ULL},
    {0x6d7b49e7e429850aULL, 0x2e3063c622a24777ULL},
    {0xbd90d5377ba1b762ULL, 0xc07317d419a7548dULL},
    {0x53d39c6752dac858ULL, 0xbcd1c5a80ab65b3eULL},
    {0xb4a24d7a84e7677bULL, 0x023ff9668e89b5c4ULL},
    {0xdffa22b534c5f608ULL, 0xb9b67517d3665ca9ULL},
    {0xd50708086cef4d7cULL, 0x6e1651ecc7f43309ULL},
};

static const double MCSEED_LACUNARITY_BY_NEGATIVE_OCTAVE[] = {
    1.0,
    0.5,
    0.25,
    1.0 / 8.0,
    1.0 / 16.0,
    1.0 / 32.0,
    1.0 / 64.0,
    1.0 / 128.0,
    1.0 / 256.0,
    1.0 / 512.0,
    1.0 / 1024.0,
    1.0 / 2048.0,
    1.0 / 4096.0,
    1.0 / 8192.0,
    1.0 / 16384.0,
    1.0 / 32768.0,
    1.0 / 65536.0,
};

static const double MCSEED_PERSISTENCE_BY_LENGTH[] = {
    0.0,
    1.0,
    2.0 / 3.0,
    4.0 / 7.0,
    8.0 / 15.0,
    16.0 / 31.0,
    32.0 / 63.0,
    64.0 / 127.0,
    128.0 / 255.0,
    256.0 / 511.0,
};

static const double MCSEED_DOUBLE_PERLIN_AMPLITUDE_BY_SPAN[] = {
    0.0,
    5.0 / 6.0,
    10.0 / 9.0,
    15.0 / 12.0,
    20.0 / 15.0,
    25.0 / 18.0,
    30.0 / 21.0,
    35.0 / 24.0,
    40.0 / 27.0,
    45.0 / 30.0,
};

static const McSeedSpawnNoiseProfile MCSEED_SPAWN_NOISE_PROFILES[] = {
    {0x080518cf6af25384ULL, 0x3f3dfb40a54febd5ULL, -3, 4, {1, 1, 1, 0}},
    {0x5c7e6b29735f0d7fULL, 0xf7d86f1bbc734988ULL, -10, 6, {1.5, 0, 1, 0, 0, 0}},
    {0x81bb4d22e8dc168eULL, 0xf1c8b4bea16303cdULL, -8, 6, {1, 1, 0, 0, 0, 0}},
    {0x83886c9d0ae3a662ULL, 0xafa638a61b42e8adULL, -9, 9, {1, 1, 2, 2, 2, 1, 1, 1, 1}},
    {0xd02491e6058f6fd8ULL, 0x4792512c94c17a80ULL, -9, 5, {1, 1, 0, 1, 1}},
    {0xefc8ef4d36102b34ULL, 0x1beeeb324a0f24eaULL, -7, 6, {1, 2, 1, 0, 0, 0}},
};

static int mcseed_append_spawn_octaves(
    McSeedGpuSpawnConfig *config,
    const McSeedSpawnNoiseProfile *profile,
    uint32_t half,
    uint32_t *offset,
    uint32_t *count
)
{
    double lacunarity;
    double persistence;
    uint32_t index;

    if (-profile->first_octave >=
            (int32_t)(sizeof(MCSEED_LACUNARITY_BY_NEGATIVE_OCTAVE) /
                sizeof(*MCSEED_LACUNARITY_BY_NEGATIVE_OCTAVE)) ||
        profile->amplitude_count >=
            sizeof(MCSEED_PERSISTENCE_BY_LENGTH) /
                sizeof(*MCSEED_PERSISTENCE_BY_LENGTH))
        return 0;

    *offset = config->perlin_count;
    *count = 0;
    lacunarity = MCSEED_LACUNARITY_BY_NEGATIVE_OCTAVE[-profile->first_octave];
    persistence = MCSEED_PERSISTENCE_BY_LENGTH[profile->amplitude_count];
    for (index = 0; index < profile->amplitude_count; index++) {
        const double amplitude = profile->amplitudes[index];
        const int32_t octave_seed_index = 16 + profile->first_octave + (int32_t)index;
        if (amplitude != 0.0) {
            McSeedGpuSpawnPerlin *perlin;
            if (config->perlin_count == MCSEED_GPU_SPAWN_MAX_PERLINS ||
                octave_seed_index < 0 ||
                octave_seed_index >=
                    (int32_t)(sizeof(MCSEED_OCTAVE_SEED_XORS) /
                        sizeof(*MCSEED_OCTAVE_SEED_XORS)))
                return 0;
            perlin = &config->perlins[config->perlin_count++];
            perlin->parameter_seed_xor_lo = profile->seed_xor_lo;
            perlin->parameter_seed_xor_hi = profile->seed_xor_hi;
            perlin->octave_seed_xor_lo = MCSEED_OCTAVE_SEED_XORS[octave_seed_index][0];
            perlin->octave_seed_xor_hi = MCSEED_OCTAVE_SEED_XORS[octave_seed_index][1];
            perlin->amplitude = amplitude * persistence;
            perlin->lacunarity = lacunarity;
            perlin->half = half;
            (*count)++;
        }
        lacunarity *= 2.0;
        persistence *= 0.5;
    }
    return 1;
}

static int mcseed_append_spawn_noise(
    McSeedGpuSpawnConfig *config,
    uint32_t noise_index,
    const McSeedSpawnNoiseProfile *profile
)
{
    McSeedGpuSpawnNoise *noise = &config->noises[noise_index];
    uint32_t first_nonzero = 0;
    uint32_t last_nonzero = profile->amplitude_count;
    uint32_t span;

    if (!mcseed_append_spawn_octaves(
            config,
            profile,
            0,
            &noise->octave_a_offset,
            &noise->octave_a_count
        ) ||
        !mcseed_append_spawn_octaves(
            config,
            profile,
            1,
            &noise->octave_b_offset,
            &noise->octave_b_count
        ))
        return 0;

    while (first_nonzero < profile->amplitude_count &&
           profile->amplitudes[first_nonzero] == 0.0)
        first_nonzero++;
    while (last_nonzero > first_nonzero &&
           profile->amplitudes[last_nonzero - 1] == 0.0)
        last_nonzero--;
    span = last_nonzero - first_nonzero;
    if (span == 0 || span >=
            sizeof(MCSEED_DOUBLE_PERLIN_AMPLITUDE_BY_SPAN) /
                sizeof(*MCSEED_DOUBLE_PERLIN_AMPLITUDE_BY_SPAN))
        return 0;
    noise->amplitude = MCSEED_DOUBLE_PERLIN_AMPLITUDE_BY_SPAN[span];
    return 1;
}

int32_t mcseed_gpu_spawn_config(McSeedGpuSpawnConfig *config)
{
    static const int64_t targets[MCSEED_GPU_SPAWN_TARGET_COUNT][2] = {
        {-10000, 10000},
        {-10000, 10000},
        {-1100, 10000},
        {-10000, 10000},
        {0, 0},
        {-10000, -1600},
        {1600, 10000},
    };
    uint32_t index;

    if (!config)
        return -1;
    memset(config, 0, sizeof(*config));
    if (MCSEED_GPU_SPAWN_ALGORITHM !=
        MCSEED_GPU_SPAWN_MULTI_NOISE_ORIGIN_BIAS)
        return 0;

    config->algorithm = MCSEED_GPU_SPAWN_ALGORITHM;
    config->noise_count = MCSEED_GPU_SPAWN_NOISE_COUNT;
    config->outer_radius = 2048;
    config->outer_step = 512;
    config->inner_radius = 512;
    config->inner_step = 32;
    config->fitness_scale = 2048ULL * 2048ULL;
    memcpy(config->targets, targets, sizeof(targets));

    for (index = 0; index < MCSEED_GPU_SPAWN_NOISE_COUNT; index++) {
        if (!mcseed_append_spawn_noise(
                config,
                index,
                &MCSEED_SPAWN_NOISE_PROFILES[index]
            )) {
            memset(config, 0, sizeof(*config));
            return -2;
        }
    }
    return 1;
}
