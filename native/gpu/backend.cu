#include "abi.h"

#include <stddef.h>
#include <stdint.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if defined(MCSEED_GPU_HIP) && defined(MCSEED_GPU_CUDA)
#error "MCSEED_GPU_HIP and MCSEED_GPU_CUDA are mutually exclusive"
#elif defined(MCSEED_GPU_HIP)
#include <hip/hip_runtime.h>
typedef hipError_t McSeedGpuError;
typedef hipStream_t McSeedGpuStream;
#define MCSEED_GPU_SUCCESS hipSuccess
#define mcseed_gpu_get_device_count hipGetDeviceCount
#define mcseed_gpu_get_device_properties hipGetDeviceProperties
#define mcseed_gpu_set_device hipSetDevice
#define mcseed_gpu_malloc hipMalloc
#define mcseed_gpu_free hipFree
#define mcseed_gpu_memcpy_async hipMemcpyAsync
#define mcseed_gpu_stream_create hipStreamCreate
#define mcseed_gpu_stream_destroy hipStreamDestroy
#define mcseed_gpu_stream_synchronize hipStreamSynchronize
#define mcseed_gpu_get_last_error hipGetLastError
#define mcseed_gpu_error_string hipGetErrorString
#define MCSEED_GPU_COPY_HOST_TO_DEVICE hipMemcpyHostToDevice
#define MCSEED_GPU_COPY_DEVICE_TO_HOST hipMemcpyDeviceToHost
#define MCSEED_GPU_BACKEND_NAME "rocm"
#elif defined(MCSEED_GPU_CUDA)
#include <cuda_runtime.h>
typedef cudaError_t McSeedGpuError;
typedef cudaStream_t McSeedGpuStream;
#define MCSEED_GPU_SUCCESS cudaSuccess
#define mcseed_gpu_get_device_count cudaGetDeviceCount
#define mcseed_gpu_get_device_properties cudaGetDeviceProperties
#define mcseed_gpu_set_device cudaSetDevice
#define mcseed_gpu_malloc cudaMalloc
#define mcseed_gpu_free cudaFree
#define mcseed_gpu_memcpy_async cudaMemcpyAsync
#define mcseed_gpu_stream_create cudaStreamCreate
#define mcseed_gpu_stream_destroy cudaStreamDestroy
#define mcseed_gpu_stream_synchronize cudaStreamSynchronize
#define mcseed_gpu_get_last_error cudaGetLastError
#define mcseed_gpu_error_string cudaGetErrorString
#define MCSEED_GPU_COPY_HOST_TO_DEVICE cudaMemcpyHostToDevice
#define MCSEED_GPU_COPY_DEVICE_TO_HOST cudaMemcpyDeviceToHost
#define MCSEED_GPU_BACKEND_NAME "cuda"
#else
#error "Define exactly one of MCSEED_GPU_HIP or MCSEED_GPU_CUDA"
#endif

#include "placement.h"
#include "spawn.h"

typedef struct McSeedGpuContext {
    int32_t device;
    McSeedGpuStream stream;
    McSeedGpuStructureConfig *configs;
    McSeedGpuPredicate *predicates;
    McSeedGpuPairPredicate *pair_predicates;
    McSeedGpuSpawnConfig *spawn_config;
    McSeedGpuSpawnOffset *outer_spawn_offsets;
    McSeedGpuSpawnOffset *inner_spawn_offsets;
    McSeedGpuCandidate *candidates;
    uint8_t *matches;
    uint32_t *spawn_state_indices;
    McSeedGpuPerlinNoise *spawn_states;
    size_t config_count;
    size_t predicate_count;
    size_t pair_predicate_count;
    uint32_t outer_spawn_offset_count;
    uint32_t inner_spawn_offset_count;
    uint32_t spawn_perlin_count;
    size_t candidate_capacity;
    size_t spawn_state_capacity;
    size_t spawn_state_generation_count;
} McSeedGpuContext;

static void mcseed_gpu_write_error(
    char *error,
    size_t error_capacity,
    const char *operation,
    McSeedGpuError status
)
{
    if (!error || error_capacity == 0)
        return;
    snprintf(
        error,
        error_capacity,
        "%s: %s",
        operation,
        mcseed_gpu_error_string(status)
    );
}

static int mcseed_gpu_check(
    McSeedGpuError status,
    char *error,
    size_t error_capacity,
    const char *operation
)
{
    if (status == MCSEED_GPU_SUCCESS)
        return 1;
    mcseed_gpu_write_error(error, error_capacity, operation, status);
    return 0;
}

__global__ static void mcseed_gpu_prefilter_kernel(
    const McSeedGpuCandidate *candidates,
    size_t candidate_count,
    const McSeedGpuStructureConfig *configs,
    const McSeedGpuPredicate *predicates,
    size_t predicate_count,
    const McSeedGpuPairPredicate *pair_predicates,
    size_t pair_predicate_count,
    uint8_t *matches
)
{
    size_t index = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= candidate_count)
        return;
    int matched = predicate_count == 0 || mcseed_gpu_candidate_matches(
        &candidates[index], configs, predicates, predicate_count
    );
    if (matched && pair_predicate_count != 0)
        matched = mcseed_gpu_candidate_matches_pairs(
            &candidates[index], configs, pair_predicates, pair_predicate_count
        );
    matches[index] = (uint8_t)matched;
}

extern "C" const char *mcseed_gpu_backend_name(void)
{
    return MCSEED_GPU_BACKEND_NAME;
}

extern "C" int32_t mcseed_gpu_device_count(
    int32_t *count,
    char *error,
    size_t error_capacity
)
{
    int devices = 0;
    if (!count)
        return -1;
    if (!mcseed_gpu_check(
            mcseed_gpu_get_device_count(&devices),
            error,
            error_capacity,
            "读取 GPU 设备数量"
        ))
        return -2;
    *count = devices;
    return 0;
}

extern "C" int32_t mcseed_gpu_device_name(
    int32_t device,
    char *name,
    size_t name_capacity,
    char *error,
    size_t error_capacity
)
{
#if defined(MCSEED_GPU_HIP)
    hipDeviceProp_t properties;
#else
    cudaDeviceProp properties;
#endif
    if (!name || name_capacity == 0)
        return -1;
    if (!mcseed_gpu_check(
            mcseed_gpu_get_device_properties(&properties, device),
            error,
            error_capacity,
            "读取 GPU 设备属性"
        ))
        return -2;
    snprintf(name, name_capacity, "%s", properties.name);
    return 0;
}

static void mcseed_gpu_context_cleanup(McSeedGpuContext *context)
{
    if (!context)
        return;
    (void)mcseed_gpu_set_device(context->device);
    if (context->stream)
        (void)mcseed_gpu_stream_synchronize(context->stream);
    if (context->candidates)
        (void)mcseed_gpu_free(context->candidates);
    if (context->matches)
        (void)mcseed_gpu_free(context->matches);
    if (context->spawn_state_indices)
        (void)mcseed_gpu_free(context->spawn_state_indices);
    if (context->spawn_states)
        (void)mcseed_gpu_free(context->spawn_states);
    if (context->configs)
        (void)mcseed_gpu_free(context->configs);
    if (context->predicates)
        (void)mcseed_gpu_free(context->predicates);
    if (context->pair_predicates)
        (void)mcseed_gpu_free(context->pair_predicates);
    if (context->spawn_config)
        (void)mcseed_gpu_free(context->spawn_config);
    if (context->outer_spawn_offsets)
        (void)mcseed_gpu_free(context->outer_spawn_offsets);
    if (context->inner_spawn_offsets)
        (void)mcseed_gpu_free(context->inner_spawn_offsets);
    if (context->stream)
        (void)mcseed_gpu_stream_destroy(context->stream);
    free(context);
}

static uint32_t mcseed_gpu_build_spawn_offsets(
    McSeedGpuSpawnOffset *offsets,
    uint32_t capacity,
    uint32_t maximum_radius,
    uint32_t step
)
{
    const double full_turn = 3.14159265358979323846 * 2.0;
    float radius;
    float angle = 0.0F;
    uint32_t count = 0;
    if (!offsets || step == 0)
        return 0;
    radius = (float)step;
    while (radius <= (float)maximum_radius) {
        if (count == capacity)
            return 0;
        offsets[count].x = (int32_t)(sin((double)angle) * radius);
        offsets[count].z = (int32_t)(cos((double)angle) * radius);
        count++;
        angle += (float)step / radius;
        if ((double)angle > full_turn) {
            angle = 0.0F;
            radius += (float)step;
        }
    }
    return count;
}

static int mcseed_gpu_prepare_spawn_estimator(
    McSeedGpuContext *context,
    const McSeedGpuSpawnConfig *spawn_config,
    char *error,
    size_t error_capacity
)
{
    McSeedGpuSpawnOffset outer_offsets[MCSEED_GPU_SPAWN_MAX_OFFSETS];
    McSeedGpuSpawnOffset inner_offsets[MCSEED_GPU_SPAWN_MAX_OFFSETS];
    uint32_t outer_count;
    uint32_t inner_count;
    if (!spawn_config)
        return 1;
    if (spawn_config->algorithm != MCSEED_GPU_SPAWN_MULTI_NOISE_ORIGIN_BIAS ||
        spawn_config->noise_count != MCSEED_GPU_SPAWN_NOISE_COUNT ||
        spawn_config->perlin_count == 0 ||
        spawn_config->perlin_count > MCSEED_GPU_SPAWN_MAX_PERLINS ||
        spawn_config->fitness_scale == 0) {
        if (error && error_capacity)
            snprintf(error, error_capacity, "GPU 出生点配置无效或不受支持");
        return 0;
    }
    outer_count = mcseed_gpu_build_spawn_offsets(
        outer_offsets,
        MCSEED_GPU_SPAWN_MAX_OFFSETS,
        spawn_config->outer_radius,
        spawn_config->outer_step
    );
    inner_count = mcseed_gpu_build_spawn_offsets(
        inner_offsets,
        MCSEED_GPU_SPAWN_MAX_OFFSETS,
        spawn_config->inner_radius,
        spawn_config->inner_step
    );
    if (outer_count == 0 || inner_count == 0) {
        if (error && error_capacity)
            snprintf(error, error_capacity, "GPU 出生点搜索偏移配置无效");
        return 0;
    }
    if (!mcseed_gpu_check(
            mcseed_gpu_malloc((void **)&context->spawn_config, sizeof(*spawn_config)),
            error,
            error_capacity,
            "分配 GPU 出生点配置"
        ) ||
        !mcseed_gpu_check(
            mcseed_gpu_malloc(
                (void **)&context->outer_spawn_offsets,
                outer_count * sizeof(*outer_offsets)
            ),
            error,
            error_capacity,
            "分配 GPU 出生点外层偏移"
        ) ||
        !mcseed_gpu_check(
            mcseed_gpu_malloc(
                (void **)&context->inner_spawn_offsets,
                inner_count * sizeof(*inner_offsets)
            ),
            error,
            error_capacity,
            "分配 GPU 出生点内层偏移"
        ) ||
        !mcseed_gpu_check(
            mcseed_gpu_memcpy_async(
                context->spawn_config,
                spawn_config,
                sizeof(*spawn_config),
                MCSEED_GPU_COPY_HOST_TO_DEVICE,
                context->stream
            ),
            error,
            error_capacity,
            "上传 GPU 出生点配置"
        ) ||
        !mcseed_gpu_check(
            mcseed_gpu_memcpy_async(
                context->outer_spawn_offsets,
                outer_offsets,
                outer_count * sizeof(*outer_offsets),
                MCSEED_GPU_COPY_HOST_TO_DEVICE,
                context->stream
            ),
            error,
            error_capacity,
            "上传 GPU 出生点外层偏移"
        ) ||
        !mcseed_gpu_check(
            mcseed_gpu_memcpy_async(
                context->inner_spawn_offsets,
                inner_offsets,
                inner_count * sizeof(*inner_offsets),
                MCSEED_GPU_COPY_HOST_TO_DEVICE,
                context->stream
            ),
            error,
            error_capacity,
            "上传 GPU 出生点内层偏移"
        ))
        return 0;
    context->outer_spawn_offset_count = outer_count;
    context->inner_spawn_offset_count = inner_count;
    context->spawn_perlin_count = spawn_config->perlin_count;
    return 1;
}

extern "C" McSeedGpuContext *mcseed_gpu_context_create(
    int32_t device,
    const McSeedGpuStructureConfig *configs,
    size_t config_count,
    const McSeedGpuPredicate *predicates,
    size_t predicate_count,
    const McSeedGpuPairPredicate *pair_predicates,
    size_t pair_predicate_count,
    const McSeedGpuSpawnConfig *spawn_config,
    char *error,
    size_t error_capacity
)
{
    McSeedGpuContext *context;
    if (!configs || config_count == 0 ||
        (!predicates && predicate_count != 0) ||
        (!pair_predicates && pair_predicate_count != 0) ||
        (predicate_count == 0 && pair_predicate_count == 0)) {
        if (error && error_capacity)
            snprintf(error, error_capacity, "GPU 预筛选计划不能为空");
        return NULL;
    }
    context = (McSeedGpuContext *)calloc(1, sizeof(*context));
    if (!context) {
        if (error && error_capacity)
            snprintf(error, error_capacity, "无法分配 GPU 上下文");
        return NULL;
    }
    context->device = device;
    context->config_count = config_count;
    context->predicate_count = predicate_count;
    context->pair_predicate_count = pair_predicate_count;

    if (!mcseed_gpu_check(
            mcseed_gpu_set_device(device), error, error_capacity, "选择 GPU 设备"
        ) ||
        !mcseed_gpu_check(
            mcseed_gpu_stream_create(&context->stream),
            error,
            error_capacity,
            "创建 GPU stream"
        ) ||
        !mcseed_gpu_check(
            mcseed_gpu_malloc((void **)&context->configs, config_count * sizeof(*configs)),
            error,
            error_capacity,
            "分配 GPU 结构配置"
        ) ||
        !mcseed_gpu_check(
            mcseed_gpu_memcpy_async(
                context->configs,
                configs,
                config_count * sizeof(*configs),
                MCSEED_GPU_COPY_HOST_TO_DEVICE,
                context->stream
            ),
            error,
            error_capacity,
            "上传 GPU 结构配置"
        )) {
        mcseed_gpu_context_cleanup(context);
        return NULL;
    }
    if (predicate_count != 0 &&
        (!mcseed_gpu_check(
            mcseed_gpu_malloc(
                (void **)&context->predicates,
                predicate_count * sizeof(*predicates)
            ),
            error,
            error_capacity,
            "分配 GPU 条件配置"
        ) ||
        !mcseed_gpu_check(
            mcseed_gpu_memcpy_async(
                context->predicates,
                predicates,
                predicate_count * sizeof(*predicates),
                MCSEED_GPU_COPY_HOST_TO_DEVICE,
                context->stream
            ),
            error,
            error_capacity,
            "上传 GPU 条件配置"
        ))) {
        mcseed_gpu_context_cleanup(context);
        return NULL;
    }
    if (pair_predicate_count != 0 &&
        (!mcseed_gpu_check(
            mcseed_gpu_malloc(
                (void **)&context->pair_predicates,
                pair_predicate_count * sizeof(*pair_predicates)
            ),
            error,
            error_capacity,
            "分配 GPU 共址条件配置"
        ) ||
        !mcseed_gpu_check(
            mcseed_gpu_memcpy_async(
                context->pair_predicates,
                pair_predicates,
                pair_predicate_count * sizeof(*pair_predicates),
                MCSEED_GPU_COPY_HOST_TO_DEVICE,
                context->stream
            ),
            error,
            error_capacity,
            "上传 GPU 共址条件配置"
        ))) {
        mcseed_gpu_context_cleanup(context);
        return NULL;
    }
    if (!mcseed_gpu_prepare_spawn_estimator(
            context, spawn_config, error, error_capacity
        )) {
        mcseed_gpu_context_cleanup(context);
        return NULL;
    }
    if (!mcseed_gpu_check(
            mcseed_gpu_stream_synchronize(context->stream),
            error,
            error_capacity,
            "同步 GPU 条件配置"
        )) {
        mcseed_gpu_context_cleanup(context);
        return NULL;
    }
    return context;
}

extern "C" void mcseed_gpu_context_destroy(McSeedGpuContext *context)
{
    mcseed_gpu_context_cleanup(context);
}

static int mcseed_gpu_reserve_candidates(
    McSeedGpuContext *context,
    size_t candidate_count,
    char *error,
    size_t error_capacity
)
{
    size_t capacity;
    McSeedGpuCandidate *new_candidates = NULL;
    uint8_t *new_matches = NULL;
    uint32_t *new_state_indices = NULL;
    if (candidate_count <= context->candidate_capacity)
        return 1;
    capacity = context->candidate_capacity ? context->candidate_capacity : 4096;
    while (capacity < candidate_count) {
        if (capacity > SIZE_MAX / 2) {
            if (error && error_capacity)
                snprintf(error, error_capacity, "GPU 候选批大小溢出");
            return 0;
        }
        capacity *= 2;
    }
    if (!mcseed_gpu_check(
            mcseed_gpu_malloc((void **)&new_candidates, capacity * sizeof(*new_candidates)),
            error,
            error_capacity,
            "分配 GPU 候选缓冲区"
        ))
        return 0;
    if (!mcseed_gpu_check(
            mcseed_gpu_malloc((void **)&new_matches, capacity * sizeof(*new_matches)),
            error,
            error_capacity,
            "分配 GPU 结果缓冲区"
        )) {
        (void)mcseed_gpu_free(new_candidates);
        return 0;
    }
    if (!mcseed_gpu_check(
            mcseed_gpu_malloc(
                (void **)&new_state_indices,
                capacity * sizeof(*new_state_indices)
            ),
            error,
            error_capacity,
            "分配 GPU 出生点状态索引缓冲区"
        )) {
        (void)mcseed_gpu_free(new_candidates);
        (void)mcseed_gpu_free(new_matches);
        return 0;
    }
    if (context->candidates)
        (void)mcseed_gpu_free(context->candidates);
    if (context->matches)
        (void)mcseed_gpu_free(context->matches);
    if (context->spawn_state_indices)
        (void)mcseed_gpu_free(context->spawn_state_indices);
    context->candidates = new_candidates;
    context->matches = new_matches;
    context->spawn_state_indices = new_state_indices;
    context->candidate_capacity = capacity;
    return 1;
}

static void mcseed_gpu_try_reserve_spawn_states(
    McSeedGpuContext *context,
    size_t candidate_count
)
{
    McSeedGpuPerlinNoise *states = NULL;
    size_t state_count;
    size_t bytes;
    size_t capacity = context->candidate_capacity;
    if (!context->spawn_perlin_count || capacity < candidate_count ||
        candidate_count <= context->spawn_state_capacity)
        return;
    if (capacity > SIZE_MAX / context->spawn_perlin_count)
        return;
    state_count = capacity * context->spawn_perlin_count;
    if (state_count > SIZE_MAX / sizeof(*states))
        return;
    bytes = state_count * sizeof(*states);
    if (mcseed_gpu_malloc((void **)&states, bytes) != MCSEED_GPU_SUCCESS) {
        (void)mcseed_gpu_get_last_error();
        return;
    }
    if (context->spawn_states)
        (void)mcseed_gpu_free(context->spawn_states);
    context->spawn_states = states;
    context->spawn_state_capacity = capacity;
}

extern "C" int32_t mcseed_gpu_filter(
    McSeedGpuContext *context,
    const McSeedGpuCandidate *candidates,
    size_t candidate_count,
    uint8_t *matches,
    char *error,
    size_t error_capacity
)
{
    const uint32_t threads = MCSEED_GPU_SPAWN_BLOCK_THREADS;
    uint32_t blocks;
    if (!context || (!candidates && candidate_count) || (!matches && candidate_count))
        return -1;
    if (candidate_count == 0)
        return 0;
    if (candidate_count > (size_t)UINT32_MAX * threads) {
        if (error && error_capacity)
            snprintf(error, error_capacity, "单批 GPU 候选数过大");
        return -2;
    }
    if (!mcseed_gpu_check(
            mcseed_gpu_set_device(context->device), error, error_capacity, "选择 GPU 设备"
        ) ||
        !mcseed_gpu_reserve_candidates(context, candidate_count, error, error_capacity) ||
        !mcseed_gpu_check(
            mcseed_gpu_memcpy_async(
                context->candidates,
                candidates,
                candidate_count * sizeof(*candidates),
                MCSEED_GPU_COPY_HOST_TO_DEVICE,
                context->stream
            ),
            error,
            error_capacity,
            "上传 GPU 候选种子"
        ))
        return -3;

    blocks = (uint32_t)((candidate_count + threads - 1) / threads);
#if defined(MCSEED_GPU_HIP)
    hipLaunchKernelGGL(
        mcseed_gpu_prefilter_kernel,
        dim3(blocks),
        dim3(threads),
        0,
        context->stream,
        context->candidates,
        candidate_count,
        context->configs,
        context->predicates,
        context->predicate_count,
        context->pair_predicates,
        context->pair_predicate_count,
        context->matches
    );
#else
    mcseed_gpu_prefilter_kernel<<<blocks, threads, 0, context->stream>>>(
        context->candidates,
        candidate_count,
        context->configs,
        context->predicates,
        context->predicate_count,
        context->pair_predicates,
        context->pair_predicate_count,
        context->matches
    );
#endif
    if (!mcseed_gpu_check(
            mcseed_gpu_get_last_error(), error, error_capacity, "启动 GPU 预筛选内核"
        ) ||
        !mcseed_gpu_check(
            mcseed_gpu_memcpy_async(
                matches,
                context->matches,
                candidate_count * sizeof(*matches),
                MCSEED_GPU_COPY_DEVICE_TO_HOST,
                context->stream
            ),
            error,
            error_capacity,
            "下载 GPU 预筛选结果"
        ) ||
        !mcseed_gpu_check(
            mcseed_gpu_stream_synchronize(context->stream),
            error,
            error_capacity,
            "同步 GPU 预筛选"
        ))
        return -4;
    return 0;
}

static int32_t mcseed_gpu_run_spawn_stage(
    McSeedGpuContext *context,
    const McSeedGpuCandidate *candidates,
    const uint32_t *state_indices,
    size_t candidate_count,
    McSeedGpuCandidate *estimates,
    uint32_t stage,
    char *error,
    size_t error_capacity
)
{
    const uint32_t threads = stage == MCSEED_GPU_SPAWN_STAGE_OUTER
        ? MCSEED_GPU_SPAWN_OUTER_THREADS
        : MCSEED_GPU_SPAWN_BLOCK_THREADS;
    uint32_t blocks;
    const uint32_t *device_state_indices = NULL;
    McSeedGpuPerlinNoise *spawn_states = NULL;
    size_t spawn_state_count = 0;
    if (!context || (!candidates && candidate_count) || (!estimates && candidate_count))
        return -1;
    if (!context->spawn_config) {
        if (error && error_capacity)
            snprintf(error, error_capacity, "当前 GPU 上下文未启用出生点估算");
        return -2;
    }
    if (candidate_count == 0)
        return 0;
    if (candidate_count > UINT32_MAX) {
        if (error && error_capacity)
            snprintf(error, error_capacity, "单批 GPU 出生点候选数过大");
        return -3;
    }
    if (!mcseed_gpu_check(
            mcseed_gpu_set_device(context->device), error, error_capacity, "选择 GPU 设备"
        ) ||
        !mcseed_gpu_reserve_candidates(context, candidate_count, error, error_capacity) ||
        !mcseed_gpu_check(
            mcseed_gpu_memcpy_async(
                context->candidates,
                candidates,
                candidate_count * sizeof(*candidates),
                MCSEED_GPU_COPY_HOST_TO_DEVICE,
                context->stream
            ),
            error,
            error_capacity,
            "上传 GPU 出生点候选种子"
        ))
        return -4;

    if (state_indices) {
        if (!mcseed_gpu_check(
                mcseed_gpu_memcpy_async(
                    context->spawn_state_indices,
                    state_indices,
                    candidate_count * sizeof(*state_indices),
                    MCSEED_GPU_COPY_HOST_TO_DEVICE,
                    context->stream
                ),
                error,
                error_capacity,
                "上传 GPU 出生点状态索引"
            ))
            return -4;
        device_state_indices = context->spawn_state_indices;
        spawn_states = context->spawn_states;
        spawn_state_count = context->spawn_state_generation_count;
    } else if (stage == MCSEED_GPU_SPAWN_STAGE_OUTER) {
        mcseed_gpu_try_reserve_spawn_states(context, candidate_count);
        if (context->spawn_state_capacity >= candidate_count) {
            spawn_states = context->spawn_states;
            spawn_state_count = candidate_count;
        }
        context->spawn_state_generation_count = 0;
    }

    blocks = (uint32_t)candidate_count;
#if defined(MCSEED_GPU_HIP)
    if (stage == MCSEED_GPU_SPAWN_STAGE_OUTER) {
        hipLaunchKernelGGL(
            mcseed_gpu_estimate_spawn_outer_kernel,
            dim3(blocks),
            dim3(threads),
            0,
            context->stream,
            context->candidates,
            candidate_count,
            context->spawn_config,
            context->outer_spawn_offsets,
            context->outer_spawn_offset_count,
            spawn_states,
            spawn_state_count
        );
    } else {
        hipLaunchKernelGGL(
            mcseed_gpu_estimate_spawn_kernel,
            dim3(blocks),
            dim3(threads),
            0,
            context->stream,
            context->candidates,
            candidate_count,
            context->spawn_config,
            context->outer_spawn_offsets,
            context->outer_spawn_offset_count,
            context->inner_spawn_offsets,
            context->inner_spawn_offset_count,
            stage,
            device_state_indices,
            spawn_states,
            spawn_state_count
        );
    }
#else
    if (stage == MCSEED_GPU_SPAWN_STAGE_OUTER) {
        mcseed_gpu_estimate_spawn_outer_kernel<<<blocks, threads, 0, context->stream>>>(
            context->candidates,
            candidate_count,
            context->spawn_config,
            context->outer_spawn_offsets,
            context->outer_spawn_offset_count,
            spawn_states,
            spawn_state_count
        );
    } else {
        mcseed_gpu_estimate_spawn_kernel<<<blocks, threads, 0, context->stream>>>(
            context->candidates,
            candidate_count,
            context->spawn_config,
            context->outer_spawn_offsets,
            context->outer_spawn_offset_count,
            context->inner_spawn_offsets,
            context->inner_spawn_offset_count,
            stage,
            device_state_indices,
            spawn_states,
            spawn_state_count
        );
    }
#endif
    if (!mcseed_gpu_check(
            mcseed_gpu_get_last_error(), error, error_capacity, "启动 GPU 出生点估算内核"
        ) ||
        !mcseed_gpu_check(
            mcseed_gpu_memcpy_async(
                estimates,
                context->candidates,
                candidate_count * sizeof(*estimates),
                MCSEED_GPU_COPY_DEVICE_TO_HOST,
                context->stream
            ),
            error,
            error_capacity,
            "下载 GPU 出生点估算结果"
        ) ||
        !mcseed_gpu_check(
            mcseed_gpu_stream_synchronize(context->stream),
            error,
            error_capacity,
            "同步 GPU 出生点估算"
        ))
        return -5;
    if (stage == MCSEED_GPU_SPAWN_STAGE_OUTER && spawn_states)
        context->spawn_state_generation_count = candidate_count;
    return 0;
}

extern "C" int32_t mcseed_gpu_estimate_spawns(
    McSeedGpuContext *context,
    const McSeedGpuCandidate *candidates,
    size_t candidate_count,
    McSeedGpuCandidate *estimates,
    char *error,
    size_t error_capacity
)
{
    return mcseed_gpu_run_spawn_stage(
        context,
        candidates,
        NULL,
        candidate_count,
        estimates,
        MCSEED_GPU_SPAWN_STAGE_FULL,
        error,
        error_capacity
    );
}

extern "C" int32_t mcseed_gpu_estimate_spawn_outer(
    McSeedGpuContext *context,
    const McSeedGpuCandidate *candidates,
    size_t candidate_count,
    McSeedGpuCandidate *estimates,
    char *error,
    size_t error_capacity
)
{
    return mcseed_gpu_run_spawn_stage(
        context,
        candidates,
        NULL,
        candidate_count,
        estimates,
        MCSEED_GPU_SPAWN_STAGE_OUTER,
        error,
        error_capacity
    );
}

extern "C" int32_t mcseed_gpu_refine_cached_spawn_estimates(
    McSeedGpuContext *context,
    const McSeedGpuCandidate *candidates,
    const uint32_t *state_indices,
    size_t candidate_count,
    McSeedGpuCandidate *estimates,
    char *error,
    size_t error_capacity
)
{
    if (!state_indices && candidate_count)
        return -1;
    return mcseed_gpu_run_spawn_stage(
        context,
        candidates,
        state_indices,
        candidate_count,
        estimates,
        MCSEED_GPU_SPAWN_STAGE_INNER,
        error,
        error_capacity
    );
}
