#include "abi.h"

#include <stddef.h>
#include <stdint.h>
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

typedef struct McSeedGpuContext {
    int32_t device;
    McSeedGpuStream stream;
    McSeedGpuStructureConfig *configs;
    McSeedGpuPredicate *predicates;
    McSeedGpuPairPredicate *pair_predicates;
    McSeedGpuCandidate *candidates;
    uint8_t *matches;
    size_t config_count;
    size_t predicate_count;
    size_t pair_predicate_count;
    size_t candidate_capacity;
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
    if (context->configs)
        (void)mcseed_gpu_free(context->configs);
    if (context->predicates)
        (void)mcseed_gpu_free(context->predicates);
    if (context->pair_predicates)
        (void)mcseed_gpu_free(context->pair_predicates);
    if (context->stream)
        (void)mcseed_gpu_stream_destroy(context->stream);
    free(context);
}

extern "C" McSeedGpuContext *mcseed_gpu_context_create(
    int32_t device,
    const McSeedGpuStructureConfig *configs,
    size_t config_count,
    const McSeedGpuPredicate *predicates,
    size_t predicate_count,
    const McSeedGpuPairPredicate *pair_predicates,
    size_t pair_predicate_count,
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
    if (context->candidates)
        (void)mcseed_gpu_free(context->candidates);
    if (context->matches)
        (void)mcseed_gpu_free(context->matches);
    context->candidates = new_candidates;
    context->matches = new_matches;
    context->candidate_capacity = capacity;
    return 1;
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
    const uint32_t threads = 256;
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
