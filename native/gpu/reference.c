#include "placement.h"

void mcseed_gpu_reference_filter(
    const McSeedGpuCandidate *candidates,
    size_t candidate_count,
    const McSeedGpuStructureConfig *configs,
    size_t config_count,
    const McSeedGpuPredicate *predicates,
    size_t predicate_count,
    uint8_t *matches
)
{
    size_t index;
    (void)config_count;
    if (!candidates || !configs || !predicates || !matches)
        return;
    for (index = 0; index < candidate_count; index++) {
        matches[index] = (uint8_t)mcseed_gpu_candidate_matches(
            &candidates[index], configs, predicates, predicate_count
        );
    }
}
