#include "jigsaw.h"
#include "version.h"

#include "finders.h"

#include <limits.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

enum {
    MCJIGSAW_MAX_SPACES = MCJIGSAW_PIECE_CAPACITY + 1,
    MCJIGSAW_MAX_OCCUPATIONS = MCJIGSAW_PIECE_CAPACITY + 1,
    MCJIGSAW_MAX_POOL_TEMPLATES = 1024,
    MCJIGSAW_MAX_CONNECTORS = 64,
    MCJIGSAW_MAX_DEPTH = 6,
    MCJIGSAW_MAX_HORIZONTAL_DISTANCE = 80,
    MCJIGSAW_MAX_VERTICAL_DISTANCE = 80,
};

typedef struct McJigsawVector {
    int32_t x;
    int32_t y;
    int32_t z;
} McJigsawVector;

typedef struct McJigsawPlacedPiece {
    McJigsawPiece output;
    int32_t ground_level_delta;
} McJigsawPlacedPiece;

typedef struct McJigsawOccupation {
    McJigsawBox box;
    int32_t next;
} McJigsawOccupation;

typedef struct McJigsawFreeSpace {
    McJigsawBox region;
    int32_t first_occupation;
} McJigsawFreeSpace;

typedef struct McJigsawPieceState {
    uint16_t piece;
    uint16_t space;
    uint8_t depth;
    uint8_t active;
    int32_t priority;
    uint32_t sequence;
} McJigsawPieceState;

typedef struct McJigsawConnector {
    McJigsawVector position;
    uint8_t front;
    uint8_t top;
    uint8_t joint;
    uint16_t pool;
    int32_t placement_priority;
    int32_t selection_priority;
    const char *name;
    const char *target;
} McJigsawConnector;

struct McJigsawWorkspace {
    const McJigsawData *data;
    uint64_t random;
    McJigsawHeightFunction height_function;
    void *height_context;
    McJigsawPlacedPiece pieces[MCJIGSAW_PIECE_CAPACITY];
    size_t piece_count;
    McJigsawFreeSpace spaces[MCJIGSAW_MAX_SPACES];
    size_t space_count;
    McJigsawOccupation occupations[MCJIGSAW_MAX_OCCUPATIONS];
    size_t occupation_count;
    McJigsawPieceState states[MCJIGSAW_PIECE_CAPACITY];
    size_t state_count;
    uint32_t next_sequence;
};

typedef struct McJigsawWorkspace McJigsawEnvironment;

#include "generated/village_26_2.inc"

static int32_t add_i32(int32_t left, int32_t right, int *ok)
{
    int64_t result = (int64_t)left + right;
    if (result < INT32_MIN || result > INT32_MAX) {
        *ok = 0;
        return 0;
    }
    return (int32_t)result;
}

static McJigsawVector vector_add(McJigsawVector left, McJigsawVector right, int *ok)
{
    McJigsawVector result;
    result.x = add_i32(left.x, right.x, ok);
    result.y = add_i32(left.y, right.y, ok);
    result.z = add_i32(left.z, right.z, ok);
    return result;
}

static McJigsawVector vector_subtract(McJigsawVector left, McJigsawVector right, int *ok)
{
    McJigsawVector result;
    int64_t x = (int64_t)left.x - right.x;
    int64_t y = (int64_t)left.y - right.y;
    int64_t z = (int64_t)left.z - right.z;
    if (x < INT32_MIN || x > INT32_MAX ||
        y < INT32_MIN || y > INT32_MAX ||
        z < INT32_MIN || z > INT32_MAX) {
        *ok = 0;
        return (McJigsawVector){0, 0, 0};
    }
    result.x = (int32_t)x;
    result.y = (int32_t)y;
    result.z = (int32_t)z;
    return result;
}

static McJigsawVector direction_step(uint8_t direction)
{
    switch (direction) {
    case MCJIGSAW_DIR_DOWN:  return (McJigsawVector){ 0, -1,  0};
    case MCJIGSAW_DIR_UP:    return (McJigsawVector){ 0,  1,  0};
    case MCJIGSAW_DIR_NORTH: return (McJigsawVector){ 0,  0, -1};
    case MCJIGSAW_DIR_SOUTH: return (McJigsawVector){ 0,  0,  1};
    case MCJIGSAW_DIR_WEST:  return (McJigsawVector){-1,  0,  0};
    case MCJIGSAW_DIR_EAST:  return (McJigsawVector){ 1,  0,  0};
    default:                 return (McJigsawVector){ 0,  0,  0};
    }
}

static uint8_t opposite_direction(uint8_t direction)
{
    switch (direction) {
    case MCJIGSAW_DIR_DOWN:  return MCJIGSAW_DIR_UP;
    case MCJIGSAW_DIR_UP:    return MCJIGSAW_DIR_DOWN;
    case MCJIGSAW_DIR_NORTH: return MCJIGSAW_DIR_SOUTH;
    case MCJIGSAW_DIR_SOUTH: return MCJIGSAW_DIR_NORTH;
    case MCJIGSAW_DIR_WEST:  return MCJIGSAW_DIR_EAST;
    case MCJIGSAW_DIR_EAST:  return MCJIGSAW_DIR_WEST;
    default:                 return direction;
    }
}

static uint8_t rotate_direction(uint8_t direction, uint8_t rotation)
{
    if (direction == MCJIGSAW_DIR_DOWN || direction == MCJIGSAW_DIR_UP)
        return direction;
    switch (rotation & 3) {
    case 0:
        return direction;
    case 1:
        switch (direction) {
        case MCJIGSAW_DIR_NORTH: return MCJIGSAW_DIR_EAST;
        case MCJIGSAW_DIR_EAST:  return MCJIGSAW_DIR_SOUTH;
        case MCJIGSAW_DIR_SOUTH: return MCJIGSAW_DIR_WEST;
        default:                 return MCJIGSAW_DIR_NORTH;
        }
    case 2:
        return opposite_direction(direction);
    default:
        switch (direction) {
        case MCJIGSAW_DIR_NORTH: return MCJIGSAW_DIR_WEST;
        case MCJIGSAW_DIR_WEST:  return MCJIGSAW_DIR_SOUTH;
        case MCJIGSAW_DIR_SOUTH: return MCJIGSAW_DIR_EAST;
        default:                 return MCJIGSAW_DIR_NORTH;
        }
    }
}

static McJigsawVector rotate_vector(McJigsawVector vector, uint8_t rotation)
{
    switch (rotation & 3) {
    case 0: return vector;
    case 1: return (McJigsawVector){-vector.z, vector.y, vector.x};
    case 2: return (McJigsawVector){-vector.x, vector.y, -vector.z};
    default:return (McJigsawVector){vector.z, vector.y, -vector.x};
    }
}

static McJigsawBox element_box(
    const McJigsawElementData *element,
    McJigsawVector position,
    uint8_t rotation,
    int *ok
)
{
    McJigsawVector corner = {
        element->size_x > 0 ? element->size_x - 1 : 0,
        element->size_y > 0 ? element->size_y - 1 : 0,
        element->size_z > 0 ? element->size_z - 1 : 0,
    };
    McJigsawVector rotated = rotate_vector(corner, rotation);
    McJigsawVector other = vector_add(position, rotated, ok);
    McJigsawBox box;
    box.min_x = position.x < other.x ? position.x : other.x;
    box.min_y = position.y < other.y ? position.y : other.y;
    box.min_z = position.z < other.z ? position.z : other.z;
    box.max_x = position.x > other.x ? position.x : other.x;
    box.max_y = position.y > other.y ? position.y : other.y;
    box.max_z = position.z > other.z ? position.z : other.z;
    return box;
}

static int box_contains_position(const McJigsawBox *box, McJigsawVector position)
{
    return position.x >= box->min_x && position.x <= box->max_x &&
           position.y >= box->min_y && position.y <= box->max_y &&
           position.z >= box->min_z && position.z <= box->max_z;
}

static int box_contains_box(const McJigsawBox *outer, const McJigsawBox *inner)
{
    return inner->min_x >= outer->min_x && inner->max_x <= outer->max_x &&
           inner->min_y >= outer->min_y && inner->max_y <= outer->max_y &&
           inner->min_z >= outer->min_z && inner->max_z <= outer->max_z;
}

static int boxes_intersect(const McJigsawBox *left, const McJigsawBox *right)
{
    return left->max_x >= right->min_x && left->min_x <= right->max_x &&
           left->max_y >= right->min_y && left->min_y <= right->max_y &&
           left->max_z >= right->min_z && left->min_z <= right->max_z;
}

static void move_box(McJigsawBox *box, int32_t x, int32_t y, int32_t z, int *ok)
{
    box->min_x = add_i32(box->min_x, x, ok);
    box->max_x = add_i32(box->max_x, x, ok);
    box->min_y = add_i32(box->min_y, y, ok);
    box->max_y = add_i32(box->max_y, y, ok);
    box->min_z = add_i32(box->min_z, z, ok);
    box->max_z = add_i32(box->max_z, z, ok);
}

static void shuffle_u16(uint16_t *values, size_t count, uint64_t *random)
{
    size_t index;
    for (index = count; index > 1; index--) {
        size_t swap_index = (size_t)nextInt(random, (int)index);
        uint16_t temporary = values[index - 1];
        values[index - 1] = values[swap_index];
        values[swap_index] = temporary;
    }
}

static void shuffle_u8(uint8_t *values, size_t count, uint64_t *random)
{
    size_t index;
    for (index = count; index > 1; index--) {
        size_t swap_index = (size_t)nextInt(random, (int)index);
        uint8_t temporary = values[index - 1];
        values[index - 1] = values[swap_index];
        values[swap_index] = temporary;
    }
}

static int connector_priority(
    const McJigsawData *data,
    const McJigsawElementData *element,
    uint16_t local_index
)
{
    size_t index = (size_t)element->connector_offset + local_index;
    if (index >= data->connector_count)
        return INT_MIN;
    return data->connectors[index].selection_priority;
}

static int shuffled_connectors(
    McJigsawEnvironment *environment,
    uint16_t element_id,
    uint16_t *indices,
    size_t *count
)
{
    const McJigsawElementData *element;
    size_t index;
    if (element_id >= environment->data->element_count)
        return 0;
    element = &environment->data->elements[element_id];
    if (element->connector_count > MCJIGSAW_MAX_CONNECTORS)
        return 0;
    *count = element->connector_count;
    for (index = 0; index < *count; index++)
        indices[index] = (uint16_t)index;
    shuffle_u16(indices, *count, &environment->random);

    /* Java performs a stable, descending sort after the random shuffle. */
    for (index = 1; index < *count; index++) {
        uint16_t value = indices[index];
        int priority = connector_priority(environment->data, element, value);
        size_t insertion = index;
        while (insertion > 0 &&
               connector_priority(environment->data, element, indices[insertion - 1]) < priority) {
            indices[insertion] = indices[insertion - 1];
            insertion--;
        }
        indices[insertion] = value;
    }
    return 1;
}

static int transformed_connector(
    const McJigsawData *data,
    uint16_t element_id,
    uint16_t connector_index,
    McJigsawVector base,
    uint8_t rotation,
    McJigsawConnector *result
)
{
    const McJigsawElementData *element;
    const McJigsawConnectorData *source;
    McJigsawVector local;
    int ok = 1;
    if (element_id >= data->element_count)
        return 0;
    element = &data->elements[element_id];
    if (connector_index >= element->connector_count ||
        (size_t)element->connector_offset + connector_index >= data->connector_count)
        return 0;
    source = &data->connectors[element->connector_offset + connector_index];
    local = rotate_vector((McJigsawVector){source->x, source->y, source->z}, rotation);
    result->position = vector_add(base, local, &ok);
    result->front = rotate_direction(source->front, rotation);
    result->top = rotate_direction(source->top, rotation);
    result->joint = source->joint;
    result->pool = source->pool;
    result->placement_priority = source->placement_priority;
    result->selection_priority = source->selection_priority;
    result->name = source->name;
    result->target = source->target;
    return ok;
}

static int append_pool_templates(
    McJigsawEnvironment *environment,
    uint16_t pool_id,
    uint16_t *templates,
    size_t *count
)
{
    const McJigsawPoolData *pool;
    size_t entry_index;
    size_t start;
    if (pool_id >= environment->data->pool_count)
        return 0;
    pool = &environment->data->pools[pool_id];
    start = *count;
    if ((size_t)pool->entry_offset + pool->entry_count > environment->data->pool_entry_count)
        return 0;
    for (entry_index = 0; entry_index < pool->entry_count; entry_index++) {
        const McJigsawPoolEntryData *entry =
            &environment->data->pool_entries[pool->entry_offset + entry_index];
        size_t repeat;
        if (entry->element >= environment->data->element_count ||
            *count + entry->weight > MCJIGSAW_MAX_POOL_TEMPLATES)
            return 0;
        for (repeat = 0; repeat < entry->weight; repeat++)
            templates[(*count)++] = entry->element;
    }
    shuffle_u16(templates + start, *count - start, &environment->random);
    return 1;
}

static int random_pool_element(
    McJigsawEnvironment *environment,
    uint16_t pool_id,
    uint16_t *element_id
)
{
    const McJigsawPoolData *pool;
    int choice;
    size_t entry_index;
    if (pool_id >= environment->data->pool_count)
        return 0;
    pool = &environment->data->pools[pool_id];
    if (pool->total_weight == 0) {
        *element_id = 0;
        return 1;
    }
    choice = nextInt(&environment->random, pool->total_weight);
    for (entry_index = 0; entry_index < pool->entry_count; entry_index++) {
        const McJigsawPoolEntryData *entry =
            &environment->data->pool_entries[pool->entry_offset + entry_index];
        if (choice < entry->weight) {
            *element_id = entry->element;
            return 1;
        }
        choice -= entry->weight;
    }
    return 0;
}

static int new_space(McJigsawEnvironment *environment, McJigsawBox region, uint16_t *space_id)
{
    if (environment->space_count >= MCJIGSAW_MAX_SPACES)
        return 0;
    *space_id = (uint16_t)environment->space_count;
    environment->spaces[environment->space_count].region = region;
    environment->spaces[environment->space_count].first_occupation = -1;
    environment->space_count++;
    return 1;
}

static int occupy_space(McJigsawEnvironment *environment, uint16_t space_id, McJigsawBox box)
{
    McJigsawFreeSpace *space;
    McJigsawOccupation *occupation;
    if (space_id >= environment->space_count ||
        environment->occupation_count >= MCJIGSAW_MAX_OCCUPATIONS)
        return 0;
    space = &environment->spaces[space_id];
    occupation = &environment->occupations[environment->occupation_count];
    occupation->box = box;
    occupation->next = space->first_occupation;
    space->first_occupation = (int32_t)environment->occupation_count;
    environment->occupation_count++;
    return 1;
}

static int space_accepts(
    const McJigsawEnvironment *environment,
    uint16_t space_id,
    const McJigsawBox *box
)
{
    int32_t occupation_index;
    if (space_id >= environment->space_count)
        return 0;
    if (!box_contains_box(&environment->spaces[space_id].region, box))
        return 0;
    occupation_index = environment->spaces[space_id].first_occupation;
    while (occupation_index >= 0) {
        const McJigsawOccupation *occupation;
        if ((size_t)occupation_index >= environment->occupation_count)
            return 0;
        occupation = &environment->occupations[occupation_index];
        if (boxes_intersect(&occupation->box, box))
            return 0;
        occupation_index = occupation->next;
    }
    return 1;
}

static int enqueue_state(
    McJigsawEnvironment *environment,
    uint16_t piece,
    uint16_t space,
    uint8_t depth,
    int32_t priority
)
{
    McJigsawPieceState *state;
    if (environment->state_count >= MCJIGSAW_PIECE_CAPACITY)
        return 0;
    state = &environment->states[environment->state_count++];
    state->piece = piece;
    state->space = space;
    state->depth = depth;
    state->active = 1;
    state->priority = priority;
    state->sequence = environment->next_sequence++;
    return 1;
}

static int pop_state(McJigsawEnvironment *environment, McJigsawPieceState *result)
{
    size_t index;
    size_t best = SIZE_MAX;
    for (index = 0; index < environment->state_count; index++) {
        const McJigsawPieceState *candidate = &environment->states[index];
        if (!candidate->active)
            continue;
        if (best == SIZE_MAX ||
            candidate->priority > environment->states[best].priority ||
            (candidate->priority == environment->states[best].priority &&
             candidate->sequence < environment->states[best].sequence))
            best = index;
    }
    if (best == SIZE_MAX)
        return 0;
    *result = environment->states[best];
    environment->states[best].active = 0;
    return 1;
}

static int connectors_attach(
    const McJigsawConnector *source,
    const McJigsawConnector *target
)
{
    return source->front == opposite_direction(target->front) &&
           (source->joint == MCJIGSAW_JOINT_ROLLABLE || source->top == target->top) &&
           strcmp(source->target, target->name) == 0;
}

static int expansion_height(
    const McJigsawEnvironment *environment,
    const McJigsawConnector *connector,
    const McJigsawBox *element_box
)
{
    const McJigsawPoolData *pool;
    const McJigsawPoolData *fallback;
    McJigsawVector next;
    int ok = 1;
    if (connector->pool >= environment->data->pool_count)
        return 0;
    next = vector_add(connector->position, direction_step(connector->front), &ok);
    if (!ok || !box_contains_position(element_box, next))
        return 0;
    pool = &environment->data->pools[connector->pool];
    if (pool->fallback >= environment->data->pool_count)
        return pool->max_height;
    fallback = &environment->data->pools[pool->fallback];
    return pool->max_height > fallback->max_height ? pool->max_height : fallback->max_height;
}

static int try_placing_children(
    McJigsawEnvironment *environment,
    uint16_t source_piece_id,
    uint16_t context_space,
    uint8_t depth
)
{
    McJigsawPlacedPiece *source_piece;
    const McJigsawElementData *source_element;
    uint16_t source_indices[MCJIGSAW_MAX_CONNECTORS];
    size_t source_count;
    size_t source_order;
    uint16_t local_space = UINT16_MAX;
    if (source_piece_id >= environment->piece_count)
        return 0;
    source_piece = &environment->pieces[source_piece_id];
    if (source_piece->output.element >= environment->data->element_count)
        return 0;
    source_element = &environment->data->elements[source_piece->output.element];
    if (!shuffled_connectors(environment, source_piece->output.element, source_indices, &source_count))
        return 0;

    for (source_order = 0; source_order < source_count; source_order++) {
        McJigsawConnector source_connector;
        McJigsawVector target_connector_position;
        uint16_t target_templates[MCJIGSAW_MAX_POOL_TEMPLATES];
        size_t target_template_count = 0;
        size_t target_template_order;
        uint16_t children_space;
        int32_t source_local_y;
        int32_t source_base_height = INT32_MIN;
        int ok = 1;

        if (!transformed_connector(
                environment->data,
                source_piece->output.element,
                source_indices[source_order],
                (McJigsawVector){source_piece->output.x, source_piece->output.y, source_piece->output.z},
                source_piece->output.rotation,
                &source_connector))
            return 0;
        target_connector_position = vector_add(
            source_connector.position,
            direction_step(source_connector.front),
            &ok
        );
        if (!ok || source_connector.pool >= environment->data->pool_count)
            continue;
        source_local_y = source_connector.position.y - source_piece->output.box.min_y;

        if (box_contains_position(&source_piece->output.box, target_connector_position)) {
            if (local_space == UINT16_MAX &&
                !new_space(environment, source_piece->output.box, &local_space))
                return 0;
            children_space = local_space;
        } else {
            children_space = context_space;
        }

        if (depth != MCJIGSAW_MAX_DEPTH &&
            !append_pool_templates(
                environment,
                source_connector.pool,
                target_templates,
                &target_template_count))
            return 0;
        {
            const McJigsawPoolData *target_pool =
                &environment->data->pools[source_connector.pool];
            if (!append_pool_templates(
                    environment,
                    target_pool->fallback,
                    target_templates,
                    &target_template_count))
                return 0;
        }

        for (target_template_order = 0;
             target_template_order < target_template_count;
             target_template_order++) {
            uint16_t target_element_id = target_templates[target_template_order];
            const McJigsawElementData *target_element;
            uint8_t rotations[4] = {0, 1, 2, 3};
            size_t rotation_order;
            if (target_element_id >= environment->data->element_count)
                return 0;
            target_element = &environment->data->elements[target_element_id];
            if (target_element->kind == MCJIGSAW_ELEMENT_EMPTY)
                break;
            shuffle_u8(rotations, 4, &environment->random);

            for (rotation_order = 0; rotation_order < 4; rotation_order++) {
                uint8_t target_rotation = rotations[rotation_order];
                uint16_t target_indices[MCJIGSAW_MAX_CONNECTORS];
                size_t target_count;
                size_t target_order;
                McJigsawBox hack_box = element_box(
                    target_element,
                    (McJigsawVector){0, 0, 0},
                    target_rotation,
                    &ok
                );
                int expand_to = 0;
                if (!ok || !shuffled_connectors(
                        environment,
                        target_element_id,
                        target_indices,
                        &target_count))
                    return 0;

                if (hack_box.max_y - hack_box.min_y + 1 <= 16) {
                    for (target_order = 0; target_order < target_count; target_order++) {
                        McJigsawConnector connector;
                        int height;
                        if (!transformed_connector(
                                environment->data,
                                target_element_id,
                                target_indices[target_order],
                                (McJigsawVector){0, 0, 0},
                                target_rotation,
                                &connector))
                            return 0;
                        height = expansion_height(environment, &connector, &hack_box);
                        if (height > expand_to)
                            expand_to = height;
                    }
                }

                for (target_order = 0; target_order < target_count; target_order++) {
                    McJigsawConnector target_connector;
                    McJigsawVector raw_target_position;
                    McJigsawVector target_position;
                    McJigsawBox raw_target_box;
                    McJigsawBox target_box;
                    int32_t target_local_y;
                    int32_t delta_y;
                    int32_t target_box_y;
                    int32_t y_offset;
                    int target_rigid;
                    int32_t target_ground_delta;
                    uint16_t piece_id;

                    if (!transformed_connector(
                            environment->data,
                            target_element_id,
                            target_indices[target_order],
                            (McJigsawVector){0, 0, 0},
                            target_rotation,
                            &target_connector))
                        return 0;
                    if (!connectors_attach(&source_connector, &target_connector))
                        continue;
                    raw_target_position = vector_subtract(
                        target_connector_position,
                        target_connector.position,
                        &ok
                    );
                    raw_target_box = element_box(
                        target_element,
                        raw_target_position,
                        target_rotation,
                        &ok
                    );
                    if (!ok)
                        return 0;
                    target_local_y = target_connector.position.y;
                    delta_y = source_local_y - target_local_y +
                        direction_step(source_connector.front).y;
                    target_rigid = target_element->projection == MCJIGSAW_PROJECTION_RIGID;
                    if (source_element->projection == MCJIGSAW_PROJECTION_RIGID && target_rigid) {
                        target_box_y = source_piece->output.box.min_y + delta_y;
                    } else {
                        if (source_base_height == INT32_MIN)
                            source_base_height = environment->height_function(
                                environment->height_context,
                                source_connector.position.x,
                                source_connector.position.z
                            );
                        target_box_y = source_base_height - target_local_y;
                    }
                    y_offset = target_box_y - raw_target_box.min_y;
                    target_box = raw_target_box;
                    move_box(&target_box, 0, y_offset, 0, &ok);
                    target_position = raw_target_position;
                    target_position.y = add_i32(target_position.y, y_offset, &ok);
                    if (!ok)
                        return 0;

                    if (expand_to > 0) {
                        int32_t current_size = target_box.max_y - target_box.min_y;
                        int32_t new_size = expand_to + 1 > current_size
                            ? expand_to + 1
                            : current_size;
                        int32_t expanded_y = add_i32(target_box.min_y, new_size, &ok);
                        if (!ok)
                            return 0;
                        if (expanded_y > target_box.max_y)
                            target_box.max_y = expanded_y;
                    }
                    if (!space_accepts(environment, children_space, &target_box))
                        continue;
                    if (environment->piece_count >= MCJIGSAW_PIECE_CAPACITY ||
                        !occupy_space(environment, children_space, target_box))
                        return 0;

                    target_ground_delta = target_rigid
                        ? source_piece->ground_level_delta - delta_y
                        : 1;
                    piece_id = (uint16_t)environment->piece_count++;
                    environment->pieces[piece_id].output.name = target_element->name;
                    environment->pieces[piece_id].output.x = target_position.x;
                    environment->pieces[piece_id].output.y = target_position.y;
                    environment->pieces[piece_id].output.z = target_position.z;
                    environment->pieces[piece_id].output.box = target_box;
                    environment->pieces[piece_id].output.element = target_element_id;
                    environment->pieces[piece_id].output.rotation = target_rotation;
                    environment->pieces[piece_id].output.depth = (uint8_t)(depth + 1);
                    environment->pieces[piece_id].ground_level_delta = target_ground_delta;
                    if (depth + 1 <= MCJIGSAW_MAX_DEPTH &&
                        !enqueue_state(
                            environment,
                            piece_id,
                            children_space,
                            (uint8_t)(depth + 1),
                            source_connector.placement_priority))
                        return 0;
                    goto source_connector_done;
                }
            }
        }
source_connector_done:
        ;
    }
    return 1;
}

const McJigsawData *mcjigsaw_village_data(const char *version)
{
    if (version && strcmp(version, MCSEED_VERSION_NAME) == 0)
        return &MCJIGSAW_VILLAGE_DATA_26_2;
    return NULL;
}

McJigsawWorkspace *mcjigsaw_workspace_create(void)
{
    return (McJigsawWorkspace *)calloc(1, sizeof(McJigsawWorkspace));
}

void mcjigsaw_workspace_destroy(McJigsawWorkspace *workspace)
{
    free(workspace);
}

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
)
{
    McJigsawEnvironment *environment = workspace;
    const McJigsawElementData *center_element;
    McJigsawVector center_position;
    McJigsawBox center_box;
    McJigsawBox global_region;
    uint16_t center_element_id;
    uint16_t global_space;
    uint8_t center_rotation;
    int32_t center_x;
    int32_t center_z;
    int32_t bottom_y;
    int32_t move_y;
    int ok = 1;
    McJigsawPieceState state;
    size_t index;

    if (!environment || !data || !height_function || !pieces || capacity == 0 ||
        style < 0 || style >= MCJIGSAW_STYLE_COUNT)
        return -1;
    if (capacity > MCJIGSAW_PIECE_CAPACITY)
        capacity = MCJIGSAW_PIECE_CAPACITY;
    memset(environment, 0, sizeof(*environment));
    environment->data = data;
    environment->height_function = height_function;
    environment->height_context = height_context;
    environment->random = chunkGenerateRnd(world_seed, start_x >> 4, start_z >> 4);

    center_rotation = (uint8_t)nextInt(&environment->random, 4);
    if (!random_pool_element(
            environment,
            data->start_pools[style],
            &center_element_id) ||
        center_element_id >= data->element_count) {
        return -3;
    }
    center_element = &data->elements[center_element_id];
    if (center_element->kind == MCJIGSAW_ELEMENT_EMPTY) {
        return 0;
    }
    center_position = (McJigsawVector){start_x, 0, start_z};
    center_box = element_box(center_element, center_position, center_rotation, &ok);
    center_x = (center_box.max_x + center_box.min_x) / 2;
    center_z = (center_box.max_z + center_box.min_z) / 2;
    bottom_y = height_function(height_context, center_x, center_z);
    move_y = bottom_y - (center_box.min_y + 1);
    center_position.y = add_i32(center_position.y, move_y, &ok);
    move_box(&center_box, 0, move_y, 0, &ok);
    if (!ok) {
        return -3;
    }

    environment->pieces[0].output.name = center_element->name;
    environment->pieces[0].output.x = center_position.x;
    environment->pieces[0].output.y = center_position.y;
    environment->pieces[0].output.z = center_position.z;
    environment->pieces[0].output.box = center_box;
    environment->pieces[0].output.element = center_element_id;
    environment->pieces[0].output.rotation = center_rotation;
    environment->pieces[0].output.depth = 0;
    environment->pieces[0].ground_level_delta = 1;
    environment->piece_count = 1;

    global_region.min_x = center_x - MCJIGSAW_MAX_HORIZONTAL_DISTANCE;
    global_region.max_x = center_x + MCJIGSAW_MAX_HORIZONTAL_DISTANCE;
    global_region.min_z = center_z - MCJIGSAW_MAX_HORIZONTAL_DISTANCE;
    global_region.max_z = center_z + MCJIGSAW_MAX_HORIZONTAL_DISTANCE;
    global_region.min_y = bottom_y - MCJIGSAW_MAX_VERTICAL_DISTANCE;
    if (global_region.min_y < -64)
        global_region.min_y = -64;
    global_region.max_y = bottom_y + MCJIGSAW_MAX_VERTICAL_DISTANCE;
    if (global_region.max_y > 319)
        global_region.max_y = 319;
    if (!new_space(environment, global_region, &global_space) ||
        !occupy_space(environment, global_space, center_box) ||
        !try_placing_children(environment, 0, global_space, 0)) {
        return -4;
    }
    while (pop_state(environment, &state)) {
        if (!try_placing_children(
                environment,
                state.piece,
                state.space,
                state.depth)) {
            return -4;
        }
    }
    if (environment->piece_count > capacity) {
        return -4;
    }
    for (index = 0; index < environment->piece_count; index++)
        pieces[index] = environment->pieces[index].output;
    index = environment->piece_count;
    return (int32_t)index;
}
