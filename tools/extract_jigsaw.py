#!/usr/bin/env python3
"""Extract compact, build-time-independent village Jigsaw metadata from a client JAR.

The generated C include contains only pool weights, template dimensions and Jigsaw
connectors. It intentionally does not contain Minecraft blocks, entities or templates.
The application builds from the checked-in generated include and never invokes this
tool automatically.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import re
import struct
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, NoReturn


NBT_END: Final[int] = 0
NBT_BYTE: Final[int] = 1
NBT_SHORT: Final[int] = 2
NBT_INT: Final[int] = 3
NBT_LONG: Final[int] = 4
NBT_FLOAT: Final[int] = 5
NBT_DOUBLE: Final[int] = 6
NBT_BYTE_ARRAY: Final[int] = 7
NBT_STRING: Final[int] = 8
NBT_LIST: Final[int] = 9
NBT_COMPOUND: Final[int] = 10
NBT_INT_ARRAY: Final[int] = 11
NBT_LONG_ARRAY: Final[int] = 12
INT16_MIN: Final[int] = -(1 << 15)
INT16_MAX: Final[int] = (1 << 15) - 1
UINT16_MAX: Final[int] = (1 << 16) - 1
UINT32_MAX: Final[int] = (1 << 32) - 1
ENGINE_MAX_CONNECTORS: Final[int] = 64
ENGINE_MAX_POOL_TEMPLATES: Final[int] = 1024

DIRECTIONS: Final[dict[str, str]] = {
    "down": "MCJIGSAW_DIR_DOWN",
    "up": "MCJIGSAW_DIR_UP",
    "north": "MCJIGSAW_DIR_NORTH",
    "south": "MCJIGSAW_DIR_SOUTH",
    "west": "MCJIGSAW_DIR_WEST",
    "east": "MCJIGSAW_DIR_EAST",
}

START_POOLS: Final[tuple[tuple[str, str], ...]] = (
    ("plains", "village/plains/town_centers"),
    ("desert", "village/desert/town_centers"),
    ("savanna", "village/savanna/town_centers"),
    ("snowy", "village/snowy/town_centers"),
    ("taiga", "village/taiga/town_centers"),
)


class ExtractionError(RuntimeError):
    """Raised when an input resource is inconsistent or unsupported."""


class NbtReader:
    """Small strict reader for the NBT types used by structure templates."""

    def __init__(self, data: bytes) -> None:
        self._data: memoryview = memoryview(data)
        self._offset: int = 0

    def _fail(self, message: str) -> NoReturn:
        raise ExtractionError(f"NBT offset {self._offset}: {message}")

    def _read(self, length: int) -> memoryview:
        if length < 0 or self._offset + length > len(self._data):
            self._fail(f"cannot read {length} bytes")
        result = self._data[self._offset : self._offset + length]
        self._offset += length
        return result

    def _unpack(self, fmt: str) -> int | float:
        size = struct.calcsize(fmt)
        return struct.unpack(fmt, self._read(size))[0]

    def _length(self) -> int:
        value = int(self._unpack(">i"))
        if value < 0:
            self._fail(f"negative array/list length {value}")
        return value

    def _string(self) -> str:
        length = int(self._unpack(">H"))
        try:
            return self._read(length).tobytes().decode("utf-8")
        except UnicodeDecodeError as error:
            self._fail(f"invalid UTF-8 string: {error}")

    def _payload(self, tag_type: int) -> Any:
        if tag_type == NBT_BYTE:
            return int(self._unpack(">b"))
        if tag_type == NBT_SHORT:
            return int(self._unpack(">h"))
        if tag_type == NBT_INT:
            return int(self._unpack(">i"))
        if tag_type == NBT_LONG:
            return int(self._unpack(">q"))
        if tag_type == NBT_FLOAT:
            return float(self._unpack(">f"))
        if tag_type == NBT_DOUBLE:
            return float(self._unpack(">d"))
        if tag_type == NBT_BYTE_ARRAY:
            return self._read(self._length()).tobytes()
        if tag_type == NBT_STRING:
            return self._string()
        if tag_type == NBT_LIST:
            child_type = int(self._unpack(">B"))
            length = self._length()
            if child_type == NBT_END and length != 0:
                self._fail("non-empty list uses TAG_End as its element type")
            return [self._payload(child_type) for _ in range(length)]
        if tag_type == NBT_COMPOUND:
            result: dict[str, Any] = {}
            while True:
                child_type = int(self._unpack(">B"))
                if child_type == NBT_END:
                    return result
                child_name = self._string()
                result[child_name] = self._payload(child_type)
        if tag_type == NBT_INT_ARRAY:
            return [int(self._unpack(">i")) for _ in range(self._length())]
        if tag_type == NBT_LONG_ARRAY:
            return [int(self._unpack(">q")) for _ in range(self._length())]
        self._fail(f"unsupported tag type {tag_type}")

    def root_compound(self) -> dict[str, Any]:
        tag_type = int(self._unpack(">B"))
        if tag_type != NBT_COMPOUND:
            self._fail(f"root tag is {tag_type}, expected compound")
        self._string()
        value = self._payload(tag_type)
        if self._offset != len(self._data):
            self._fail(f"{len(self._data) - self._offset} trailing bytes")
        if not isinstance(value, dict):
            self._fail("root payload is not a compound")
        return value


@dataclass(frozen=True)
class Connector:
    x: int
    y: int
    z: int
    front: str
    top: str
    pool: str
    name: str
    target: str
    joint: str
    placement_priority: int
    selection_priority: int


@dataclass(frozen=True)
class Element:
    name: str
    size_x: int
    size_y: int
    size_z: int
    projection: str
    kind: str
    connectors: tuple[Connector, ...]


@dataclass(frozen=True)
class PoolEntry:
    element: int
    weight: int


@dataclass(frozen=True)
class Pool:
    name: str
    fallback: str
    entries: tuple[PoolEntry, ...]


def normalize_resource(value: str) -> str:
    value = value.strip()
    return value.removeprefix("minecraft:")


def require_dict(value: Any, context: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ExtractionError(f"{context} must be an object")
    return value


def require_list(value: Any, context: str) -> list[Any]:
    if not isinstance(value, list):
        raise ExtractionError(f"{context} must be a list")
    return value


def require_str(value: Any, context: str) -> str:
    if not isinstance(value, str):
        raise ExtractionError(f"{context} must be a string")
    return value


def require_int(value: Any, context: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ExtractionError(f"{context} must be an integer")
    return value


def require_range(value: int, minimum: int, maximum: int, context: str) -> None:
    if not minimum <= value <= maximum:
        raise ExtractionError(f"{context}={value} is outside {minimum}..{maximum}")


def orientation_parts(value: str, context: str) -> tuple[str, str]:
    try:
        front, top = value.split("_", maxsplit=1)
    except ValueError as error:
        raise ExtractionError(f"{context}: invalid orientation {value!r}") from error
    if front not in DIRECTIONS or top not in DIRECTIONS:
        raise ExtractionError(f"{context}: unknown orientation {value!r}")
    return front, top


def read_template(archive: zipfile.ZipFile, location: str, projection: str) -> Element:
    path = f"data/minecraft/structure/{location}.nbt"
    try:
        compressed = archive.read(path)
    except KeyError as error:
        raise ExtractionError(f"missing structure template {path}") from error
    try:
        raw = gzip.decompress(compressed)
    except gzip.BadGzipFile as error:
        raise ExtractionError(f"template {path} is not gzip-compressed NBT") from error
    try:
        root = NbtReader(raw).root_compound()
    except ExtractionError as error:
        raise ExtractionError(f"cannot parse {path}: {error}") from error
    size = require_list(root.get("size"), f"{path}.size")
    if len(size) != 3:
        raise ExtractionError(f"{path}.size must have three coordinates")
    size_x, size_y, size_z = (require_int(value, f"{path}.size") for value in size)

    raw_palettes = root.get("palettes")
    palettes: list[list[Any]]
    if raw_palettes is not None:
        palettes = [require_list(value, f"{path}.palettes") for value in require_list(raw_palettes, f"{path}.palettes")]
    else:
        palettes = [require_list(root.get("palette"), f"{path}.palette")]
    if not palettes:
        raise ExtractionError(f"{path} contains no palettes")

    jigsaw_states: list[dict[int, tuple[str, str]]] = []
    for palette_index, palette in enumerate(palettes):
        states: dict[int, tuple[str, str]] = {}
        for state_id, raw_state in enumerate(palette):
            state = require_dict(raw_state, f"{path}.palette[{palette_index}][{state_id}]")
            if state.get("Name") != "minecraft:jigsaw":
                continue
            properties = require_dict(state.get("Properties"), f"{path}.palette[{palette_index}][{state_id}].Properties")
            orientation = require_str(properties.get("orientation"), f"{path}.palette[{palette_index}][{state_id}].orientation")
            states[state_id] = orientation_parts(orientation, path)
        jigsaw_states.append(states)
    if any(states != jigsaw_states[0] for states in jigsaw_states[1:]):
        raise ExtractionError(f"{path} changes Jigsaw states between palettes")

    connectors: list[Connector] = []
    for block_index, raw_block in enumerate(require_list(root.get("blocks"), f"{path}.blocks")):
        block = require_dict(raw_block, f"{path}.blocks[{block_index}]")
        state_id = require_int(block.get("state"), f"{path}.blocks[{block_index}].state")
        orientation = jigsaw_states[0].get(state_id)
        if orientation is None:
            continue
        position = require_list(block.get("pos"), f"{path}.blocks[{block_index}].pos")
        if len(position) != 3:
            raise ExtractionError(f"{path}.blocks[{block_index}].pos must have three coordinates")
        x, y, z = (require_int(value, f"{path}.blocks[{block_index}].pos") for value in position)
        nbt = require_dict(block.get("nbt"), f"{path}.blocks[{block_index}].nbt")
        front, top = orientation
        default_joint = "aligned" if front in {"north", "south", "east", "west"} else "rollable"
        joint = require_str(nbt.get("joint", default_joint), f"{path}.blocks[{block_index}].joint")
        if joint not in {"aligned", "rollable"}:
            raise ExtractionError(f"{path}: unsupported joint {joint!r}")
        connectors.append(
            Connector(
                x=x,
                y=y,
                z=z,
                front=front,
                top=top,
                pool=normalize_resource(require_str(nbt.get("pool", "minecraft:empty"), f"{path}.pool")),
                name=normalize_resource(require_str(nbt.get("name", "minecraft:empty"), f"{path}.name")),
                target=normalize_resource(require_str(nbt.get("target", "minecraft:empty"), f"{path}.target")),
                joint=joint,
                placement_priority=require_int(nbt.get("placement_priority", 0), f"{path}.placement_priority"),
                selection_priority=require_int(nbt.get("selection_priority", 0), f"{path}.selection_priority"),
            )
        )
    return Element(location, size_x, size_y, size_z, projection, "template", tuple(connectors))


def feature_element(feature: str, projection: str) -> Element:
    connector = Connector(
        x=0,
        y=0,
        z=0,
        front="down",
        top="south",
        pool="empty",
        name="bottom",
        target="empty",
        joint="rollable",
        placement_priority=0,
        selection_priority=0,
    )
    return Element(f"feature/{feature}", 1, 1, 1, projection, "feature", (connector,))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jar", required=True, type=Path, help="Minecraft client JAR")
    parser.add_argument("--version", required=True, help="Minecraft version label, for example 26.2")
    parser.add_argument("--output", required=True, type=Path, help="generated C include path")
    return parser.parse_args()


def c_string(value: str) -> str:
    return json.dumps(value, ensure_ascii=True)


def symbol_suffix(version: str) -> str:
    result = re.sub(r"[^A-Za-z0-9]+", "_", version).strip("_")
    if not result:
        raise ExtractionError("version must contain at least one letter or digit")
    return result.upper()


def load_data(jar: Path) -> tuple[list[Element], list[Pool]]:
    elements: list[Element] = [Element("empty", 0, 0, 0, "rigid", "empty", ())]
    element_ids: dict[tuple[str, str, str], int] = {("empty", "empty", "rigid"): 0}
    raw_pools: list[tuple[str, dict[str, Any]]] = []

    with zipfile.ZipFile(jar) as archive:
        prefix = "data/minecraft/worldgen/template_pool/village/"
        for member in sorted(archive.namelist()):
            if not member.startswith(prefix) or not member.endswith(".json"):
                continue
            pool_name = member.removeprefix("data/minecraft/worldgen/template_pool/").removesuffix(".json")
            try:
                document = json.loads(archive.read(member))
            except (UnicodeDecodeError, json.JSONDecodeError) as error:
                raise ExtractionError(f"invalid template pool JSON {member}: {error}") from error
            raw_pools.append((pool_name, require_dict(document, member)))

        pools: list[Pool] = []
        for pool_name, document in raw_pools:
            entries: list[PoolEntry] = []
            for entry_index, raw_entry in enumerate(require_list(document.get("elements"), f"{pool_name}.elements")):
                entry = require_dict(raw_entry, f"{pool_name}.elements[{entry_index}]")
                raw_element = require_dict(entry.get("element"), f"{pool_name}.elements[{entry_index}].element")
                element_type = normalize_resource(require_str(raw_element.get("element_type"), f"{pool_name}.element_type"))
                weight = require_int(entry.get("weight"), f"{pool_name}.elements[{entry_index}].weight")
                if weight <= 0 or weight > 65_535:
                    raise ExtractionError(f"{pool_name}: weight {weight} is outside 1..65535")
                if element_type == "empty_pool_element":
                    element_id = 0
                else:
                    projection = require_str(raw_element.get("projection"), f"{pool_name}.projection")
                    if projection not in {"rigid", "terrain_matching"}:
                        raise ExtractionError(f"{pool_name}: unsupported projection {projection!r}")
                    if element_type == "legacy_single_pool_element":
                        location = normalize_resource(require_str(raw_element.get("location"), f"{pool_name}.location"))
                        key = ("template", location, projection)
                        if key not in element_ids:
                            element_ids[key] = len(elements)
                            elements.append(read_template(archive, location, projection))
                        element_id = element_ids[key]
                    elif element_type == "feature_pool_element":
                        feature = normalize_resource(require_str(raw_element.get("feature"), f"{pool_name}.feature"))
                        key = ("feature", feature, projection)
                        if key not in element_ids:
                            element_ids[key] = len(elements)
                            elements.append(feature_element(feature, projection))
                        element_id = element_ids[key]
                    else:
                        raise ExtractionError(f"{pool_name}: unsupported element type {element_type!r}")
                entries.append(PoolEntry(element_id, weight))
            fallback = normalize_resource(require_str(document.get("fallback"), f"{pool_name}.fallback"))
            pools.append(Pool(pool_name, fallback, tuple(entries)))

    pools.append(Pool("empty", "empty", ()))
    pool_names = {pool.name for pool in pools}
    if len(pool_names) != len(pools):
        raise ExtractionError("duplicate template pool name")
    for pool in pools:
        if pool.fallback not in pool_names:
            raise ExtractionError(f"pool {pool.name} references missing fallback {pool.fallback}")
    for element in elements:
        for connector in element.connectors:
            if connector.pool not in pool_names:
                raise ExtractionError(f"template {element.name} references missing pool {connector.pool}")
    element_names = [element.name for element in elements if element.kind != "empty"]
    if len(element_names) != len(set(element_names)):
        raise ExtractionError("different village elements resolve to the same public piece name")
    for _, start_pool in START_POOLS:
        if start_pool not in pool_names:
            raise ExtractionError(f"missing start pool {start_pool}")
    validate_generated_ranges(elements, pools)
    return elements, pools


def validate_generated_ranges(elements: list[Element], pools: list[Pool]) -> None:
    """Reject data that cannot fit the checked C representation or engine scratch arrays."""
    if len(elements) > UINT16_MAX + 1:
        raise ExtractionError(f"{len(elements)} elements exceed the uint16 ID space")
    if len(pools) > UINT16_MAX + 1:
        raise ExtractionError(f"{len(pools)} pools exceed the uint16 ID space")

    connector_offset = 0
    for element in elements:
        for axis, value in (
            ("size_x", element.size_x),
            ("size_y", element.size_y),
            ("size_z", element.size_z),
        ):
            require_range(value, 0, INT16_MAX, f"{element.name}.{axis}")
        if len(element.connectors) > ENGINE_MAX_CONNECTORS:
            raise ExtractionError(
                f"{element.name} has {len(element.connectors)} connectors; "
                f"engine maximum is {ENGINE_MAX_CONNECTORS}"
            )
        require_range(connector_offset, 0, UINT32_MAX, f"{element.name}.connector_offset")
        for index, connector in enumerate(element.connectors):
            for axis, value in (("x", connector.x), ("y", connector.y), ("z", connector.z)):
                require_range(
                    value,
                    INT16_MIN,
                    INT16_MAX,
                    f"{element.name}.connectors[{index}].{axis}",
                )
        connector_offset += len(element.connectors)

    entry_offset = 0
    for pool in pools:
        require_range(len(pool.entries), 0, UINT16_MAX, f"{pool.name}.entry_count")
        require_range(entry_offset, 0, UINT32_MAX, f"{pool.name}.entry_offset")
        expanded_size = sum(entry.weight for entry in pool.entries)
        if expanded_size > ENGINE_MAX_POOL_TEMPLATES:
            raise ExtractionError(
                f"{pool.name} expands to {expanded_size} templates; "
                f"engine maximum is {ENGINE_MAX_POOL_TEMPLATES}"
            )
        require_range(expanded_size, 0, UINT16_MAX, f"{pool.name}.total_weight")
        entry_offset += len(pool.entries)


def render(version: str, jar: Path, elements: list[Element], pools: list[Pool]) -> str:
    suffix = symbol_suffix(version)
    pool_ids = {pool.name: index for index, pool in enumerate(pools)}
    connector_rows: list[str] = []
    element_rows: list[str] = []
    connector_offset = 0
    for element in elements:
        for connector in element.connectors:
            connector_rows.append(
                "    {"
                f"{connector.x}, {connector.y}, {connector.z}, "
                f"{DIRECTIONS[connector.front]}, {DIRECTIONS[connector.top]}, "
                f"MCJIGSAW_JOINT_{connector.joint.upper()}, {pool_ids[connector.pool]}, "
                f"{connector.placement_priority}, {connector.selection_priority}, "
                f"{c_string(connector.name)}, {c_string(connector.target)}"
                "},"
            )
        kind = f"MCJIGSAW_ELEMENT_{element.kind.upper()}"
        projection = f"MCJIGSAW_PROJECTION_{element.projection.upper()}"
        element_rows.append(
            "    {"
            f"{c_string(element.name)}, {element.size_x}, {element.size_y}, {element.size_z}, "
            f"{connector_offset}, {len(element.connectors)}, {kind}, {projection}"
            "},"
        )
        connector_offset += len(element.connectors)

    entry_rows: list[str] = []
    pool_rows: list[str] = []
    entry_offset = 0
    for pool in pools:
        total_weight = sum(entry.weight for entry in pool.entries)
        max_height = max(
            (elements[entry.element].size_y for entry in pool.entries if elements[entry.element].kind != "empty"),
            default=0,
        )
        entry_rows.extend(f"    {{{entry.element}, {entry.weight}}}," for entry in pool.entries)
        pool_rows.append(
            "    {"
            f"{c_string(pool.name)}, {pool_ids[pool.fallback]}, {entry_offset}, {len(pool.entries)}, "
            f"{total_weight}, {max_height}"
            "},"
        )
        entry_offset += len(pool.entries)

    digest = hashlib.sha256(jar.read_bytes()).hexdigest()
    start_pool_ids = ", ".join(str(pool_ids[name]) for _, name in START_POOLS)
    lines = [
        "/* Generated by tools/extract_jigsaw.py; do not edit manually.",
        f" * Minecraft version: {version}",
        f" * Source JAR SHA-256: {digest}",
        " * Contains only pool weights, dimensions and Jigsaw connector metadata.",
        " */",
        "",
        f"static const McJigsawConnectorData MCJIGSAW_CONNECTORS_{suffix}[] = {{",
        *connector_rows,
        "};",
        "",
        f"static const McJigsawElementData MCJIGSAW_ELEMENTS_{suffix}[] = {{",
        *element_rows,
        "};",
        "",
        f"static const McJigsawPoolEntryData MCJIGSAW_POOL_ENTRIES_{suffix}[] = {{",
        *entry_rows,
        "};",
        "",
        f"static const McJigsawPoolData MCJIGSAW_POOLS_{suffix}[] = {{",
        *pool_rows,
        "};",
        "",
        f"const McJigsawData MCJIGSAW_VILLAGE_DATA_{suffix} = {{",
        f"    {c_string(version)},",
        f"    MCJIGSAW_CONNECTORS_{suffix},",
        f"    sizeof(MCJIGSAW_CONNECTORS_{suffix}) / sizeof(MCJIGSAW_CONNECTORS_{suffix}[0]),",
        f"    MCJIGSAW_ELEMENTS_{suffix},",
        f"    sizeof(MCJIGSAW_ELEMENTS_{suffix}) / sizeof(MCJIGSAW_ELEMENTS_{suffix}[0]),",
        f"    MCJIGSAW_POOL_ENTRIES_{suffix},",
        f"    sizeof(MCJIGSAW_POOL_ENTRIES_{suffix}) / sizeof(MCJIGSAW_POOL_ENTRIES_{suffix}[0]),",
        f"    MCJIGSAW_POOLS_{suffix},",
        f"    sizeof(MCJIGSAW_POOLS_{suffix}) / sizeof(MCJIGSAW_POOLS_{suffix}[0]),",
        f"    {{{start_pool_ids}}},",
        "};",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    arguments = parse_args()
    jar: Path = arguments.jar
    output: Path = arguments.output
    version: str = arguments.version
    if not jar.is_file():
        raise ExtractionError(f"client JAR does not exist: {jar}")
    elements, pools = load_data(jar)
    generated = render(version, jar, elements, pools)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(generated, encoding="utf-8")
    print(
        f"wrote {output}: {len(elements)} elements, "
        f"{sum(len(element.connectors) for element in elements)} connectors, {len(pools)} pools"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
