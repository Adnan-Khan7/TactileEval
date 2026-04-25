#!/usr/bin/env python3
"""Task and option metadata for the consolidated AMT dataset."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple


@dataclass(frozen=True)
class OptionSpec:
    option_id: str
    description: str


@dataclass(frozen=True)
class TaskSpec:
    task_id: str
    family: str
    options: Tuple[OptionSpec, ...]

    @property
    def option_ids(self) -> Tuple[str, ...]:
        return tuple(opt.option_id for opt in self.options)


FAMILY_NAMES: Dict[str, str] = {
    "F1": "Animals & Creatures",
    "F2": "Food, Nature & Simple Objects",
    "F3": "Furniture & Structures",
    "F4": "Wearables & Accessories",
    "F5": "Vehicles & Flight Systems",
    "F6": "Tools, Instruments & Appliances",
}


def opts(*items: Tuple[str, str]) -> Tuple[OptionSpec, ...]:
    return tuple(OptionSpec(option_id=i, description=d) for i, d in items)


LINE_OPTIONS = opts(
    ("broken_lines", "Lines have gaps or discontinuities."),
    ("too_thick", "Lines are too thick/bold."),
    ("fuzzy_lines", "Lines look fuzzy or blurry."),
    ("no_line_issues", "Lines are clean / no issues."),
)

CONTENT_OPTIONS = opts(
    ("missing_parts", "Some parts from the photo are missing."),
    ("extra_parts", "Extra or hallucinated parts exist."),
    ("wrong_location", "Parts exist but are misplaced."),
    ("all_correct", "All parts present and correctly located."),
)

TEXTURE_OPTIONS = opts(
    ("missing_texture", "Texture missing where expected."),
    ("inconsistent_within_elements", "Texture inconsistent within an element."),
    ("far_near_wrong", "Near vs. far textures not differentiated."),
    ("too_dense_cluttered", "Textures overly dense or cluttered."),
    ("too_similar_adjacent", "Textures too similar to adjacent regions."),
    ("bleeds_boundaries", "Textures bleed across boundaries."),
    ("no_issues_good", "Textures look good / no issues."),
)

VIEW_OPTIONS = opts(
    ("orientation_match", "Orientation/view matches the photo."),
    ("view_match", "Overall viewpoint alignment."),
    ("view_frontal", "Front view."),
    ("view_side", "Side view."),
    ("view_top", "Top view."),
    ("view_perspective", "Perspective / three-quarter view."),
    ("view_undefined", "View is ambiguous/undefined."),
)

MATCHING_OPTIONS = opts(
    ("posture_match", "Pose/orientation matches the photo."),
    ("species_match", "Object/species identity matches."),
    ("background_clean", "Background is clean/solid without clutter."),
)

CONFIG_OPTIONS = opts(
    ("background_clean", "Background is clean/solid without clutter."),
    ("configuration_match", "Configuration/layout matches the photo."),
    ("object_match", "Object identity matches the reference."),
)

TRIAGE_OPTIONS = opts(
    ("bigfix", "Major issue present (needs large fix)."),
    ("smallfix", "Minor issue present."),
    ("good", "Looks good / no fix needed."),
)


TASK_SPECS: Dict[str, TaskSpec] = {}


def add_task(task_id: str, family: str, options: Tuple[OptionSpec, ...]) -> None:
    TASK_SPECS[task_id] = TaskSpec(task_id, family, options)


# F1
add_task("F1QB", "F1", MATCHING_OPTIONS)
add_task("F1QL", "F1", LINE_OPTIONS)
add_task("F1QP", "F1", CONTENT_OPTIONS)
add_task("F1QT", "F1", TEXTURE_OPTIONS)
add_task("F1QV", "F1", opts(
    ("angle_match", "Viewing angle matches the natural image."),
    ("view_frontal", "Frontal view."),
    ("view_side", "Side view."),
    ("view_top", "Top view."),
    ("view_perspective", "Perspective / three-quarter view."),
))

# F2
add_task("F2QB", "F2", MATCHING_OPTIONS)
add_task("F2QL", "F2", LINE_OPTIONS)
add_task("F2QP", "F2", CONTENT_OPTIONS)
add_task("F2QT", "F2", TEXTURE_OPTIONS)
add_task("F2QV", "F2", opts(
    ("view_match", "Overall viewpoint alignment."),
    ("view_frontal", "Front view."),
    ("view_side", "Side view."),
    ("view_top", "Top view."),
    ("view_perspective", "Perspective / three-quarter view."),
    ("view_undefined", "View is ambiguous/undefined."),
))

# F3
add_task("F3QB", "F3", opts(
    ("posture_match", "Pose/orientation matches the photo."),
    ("object_match", "Object identity matches the reference."),
    ("background_clean", "Background is clean/solid without clutter."),
))
add_task("F3QL", "F3", LINE_OPTIONS)
add_task("F3QP", "F3", CONTENT_OPTIONS)
add_task("F3QT", "F3", TEXTURE_OPTIONS)
add_task("F3QV", "F3", opts(
    ("orientation_match", "Orientation/view matches the photo."),
    ("view_frontal", "Front view."),
    ("view_side", "Side view."),
    ("view_top", "Top view."),
    ("view_perspective", "Perspective / three-quarter view."),
))

# F4
add_task("F4QB", "F4", CONFIG_OPTIONS)
add_task("F4QL", "F4", LINE_OPTIONS)
add_task("F4QP", "F4", CONTENT_OPTIONS)
add_task("F4QT", "F4", TEXTURE_OPTIONS)
add_task("F4QV", "F4", TASK_SPECS["F3QV"].options)

# F5
add_task("F5QB", "F5", CONFIG_OPTIONS)
add_task("F5QL", "F5", LINE_OPTIONS)
add_task("F5QP", "F5", CONTENT_OPTIONS)
add_task("F5QT", "F5", TEXTURE_OPTIONS)
add_task("F5QV", "F5", TASK_SPECS["F3QV"].options)

# F6
add_task("F6QB", "F6", CONFIG_OPTIONS)
add_task("F6QL", "F6", LINE_OPTIONS)
add_task("F6QP", "F6", CONTENT_OPTIONS)
add_task("F6QT", "F6", TEXTURE_OPTIONS)
add_task("F6QV", "F6", TASK_SPECS["F3QV"].options)


def infer_task_spec(task_id: str) -> TaskSpec:
    try:
        return TASK_SPECS[task_id]
    except KeyError as exc:
        raise KeyError(f"Unknown task '{task_id}'. Update TASK_SPECS.") from exc


__all__ = [
    "OptionSpec",
    "TaskSpec",
    "TASK_SPECS",
    "FAMILY_NAMES",
    "infer_task_spec",
]
