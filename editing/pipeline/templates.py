"""Issue templates and polarity metadata for tactile editing."""

ISSUE_POLARITY = {
    # True => label==1 means issue present. False => label==1 means good, so issue when label==0.
    "broken_lines": True,
    "too_thick": True,
    "fuzzy_lines": True,
    "missing_parts": True,
    "extra_parts": True,
    "wrong_location": True,
    "missing_texture": True,
    "inconsistent_within_elements": True,
    "far_near_wrong": True,
    "too_dense_cluttered": True,
    "too_similar_adjacent": True,
    "bleeds_boundaries": True,
    # Positive statements
    "background_clean": False,
    "posture_match": False,
    "species_match": False,
    "angle_match": False,
    "top_view": False,
    "view_frontal": False,
    "view_side": False,
    "view_perspective": False,
    "no_line_issues": False,
    "no_issues_good": False,
    "all_correct": False,
}

NON_ACTION_OPTIONS = {
    "all_correct",
    "no_line_issues",
    "no_issues_good",
    "good",
}

ISSUE_TEMPLATES = {
    "broken_lines": "Repair every broken or gapped outline so each contour becomes a single continuous stroke while maintaining the existing silhouette.",
    "too_thick": "Thin every oversized stroke so adjacent parts never merge, scrape away any solid fills created by the thickness, and redraw each outline as a clean contour with crisp negative space separating teeth, limbs, and body segments.",
    "fuzzy_lines": "Sharpen blurry or feathered lines into crisp high-contrast strokes without shifting their position.",
    "missing_parts": "Add the missing anatomy indicated by the natural photo using clean tactile lines, aligning with the current pose.",
    "extra_parts": "Remove extra or duplicated anatomy that does not exist in the reference photo while leaving valid parts untouched.",
    "wrong_location": "Reposition misplaced anatomy (limbs, facial features) to the anatomically correct location without altering scale.",
    "missing_texture": "Add the absent tactile textures with consistent fill patterns, keeping near and far elements distinguishable.",
    "inconsistent_within_elements": "Make texture fills uniform within each region so a single element never mixes conflicting patterns.",
    "far_near_wrong": "Adjust texture density to reinforce depth cues—closer elements get bolder fills, distant elements lighter ones.",
    "too_dense_cluttered": "Simplify dense texture fills by removing redundant hatching while preserving the underlying shape.",
    "too_similar_adjacent": "Differentiate adjacent textures so neighboring regions feel distinct under touch.",
    "bleeds_boundaries": "Trim any texture that spills over outlines so fills stop exactly at their boundaries.",
    "background_clean": "Erase stray marks or clutter from the background while preserving the subject contours.",
    "posture_match": "Align the limb and body angles with the natural photo so the pose matches without changing proportions.",
    "species_match": "Adjust feature proportions (ear shape, snout, tail) so the tactile drawing unmistakably matches the correct species.",
    "angle_match": "Rotate or redraw the subject so the viewing angle matches the natural reference (front, side, or 3/4 view).",
    "top_view": "Re-render the subject from a top-down orientation so overlapping parts layer correctly.",
    "view_frontal": "Ensure the subject faces forward—center the torso and symmetrize both sides to depict a frontal view.",
    "view_side": "Redraw strictly in profile, hiding far limbs so only the nearer side is visible.",
    "view_perspective": "Replace the perspective rendering with a clean orthographic view (front, side, or top) that blind readers can interpret; simplify foreshortening into clear overlapping layers.",
}

GENERIC_TEMPLATE = "Address issue '{option_description}' while preserving the existing silhouette and proportions."

TASK_PROMPT_FRAMES = {
    "QB": {
        "header": "Background / posture / species alignment cleanup. Keep anatomy faithful to the natural reference while removing stray clutter.",
        "footer": "Do not invent new scenery or new objects; only adjust alignment and background cleanliness.",
    },
    "QL": {
        "header": "Line quality remediation. Redraw outlines entirely if needed so strokes become even, clean, and easy to trace.",
        "footer": "You may restyle all lines provided the silhouette and pose stay recognizable; prioritize tactile legibility over matching the original ink exactly.",
    },
    "QP": {
        "header": "Part completeness and placement adjustments. Restore missing anatomy and correct misplaced components.",
        "footer": "Maintain the current silhouette; do not add brand new features beyond those implied in the photo.",
    },
    "QT": {
        "header": "Texture fidelity fixes. Balance tactile fills so each region remains legible under touch.",
        "footer": "Keep textures embossable—no photorealistic shading or new decorative fills.",
    },
    "QV": {
        "header": "Viewpoint verification. Align the drawing with the natural image's viewing angle.",
        "footer": "Do not recompose the scene; only rotate or adjust the subject to match the requested view.",
    },
}
