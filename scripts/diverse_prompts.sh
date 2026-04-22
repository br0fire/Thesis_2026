#!/bin/bash
# 16 diverse (non-bgrich) prompt triplets to test prior generalization.
# 4 categories × 4 each: object swap, landscape, portrait, style/scene.
# Sourced by other scripts.

declare -gA DIV_SRC DIV_TGT DIV_SEG

# ─── Object swap (4) ───
DIV_SRC[cat_dog]="A high-resolution photo of a tabby cat sitting on a stone pavement in a city park, soft afternoon light, shallow depth of field, 4k."
DIV_TGT[cat_dog]="A high-resolution photo of a golden retriever dog sitting on a stone pavement in a city park, soft afternoon light, shallow depth of field, 4k."
DIV_SEG[cat_dog]="cat"

DIV_SRC[apple_orange]="A high-resolution photo of a single red apple on a wooden table, natural window light, minimalist composition, 4k."
DIV_TGT[apple_orange]="A high-resolution photo of a single orange on a wooden table, natural window light, minimalist composition, 4k."
DIV_SEG[apple_orange]="apple"

DIV_SRC[rose_tulip]="A high-resolution photo of a single red rose in a clear glass vase on a white shelf, soft diffused light, 4k, still life photography."
DIV_TGT[rose_tulip]="A high-resolution photo of a single pink tulip in a clear glass vase on a white shelf, soft diffused light, 4k, still life photography."
DIV_SEG[rose_tulip]="rose"

DIV_SRC[mug_teacup]="A high-resolution photo of a white ceramic coffee mug steaming on a wooden desk, morning light, soft focus background, 4k."
DIV_TGT[mug_teacup]="A high-resolution photo of a delicate porcelain teacup on a saucer steaming on a wooden desk, morning light, soft focus background, 4k."
DIV_SEG[mug_teacup]="mug"

# ─── Landscape (4) — no clear fg object ───
DIV_SRC[beach_mountain]="A high-resolution landscape photo of a sunny tropical beach with turquoise water and palm trees, bright blue sky, 4k, travel photography."
DIV_TGT[beach_mountain]="A high-resolution landscape photo of a rocky snow-capped mountain range with a glacial lake in the foreground, bright blue sky, 4k, travel photography."
DIV_SEG[beach_mountain]="sandy beach"

DIV_SRC[summer_winter_field]="A high-resolution photo of a green summer wheat field under a bright sunny sky, wildflowers in the foreground, warm golden light, 4k."
DIV_TGT[summer_winter_field]="A high-resolution photo of a snow-covered winter field under an overcast grey sky, bare stubble sticking out of snow, cold blue light, 4k."
DIV_SEG[summer_winter_field]="green field"

DIV_SRC[calm_stormy_sea]="A high-resolution photo of a calm turquoise ocean at sunset with gentle reflections on flat water, clear orange sky, 4k, seascape photography."
DIV_TGT[calm_stormy_sea]="A high-resolution photo of a stormy grey ocean with tall crashing waves and whitecaps, dark overcast sky with lightning, 4k, seascape photography."
DIV_SEG[calm_stormy_sea]="calm water"

DIV_SRC[forest_desert]="A high-resolution photo of a lush green pine forest path with tall trees, soft morning mist, dappled sunlight on moss-covered ground, 4k."
DIV_TGT[forest_desert]="A high-resolution photo of a vast sandy desert with tall rolling dunes, harsh midday sun, no vegetation, heat shimmer, 4k."
DIV_SEG[forest_desert]="forest"

# ─── Portrait (4) — face-centric ───
DIV_SRC[young_old_woman]="A high-resolution portrait photo of a young woman in her twenties with smooth skin and dark hair, neutral expression, studio lighting, plain grey background, 4k."
DIV_TGT[young_old_woman]="A high-resolution portrait photo of an elderly woman in her seventies with wrinkled skin and silver hair, neutral expression, studio lighting, plain grey background, 4k."
DIV_SEG[young_old_woman]="young woman"

DIV_SRC[shaved_bearded]="A high-resolution portrait photo of a clean-shaven man with short brown hair looking directly at the camera, studio lighting, plain white background, 4k."
DIV_TGT[shaved_bearded]="A high-resolution portrait photo of a man with a thick full beard and short brown hair looking directly at the camera, studio lighting, plain white background, 4k."
DIV_SEG[shaved_bearded]="man face"

DIV_SRC[smile_frown]="A high-resolution portrait photo of a child with a big happy smile showing teeth, bright natural light, blurred outdoor background, 4k."
DIV_TGT[smile_frown]="A high-resolution portrait photo of a child with a serious frowning expression, bright natural light, blurred outdoor background, 4k."
DIV_SEG[smile_frown]="child face"

DIV_SRC[blonde_brunette]="A high-resolution portrait photo of a woman with long blonde hair in soft waves, looking slightly to the side, natural outdoor light, 4k."
DIV_TGT[blonde_brunette]="A high-resolution portrait photo of a woman with long dark brown hair in soft waves, looking slightly to the side, natural outdoor light, 4k."
DIV_SEG[blonde_brunette]="blonde hair"

# ─── Style / Scene (4) — compositional changes ───
DIV_SRC[photo_painting]="A high-resolution realistic photograph of a red vintage convertible car parked in front of a Parisian cafe on a sunny day, 4k."
DIV_TGT[photo_painting]="An oil painting in the style of impressionism of a red vintage convertible car parked in front of a Parisian cafe on a sunny day, visible brushstrokes."
DIV_SEG[photo_painting]="red car"

DIV_SRC[modern_medieval]="A high-resolution photo of a sleek modern glass skyscraper with reflective windows against a clear blue sky, 4k, architecture photography."
DIV_TGT[modern_medieval]="A high-resolution photo of a medieval stone castle with turrets and a drawbridge against a clear blue sky, 4k, architecture photography."
DIV_SEG[modern_medieval]="glass building"

DIV_SRC[empty_cluttered]="A high-resolution photo of a minimalist empty room with white walls, polished wooden floor, one chair in the corner, soft natural light, 4k, interior photography."
DIV_TGT[empty_cluttered]="A high-resolution photo of a cluttered bohemian room with colorful rugs, scattered plants, stacks of books, tapestries, many cushions on the floor, soft natural light, 4k, interior photography."
DIV_SEG[empty_cluttered]="minimalist room"

DIV_SRC[day_night_city]="A high-resolution photo of a busy city street during bright daytime with shops open, pedestrians and cars, clear blue sky, 4k, street photography."
DIV_TGT[day_night_city]="A high-resolution photo of a busy city street at night with glowing neon signs, illuminated shop windows, car headlights, dark sky, 4k, street photography."
DIV_SEG[day_night_city]="city street"

DIV_EXPS=(cat_dog apple_orange rose_tulip mug_teacup \
          beach_mountain summer_winter_field calm_stormy_sea forest_desert \
          young_old_woman shaved_bearded smile_frown blonde_brunette \
          photo_painting modern_medieval empty_cluttered day_night_city)
