#!/bin/bash
# Run 10 diverse FLUX binary path experiments sequentially.
set -e
cd /home/jovyan/shares/SR006.nfs2/svgrozny/project/clear_project

echo "Starting 10 FLUX experiments at $(date)"
echo "================================================"

# 1. Sports car → Taxi
NAME="car_taxi" \
SOURCE_PROMPT="A high-resolution photo of a red sports car parked on a wet cobblestone city street at dusk, with warm streetlamp reflections on the pavement, old European brick buildings with glowing windows on both sides, a few pedestrians with umbrellas in the background, moody cinematic lighting, 4k, street photography style." \
TARGET_PROMPT="A high-resolution photo of a yellow taxi cab parked on a wet cobblestone city street at dusk, with warm streetlamp reflections on the pavement, old European brick buildings with glowing windows on both sides, a few pedestrians with umbrellas in the background, moody cinematic lighting, 4k, street photography style." \
SEG_PROMPT="car" \
./scripts/run_flux_pipeline.sh

# 2. Sunflower field → Lavender field
NAME="sunflower_lavender" \
SOURCE_PROMPT="A high-resolution landscape photo of a vast sunflower field stretching to the horizon, under a deep blue sky with scattered white cumulus clouds, a single dirt path winding through the flowers, rolling green hills in the far background, golden hour sunlight casting long shadows, 4k, nature photography style." \
TARGET_PROMPT="A high-resolution landscape photo of a vast lavender field stretching to the horizon, under a deep blue sky with scattered white cumulus clouds, a single dirt path winding through the flowers, rolling green hills in the far background, golden hour sunlight casting long shadows, 4k, nature photography style." \
SEG_PROMPT="sunflower" \
./scripts/run_flux_pipeline.sh

# 3. Wooden chair → Ornate throne
NAME="chair_throne" \
SOURCE_PROMPT="A high-resolution photo of a simple wooden chair placed in the center of a grand empty hall with polished marble floors, tall arched windows letting in soft diffused daylight, ornate ceiling moldings, dust particles floating in the light beams, minimalist composition, 4k, fine art photography style." \
TARGET_PROMPT="A high-resolution photo of an ornate golden throne with red velvet cushions placed in the center of a grand empty hall with polished marble floors, tall arched windows letting in soft diffused daylight, ornate ceiling moldings, dust particles floating in the light beams, minimalist composition, 4k, fine art photography style." \
SEG_PROMPT="chair" \
./scripts/run_flux_pipeline.sh

# 4. Penguin → Flamingo
NAME="penguin_flamingo" \
SOURCE_PROMPT="A high-resolution wildlife photo of an emperor penguin standing on a flat icy shore, with a calm dark blue ocean behind it, distant icebergs and snow-covered mountains on the horizon, overcast sky with soft gray clouds, crisp polar light, 4k, National Geographic photography style." \
TARGET_PROMPT="A high-resolution wildlife photo of a pink flamingo standing on a flat icy shore, with a calm dark blue ocean behind it, distant icebergs and snow-covered mountains on the horizon, overcast sky with soft gray clouds, crisp polar light, 4k, National Geographic photography style." \
SEG_PROMPT="penguin" \
./scripts/run_flux_pipeline.sh

# 5. Birthday cake → Stack of books
NAME="cake_books" \
SOURCE_PROMPT="A high-resolution photo of a white birthday cake with colorful candles on a rustic wooden kitchen table, a checkered tablecloth underneath, warm morning sunlight streaming through a nearby window, blurred kitchen shelves with jars and plants in the background, cozy domestic atmosphere, 4k, food photography style." \
TARGET_PROMPT="A high-resolution photo of a tall stack of old leather-bound books on a rustic wooden kitchen table, a checkered tablecloth underneath, warm morning sunlight streaming through a nearby window, blurred kitchen shelves with jars and plants in the background, cozy domestic atmosphere, 4k, still life photography style." \
SEG_PROMPT="cake" \
./scripts/run_flux_pipeline.sh

# 6. Lighthouse → Castle
NAME="lighthouse_castle" \
SOURCE_PROMPT="A high-resolution photo of a tall white-and-red lighthouse standing on a dramatic rocky cliff edge, crashing ocean waves below, a winding gravel path leading up to it, wild grass and wildflowers on the hillside, a vivid orange and purple sunset sky with wispy clouds, 4k, landscape photography style." \
TARGET_PROMPT="A high-resolution photo of a medieval stone castle with towers and battlements standing on a dramatic rocky cliff edge, crashing ocean waves below, a winding gravel path leading up to it, wild grass and wildflowers on the hillside, a vivid orange and purple sunset sky with wispy clouds, 4k, landscape photography style." \
SEG_PROMPT="lighthouse" \
./scripts/run_flux_pipeline.sh

# 7. Violin → Electric guitar
NAME="violin_guitar" \
SOURCE_PROMPT="A high-resolution photo of a polished wooden violin resting on a dark velvet cloth draped over an antique wooden table, soft warm spotlight from above, a blurred concert hall with red curtains and empty seats in the background, dramatic chiaroscuro lighting, 4k, product photography style." \
TARGET_PROMPT="A high-resolution photo of a sleek black electric guitar resting on a dark velvet cloth draped over an antique wooden table, soft warm spotlight from above, a blurred concert hall with red curtains and empty seats in the background, dramatic chiaroscuro lighting, 4k, product photography style." \
SEG_PROMPT="violin" \
./scripts/run_flux_pipeline.sh

# 8. Snowy mountains → Volcanic landscape
NAME="snow_volcano" \
SOURCE_PROMPT="A high-resolution landscape photo of a majestic snow-covered mountain peak towering above a dense pine forest, a frozen lake in the foreground reflecting the mountains, fresh powder snow on the ground, clear winter sky with a few high cirrus clouds, crisp cold atmosphere, 4k, landscape photography style." \
TARGET_PROMPT="A high-resolution landscape photo of an active volcanic mountain with glowing lava streams towering above a dense pine forest, a frozen lake in the foreground reflecting the mountains, fresh powder snow on the ground, clear winter sky with a few high cirrus clouds, crisp cold atmosphere, 4k, landscape photography style." \
SEG_PROMPT="mountain peak" \
./scripts/run_flux_pipeline.sh

# 9. Butterfly → Hummingbird
NAME="butterfly_hummingbird" \
SOURCE_PROMPT="A high-resolution macro photo of a monarch butterfly with orange and black wings perched on a vibrant pink hibiscus flower, lush green garden foliage blurred in the background, tiny water droplets on the petals, soft natural daylight, shallow depth of field, 4k, macro photography style." \
TARGET_PROMPT="A high-resolution macro photo of a tiny iridescent green hummingbird hovering near a vibrant pink hibiscus flower, lush green garden foliage blurred in the background, tiny water droplets on the petals, soft natural daylight, shallow depth of field, 4k, macro photography style." \
SEG_PROMPT="butterfly" \
./scripts/run_flux_pipeline.sh

# 10. Sailboat → Pirate ship
NAME="sail_pirate" \
SOURCE_PROMPT="A high-resolution photo of a small white sailboat with a single mast gliding on a calm turquoise tropical ocean, a distant sandy beach with palm trees on the horizon, a few white seagulls in the bright blue sky, sparkling sunlight reflections on the water surface, 4k, nautical photography style." \
TARGET_PROMPT="A high-resolution photo of a large old wooden pirate ship with tattered black sails and a skull flag gliding on a calm turquoise tropical ocean, a distant sandy beach with palm trees on the horizon, a few white seagulls in the bright blue sky, sparkling sunlight reflections on the water surface, 4k, nautical photography style." \
SEG_PROMPT="sailboat" \
./scripts/run_flux_pipeline.sh

echo ""
echo "================================================"
echo "All 10 experiments DONE at $(date)"
