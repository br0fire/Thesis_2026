#!/bin/bash
# 16 bg-rich prompt triplets for clean_v1 run (canonical source, α=0.5, PNG-raw).
# Sourced by other scripts — not run directly.

declare -gA SRC TGT SEG

# 15 from new_bgrich (reuse exact prompts — same scenes)
source "$(dirname "${BASH_SOURCE[0]}")/new_bgrich_prompts.sh"
for k in "${!NEW_SRC[@]}"; do SRC[$k]="${NEW_SRC[$k]}"; done
for k in "${!NEW_TGT[@]}"; do TGT[$k]="${NEW_TGT[$k]}"; done
for k in "${!NEW_SEG[@]}"; do SEG[$k]="${NEW_SEG[$k]}"; done

# 16th: jewelry_music_box (new)
SRC[bgrich_jewelry_music_box]="A high-resolution photo of an open carved wooden jewelry box overflowing with a tangle of pearl necklaces on a Victorian vanity table, surrounded by cut-glass perfume bottles with silver atomizers, a silver hand mirror, bone-handled hairbrush, scattered lipstick tubes, a powder puff in a china dish, silk scarves draped over the edge, a vase with fresh peonies, brass candlesticks, embroidered doilies, and a gilt-framed oval mirror behind, soft rose-gold light through sheer curtains, 4k, boudoir photography."
TGT[bgrich_jewelry_music_box]="A high-resolution photo of an ornate golden music box with a spinning ballerina on top on a Victorian vanity table, surrounded by cut-glass perfume bottles with silver atomizers, a silver hand mirror, bone-handled hairbrush, scattered lipstick tubes, a powder puff in a china dish, silk scarves draped over the edge, a vase with fresh peonies, brass candlesticks, embroidered doilies, and a gilt-framed oval mirror behind, soft rose-gold light through sheer curtains, 4k, boudoir photography."
SEG[bgrich_jewelry_music_box]="jewelry box"

# Canonical list of the 16 experiment names (ordering matters for GPU assignment)
EXPS=(bgrich_globe_orrery bgrich_telescope_microscope bgrich_camera_binoculars \
      bgrich_pocketwatch_compass bgrich_teapot_samovar bgrich_chess_checkers \
      bgrich_violin_cello bgrich_guitar_banjo bgrich_lantern_candelabra \
      bgrich_wine_whiskey bgrich_revolver_flintlock bgrich_sewing_typewriter \
      bgrich_piano_harp bgrich_crown_tiara bgrich_skull_hourglass \
      bgrich_jewelry_music_box)
