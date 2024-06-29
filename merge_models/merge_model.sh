#!/bin/bash

# Arrays of model names and corresponding output file names
models=("slate_pencil_mix.safetensors" "fashion_cloth.safetensors" "background_beautiful_outdoor.safetensors" "adm_interior.safetensors" "cafe_background.safetensors" "exterior_design.safetensors" "fashion_sketch.safetensors" "game_wallpaper.safetensors" "interior_design.safetensors" "sketch_achitectural_design.safetensors" "sketch_car.safetensors")
outputs=("slate_pencil_mix.safetensors" "fashion_cloth.safetensors" "background_beautiful_outdoor.safetensors" "adm_interior.safetensors" "cafe_background.safetensors" "exterior_design.safetensors" "fashion_sketch.safetensors" "game_wallpaper.safetensors" "interior_design.safetensors" "sketch_achitectural_design.safetensors" "sketch_car.safetensors")

# Loop through the arrays
for i in "${!models[@]}"; do
  python -m sd_scripts.networks.merge_lora \
    --sd_model civitai_models/v1-5-pruned-emaonly.safetensors \
    --save_to models/${outputs[$i]} \
    --model civitai_models/${models[$i]} \
    --ratios 1 || true
done



# python -m sd_scripts.networks.merge_lora \
#     --sd_model civitai_models/v1-5-pruned-emaonly.safetensors \
#     --save_to models/slate_pencil_mix.safetensors \
#     --model civitai_models/slate_pencil_mix.safetensors \
#     --ratios 1

# python -m sd_scripts.networks.merge_lora \
#     --sd_model civitai_models/v1-5-pruned-emaonly.safetensors \
#     --save_to models/fashion_cloth.safetensors \
#     --model civitai_models/fashion_cloth.safetensors \
#     --ratios 1