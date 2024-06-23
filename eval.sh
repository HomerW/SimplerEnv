python simpler_env/simple_inference_visual_matching_prepackaged_envs.py \
    --policy octo-base \
    --task google_robot_pick_coke_can \
    --ckpt-path gs://multi-robot-bucket2/runs/octo/fractal_baseline_20240602_011114 \
    --logging-root ./results_simple_eval/ \
    --n-trajs 10 \
    --step 300000

