
python -m scripts.unwrap experiment=mossland-codec  \
 experiment_ckpt_path=/inspire/qb-ilm2/project/embodied-multimodality/public/zhaoguojie/Mossland/logs/mossland-codec/runs/2026-06-09_16-52-30/checkpoints/last.ckpt \
 output_path=./ckpt/mossland-codec0610 \
 deepspeed=false \
 export_ema=true
# resume_from_ckpt=/home/gjzhao/workspace2024/heartstrings-version4/logs/vae_113/runs/2024-09-24_13-16-53/checkpoints/last.ckpt

# python -m scripts.unwrap experiment=compose_music_net_ultra \
#  experiment_ckpt_path=/inspire/hdd/global_user/p-shangli/zgj/loopland2/logs/compose_music_net_ultra_current_training/checkpoints/0-108000.ckpt/checkpoint/mp_rank_00_model_states.pt \
#  output_path=./ckpt/compose_music_net_0826 \
#  deepspeed=true \
#  export_ema=true

#  python -m scripts.unwrap experiment=plan_music_net  \
#  experiment_ckpt_path=/inspire/hdd/global_user/p-shangli/zgj/loopland2/logs/plan_music_net_current_training/checkpoints/last.ckpt \
#  output_path=./ckpt/plan_music_net0825 \
#  deepspeed=false \
#  export_ema=true

 