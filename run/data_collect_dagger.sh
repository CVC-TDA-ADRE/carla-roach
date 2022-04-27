#!/bin/bash

# * Use `Roach` to label the on-policy data generated by the `L_K+L_V+L_F(c)` agent on the Leaderboard benchmark.
# data_collect_dagger () {
#   python -u data_collect.py resume=true log_video=false save_to_wandb=true \
#   agent.cilrs.wb_run_path=iccv21-roach/trained-models/21trg553 \
#   wb_group=dagger5 \
#   test_suites=lb_data \
#   dataset_root=/home/ubuntu/dagger \
#   actors.hero.coach=ppo \
#   agent.ppo.wb_run_path=iccv21-roach/trained-models/1929isj0 \
#   agent.ppo.wb_ckpt_step=null \
#   actors.hero.driver=cilrs \
#   n_episodes=80 inject_noise=false \
#   dagger_thresholds.acc=0.2 \
#   remove_final_steps=false \
#   actors.hero.terminal.kwargs.max_time=300 \
#   actors.hero.terminal.kwargs.no_collision=true \
#   actors.hero.terminal.kwargs.no_run_rl=false \
#   actors.hero.terminal.kwargs.no_run_stop=false \
#   carla_sh_path=${CARLA_ROOT}/CarlaUE4.sh
# }

# * Use `Autopilot` to label the on-policy data generated by the `L_A(AP)` agent on the NoCrash benchmark.
data_collect_dagger () {
  python -u data_collect.py resume=true log_video=false save_to_wandb=true \
  agent.cilrs.wb_run_path=iccv21-roach/trained-models/39o1h862 \
  wb_group=dagger5 \
  test_suites=eu_data \
  dataset_root=/home/ubuntu/dagger \
  actors.hero.coach=roaming \
  actors.hero.driver=cilrs \
  n_episodes=80 inject_noise=false \
  dagger_thresholds.acc=0.2 \
  remove_final_steps=false \
  actors.hero.terminal.kwargs.max_time=300 \
  actors.hero.terminal.kwargs.no_collision=true \
  actors.hero.terminal.kwargs.no_run_rl=false \
  actors.hero.terminal.kwargs.no_run_stop=false \
  carla_sh_path=${CARLA_ROOT}/CarlaUE4.sh
}



# NO NEED TO MODIFY THE FOLLOWING
# actiate conda env
source ~/miniconda3/etc/profile.d/conda.sh
conda activate roach

# remove checkpoint files
rm outputs/checkpoint.txt
rm outputs/wb_run_id.txt
rm outputs/ep_stat_buffer_*.json


# resume benchmark in case carla is crashed.
RED=$'\e[0;31m'
NC=$'\e[0m'
PYTHON_RETURN=1
until [ $PYTHON_RETURN == 0 ]; do
  data_collect_dagger
  PYTHON_RETURN=$?
  echo "${RED} PYTHON_RETURN=${PYTHON_RETURN}!!! Start Over!!!${NC}" >&2
  sleep 2
done

# To shut down the aws instance after the script is finished
# sleep 10
# sudo shutdown -h now