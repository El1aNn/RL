  1. 数据文件：Final_project/data/scenarios_rl_5k/{train,val,test}.jsonl — 需要在执行机器上准备
  2. 模型 checkpoint：./checkpoints/sft_base/ — 需要 SFT 产出
  3. Stage 2/3 脚本 adapter 路径 Bug：--override adapter_init.xxx=.../best 应改为 .../best/buyer 或 .../best/seller（如需我修复可以告知）