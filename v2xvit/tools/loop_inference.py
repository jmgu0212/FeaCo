import os   

for index in range(15,40,2):
  cmd = f"CUDA_VISIBLE_DEVICES=1 python /home/gaojing/zjy/v2x-vit/v2xvit/tools/inference.py --eval_epoch {index} "
  print(f"Running command: {cmd}")
  os.system(cmd)