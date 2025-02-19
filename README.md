# GPUManager
Automatically preempt idle GPUs and automatically run scripts. 自动抢显卡和自动不间断挂脚本

# Warning
This tool can be used when there are a large number of scripts that need to be run in batches and it's not possible to immediately start a new command after finishing one.  
**Caution!** Please use this tool with caution. Do not occupy all the GPUs, otherwise your peers may become dissatisfied with you.  
在脚本数量极大且需要批量跑，没法在跑完一个命令马上重新挂上新命令时，可以使用此工具。  
**注意**请慎用这个工具，不要占用所有的显卡，否则你的同学可能会对你感到不满

**How to use** :
The original parameters must include the --gpu option.  
必须保证原有的参数里有 --gpu 这个选项  
Method 1:  
```python
from gpu_manager import GPUManager

# 创建 GPU 管理器对象，设定最大使用 2 张 GPU，每张 GPU 至少剩余 3GB 显存
gpu_manager = GPUManager(min_memory=3000, max_gpus=2)

# 定义你的任务命令列表
commands = [
    "nohup python -m myfile --gpu 1 --data_path /my/datasets --dataset cifar10 --arch resnet18 --epochs 20 --learning_rate 0.1 --batch_size 128 --pretrained > my_log.log 2>&1 &",
]

# 调用库中的方法运行任务
gpu_manager.run_task_on_gpus(commands)
```
Method 2:
run the file directly and path the parameter (keep it in the same directory)  
也可以直接运行此文件，传入参数(保证同目录下)  
```python
python gpu_manager.py --min_memory 3000 --max_gpus 2 --gpu_check_interval 30 --shell_path my/shell/path.sh
```
后续再更新多机抢卡和分布式... 

