import subprocess
import time

class GPUManager:
    def __init__(self, min_memory=3000, max_gpus=1, gpu_check_interval=5):
        """
        Initialize the GPU manager
        :param min_memory: The minimum amount of remaining memory required per GPU (in MB), default 3000MB
        :param max_gpus: The maximum number of GPUs allowed to be preempted, default 1
        :param gpu_check_interval: The time interval for checking GPU usage (in seconds), default 5 seconds
        """
        self.min_memory = min_memory  # Memory threshold
        self.max_gpus = max_gpus  # Maximum number of GPUs
        self.gpu_check_interval = gpu_check_interval  # Check interval

    def get_gpu_memory(self):
        """
        Get the remaining memory of GPUs and return a list of GPUs with memory greater than the threshold
        """
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,memory.free", "--format=csv,noheader,nounits"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        gpu_info = result.stdout.decode('utf-8').strip().split('\n')
        gpus = []
        for info in gpu_info:
            index, memory_free = map(int, info.split(','))
            if memory_free > self.min_memory:  # Check if memory is greater than the threshold
                gpus.append(index)
        return gpus

    def select_idle_gpus(self):
        """
        Dynamically select idle GPUs, selecting up to max_gpus GPUs that meet the condition
        """
        while True:
            idle_gpus = self.get_gpu_memory()
            if len(idle_gpus) >= self.max_gpus:
                return idle_gpus[:self.max_gpus]
            time.sleep(self.gpu_check_interval)  # Check every gpu_check_interval seconds

    def run_task_on_gpus(self, commands):
        """
        Run tasks on the most idle GPUs
        :param commands: List of commands to execute, each command must include the --gpu parameter
        """
        selected_gpus = self.select_idle_gpus()
        for i, cmd in enumerate(commands):
            selected_gpu = selected_gpus[i % len(selected_gpus)]  # Assign GPU
            cmd = cmd.replace('--gpu 1', f'--gpu {selected_gpu}')  # Replace GPU parameter
            subprocess.run(cmd, shell=True)
    def load_commands_from_sh(self, shell_path):
        """
        load command from shell file
        :param shell_path: .sh file path
        :return: command list
        """
        with open(shell_path, 'r') as f:
            commands = []
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'): 
                    commands.append(line)
        return commands

def main():
    # Use argparse to handle command line arguments
    parser = argparse.ArgumentParser(description="Automatically preempt GPUs and run script tasks")
    parser.add_argument('--min_memory', type=int, default=3000, help="Minimum GPU memory per card (MB)")
    parser.add_argument('--max_gpus', type=int, default=1, help="Maximum number of GPUs to use")
    parser.add_argument('--gpu_check_interval', type=int, default=5, help="Interval for checking GPU status (seconds)")
    parser.add_argument('--shell_path', type=str, required=True, help="Path to the shell script file")
    args = parser.parse_args()

    # Initialize GPU manager
    gpu_manager = GPUManager(min_memory=args.min_memory, max_gpus=args.max_gpus, gpu_check_interval=args.gpu_check_interval)

    # Load commands from shell script
    commands = gpu_manager.load_commands_from_sh(args.shell_path)

    # Run tasks
    gpu_manager.run_task_on_gpus(commands)

if __name__ == '__main__':
    main()
