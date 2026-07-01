build_image:
	clear
	srun -p H100,H200,A100-80GB,A100-IML --ntasks 1 --gpus-per-task 1 --immediate=3600 --time=04:00:00 --cpus-per-gpu=64 --mem-per-cpu 4G --container-mounts=/netscratch/duynguyen:/netscratch/duynguyen --container-image=/netscratch/duynguyen/Research/Nghiem_LLaVA-Med/VLA-Humanoid/vla_p0.sqsh --container-save=/netscratch/duynguyen/Research/Nghiem_LLaVA-Med/VLA-Humanoid/vla_p0.sqsh --pty /bin/bash	
.PHONY: build_image


build_image_eval:
	clear
	srun -p H100,H200,A100-80GB,A100-IML --ntasks 1 --gpus-per-task 1 --immediate=3600 --time=04:00:00 --cpus-per-gpu=64 --mem-per-cpu 4G --container-mounts=/netscratch/duynguyen:/netscratch/duynguyen --container-image=/netscratch/duynguyen/Research/Nghiem_LLaVA-Med/VLA-Humanoid/vla_p0_eval.sqsh --container-save=/netscratch/duynguyen/Research/Nghiem_LLaVA-Med/VLA-Humanoid/vla_p0_eval.sqsh --pty /bin/bash	
.PHONY: build_image_eval