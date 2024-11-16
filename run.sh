torchrun --nproc_per_node=2 \
 --nnodes=1 \
 --node_rank=0 \
 --master_add="192.168.0.246" \
 --master_port=2655 \
 runtime.py