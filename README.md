# ElasticPipe
## Dataset
```raw_datasets = load_dataset('code_search_net', 'python')```

```datasets = raw_datasets['train'].filter(lambda x: 'apache/spark' in x['repository_name'])```

## Model(GPT)
```class MaskedAttention(nn.Module)```

```class MaskedMultiHeadAttention(nn.Module)```

```class FeedForward(nn.Module)```

```class Block(nn.Module)```

## Communication 
```
env_dict = {
        key: os.environ[key]
        for key in ("MASTER_ADDR", "MASTER_PORT", "WORLD_SIZE", "LOCAL_WORLD_SIZE")
    }
print(f"[{os.getpid()}] Initializing process group with: {env_dict}")
dist.init_process_group(backend="NCCL", timeout=datetime.timedelta(seconds=30)) # gloo
global_rank = int(os.environ["RANK"])
```
