# ElasticPipe
## Dataset
```raw_datasets = load_dataset('code_search_net', 'python')```

```datasets = raw_datasets['train'].filter(lambda x: 'apache/spark' in x['repository_name'])```

## Model(GPT)
``` class MaskedAttention(nn.Module)```

```class MaskedMultiHeadAttention(nn.Module)```

```class FeedForward(nn.Module)```

```class Block(nn.Module)```

