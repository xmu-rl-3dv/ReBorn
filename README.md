# ReBorn

This code is build based on PyMARL2 and PyMARL. We assume that you have experience with PyMARL. The requirements are the same as PyMARL2.

## Run an experiment 

```shell
python src/main.py --config=qmix --env-config=sc2 with env_args.map_name=2s3z
python src/main.py --config=qmix_b --env-config=sc2 with env_args.map_name=2s3z #ReBorn
python src/main.py --config=qmix_r --env-config=sc2 with env_args.map_name=2s3z #Redo
```

The config files act as defaults for an algorithm or environment. 

They are all located in `src/config`.
`--config` refers to the config files in `src/config/algs`
`--env-config` refers to the config files in `src/config/envs`

All results will be stored in the `Results` folder.

The previous config files used for the SMAC Beta have the suffix `_beta`.

## License

Code licensed under the Apache License v2.0
