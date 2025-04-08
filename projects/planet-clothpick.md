This is a description for traning and evaluating `PlaNet-ClohPick` models in `mono-square-fabric` environment of `SoftGym`.

Make sure you setup the [`agent-arena-v0`](https://github.com/halid1020/agent-arena-v0) and [`softgym`](https://github.com/halid1020/softgym) as instructed.

# Download Training Data

1. Create a `data` directory under `agent-arena-v0`
```
cd <path-to-agent-arena>/agent-arena-v0

# if the `data` directory was not created, create one as follows
mkdir data

```

2. Download and set up training data.

```
cd <path-to-agent-arena>/agent-arena-v0/data

gdown https://drive.google.com/uc?id=1gBDrYKjtD0Qga9sjcJpybYIXKbtWxiIW

unzip MonoSquareFabric-VisionPickAndPlace-TrajectorySim.zip

```

# Train and Evaluate PlaNet-ClothPick from Scratch

After downloading the data, you can start traning `planet-clothpick` agent. You can find the implementation of the agent [here](https://github.com/halid1020/agent-arena-v0/tree/main/agent_arena/agent/drl/planet)

Under the `softgym`'s root directory, please run
```
. ./setup.sh
```

Then, using the same terminal, under the `agent-arena-v0`'s root directory, please run

```
. ./setup.sh

cd tool

python train_and_evaluate.py \
    --agent planet-clothpick \
    --arena "softgym|domain:mono-square-fabric,initial:crumpled,action:pixel-pick-and-place(1),task:flattening" \
    --config <config-file-name> \
    --log_dir <save-path> \
```

The above program will also evaluate the agent in the test trials of the arena.

Provided config files are `D2M` and `RGB2RGB`, and you can find them [here](https://github.com/halid1020/agent-arena-v0/tree/main/agent_arena/configuration/train_and_evaluate/planet-clothpick/softgym%7Cdomain%3Amono-square-fabric%2Cinitial%3Acrumpled%2Caction%3Apixel-pick-and-place(1)%2Ctask%3Aflattening)


# Download Provided Weights

Download the provided weights of the model to a save directory.

```
cd <save-dir>

gdown https://drive.google.com/uc?id=1USVozXZzkVlLiaPNGEdPfpMSmlC6n_Sx

unzip planet-clothpick-weights.zip  
```

Provided weights are checkpoint `50000` of `D2M`. Note that this checkpoint cannot be transfered to real-world setup.

# Evaluate PlaNet-ClothPick on Test Trials

After training the agent or downloading the weights, you can evaluate the agent on its environment:

```
python evaluate.py \
    --agent planet-clothpick \
    --arena "softgym|domain:mono-square-fabric,initial:crumpled,action:pixel-pick-and-place(1),task:flattening" \
    --config <config-file-name> \
    --eval_checkpoint <checkpoint-num> \
    --log_dir <save-dir> # or <save-dir>/planet-clothpick-weights
```


# Paper citation

```
@inproceedings{kadi2024planet,
  title={PlaNet-ClothPick: effective fabric flattening based on latent dynamic planning},
  author={Kadi, Halid Abdulrahim and Terzi{\'c}, Kasim},
  booktitle={2024 IEEE/SICE International Symposium on System Integration (SII)},
  pages={972--979},
  year={2024},
  organization={IEEE}
}
```
