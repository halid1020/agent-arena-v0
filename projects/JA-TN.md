This is a description for traning and evaluating `JA-TN` models in `mono-square-fabric` and ``rainbow-rectangular-fabrics` environment of `SoftGym`.

Make sure you setup the [`agent-arena-v0`](https://github.com/halid1020/agent-arena-v0) and [`softgym`](https://github.com/halid1020/softgym) as instructed.


# Train and Evaluate JA-TN from Scratch


Under the `softgym`'s root directory, please run
```
. ./setup.sh
```

Then, using the same terminal, under the `agent-arena-v0`'s root directory, please run

```
. ./setup.sh

cd tool

python train_and_evaluate.py \
    --agent transporter \
    --arena "softgym|domain:rainbow-rectangular-fabrics,initial:crumpled,action:pixel-pick-and-place(1),task:flattening" \
    --config JA-TN-2000 \
    --log_dir <save-path> \
```

The above program will also evaluate the agent in the test trials of the arena.

# Evaluate PlaNet-ClothPick on Test Trials

After training the agent or downloading the weights, you can evaluate the agent on its environment:

```
python evaluate.py \
    --agent transporter \
    --arena "softgym|domain:mono-rectangular-fabrics,initial:crumpled,action:pixel-pick-and-place(1),task:flattening" \
    --config <config-file-name> \
    --eval_checkpoint <checkpoint-num> \
    --log_dir <save-dir> # or <save-dir>/planet-clothpick-weights
```


# Paper citation

```
?
```
