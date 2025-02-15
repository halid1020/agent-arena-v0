## Install Packages for Raven

```
cd arena/raven
pip install -r requirements.txt
```

# Transporter Net
## 1. raven block-insertion (Tested)
```
python train_and_evaluate.py \
--agent transporter \
--arena "raven|task:block-insertion" \
--log_dir /data/test
```

## 2. raven place-red-in-green (Tested)
```
confipython train_and_evaluate.py \
--agent transporter \
--arena "raven|task:place-red-in-green" \
--log_dir /data/test
```

## 3. raven stack-block-pyramid (Tested)
```
confipython train_and_evaluate.py \
--agent transporter \
--arena "raven|task:stack-block-pyramid" \
--log_dir /data/test
```





# PlaNet

https://github.com/Kaixhin/PlaNet/blob/master/env.py

## 1. DM-Control Walker-Walk (Testing)
```
python train_and_run.py \
--agent planet \
--arena "dm-control-suite|domain:walker,task:walk" \
--config default \
--log_dir /data/test
```


## 2. DM-Control reacher-easy (Testing)
```
confipython train_and_evaluate.py \
--agent planet \
--arena "dm-control-suite|domain:reacher,task:easy" \
--config default \
--log_dir /data/test
```

## 2. DM-Control reacher-easy (Testing)
```
confipython train_and_evaluate.py \
--agent planet \
--arena "dm-control-suite|domain:reacher,task:easy" \
--config default \
--log_dir /data/test
```

# Dreamer V2

1. DM Control Suite
```
confipython train_and_evaluate.py \
--agent dreamer \
--arena "dm-control-suite|domain:walker,task:walk" \
--config default \
--log_dir /data/test
```
Avialble (domain, task)s are [(walker-walk), ()]




## Goal Split Transporter Net on Fabric Flattening

```
confipython train_and_evaluate.py \
--agent transporter \
--arena "mono-square-fabric-sim|task:flattening,action:pixel-pick-and-place(1),initial:crumple" \
--config goal-split \
--log_dir /data/test

```

Testing

## PlaNet-ClothPick, RGB2RGB, on Fabric Flattening

```
confipython train_and_evaluate.py \
--agent planet-clothpick \
--arena "mono-square-fabric-sim|task:flattening,action:pixel-pick-and-place(1),initial:crumple" \
--config RGB2RGB \
--log_dir /data/test
```

Testing

### Test Oracle Policy for fabric flattening in SoftGym
```
python test_oracle_policy.py \
--eid 20 \
--log_dir /data/test
```

Tested

### Test Oracle Straight Policy for fabric flattening velocity grapsin SoftGym
```
python test_oracle_policy.py \
--eid 20 \
--policy "max_action" \
--arena "mono-square-fabric-sim|task:flattening,action:velocity-grasp(1),initial:crumple" \
--log_dir /data/test
```

Tested

### Test Oracle Wrinckle Policy for Fabric flattening in SoftGym
```
python test_oracle_policy.py \
--policy "rect_fabric_wrinkels" \
--eid 20 \
--log_dir /data/test
```

Tested

### Test Oracle Policy for raven block insertion
```
python test_oracle_policy.py \
--arena "raven|task:block-insertion" \
--policy "raven|task:block-insertion" \
--logger "pick_and_place_raven_logger" \
--log_dir /data/test
```

Tested

### Test Oracle Policy for deformable raven environments.
```
python test_oracle_policy.py \
--arena "deformable-raven|task:cloth-flat" \
--policy "deformbale-raven|task:cloth-flat" \
--log_dir /data/test
```

tasks: [sorting, insertion, insertion-translation, hanoi, 
        aligning, stacking, sweeping, pushing, palletizing, 
        kitting, packing, cable, insertion-goal, cable-shape, 
        cable-shape-notarget, cable-line-notarget, cable-ring, 
        cable-ring-notarget, cloth-flat[tested], cloth-flat-notarget, 
        cloth-cover, bag-alone-open[tested], bag-items-easy, bag-items-hard,
        bag-color-goal]

TODO: Develop

## Test Oracle policy diagonal folding in SoftGym
```
python test_oracle_policy.py --gui 1 \
--env "mono-square-fabric-sim|task:diagonal-folding,action:pixel-pick-and-place-z(1),initial:flatten" \
--policy "oracle-rect-fabric|action:pixel-pick-and-place-z(1),strategy:expert-diagonal-folding" \
--logger "empty_logger" 

```

TODO

### Test Tranporter on Diagoanl folding (from flattening) softgym
```
confipython train_and_evaluate.py \
--agent transporter \
--hyper_para TN \
--log_dir /data/planet-clothpick-v2 \
--setting "mono-square-fabric-sim|task:diagonal-folding,action:pixel-pick-and-place(1),initial:flatten"

```

TODO

### Generate Goals for folding
```
python test_oracle_policy.py --gui 0 \
--arena "mono-square-fabric-sim|task:cross-folding,action:pixel-pick-and-place(1),initial:crumple" \
--policy "oracle-rect-fabric|action:pixel-pick-and-place(1),strategy:expert-cross-folding"
--logger "save_goal_logger"
```

TODO