Test softgym environments

```
python test_env.py --env "mono-square-fabric-sim|task:flattening,observation:RGB,action:pixel-pick-and-place(1),initial:crumple" --gui 1
```

Test raven environments

```
python test_env.py --env "raven|task:block-insertion" --gui 1

```

Test deformable raven environments

```
python test_env.py --env "deformable-raven|task:cloth-flat" --gui 1

```