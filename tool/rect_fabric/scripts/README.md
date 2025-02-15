### Train and run **planet-pick** on **old pick-and-place cloth-flattening** setting
```
python train_and_run_agent.py --setting pick-and-place-cloth-flattening-old --agent planet-pick --hyper_para default
```

The loss, evaluation and manupilation data will be saved in `tmp/pick-and-place-cloth-flattening-old/planet-pick/default` directory.

### Train and Run a certain agant in a certain envrionmetn setting

```
python train_and_run_agent.py --setting <environment&dataset setting anem> --agent <agent name> --hyper_para <config name> --policy <policy name>
```

### Run a certain agant in a certain envrionmetn setting

```
python run_agent.py --setting <environment setting anem> --agent <agent name> --hyper_para <config name> --policy <policy name>
```