# The retired npz collection pipeline

`actoris_harena/data/` held a from-scratch multi-modal collection stack —
`DataStream`, `DataCollectionPipeline`, `DynamicTrajectoryDataset`,
`StaticTrajectoryDataset` and a Flask `dataset_visualiser` — writing one `.npz`
per episode beside a `metadata.json`. It was committed in
`306d5e0` and removed in the commit that follows it. This note is why, and what
of it is worth keeping.

## Why it was retired

The shared pipeline that replaces it does the same job on **LeRobotDataset**:
parquet + encoded video + `meta/`, which is the format `lerobot-train` reads
directly. That removes a conversion step that nothing was paying for. Concretely:

- Storage was `np.savez` of whole episodes held in memory, with JPEG bytes
  stuffed into object arrays and decoded by name-substring dispatch
  (`*camera*`, `*rgb*`, `*action*`). The replacement writes video features that
  a policy's dataloader decodes, and names its features explicitly.
- There was no `episode_index`, no per-episode metadata file, no integrity
  model. The replacement has `recording/dataset_check.py`, which can describe a
  dataset LeRobot itself refuses to open.
- It had never run against hardware. Its one dual-arm test declares
  `"robot_type": "Dual UR5e with Robotiq 2F-85"` and then drives every stream
  with `np.sin` and `np.random.randn`.
- `dataset_visualiser.py:8` carried its own note that the camera panel and the
  other panels showed nothing.

## What was good, and is owed

Two ideas in it are better than what the replacement currently has, and should
be ported into the console rather than forgotten:

1. **A language instruction per recording.** `metadata.json` carried
   `recordings.<id>.language_instruction` — "Fold the blue towel in half." A
   LeRobot dataset has a task string per frame, but it is set at collection time
   and awkward to revise afterwards; being able to write and correct the
   instruction per episode, after the fact, from the browser, is the feature.
2. **Subtask annotation.** `recordings.<id>.subtasks` was an empty list waiting
   for spans — a way to mark, within one episode, where the reach ends and the
   grasp begins. Nothing in the replacement records that, and the input-analysis
   work already shows why it matters: tactile share runs at 0.6 % while the arms
   travel and peaks at 21–31 % on contact, so a mean over a whole episode hides
   the result. Phase spans are how that stops being a hand-read plot.

Both belong in the dataset-curation tab, beside episode marking, where the
person who can judge the episode is already looking at it.
