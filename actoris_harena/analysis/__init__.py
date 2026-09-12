"""What each input stream contributes to the actions a policy plans.

Separate from the inference pipeline on purpose. This package imports from
``actoris_harena.deploy.policy_client`` and from the LeRobot policies; nothing on the inference
path imports it back, so an attribution study cannot slow a rollout down or
change what the arms do.

Two questions are easy to confuse, and only one of them lives here.

*What could a policy learn from each stream?* is answered by TRAINING one per
camera set and comparing -- the rows in ``hpc/runs.tsv``. It costs GPU-days.

*What does THIS trained policy use?* is answered here, from one checkpoint, in
minutes. The two can disagree, and where they do, that is the finding: a stream
whose absence a retrained policy compensates for is not the same as a stream the
trained policy ignores.

The modules, in the order they depend on each other:

``streams``      which part of a policy's conditioning belongs to which input.
                 Pure arithmetic over a config; no torch, no weights.
``perturb``      replace a stream and re-infer. The behavioural ground truth,
                 and what the gradient methods are scored against.
``gradients``    integrated gradients, SmoothGrad and Grad-CAM.
``attention``    ACT only: the decoder's cross-attention, per action step.
``diffusion``    what a stochastic multi-step sampler needs that ACT does not.
``phases``       when in an episode a stream mattered, against what the grippers
                 were doing at the time.
``report``       the figures.

Driven by ``tool/analyse_policy_inputs.py``.
"""
