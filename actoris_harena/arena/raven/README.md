
avaliable tasks

'align-box-corner'
'assembling-kits'
'assembling-kits-easy'
'block-insertion'
'block-insertion-easy'
'block-insertion-nofixture'
'block-insertion-sixdof'
'block-insertion-translation'
'manipulating-rope'
'packing-boxes'
'palletizing-boxes'
'place-red-in-green'
'stack-block-pyramid'
'sweeping-piles'
'towers-of-hanoi'


export RAVENS_ASSETS_DIR=${actoris_harena_PATH}/arena/raven/environments/assets

initate the arena with string ag_ar.build_arena('raven|task:<task-name>')

run the oracle agent: agent = ag_ar.build_agent('raven-oracle', DotMap({}))

