import os
from tqdm import tqdm
from agent.transporter.dataset import Dataset
from src.utilities.visualisation_utils import plot_image_trajectory as pt

target_dataset_dir = '/data/fast-ldm-fabric-shaper/datasets/OTS-OTF-RSF-all-corner-inward-folding' 

train_dataset_target = Dataset(os.path.join(target_dataset_dir,'{}_dataset'.format('train')), 
    swap_action=False, 
    img_shape=(128, 128))

for i in tqdm(range(2000, 2005)):
    episode, _ = train_dataset_target.load(i)
    rgbs = []
    actions = []
    for j in range(len(episode)):
        (obs, act_, _, _) = episode[j]
        rgbs.append(obs['color'])
        actions.append(act_)
    
    pt(
        rgbs, # TODO: this is envionrment specific
        title='Episode {}'.format(i), 
        # rewards=result['rewards'], 
        save_png=True, col=5, save_path=".")