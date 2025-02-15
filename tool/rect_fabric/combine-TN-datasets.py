import os
from tqdm import tqdm
from agent.transporter.dataset import Dataset

dataset_dir_1 = '/data/fast-ldm-fabric-shaper/datasets/OTS-RRF-flattening'
dataset_dir_2 = '/data/fast-ldm-fabric-shaper/datasets/OTF-RRF-rectangular-folding'
target_dataset_dir = '/data/fast-ldm-fabric-shaper/datasets/OTS-OTF-RRF-rectangular-folding' 

if os.path.exists(target_dataset_dir):
    #remove the target dataset
    os.system('rm -r {}'.format(target_dataset_dir))

train_dataset_1 = Dataset(os.path.join(dataset_dir_1,'{}_dataset'.format('train')), 
    swap_action=False, 
    img_shape=(128, 128), n_sample=2000)

train_dataset_2 = Dataset(os.path.join(dataset_dir_2,'{}_dataset'.format('train')), 
    swap_action=False, 
    img_shape=(128, 128))

train_dataset_target = Dataset(os.path.join(target_dataset_dir,'{}_dataset'.format('train')), 
    swap_action=False, 
    img_shape=(128, 128))

for i in tqdm(range(2000)):
    episode, _ = train_dataset_1.load(i, images=True, cache=False)
    train_dataset_target.add(0, episode)

for i in tqdm(range(1000)):
    episode, _ = train_dataset_2.load(i, images=True, cache=False)
    train_dataset_target.add(0, episode)

for i in tqdm(range(1000)):
    episode, _ = train_dataset_2.load(i, images=True, cache=False)
    train_dataset_target.add(0, episode)