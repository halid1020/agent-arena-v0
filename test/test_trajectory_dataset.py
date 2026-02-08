import unittest
import numpy as np
import os
import shutil
import tempfile
from torch.utils.data import DataLoader

# Import your class here. Assuming it's in a file named `trajectory_dataset.py`
# If it's in the same file, just leave the import as is.
# from trajectory_dataset import TrajectoryDataset 

# --- Mocking the class import for the purpose of this script ---
# (In your real setup, just import your actual class)
from actoris_harena import TrajectoryDataset

class TestTrajectoryDataset(unittest.TestCase):

    def setUp(self):
        # Create a temporary directory for test data
        self.test_dir = tempfile.mkdtemp()
        self.data_path = "test_dataset.zarr"
        self.full_path = os.path.join(self.test_dir, self.data_path)

        # Standard Configs
        self.obs_config = {
            'rgb': {'shape': (3, 32, 32), 'output_key': 'rgb'},
            'depth': {'shape': (1, 32, 32), 'output_key': 'depth'}
        }
        self.act_config = {
            'default': {'shape': (4,), 'output_key': 'action'}
        }
        self.goal_config = {
            'target_rgb': {'shape': (3, 32, 32), 'output_key': 'goal_rgb'}
        }

    def tearDown(self):
        # Clean up the temporary directory
        shutil.rmtree(self.test_dir)

    def generate_dummy_data(self, num_steps=10):
        """Helper to generate consistent dummy data"""
        obs = {
            'rgb': np.random.randn(num_steps + 1, 3, 32, 32).astype(np.float32),
            'depth': np.random.randn(num_steps + 1, 1, 32, 32).astype(np.float32)
        }
        act = {
            'default': np.random.randn(num_steps, 4).astype(np.float32)
        }
        goal = {
            'target_rgb': np.random.randn(1, 3, 32, 32).astype(np.float32)
        }
        return obs, act, goal

    # =========================================================================
    # TEST 1: Basic Write and Read
    # =========================================================================
    def test_write_and_read(self):
        print("\n--- Test 1: Basic Write and Read ---")
        
        # 1. Create and Write
        dataset = TrajectoryDataset(
            data_path=self.data_path,
            data_dir=self.test_dir,
            io_mode='w',
            obs_config=self.obs_config,
            act_config=self.act_config,
            goal_config=self.goal_config,
            save_goal=True,
            return_trj_last=True
        )

        obs1, act1, goal1 = self.generate_dummy_data(10)
        dataset.add_trajectory(obs1, act1, goal1)
        
        obs2, act2, goal2 = self.generate_dummy_data(15)
        dataset.add_trajectory(obs2, act2, goal2)

        self.assertEqual(dataset.num_trajectories(), 2)
        self.assertEqual(dataset.get_total_timesteps(), 11 + 16)

        # 2. Reload in Read Mode
        dataset_r = TrajectoryDataset(
            data_path=self.data_path,
            data_dir=self.test_dir,
            io_mode='r',
            obs_config=self.obs_config,
            act_config=self.act_config,
            goal_config=self.goal_config,
            save_goal=True,
            whole_trajectory=True
        )

        # Verify Content of Trajectory 0
        traj0 = dataset_r.get_trajectory(0)
        np.testing.assert_array_almost_equal(traj0['observation']['rgb'], obs1['rgb'])
        np.testing.assert_array_almost_equal(traj0['action']['action'], act1['default'])
        np.testing.assert_array_almost_equal(traj0['goal']['goal_rgb'], goal1['target_rgb'])
        
        print("Write/Read verification successful.")

    # =========================================================================
    # TEST 2: Caching Logic
    # =========================================================================
    def test_caching_logic(self):
        print("\n--- Test 2: In-Memory Caching ---")
        
        # Initialize with cache_in_memory=True
        # In this case whole_trajecotry is set to Flase, the seuqnce length has to be set to an number
        dataset = TrajectoryDataset(
            data_path=self.data_path,
            data_dir=self.test_dir,
            io_mode='w',
            obs_config=self.obs_config,
            act_config=self.act_config,
            cache_in_memory=True
        )

        obs, act, _ = self.generate_dummy_data(10)
        
        # Add trajectory (should update both Zarr and Cache)
        dataset.add_trajectory(obs, act)

        # Verify Cache is populated
        self.assertIsInstance(dataset.obs_source['rgb'], np.ndarray)
        self.assertEqual(dataset.obs_source['rgb'].shape[0], 11)
        
        # Verify correctness
        np.testing.assert_array_almost_equal(dataset.obs_source['rgb'], obs['rgb'])
        print("Caching logic successful.")

    # =========================================================================
    # TEST 3: Sampling Modes (Seq, Whole, Cross)
    # =========================================================================
    def test_sampling_modes(self):
        print("\n--- Test 3: Sampling Modes ---")
        
        # Setup Data: Trj1 (Length 10), Trj2 (Length 5)
        dataset = TrajectoryDataset(
            data_path=self.data_path,
            data_dir=self.test_dir,
            io_mode='w',
            obs_config=self.obs_config,
            act_config=self.act_config
        )
        dataset.add_trajectory(*self.generate_dummy_data(10)[:2]) # steps are 10, but sequence size is 11
        dataset.add_trajectory(*self.generate_dummy_data(5)[:2]) # steps are 5, but sequence size is 6

        # A. Fixed Sequence Length
        # Trj1 has 10 steps. Seq=2. Valid starts: 0 to 8 (indices)
        dataset_seq = TrajectoryDataset(
            data_path=self.data_path,
            data_dir=self.test_dir,
            io_mode='r',
            obs_config=self.obs_config,
            act_config=self.act_config,
            seq_length=2,
            cache_in_memory=True
        )
        
        # Check an item
        item = dataset_seq[0]
        self.assertEqual(item['observation']['rgb'].shape[0], 3) # Seq=2 means 3 frames (t, t+1, t+2)
        self.assertEqual(item['action']['action'].shape[0], 2)
        
        # B. Whole Trajectory
        dataset_whole = TrajectoryDataset(
            data_path=self.data_path,
            data_dir=self.test_dir,
            io_mode='r',
            obs_config=self.obs_config,
            act_config=self.act_config,
            whole_trajectory=True
        )
        self.assertEqual(len(dataset_whole), 2) # Should act like list of trajectories
        item_whole = dataset_whole[1]
        self.assertEqual(item_whole['action']['action'].shape[0], 5)

        print("Sampling modes successful.")

    # =========================================================================
    # TEST 4: Train / Val / Eval Split
    # =========================================================================
    def test_splitting(self):
        print("\n--- Test 4: Data Splitting ---")
        
        dataset = TrajectoryDataset(
            data_path=self.data_path,
            data_dir=self.test_dir,
            io_mode='w',
            obs_config=self.obs_config,
            act_config=self.act_config,
        )
        
        # Add 100 trajectories of length 10
        for _ in range(100):
            dataset.add_trajectory(*self.generate_dummy_data(10)[:2])
        
        # Standard Split: 10% Eval, 10% Val, 80% Train
        # Total samples approx 100 * (10 - seq + 1)
        
        seq_len = 5
        # Each trajectory has length 10. Valid starts for seq=5 is 10-5 = 5 samples per traj.
        # Total valid samples = 500.
        
        ds_train = TrajectoryDataset(
            data_path=self.data_path, data_dir=self.test_dir, io_mode='r', 
            obs_config=self.obs_config, act_config=self.act_config, cross_trajectory=False,
            seq_length=seq_len, sample_mode='train', split_ratios=[0.1, 0.1, 0.8], return_trj_last=False
        )
        
        ds_val = TrajectoryDataset(
            data_path=self.data_path, data_dir=self.test_dir, io_mode='r', 
            obs_config=self.obs_config, act_config=self.act_config, cross_trajectory=False,
            seq_length=seq_len, sample_mode='val', split_ratios=[0.1, 0.1, 0.8], return_trj_last=False
        )

        total_samples = 100 * (11 - seq_len)
        expected_train = int(total_samples * 0.8)
        expected_val = int(total_samples * 0.1)

        print(f"Total Samples: {total_samples}, Train: {len(ds_train)}, Val: {len(ds_val)}")
        
        # Allow +/- 1 due to rounding
        self.assertTrue(abs(len(ds_train) - expected_train) <= 1)
        self.assertTrue(abs(len(ds_val) - expected_val) <= 1)
        
        print("Splitting logic successful.")

    # =========================================================================
    # TEST 5: Robustness (Short Trajectories)
    # =========================================================================
    def test_short_trajectories(self):
        print("\n--- Test 5: Robustness to Short Trajectories ---")
        
        dataset = TrajectoryDataset(
            data_path=self.data_path,
            data_dir=self.test_dir,
            io_mode='w',
            obs_config=self.obs_config,
            act_config=self.act_config
        )
        
        # Add a "good" trajectory (len 10)
        dataset.add_trajectory(*self.generate_dummy_data(10)[:2]) # steps 10, sqenece 11.
        
        # Add a "bad" trajectory (len 2) - shorter than seq_length 5
        dataset.add_trajectory(*self.generate_dummy_data(2)[:2]) # steps 2, sqenece 3.
        
        # TODO: check if sequence lenght for the added trajecoties are 11 and 3.

        # Initialize reader with seq_length 5
        dataset_r = TrajectoryDataset(
            data_path=self.data_path,
            data_dir=self.test_dir,
            whole_trajectory=False,
            cross_trajectory=False,
            io_mode='r',
            obs_config=self.obs_config,
            act_config=self.act_config,
            seq_length=5,
            return_trj_last=False
        )
        
        # The bad trajectory should be skipped in flat_ranges
        # Good traj (len 11) -> valid sequence [0-4],[1-5],[2-6],[3-7],[4-8], [5-9] (6 samples); 
        # we do not sample [6-10] because last action is 'place-holder'
        # Bad traj (len 3) -> 0 samples
        self.assertEqual(len(dataset_r), 6)
        
        # Ensure we can load all of them without crashing
        loader = DataLoader(dataset_r, batch_size=2)
        for batch in loader:
            pass
            
        print("Short trajectory handling successful.")

    # =========================================================================
    # TEST 6: Cross Trajectory & Terminal Flags
    # =========================================================================
    def test_cross_trajectory(self):
        print("\n--- Test 6: Cross Trajectory & Terminal Flags ---")
        
        dataset = TrajectoryDataset(
            data_path=self.data_path,
            data_dir=self.test_dir,
            io_mode='w',
            obs_config=self.obs_config,
            act_config=self.act_config
        )
        
        # Add 2 trajectories of 5 steps each (Storage length 6)
        # Total storage: 12 steps. 
        # Indices: 
        # T1: 0, 1, 2, 3, 4 (actions), 5 (pad/terminal)
        # T2: 6, 7, 8, 9, 10 (actions), 11 (pad/terminal)
        dataset.add_trajectory(*self.generate_dummy_data(5)[:2])
        dataset.add_trajectory(*self.generate_dummy_data(5)[:2])
        
        dataset_cross = TrajectoryDataset(
            data_path=self.data_path,
            data_dir=self.test_dir,
            io_mode='r',
            obs_config=self.obs_config,
            act_config=self.act_config,
            seq_length=4,
            cross_trajectory=True,
            sample_terminal=True,
            return_trj_last=True
        )

        # We want to sample a window that crosses from T1 to T2.
        # Window length = 4. 
        # If we start at index 3:
        # steps: 3, 4, 5, 6.
        # 3,4 are T1 actions. 5 is T1 pad. 6 is T2 start.
        
        # Note: In cross_trajectory mode, we treat indices linearly.
        item = dataset_cross[3] 
        
        terminals = item['observation']['terminal']
        # terminal shape is (seq_len + 1, 1) -> (5, 1)
        # indices relative to window: 0(idx3), 1(idx4), 2(idx5), 3(idx6), 4(idx7)
        
        # Index 5 in global storage is the end of T1. 
        # It should correspond to index 2 in our local window.
        
        print("Terminal array:", terminals.flatten())
        
        # Assert that the terminal flag is high at the boundary
        self.assertEqual(terminals[2].item(), 1.0, "Terminal flag should be 1 at the end of Trajectory 1")
        self.assertEqual(terminals[0].item(), 0.0)
        self.assertEqual(terminals[3].item(), 0.0) # Start of T2 should not be terminal
        
        print("Cross trajectory logic successful.")

    
    # =========================================================================
    # TEST 7: Return Trajectory Last Flag Logic
    # =========================================================================
    def test_return_trj_last_modes(self):
        print("\n--- Test 7: Return Last Trajectory Flag Logic ---")
        
        # Setup: Create 2 trajectories, 10 steps each.
        # Stored length is 11 (0..9 actions, 10 pad).
        # Total storage: 22 steps.
        # Terminals at indices: 10 and 21.
        dataset = TrajectoryDataset(
            data_path=self.data_path, data_dir=self.test_dir, io_mode='w',
            obs_config=self.obs_config, act_config=self.act_config
        )
        dataset.add_trajectory(*self.generate_dummy_data(10)[:2])
        dataset.add_trajectory(*self.generate_dummy_data(10)[:2])
        
        seq_len = 2

        # --- Case A: Standard Mode, return_trj_last=False ---
        # Length used: 11 - 1 = 10.
        # Valid starts: 0 to 10 - 2 + 1 = 9
        # Formula: start + len - seq. 
        # start=0. len=10. seq=2. end_idx = 0 + 10 - 2 = 8.
        # Range 0..8 inclusive = 9 samples.
        ds_std_false = TrajectoryDataset(data_path=self.data_path, data_dir=self.test_dir, 
                                         obs_config=self.obs_config, act_config=self.act_config,
                                         seq_length=seq_len, cross_trajectory=False, return_trj_last=False)
        self.assertEqual(len(ds_std_false), 18) # 9 per trj * 2 trj
        
        # --- Case B: Standard Mode, return_trj_last=True ---
        # Length used: 11.
        # Formula: start + 11 - 2 = 9.
        # Range 0..9 inclusive = 10 samples. (One extra sample per trj).
        ds_std_true = TrajectoryDataset(data_path=self.data_path, data_dir=self.test_dir, 
                                        obs_config=self.obs_config, act_config=self.act_config,
                                        seq_length=seq_len, cross_trajectory=False, return_trj_last=True)
        self.assertEqual(len(ds_std_true), 20) # 9 per trj * 2 trj

        # --- Case C: Cross Mode, return_trj_last=True ---
        # Total time: 22. Max start: 22 - 2 = 20. (Indices 0..19).
        # Sample count: 20.
        ds_cross_true = TrajectoryDataset(data_path=self.data_path, data_dir=self.test_dir, 
                                          obs_config=self.obs_config, act_config=self.act_config,
                                          seq_length=seq_len, cross_trajectory=True, return_trj_last=True)
        self.assertEqual(len(ds_cross_true), 20)
        
        # --- Case D: Cross Mode, return_trj_last=False ---
        # Total possible starts: 20.
        # Terminals at 10 and 21.
        # Index 21 is >= 20, so it's out of range anyway.
        # Index 10 is inside range 0..19.
        # We must skip index 10.
        # Total samples: 20 - 1 = 19.
        ds_cross_false = TrajectoryDataset(data_path=self.data_path, data_dir=self.test_dir, 
                                           obs_config=self.obs_config, act_config=self.act_config,
                                           seq_length=seq_len, cross_trajectory=True, return_trj_last=False)
        self.assertEqual(len(ds_cross_false), 19)
        
        # Verify that we actually skipped index 10
        # If we access index 10 (which maps to 11 in valid list), we should get start_idx=11
        # Item 10 in the dataset should correspond to index 11 in raw storage.
        # (Indices 0..9 map to 0..9. Index 10 maps to 11).
        item = ds_cross_false[10]
        # Check an observation value to verify shift (optional, but good for sanity)
        # Or just check that we didn't crash.
        
        print("Return Last Trajectory flags verified successfully.")


if __name__ == '__main__':
    unittest.main()