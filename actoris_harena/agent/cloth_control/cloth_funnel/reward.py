def deformable_reward(env, log_prefix=None):
    self.episode_memory.add_value(
        key='init_verts', value=self.init_pos
    )
    self.current_pos = pyflex.get_positions().reshape((-1, 4))[:self.num_particles, :3]
    self.episode_memory.add_value(
        key=f"{log_prefix}_verts", value=self.current_pos)

    weighted_distance, l2_distance, icp_distance, real_l2_distance, _ = \
        deformable_distance(self.init_pos, self.current_pos, self.current_task.get_config()["cloth_area"], self.deformable_weight)

    task_image_dict = self.current_task.get_images()

    init_img = task_image_dict["init_rgb"]
    curr_img = np.array(self.pretransform_rgb).transpose(2, 0, 1)[:3]

    try:
        init_mask = self.get_cloth_mask(init_img.transpose(2, 0, 1))
    except:
        print("[SimEnv] Init mask failed, using curr_img.")
        init_mask = self.get_cloth_mask(curr_img)
        
    curr_mask = self.get_cloth_mask(curr_img)

    intersection = np.logical_and(curr_mask, init_mask)
    union = np.logical_or(curr_mask, init_mask)
    iou = np.sum(intersection) / np.sum(union)
    coverage = np.sum(curr_mask) / np.sum(init_mask)

    deformable_dict = {
        "weighted_distance": weighted_distance,
        "l2_distance": l2_distance,
        "icp_distance": icp_distance,
        "pointwise_distance": real_l2_distance,
        "iou": iou,
        "coverage": coverage
    }

    self.episode_memory.add_value(
                    log_prefix + "_init_mask", init_mask)
    self.episode_memory.add_value(
                    log_prefix + "_curr_mask", curr_mask)
    
    if log_prefix is not None:
        for k, v in deformable_dict.items():
            if type(v) == float or type(v) == int \
                or type(v) == np.float64 or type(v) == np.float32:
                self.episode_memory.add_value(
                    log_prefix + "_" + k, float(v))
            elif type(v) == np.ndarray:
                self.episode_memory.add_value(
                    log_prefix + "_" + k, v)

    if self.init_deformable_data is None:
        self.init_deformable_data = deformable_dict

    for k, v in self.init_deformable_data.items():
        if type(v) == float or type(v) == int or \
            type(v) == np.float64 or type(v) == np.float32:
            self.episode_memory.add_value(
                "init_" + k, float(v))

    # print("Coverage in memory", self.episode_memory.data['init_coverage'])
    # print("IoU in memory", self.episode_memory.data['init_iou'])


    return l2_distance, icp_distance, weighted_distance