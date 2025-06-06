# coding=utf-8
# Adapted from Ravens - Transporter Networks, Zeng et al., 2021
# https://github.com/google-research/ravens

"""Transport 6DoF models."""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange

from ..models.regression import Regression
from ..models.transport import Transport
from ..utils import MeanMetrics


class TransportHybrid6DoF(Transport):
    """Transport + 6DoF regression hybrid."""

    def __init__(self, in_channels, n_rotations, crop_size, preprocess, verbose=False):
        self.output_dim = 24
        self.kernel_dim = 24
        super().__init__(in_channels, n_rotations, crop_size,
                         preprocess, verbose, name="Transport 6DoF")

        self.regress_loss = nn.HuberLoss()

        self.z_regressor = Regression(in_channels, verbose=verbose)
        self.roll_regressor = Regression(in_channels, verbose=verbose)
        self.pitch_regressor = Regression(in_channels, verbose=verbose)

        self.z_metric = MeanMetrics()
        self.roll_metric = MeanMetrics()
        self.pitch_metric = MeanMetrics()

    def correlate(self, in0, in1, softmax):
        in0 = Rearrange('b h w c -> b c h w')(in0)
        in1 = Rearrange('b h w c -> b c h w')(in1)

        output = F.conv2d(in0[Ellipsis, :3], in1[:, :, :3, :])
        z_tensor = F.conv2d(in0[Ellipsis, :8], in1[:, :, :8, :])
        roll_tensor = F.conv2d(in0[Ellipsis, 8:16], in1[:, :, 8:16, :])
        pitch_tensor = F.conv2d(in0[Ellipsis, 16:24], in1[:, :, 16:24, :])

        if softmax:
            output_shape = output.shape
            output = Rearrange('b c h w -> b (c h w)')(output)
            output = self.softmax(output)
            output = Rearrange(
                'b (c h w) -> b c h w',
                c=output_shape[1],
                h=output_shape[2],
                w=output_shape[3])(output)
            output = output.detach().cpu().numpy()

        return output, z_tensor, roll_tensor, pitch_tensor

    def train_block(self, in_img, p, q, theta, z, roll, pitch):
        output = self.forward(in_img, p, softmax=False)
        output, z_tensor, roll_tensor, pitch_tensor = output

        # Get one-hot pixel label map and 6DoF labels.
        itheta = theta / (2 * np.pi / self.n_rotations)
        itheta = np.int32(np.round(itheta)) % self.n_rotations
        label_size = in_img.shape[:2] + (self.n_rotations,)
        label = np.zeros(label_size)
        label[q[0], q[1], itheta] = 1
        z_label, roll_label, pitch_label = z, roll, pitch

        # Use a window for regression rather than only exact.
        u_window = 7
        v_window = 7
        theta_window = 1
        u_min = max(q[0] - u_window, 0)
        u_max = min(q[0] + u_window + 1, z_tensor.shape[1])
        v_min = max(q[1] - v_window, 0)
        v_max = min(q[1] + v_window + 1, z_tensor.shape[2])
        theta_min = max(itheta - theta_window, 0)
        theta_max = min(itheta + theta_window + 1, z_tensor.shape[3])

        z_est_at_xytheta = z_tensor[0, u_min:u_max, v_min:v_max,
                                    theta_min:theta_max]
        roll_est_at_xytheta = roll_tensor[0, u_min:u_max, v_min:v_max,
                                          theta_min:theta_max]
        pitch_est_at_xytheta = pitch_tensor[0, u_min:u_max, v_min:v_max,
                                            theta_min:theta_max]

        z_est_at_xytheta = Rearrange('b c h w -> b (h w c)')(z_est_at_xytheta)
        roll_est_at_xytheta = Rearrange(
            'b c h w -> b (h w c)')(roll_est_at_xytheta)
        pitch_est_at_xytheta = Rearrange(
            'b c h w -> b (h w c)')(pitch_est_at_xytheta)

        z_est_at_xytheta = self.z_regressor(z_est_at_xytheta)
        roll_est_at_xytheta = self.roll_regressor(roll_est_at_xytheta)
        pitch_est_at_xytheta = self.pitch_regressor(pitch_est_at_xytheta)

        z_weight = 10.0
        roll_weight = 10.0
        pitch_weight = 10.0

        z_label = torch.tensor(z_label[np.newaxis, ...]).to(self.device)
        roll_label = torch.tensor(roll_label[np.newaxis, ...]).to(self.device)
        pitch_label = torch.tensor(
            pitch_label[np.newaxis, ...]).to(self.device)

        z_loss = z_weight * self.regress_loss(z_label, z_est_at_xytheta)
        roll_loss = roll_weight * \
            self.regress_loss(roll_label, roll_est_at_xytheta)
        pitch_loss = pitch_weight * \
            self.regress_loss(pitch_label, pitch_est_at_xytheta)

        return z_loss, roll_loss, pitch_loss

    def train(self, in_img, p, q, theta, z, roll, pitch):
        self.metric.reset()
        self.z_metric.reset()
        self.roll_metric.reset()
        self.pitch_metric.reset()

        self.model_query.train()
        self.model_key.train()
        self.z_regressor.model.train()
        self.roll_regressor.model.train()
        self.pitch_regressor.model.train()

        self.optimizer_query.zero_grad()
        self.optimizer_key.zero_grad()
        self.z_regressor.optimizer.zero_grad()
        self.roll_regressor.optimizer.zero_grad()
        self.pitch_regressor.optimizer.zero_grad()

        z_loss, roll_loss, pitch_loss = self.train_block(
            in_img, p, q, theta, z, roll, pitch)
        loss = z_loss + roll_loss + pitch_loss

        loss.backward()
        # MAYBE NOT FOLLOWING TWO LINES
        self.optimizer_query.step()
        self.optimizer_key.step()
        self.z_regressor.optimizer.step()
        self.roll_regressor.optimizer.step()
        self.pitch_regressor.optimizer.step()

        self.z_metric(z_loss)
        self.roll_metric(roll_loss)
        self.pitch_metric(pitch_loss)

        self.iters += 1
        return np.float32(loss.detach().cpu().numpy())

    def test(self, in_img, p, q, theta, z, roll, pitch):
        self.model_query.eval()
        self.model_key.eval()
        self.z_regressor.model.eval()
        self.roll_regressor.model.eval()
        self.pitch_regressor.model.eval()

        with torch.no_grad():
            z_loss, roll_loss, pitch_loss = self.train_block(
                in_img, p, q, theta, z, roll, pitch)
            loss = z_loss + roll_loss + pitch_loss

        self.z_metric(z_loss)
        self.roll_metric(roll_loss)
        self.pitch_metric(pitch_loss)

        self.iters += 1
        return np.float32(loss.detach().cpu().numpy())

    # ---------------------------------------------------------------------------
    # Visualization methods.
    # ---------------------------------------------------------------------------

    # # For visualization
    # Need to pass in theta to use this visualization.
    # self.feature_visualize = False
    # if self.feature_visualize:
    #   self.fig, self.ax = plt.subplots(5, 1)
    # self.write_visualize = False
    # self.plot_interval = 20

    # visualize_input = False
    # if visualize_input and self.six_dof:  # only supported for six dof model
    #   self.visualize_train_input(in_img, p, q, theta, z, roll, pitch)

    # if theta is not None and self.feature_visualize and
    # self.iters % self.plot_interval == 0:
    #   self.visualize_introspection(img_unprocessed, p, rvecs, in_shape,
    #                                theta, logits, kernel_raw, output)

    # def visualize_introspection(self, img_unprocessed, p, rvecs, in_shape,
    #                             theta, logits, kernel, output):
    #   """Utils for visualizing features at
    #      the end of the Background and Foreground networks."""

    #   # Do this again, for visualization
    #   crop_rgb = tf.convert_to_tensor(
    #       img_unprocessed.copy().reshape(in_shape), dtype=tf.float32)
    #   crop_rgb = tf.repeat(crop_rgb, repeats=self.n_rotations, axis=0)
    #   crop_rgb = tfa.image.transform(crop_rgb, rvecs, interpolation="NEAREST")
    #   crop_rgb = crop_rgb[:, p[0]:(p[0] + self.crop_size),
    #                       p[1]:(p[1] + self.crop_size), :]
    #   crop_rgb = crop_rgb.numpy()

    #   self.ax[0].cla()
    #   self.ax[1].cla()
    #   self.ax[2].cla()
    #   self.ax[3].cla()
    #   itheta = theta / (2 * np.pi / self.n_rotations)
    #   itheta = np.int32(np.round(itheta)) % self.n_rotations

    #   self.ax[0].imshow(crop_rgb[itheta, :, :, :3].transpose(1, 0, 2) / 255.)

    #   if self.write_visualize:
    #     # delete first:
    #     try:
    #       shutil.rmtree("vis/crop_rgb")
    #       shutil.rmtree("vis/crop_kernel")
    #     except:  # pylint: disable=bare-except
    #       print("Warning: couldn't delete folder for visualization.")

    #     os.system("mkdir -p vis/crop_rgb")
    #     os.system("mkdir -p vis/crop_kernel")

    #     for theta_idx in range(self.n_rotations):
    #       filename = "itheta_" + str(theta_idx).zfill(4) + ".png"
    #       if itheta == theta_idx:
    #         filename = "label-" + filename
    #       imageio.imwrite(
    #           os.path.join("vis/crop_rgb/", filename),
    #           crop_rgb[theta_idx, :, :, :3].transpose(1, 0, 2))

    #   self.ax[1].imshow(img_unprocessed[:, :, :3].transpose(1, 0, 2) / 255.)
    #   if self.write_visualize:
    #     filename = "img_rgb.png"
    #     imageio.imwrite(
    #         os.path.join("vis/", filename),
    #         img_unprocessed[:, :, :3].transpose(1, 0, 2))

    #   logits_numpy = logits.numpy()
    #   kernel_numpy = kernel.numpy()

    #   for c in range(3):
    #     channel_mean = np.mean(logits_numpy[:, :, :, c])
    #     channel_std = np.std(logits_numpy[:, :, :, c])
    #     channel_1std_max = channel_mean + channel_std
    #     # channel_1std_max = np.max(logits_numpy[:, :, :, c])
    #     channel_1std_min = channel_mean - channel_std
    #     # channel_1std_min = np.min(logits_numpy[:, :, :, c])
    #     logits_numpy[:, :, :, c] -= channel_1std_min
    #     logits_numpy[:, :, :, c] /= (channel_1std_max - channel_1std_min)
    #     for theta_idx in range(self.n_rotations):
    #       channel_mean = np.mean(kernel_numpy[theta_idx, :, :, c])
    #       channel_std = np.std(kernel_numpy[theta_idx, :, :, c])
    #       channel_1std_max = channel_mean + channel_std
    #       # channel_1std_max = np.max(kernel_numpy[itheta, :, :, c])
    #       channel_1std_min = channel_mean - channel_std
    #       # channel_1std_min = np.min(kernel_numpy[itheta, :, :, c])
    #       kernel_numpy[theta_idx, :, :, c] -= channel_1std_min
    #       kernel_numpy[theta_idx, :, :, c] /= (
    #           channel_1std_max - channel_1std_min)

    #   self.ax[2].imshow(logits_numpy[0, :, :, :3].transpose(1, 0, 2))
    #   self.ax[3].imshow(kernel_numpy[itheta, :, :, :3].transpose(1, 0, 2))

    #   if self.write_visualize:
    #     imageio.imwrite(
    #         os.path.join("vis", "img_features.png"),
    #         logits_numpy[0, :, :, :3].transpose(1, 0, 2))

    #     for theta_idx in range(self.n_rotations):
    #       filename = "itheta_" + str(theta_idx).zfill(4) + ".png"
    #       if itheta == theta_idx:
    #         filename = "label-" + filename
    #       imageio.imwrite(
    #           os.path.join("vis/crop_kernel/", filename),
    #           kernel_numpy[theta_idx, :, :, :3].transpose(1, 0, 2))

    #   heatmap = output[0, :, :, itheta].numpy().transpose()
    #   # variance = 0.1
    #   heatmap = -np.exp(-heatmap / 0.1)

    #   cmap = plt.cm.jet
    #   norm = plt.Normalize(vmin=heatmap.min(), vmax=heatmap.max())
    #   heatmap = cmap(norm(heatmap))

    #   self.ax[4].imshow(heatmap)
    #   if self.write_visualize:
    #     imageio.imwrite("vis/heatmap.png", heatmap)

    #   # non-blocking
    #   plt.draw()
    #   plt.pause(0.001)

    #   # blocking
    #   # plt.show()

    # def visualize_train_input(self, in_img, p, q, theta, z, roll, pitch):
    #   """Visualize the training input."""
    #   points = []
    #   colors = []
    #   height = in_img[:, :, 3]

    #   for i in range(in_img.shape[0]):
    #     for j in range(in_img.shape[1]):
    #       pixel = (i, j)
    #       position = utils.pix_to_xyz(pixel, height, self.bounds,
    #                                          self.pixel_size)
    #       points.append(position)
    #       colors.append(in_img[i, j, :3])

    #   points = np.array(points).T  # shape (3, N)
    #   colors = np.array(colors).T / 255.0  # shape (3, N)

    #   self.vis["pointclouds/scene"].set_object(
    #       g.PointCloud(position=points, color=colors))

    #   pick_position = utils.pix_to_xyz(p, height, self.bounds,
    #                                           self.pixel_size)
    #   label = "pick"
    #   utils.make_frame(self.vis, label, h=0.05, radius=0.0012, o=0.1)

    #   pick_transform = np.eye(4)
    #   pick_transform[0:3, 3] = pick_position
    #   self.vis[label].set_transform(pick_transform)

    #   place_position = utils.pix_to_xyz(q, height, self.bounds,
    #                                            self.pixel_size)
    #   label = "place"
    #   utils.make_frame(self.vis, label, h=0.05, radius=0.0012, o=0.1)

    #   place_transform = np.eye(4)
    #   place_transform[0:3, 3] = place_position
    #   place_transform[2, 3] = z

    #   rotation = utils.eulerXYZ_to_quatXYZW((roll, pitch, -theta))
    #   quaternion_wxyz = np.asarray(
    #       [rotation[3], rotation[0], rotation[1], rotation[2]])

    #   place_transform[0:3, 0:3] =
    #   mtf.quaternion_matrix(quaternion_wxyz)[0:3, 0:3]
    #   self.vis[label].set_transform(place_transform)

    #   _, ax = plt.subplots(2, 1)
    #   ax[0].imshow(in_img.transpose(1, 0, 2)[:, :, :3] / 255.0)
    #   ax[0].scatter(p[0], p[1])
    #   ax[0].scatter(q[0], q[1])
    #   ax[1].imshow(in_img.transpose(1, 0, 2)[:, :, 3])
    #   ax[1].scatter(p[0], p[1])
    #   ax[1].scatter(q[0], q[1])
    #   plt.show()
