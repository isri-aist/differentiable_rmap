#! /usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from sklearn import mixture


# Estimate the probability density from the task-space sample set
class TaskSpaceDensityEstimation(object):
    def __init__(self):
        # Setup plot
        fig = plt.figure()
        fig.set_size_inches(14, 10)
        plt.subplots_adjust(wspace=0.3, hspace=0.5)
        plt.rcParams["font.size"] = 18

        # Load sampling result
        sample_path = "/tmp/uniform_joint_sampling.npz"
        print("Load sampling result from {}".format(sample_path))
        sample_data = np.load(sample_path)
        joint_pos_list = sample_data["joint_pos_list"]
        body_pos_list = sample_data["body_pos_list"]

        # Setup task-space plot
        pos_max = body_pos_list.max(axis=0)
        pos_min = body_pos_list.min(axis=0)
        pos_range = pos_max - pos_min
        padding_rate = 0.1
        pos_max_with_margin = pos_max + padding_rate * pos_range
        pos_min_with_margin = pos_min - padding_rate * pos_range
        xy_linspace = [np.linspace(pos_min_with_margin[i], pos_max_with_margin[i], 100) for i in range(2)]
        xy_mesh = np.meshgrid(xy_linspace[0], xy_linspace[1])
        xy_mesh_plot = np.vstack([xy_mesh[0].ravel(), xy_mesh[1].ravel()])

        # Plot joint-space sampling
        ax = fig.add_subplot(221)
        ax.scatter(joint_pos_list[:, 0], joint_pos_list[:, 1], s=10)
        ax.set_title("(A) Joint-space sampling", y=-0.375)
        ax.set_xlabel("Joint1 position [rad]")
        ax.set_ylabel("Joint2 position [rad]")
        ax.set_aspect("equal")

        # Plot task-space sampling
        ax = fig.add_subplot(222)
        ax.scatter(body_pos_list[:, 0], body_pos_list[:, 1], s=10)
        ax.set_title("(B) Task-space sampling", y=-0.375)
        ax.set_xlabel("X position [m]")
        ax.set_ylabel("Y position [m]")
        ax.set_aspect("equal")

        # KDE
        kde = gaussian_kde(body_pos_list[:,:2].T)
        kde_pred_list = kde.evaluate(xy_mesh_plot)
        ax = fig.add_subplot(223)
        ax.scatter(body_pos_list[:, 0], body_pos_list[:, 1], s=5)
        cont = ax.contourf(xy_mesh[0], xy_mesh[1], kde_pred_list.reshape(len(xy_linspace[1]), len(xy_linspace[0])),
                           cmap="jet", alpha=0.5)
        ax.set_title("(C) Kernel density estimation", y=-0.375)
        ax.set_xlabel("X position [m]")
        ax.set_ylabel("Y position [m]")
        ax.set_aspect("equal")
        plt.colorbar(cont, ax=ax, fraction=0.046, pad=0.04)

        # GMM
        gmm = mixture.GaussianMixture(n_components=4, covariance_type="full")
        gmm.fit(body_pos_list[:,:2])
        gmm_pred_list = np.exp(gmm.score_samples(xy_mesh_plot.T))
        ax = fig.add_subplot(224)
        ax.scatter(body_pos_list[:, 0], body_pos_list[:, 1], s=5)
        cont = ax.contourf(xy_mesh[0], xy_mesh[1], gmm_pred_list.reshape(len(xy_linspace[1]), len(xy_linspace[0])),
                           cmap="jet", alpha=0.5)
        ax.set_title("(D) Gaussian mixture model", y=-0.375)
        ax.set_xlabel("X position [m]")
        ax.set_ylabel("Y position [m]")
        ax.set_aspect("equal")
        plt.colorbar(cont, ax=ax, fraction=0.046, pad=0.04)

        # Plot
        plt.show()


if __name__ == "__main__":
    estimation = TaskSpaceDensityEstimation()
