#! /usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from scipy import ndimage
from sklearn import mixture
from sklearn import svm


# Estimate the probability density from the task-space sample set
class TaskSpaceDensityEstimation(object):
    def __init__(self):
        # Setup plot
        fig = plt.figure()
        fig.set_size_inches(12, 18)
        plt.subplots_adjust(wspace=0.3, hspace=0.55)
        plt.rcParams["font.size"] = 16
        plt.rcParams["image.origin"] = "lower"

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
        ax = fig.add_subplot(321)
        ax.scatter(joint_pos_list[:, 0], joint_pos_list[:, 1], s=7.5)
        ax.set_title("(A) Joint-space sampling", y=-0.4)
        ax.set_xlabel("Joint1 position [rad]")
        ax.set_ylabel("Joint2 position [rad]")
        ax.set_aspect("equal")

        # Plot task-space sampling
        ax = fig.add_subplot(322)
        ax.scatter(body_pos_list[:, 0], body_pos_list[:, 1], s=7.5)
        ax.set_title("(B) Task-space sampling", y=-0.4)
        ax.set_xlabel("X position [m]")
        ax.set_ylabel("Y position [m]")
        ax.set_aspect("equal")

        # KDE
        kde = gaussian_kde(body_pos_list[:,:2].T)
        kde_pred_list = kde.evaluate(xy_mesh_plot)
        ax = fig.add_subplot(323)
        ax.scatter(body_pos_list[:, 0], body_pos_list[:, 1], s=5)
        cont = ax.contourf(xy_mesh[0], xy_mesh[1], kde_pred_list.reshape(len(xy_linspace[1]), len(xy_linspace[0])),
                           cmap="jet", alpha=0.5)
        ax.set_title("(C) Kernel density estimation", y=-0.4)
        ax.set_xlabel("X position [m]")
        ax.set_ylabel("Y position [m]")
        ax.set_aspect("equal")
        plt.colorbar(cont, ax=ax, fraction=0.046, pad=0.04)

        # GMM
        gmm = mixture.GaussianMixture(n_components=4, covariance_type="full")
        gmm.fit(body_pos_list[:,:2])
        gmm_pred_list = np.exp(gmm.score_samples(xy_mesh_plot.T))
        ax = fig.add_subplot(324)
        ax.scatter(body_pos_list[:, 0], body_pos_list[:, 1], s=5)
        cont = ax.contourf(xy_mesh[0], xy_mesh[1], gmm_pred_list.reshape(len(xy_linspace[1]), len(xy_linspace[0])),
                           cmap="jet", alpha=0.5)
        ax.set_title("(D) Gaussian mixture model", y=-0.4)
        ax.set_xlabel("X position [m]")
        ax.set_ylabel("Y position [m]")
        ax.set_aspect("equal")
        plt.colorbar(cont, ax=ax, fraction=0.046, pad=0.04)

        # One-class SVM
        ocsvm = svm.OneClassSVM(nu=0.05, kernel="rbf", gamma=20)
        ocsvm.fit(body_pos_list[:,:2])
        ocsvm_pred_list = ocsvm.decision_function(xy_mesh_plot.T)
        ax = fig.add_subplot(325)
        ax.scatter(body_pos_list[:, 0], body_pos_list[:, 1], s=5)
        cont = ax.contourf(xy_mesh[0], xy_mesh[1], ocsvm_pred_list.reshape(len(xy_linspace[1]), len(xy_linspace[0])),
                           cmap="jet", alpha=0.5)
        ax.set_title("(E) One-class SVM", y=-0.4)
        ax.set_xlabel("X position [m]")
        ax.set_ylabel("Y position [m]")
        ax.set_aspect("equal")
        plt.colorbar(cont, ax=ax, fraction=0.046, pad=0.04)

        # Signed distance function
        pos_range_with_margin = pos_max_with_margin - pos_min_with_margin
        img_resolution = 0.01 # [m]

        def idxToPos(idx):
            return img_resolution * np.array(idx) + pos_min_with_margin[:2]

        # This is specific to Simple2DoFManipulator.urdf.xacro
        def posToBool(pos):
            pos_norm = np.linalg.norm(pos)
            return ((pos[0] >= 0) \
                    and (pos[1] >= 0) \
                    and (0.5 <= pos_norm <= 1.5) \
                    and (0.5 <= np.linalg.norm(pos - np.array([1.0, 0.0])))) \
                    or (np.linalg.norm(pos - np.array([0.0, 1.0])) <= 0.5)

        # Generate bool image
        bool_img = np.zeros((pos_range_with_margin / img_resolution).astype(int)[:2], dtype=bool)
        for ix in range(bool_img.shape[0]):
            for iy in range(bool_img.shape[1]):
                bool_img[ix, iy] = posToBool(idxToPos([ix, iy]))

        # Generate SDF image
        # See https://github.com/pmneila/morphsnakes/issues/5#issuecomment-203014643
        sdf_img = img_resolution * np.where(
            bool_img,
            ndimage.distance_transform_edt(bool_img) - 0.5,
            - (ndimage.distance_transform_edt(1 - bool_img) - 0.5))

        ax = fig.add_subplot(326)
        ax.scatter(body_pos_list[:, 0], body_pos_list[:, 1], s=5)
        im = ax.imshow(
            sdf_img.T,
            cmap="jet",
            alpha=0.8,
            extent=[pos_min_with_margin[0], pos_max_with_margin[0], pos_min_with_margin[1], pos_max_with_margin[1]])
        ax.set_title("(F) Signed distance function", y=-0.4)
        ax.set_xlabel("X position [m]")
        ax.set_ylabel("Y position [m]")
        ax.set_aspect("equal")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        # Plot
        plt.show()


if __name__ == "__main__":
    estimation = TaskSpaceDensityEstimation()
