#! /usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm


# Estimate the probability density from the task-space sample set
class SVMKernelComparison(object):
    def __init__(self):
        # Setup plot
        fig = plt.figure()
        fig.set_size_inches(8, 18)
        plt.subplots_adjust(wspace=0.50, hspace=0.70)
        plt.rcParams["font.size"] = 16
        plt.rcParams["image.origin"] = "lower"

        # To avoid Type 3 fonts
        # See http://phyletica.org/matplotlib-fonts/
        plt.rcParams['pdf.fonttype'] = 42
        plt.rcParams['ps.fonttype'] = 42

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
        padding_rate = 0.5
        pos_max_with_margin = pos_max + padding_rate * pos_range
        pos_min_with_margin = pos_min - padding_rate * pos_range
        xy_linspace = [np.linspace(pos_min_with_margin[i], pos_max_with_margin[i], 100) for i in range(2)]
        xy_mesh = np.meshgrid(xy_linspace[0], xy_linspace[1])
        xy_mesh_plot = np.vstack([xy_mesh[0].ravel(), xy_mesh[1].ravel()])

        # Plot joint-space sampling
        ax = fig.add_subplot(4, 2, 1)
        ax.scatter(joint_pos_list[:, 0], joint_pos_list[:, 1], s=7.5)
        ax.set_title("(A) Joint-space samples", y=-0.55)
        ax.set_xlabel("Joint1 position [rad]")
        ax.set_ylabel("Joint2 position [rad]")
        ax.set_aspect("equal")

        # Plot task-space sampling
        ax = fig.add_subplot(4, 2, 2)
        ax.scatter(body_pos_list[:, 0], body_pos_list[:, 1], s=7.5)
        ax.set_title("(B) Task-space samples", y=-0.55)
        ax.set_xlabel("X position [m]")
        ax.set_ylabel("Y position [m]")
        ax.set_aspect("equal")

        # One-class SVM
        def kernel_func(x1, x2, kernel_type="SquaredExponential"):
            x1 = x1.reshape(x1.shape[0], -1, x1.shape[1])
            x1_grid = x1.repeat(x2.shape[0], axis=1)
            x2 = x2.reshape(x2.shape[0], -1, x2.shape[1])
            x2_grid = np.transpose(x2.repeat(x1.shape[0], axis=1), (1, 0, 2))
            norm_grid = np.linalg.norm(x1_grid - x2_grid, axis=2)

            # https://www.mathworks.com/help/stats/kernel-covariance-function-options.html
            # https://www.cs.toronto.edu/~duvenaud/cookbook/
            if kernel_type == "SquaredExponential": # RBF
                l = 0.15
                return np.exp(-1 * norm_grid**2 / (2 * l**2))
            elif kernel_type == "RationalQuadratic":
                alpha = 1.
                l = 0.15
                return (1 + norm_grid**2 / (2 * alpha * l**2))**(-alpha)
            elif kernel_type == "Exponential":
                l = 0.15
                return np.exp(-1 * norm_grid / l)
            elif kernel_type == "Matern3":
                l = 0.15
                return (1 + 3.**0.5 * norm_grid / l) * np.exp(-1 * 3.**0.5 * norm_grid / l)
            elif kernel_type == "Matern5":
                l = 0.15
                return (1 + 5.**0.5 * norm_grid / l + 5. * norm_grid**2 / (3. * l**2)) * np.exp(-1 * 5.**0.5 * norm_grid / l)
            elif kernel_type == "Abs":
                l = 0.15
                return -1 * norm_grid / l
            elif kernel_type == "Huber":
                l = 0.15
                delta = 5.0
                return np.where(norm_grid < l * delta,
                                -0.5 * (norm_grid / l)**2,
                                -1 * delta * (norm_grid / l - 0.5 * delta))
            elif kernel_type == "RationalQuadratic+Abs":
                alpha = 1.
                l_rq = 0.15
                l_abs = 0.15
                w_abs = 0.01
                return (1 + norm_grid**2 / (2 * alpha * l_rq**2))**(-alpha) + w_abs * -1 * norm_grid / l_abs
            else:
                print("Invalid kernel_type: {}".format(kernel_type))
                return 0.

        kernel_type_list = [
            "SquaredExponential",
            "RationalQuadratic",
            "Exponential",
            "Abs",
            "Huber",
            "RationalQuadratic+Abs"]
        for i, kernel_type in enumerate(kernel_type_list):
            ocsvm = svm.OneClassSVM(nu=0.05, kernel=lambda x1, x2: kernel_func(x1, x2, kernel_type=kernel_type))
            ocsvm.fit(body_pos_list[:,:2])
            ocsvm_pred_list = ocsvm.decision_function(xy_mesh_plot.T)
            ax = fig.add_subplot(4, 2, 3+i)
            ax.scatter(body_pos_list[:, 0], body_pos_list[:, 1], s=5)
            cont = ax.contourf(xy_mesh[0], xy_mesh[1], ocsvm_pred_list.reshape(len(xy_linspace[1]), len(xy_linspace[0])),
                               cmap="jet", alpha=0.5)
            ax.set_title("(C{}) {}".format(i, kernel_type), y=-0.55)
            ax.set_xlabel("X position [m]")
            ax.set_ylabel("Y position [m]")
            ax.set_aspect("equal")
            plt.colorbar(cont, ax=ax, fraction=0.046, pad=0.04)

        # Plot
        plt.show()


if __name__ == "__main__":
    comparison = SVMKernelComparison()
