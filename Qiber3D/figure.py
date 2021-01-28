import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm, lines, colorbar, colors
from Qiber3D import helper, config
import logging


class Figure:
    """
    Generate matplotlib figures

    :param network: a :class:`Qiber3D.Network`
    """

    def __init__(self, network):
        self.logger = helper.get_logger()
        self.network = network

    def z_drop(self, out_path='.', overwrite=False):
        """
        Plot the z_drop correction.

        :param out_path: file or folder path where to save the network, if `None` show the plot.
        :type out_path: str, Path
        :param bool overwrite: allow file overwrite
        :return: path to saved file
        :rtype: Path
        """
        if self.network.extractor_data is None:
            self.logger.warn('Not Available! The network was not initialized with an image.')
            return
        data = self.network.extractor_data['processing_data']['z_drop']

        fig, ax = plt.subplots()
        ax.plot(data['x'], data['y'], ':', color='orange', lw=2, label='original')
        ax.plot(data['x'], data['z_fit'], color='black', lw=2, label='z-fit')
        ax.plot(data['x'], data['y_corrected'], color='darkblue', lw=2, label='corrected')
        ax.set_xlabel('z')
        ax.set_ylabel('average intensity')
        ax.legend()
        if out_path is not None:
            prefix = f'z-drop_'
            out_path, needs_unlink = helper.out_path_check(out_path, network=self.network, prefix=prefix,
                                                           suffix=config.figure.format,
                                                           overwrite=overwrite, logger=self.logger)
            if out_path is None:
                return

        if out_path is not None:
            fig.tight_layout()
            fig.savefig(out_path, dpi=config.figure.dpi)
            self.logger.info(f'Figure saved at: {str(out_path.absolute())}')
            return out_path
        else:
            fig.show()

    def histogram(self, out_path='.', overwrite=False, attribute='length', bins=None):
        """
        Plot a histogram for a specific attribute of :class:`Qiber3D.Fiber`.

        :param out_path: file or folder path where to save the network, if `None` show the plot.
        :type out_path: str, Path
        :param bool overwrite: allow file overwrite
        :param str attribute: one of ['length', 'volume', 'average_diameter', 'average_radius', 'cylinder_radius']
        :param int bins: number of bins
        :return: path to saved file
        :rtype: Path
        """
        if attribute not in ['length', 'volume', 'average_diameter', 'average_radius', 'cylinder_radius']:
            logging.error(f'Histogram figure mode "{attribute}" is not supported')
            return None
        data = [getattr(fiber, attribute) for fiber in self.network.fiber.values()]
        fig, ax = plt.subplots()
        ax2 = ax.twinx()
        if bins is None:
            bins = 'auto'
        ax.hist(data, bins=bins, density=True, cumulative=False)
        y, x = np.histogram(data, bins=300)
        ax2.plot(x[1:]-np.diff(x), np.cumsum(y)/np.sum(y), c='black')
        ax.set_xlabel(attribute.replace('_', ' '))
        ax.set_ylabel('probability')
        ax2.set_ylabel('cumulative probability')
        if out_path is not None:
            prefix = f'histogram_{attribute}_'
            out_path, needs_unlink = helper.out_path_check(out_path, network=self.network, prefix=prefix,
                                                           suffix=config.figure.format,
                                                           overwrite=overwrite, logger=self.logger)
            if out_path is None:
                return

        if out_path is not None:
            fig.tight_layout()
            fig.savefig(out_path, dpi=config.figure.dpi)
            self.logger.info(f'Figure saved at: {str(out_path.absolute())}')
        else:
            fig.show()

    def directions(self, out_path='.', overwrite=False, grid=True, color_map='RdYlGn', mode='fine', bins=None):
        """
        Plot the dominat directions of a :class:`Qiber3D.Network`

        :param out_path: file or folder path where to save the network, if `None` show the plot.
        :type out_path: str, Path
        :param bool overwrite: allow file overwrite
        :param bool grid: show a grid
        :param str color_map: name of a matplotlib color map
        :param str mode: `'fine'`: use the vectors between every point,
                         `'main'`: use the vectors between the start and end point of a segment
        :param int bins: number of bins
        :return: path to saved file
        :rtype: Path
        """
        if mode == 'fine':
            data = self.network.spherical_vector
        elif mode == 'main':
            data = self.network.spherical_direction
        else:
            self.network.logging.error(f'Directions figure mode "{mode}" is not supported')
            return

        if bins is None:
            bins = int(np.ceil(np.sqrt(len(data)))) + 1

        color_map = cm.get_cmap(color_map)

        fig = plt.figure()
        ax1 = plt.subplot2grid((24, 5), (0, 0), rowspan=20, colspan=5, projection='polar')
        ax2 = plt.subplot2grid((24, 5), (23, 1), colspan=3)

        ax1.set_theta_zero_location("N")
        ax1.set_theta_direction(-1)
        ax1.set_ylim((0, np.deg2rad(90)))
        theta_n = int(np.ceil(np.sqrt(bins) / 2.0))

        segment_area_goal = 2 * np.pi / bins
        plot_list = []

        theta_list = np.linspace(0,  0.5 * np.pi, theta_n + 1)
        for n in range(theta_n):
            theta_start = theta_list[n]
            theta_end = theta_list[n + 1]
            h = np.cos(theta_start) - np.cos(theta_end)
            phi_n = int(np.ceil(2 * np.pi / (segment_area_goal / h)))
            delta_phi = 2 * np.pi / phi_n
            phi_edges = np.linspace(-np.pi, np.pi, phi_n + 1)
            phi_center = phi_edges[:-1] + np.diff(phi_edges) / (0.5 * np.pi)
            segment_area = delta_phi * h
            histogram_data, te, pe = np.histogram2d(data[:, 1], data[:, 2],
                                                    bins=((theta_start, theta_end), phi_edges),
                                                    weights=data[:, 0])
            for k, phi in enumerate(phi_center):
                plot_list.append([phi, theta_end - theta_start, delta_phi, theta_start,
                                  histogram_data[0, k] / segment_area, segment_area])

        if out_path is not None:
            prefix = f'directions_{mode}_{len(plot_list)}_'
            out_path, needs_unlink = helper.out_path_check(out_path, network=self.network, prefix=prefix,
                                                           suffix=config.figure.format,
                                                           overwrite=overwrite, logger=self.logger)
            if out_path is None:
                return

        plot_list = np.array(plot_list)
        plot_max, plot_min = np.max(plot_list[:, 4]), np.min(plot_list[:, 4])
        plot_average = sum(data[:, 0]) / (2 * np.pi)
        plot_list[:, 4] = plot_list[:, 4] / plot_average
        norm = colors.TwoSlopeNorm(vcenter=1, vmin=plot_min/(self.network.length/(2*np.pi)),
                                   vmax=plot_max/(self.network.length/(2*np.pi)))
        color_list = cm.ScalarMappable(norm=norm, cmap=color_map)

        for work in plot_list:
            ax1.bar(work[0], work[1], width=work[2], bottom=work[3], color=color_list.to_rgba(work[4]),
                    linewidth=0, zorder=-5)

        if grid:
            xticks_a = np.deg2rad(np.linspace(0, 90, 7))
            xticks_b = np.deg2rad(np.linspace(90, 360, 10))
            x_tick_res = 100
            xticks_all = np.hstack((xticks_a[:-1], xticks_b[:-1]))
            for tick_pos in xticks_b:
                tick_line = lines.Line2D((tick_pos, tick_pos), (0, np.pi * 0.52),
                                         color=config.figure.grid_color,
                                         lw=config.figure.grid_lw)
                tick_line.set_clip_on(False)
                ax1.add_line(tick_line)

            for tick_pos in xticks_a[1:-1]:
                tick_line = lines.Line2D((tick_pos, tick_pos), (np.pi * 0.5-tick_pos, np.pi * 0.5),
                                         color=config.figure.grid_color,
                                         lw=config.figure.grid_lw)
                tick_line.set_clip_on(False)
                ax1.add_line(tick_line)

            for tick_pos in xticks_a[1:-1]:
                x_tick = np.linspace(tick_pos, -np.deg2rad(270), x_tick_res)
                ax1.plot(x_tick, [0.5*np.pi-tick_pos]*x_tick_res, zorder=100,
                         color=config.figure.grid_color,
                         lw=config.figure.grid_lw)
        else:
            xticks_all = np.deg2rad(np.linspace(0, 360, 13))[:-1]

        for tick_pos in xticks_all:
            tick_line = lines.Line2D((tick_pos, tick_pos), (np.pi * 0.48, np.pi * 0.52),
                                     color='black', lw=0.75, zorder=200)
            tick_line.set_clip_on(False)
            ax1.add_line(tick_line)

        ax1.set_xticks(xticks_all)
        ax1.grid(False)
        ax1.tick_params(axis="x", direction="out", length=8)
        ax1.set_yticks([])
        ax1.set_yticklabels([])
        cb1 = colorbar.ColorbarBase(ax2, cmap=color_map, norm=norm,
                                    orientation='horizontal')
        cb1.set_label(r'$\overline{\varrho}$ (density / average density)')

        if out_path is not None:
            fig.tight_layout()
            fig.savefig(out_path, dpi=config.figure.dpi)
            self.logger.info(f'Figure saved at: {str(out_path.absolute())}')
        else:
            fig.show()

        self.logger.debug(f'Figure parameters: bins|{len(plot_list)}; data points|{len(data)}; '
                          f'average|{plot_average:.3f}; max|{plot_max:.3f} ({plot_max/plot_average:.3f}); '
                          f'min|{plot_min:.3f} ({plot_min/plot_average:.3f})')
