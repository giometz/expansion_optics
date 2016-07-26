import skfmm as fmm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import skimage as ski
import skimage.morphology

class Selective_Sweep(object):

    def __init__(self, speeds=None, widths=None,
                 Lx=None, Ly = None, Nx=None, innoc_width=5):

        self.initial_speeds = speeds
        self.initial_widths = widths

        self.num_widths = self.initial_widths.shape[0]
        self.num_pops = np.unique(self.initial_speeds).shape[0]

        self.Lx = Lx
        self.Ly = Ly
        self.Nx = Nx
        self.Ny = int(round(float(Ly)/Lx * self.Nx))

        self.innoc_width = innoc_width

        xvalues = np.linspace(0, self.Lx, self.Nx, dtype=np.double)
        yvalues = np.linspace(0, self.Ly, self.Ny, dtype=np.double)

        self.dx = xvalues[1] - xvalues[0]
        self.dy = yvalues[1] - yvalues[0]

        self.X, self.Y = np.meshgrid(xvalues, yvalues)

        # Setup labels
        labels = np.unique(self.initial_speeds)
        self.pop_type = np.zeros_like(self.initial_speeds, dtype=np.int)
        for label_num, cur_label in enumerate(labels):
            label_positions = self.initial_speeds == cur_label
            self.pop_type[np.where(label_positions)[0]] = label_num

        self.num_labels = labels.shape[0]

        # Initialize the lattice
        self.lattice_mesh = -1 * np.ones((self.num_labels, self.Ny, self.Nx), dtype=np.int)
        self.speed_mesh = np.ones((self.num_labels, self.Ny, self.Nx), dtype=np.double)

        self.initialize_meshes()

        # Run the fast marching

        self.travel_times = None
        self.all_obstacles = None
        #self.run_travel_times()

    def initialize_meshes(self):

        sites_occupied = 0
        count = 0

        for cur_speed, cur_width in zip(self.initial_speeds, self.initial_widths):

            cur_pop = self.pop_type[count]

            num_to_occupy = int(cur_width * self.Ny)

            if count == self.num_widths - 1:  # As the sum of all should equal one, but may not due to FP
                num_to_occupy = self.Ny - sites_occupied

            self.speed_mesh[cur_pop] *= cur_speed

            self.lattice_mesh[cur_pop, sites_occupied:sites_occupied + num_to_occupy, 0:self.innoc_width + 1] = 1

            sites_occupied += num_to_occupy
            count += 1

    def run_travel_times(self, max_time, num_intervals):
        # create time intervals to recalculate obstacles
        times_to_run = np.linspace(0, max_time, num_intervals + 1)
        # Get rid of zero
        times_to_run = times_to_run[1:]

        # Include a dummy travel time background
        cur_travel_times = np.zeros((self.num_pops + 1,
                                     self.lattice_mesh.shape[1],
                                     self.lattice_mesh.shape[2]),
                                    dtype=np.double)

        cur_travel_times[-1, :, :] = 10*max_time
        self.all_obstacles = np.ones_like(self.lattice_mesh, dtype=np.bool) * False

        for cur_time in times_to_run:
            print cur_time
            for i in range(self.num_pops):
                cur_lattice = self.lattice_mesh[i]
                cur_obstacle = self.all_obstacles[i]
                # Mask the lattice by the obstacles...other strains
                cur_lattice = np.ma.MaskedArray(cur_lattice, cur_obstacle)

                cur_speed = self.speed_mesh[i]

                t = fmm.travel_time(cur_lattice, cur_speed, float(self.dx))

                cur_travel_times[i, :, :] = t

            # Based on the travel times, create obstacle masks for each strain
            non_background = cur_travel_times[0:self.num_pops, : , :]
            non_background[non_background > cur_time] = np.inf
            # If cur_travel_times = 0, you are in an obstacle
            non_background[non_background == 0] = np.inf

            expansion_history = np.nanargmin(cur_travel_times, axis=0)

            for i in range(self.num_pops): # Loop over strains, locate obstacles
                # Make sure nan's do not interfere with future
                not_current_strain = (expansion_history != i)
                not_background = (expansion_history != self.num_pops) # Dummy background strain

                self.all_obstacles[i, :, :] = not_current_strain & not_background

        self.travel_times = cur_travel_times[0:self.num_pops, :, :]

    def get_wall_df(self, ii, jj, expansion_size = 3):

        frozen_field= self.get_expansion_history()
        frozen_pops = np.zeros((frozen_field.shape[0], frozen_field.shape[1], self.num_pops), dtype=np.bool)
        for i in range(self.num_pops):
            frozen_pops[:, :, i] = (frozen_field == i)

        expanded_pops = np.zeros_like(frozen_pops)

        expander = ski.morphology.disk(expansion_size)

        for i in range(self.num_pops):
            cur_slice = frozen_pops[:, :, i]
            expanded_pops[:, :, i] = ski.morphology.binary_dilation(cur_slice, selem=expander)

        walls = expanded_pops[:, :, ii] * expanded_pops[:, :, jj]

        labeled_walls = ski.measure.label(walls, connectivity=2)

        df_list = []

        for cur_label in range(1, np.max(labeled_walls) + 1):
            r, c = np.where(labeled_walls == cur_label)

            x = self.X[r, c]
            y = self.Y[r, c]

            df = pd.DataFrame(data={'i': ii, 'j': jj, 'label_num': cur_label, 'x': x, 'y': y})

            # Group the df so that there is only one y for each x

            df = df.groupby('x').agg(np.mean).reset_index()

            df_list.append(df)
        return pd.concat(df_list)

    def get_expansion_shape(self, time):
        slice_at_time = self.travel_times.copy()
        slice_at_time[slice_at_time > time] = np.inf
        # Insert a dummy, background slice
        background = np.zeros_like(slice_at_time[0])
        background[...] = time + 9999
        background = np.array([background])

        all_slices = np.concatenate([slice_at_time, background])

        expansion_shape = np.nanargmin(all_slices, axis=0)

        # Label the expansion with appropriate labels
        return expansion_shape

    def get_expansion_history(self):
        expansion_history = self.travel_times.copy()
        expansion_history[expansion_history == 0] = np.inf
        expansion_history = np.argmin(expansion_history, axis=0) # Labeled by unique strain already

        return expansion_history

    def get_min_times(self):
        min_expansion_times = np.min(self.travel_times, axis=0)

        return min_expansion_times

class Radial_Selective_Sweep(Selective_Sweep):

    def __init__(self, center_x_frac=None, center_y_frac=None, **kwargs):
        self.phi = None
        self.radius = None

        self.center_x_frac = center_x_frac
        self.center_y_frac = center_y_frac

        self.center_X = None
        self.center_Y = None

        super(Radial_Selective_Sweep, self).__init__(**kwargs)

    # Innoc width is now the initial radius

    def initialize_meshes(self):

        # Convert to radial coordinates
        center_r = int(round(self.center_y_frac*self.Ny))
        center_c = int(round(self.center_x_frac*self.Nx))

        self.center_X = self.X[center_r, center_c]
        self.center_Y = self.Y[center_r, center_c]

        # Convert to polar coordinates
        self.DX = self.center_X - self.X
        self.DY = self.center_Y - self.Y

        self.phi = np.arctan2(self.DY, self.DX) + np.pi
        self.radius = np.sqrt(self.DX**2 + self.DY**2)

        phi_occupied = 0
        count = 0

        for cur_speed, cur_width in zip(self.initial_speeds, self.initial_widths):
            # Assumes cur_width is in degrees
            cur_pop = self.pop_type[count]

            phi_to_occupy = cur_width*(np.pi/180.)

            if count == self.num_widths - 1:  # As the sum of all should equal one, but may not due to FP
                phi_to_occupy = 2*np.pi - phi_occupied

            self.speed_mesh[cur_pop] *= cur_speed

            radius_mask = self.radius <= self.innoc_width
            phi_mask = (self.phi >= phi_occupied) & (self.phi <= phi_occupied + phi_to_occupy)

            self.lattice_mesh[cur_pop, radius_mask & phi_mask] = 1

            phi_occupied += phi_to_occupy
            count += 1

    def run_travel_times(self, max_time, num_intervals):
        super(Radial_Selective_Sweep, self).run_travel_times(max_time, num_intervals)
        self.travel_times[:, self.radius < self.innoc_width] = np.nan

    def get_wall_df(self, ii, jj, expansion_size = 2):

        frozen_field= self.get_expansion_history()
        frozen_pops = np.zeros((frozen_field.shape[0], frozen_field.shape[1], self.num_pops), dtype=np.bool)
        for i in range(self.num_pops):
            frozen_pops[:, :, i] = (frozen_field == i)

        expanded_pops = np.zeros_like(frozen_pops)

        expander = ski.morphology.disk(expansion_size)

        for i in range(self.num_pops):
            cur_slice = frozen_pops[:, :, i]
            expanded_pops[:, :, i] = ski.morphology.binary_dilation(cur_slice, selem=expander)

        walls = expanded_pops[:, :, ii] * expanded_pops[:, :, jj]

        labeled_walls = ski.measure.label(walls, connectivity=2)

        df_list = []

        num_bins = np.max(self.radius)/self.dx
        num_bins = int(num_bins)

        for cur_label in range(1, np.max(labeled_walls) + 1):
            r, c = np.where(labeled_walls == cur_label)

            radius = self.radius[r, c]
            phi = self.phi[r, c]

            df = pd.DataFrame(data={'i': ii, 'j': jj, 'label_num': cur_label, 'radius': radius, 'phi': phi})

            # Group the df so that there is only one phi for each radius
            min_radius = self.innoc_width
            max_radius = np.max(self.radius)

            bins = np.linspace(min_radius, max_radius, num_bins)

            cut = pd.cut(df['radius'], bins)
            mean_df = df.groupby(cut).agg(np.mean)

            #df = df.groupby('x').agg(np.mean).reset_index()

            df_list.append(mean_df)
        return pd.concat(df_list)
