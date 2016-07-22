import skfmm as fmm
import numpy as np
import pandas as pd

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

        # Initialize the lattice
        self.lattice_mesh = -1*np.ones((self.num_widths, self.Ny, self.Nx), dtype=np.int)
        self.speed_mesh = np.ones((self.num_widths, self.Ny, self.Nx), dtype=np.double)

        self.initialize_meshes()

        # Setup labels
        labels = np.unique(self.initial_speeds)
        self.pop_type = np.zeros_like(self.initial_speeds, dtype=np.int)
        for label_num, cur_label in enumerate(labels):
            label_positions = self.initial_speeds == cur_label
            self.pop_type[np.where(label_positions)[0]] = label_num

        # Run the fast marching

        self.travel_times = None
        #self.run_travel_times()

    def initialize_meshes(self):

        sites_occupied = 0
        count = 0

        for cur_speed, cur_width in zip(self.initial_speeds, self.initial_widths):

            num_to_occupy = int(cur_width * self.Ny)

            if count == self.num_widths - 1:  # As the sum of all should equal one, but may not due to FP
                num_to_occupy = self.Ny - sites_occupied

            self.speed_mesh[count] *= cur_speed

            self.lattice_mesh[count, sites_occupied:sites_occupied + num_to_occupy, 0:self.innoc_width] = 1

            sites_occupied += num_to_occupy
            count += 1

    def run_travel_times(self, max_time, num_intervals):
        # create time intervals to recalculate obstacles
        times_to_run = np.linspace(0, max_time, num_intervals + 1)
        # Get rid of zero
        times_to_run = times_to_run[1:]

        cur_travel_times = np.zeros_like(self.lattice_mesh, dtype=np.double)
        all_obstacles = np.ones_like(self.lattice_mesh, dtype=np.bool) * False

        for cur_time in times_to_run:
            # Assumes innoculation width ~ 0
            for i in range(self.lattice_mesh.shape[0]):
                cur_lattice = self.lattice_mesh[i]
                cur_obstacle = all_obstacles[i]
                # Mask the lattice by the obstacles...other strains
                cur_lattice = np.ma.MaskedArray(cur_lattice, cur_obstacle)

                cur_speed = self.speed_mesh[i]

                t = fmm.travel_time(cur_lattice, cur_speed, float(self.dx))

                cur_travel_times[i, :, :] = t

            # Based on the travel times, create obstacle masks for each strain
            cur_travel_times[cur_travel_times > cur_time] = np.nan
            expansion_history = np.argmin(cur_travel_times, axis=0)

            for i in range(self.lattice_mesh.shape[0]): # Loop over strains, locate obstacles
                all_obstacles[i, :, :] = (expansion_history != i)

        self.travel_times = cur_travel_times

    def get_wall_df(self, i, j, tolerance = 0.5):
        diff = np.abs(self.travel_times[i] - self.travel_times[j])
        wall = diff < tolerance

        r, c = np.where(wall)

        xpos = self.X[r, c]
        ypos = self.Y[r, c]

        wall_df = pd.DataFrame(data={'x': xpos, 'y': ypos})
        filtered_df = wall_df.groupby(['x']).agg(np.mean).reset_index()

        return filtered_df

    def get_expansion_shape(self, time):
        slice_at_time = self.travel_times.copy()
        slice_at_time[slice_at_time > time] = np.inf
        # Insert a dummy, background slice
        background = np.zeros_like(slice_at_time[0])
        background[...] = time + 9999
        background = np.array([background])

        all_slices = np.concatenate([slice_at_time, background])

        expansion_shape = np.nanargmin(all_slices, axis=0)

        cur_pop_types = self.pop_type.copy()
        cur_pop_types = np.append(cur_pop_types, np.max(cur_pop_types) + 1)

        # Label the expansion with appropriate labels
        expansion_shape_labeled = cur_pop_types[expansion_shape]

        return expansion_shape_labeled

    def get_expansion_history(self):
        expansion_history = np.argmin(self.travel_times, axis=0)
        expansion_history_labeled = self.pop_type[expansion_history]

        return expansion_history_labeled

    def get_min_times(self):
        min_expansion_times = np.min(self.travel_times, axis=0)

        return min_expansion_times