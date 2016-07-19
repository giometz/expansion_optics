import skfmm as fmm
import numpy as np

class Selective_Sweep(object):

    def __init__(self, initial_speeds=None, initial_widths=None,
                 Lx=None, Ly = None, Nx=None, innoc_width=5):

        self.initial_speeds = np.array([1., 1.1, 1., 1.15, 1.])
        self.initial_widths = np.array([.2, .2, .2, .2, .2])

        self.num_widths = initial_widths.shape[0]
        self.num_pops = np.unique(initial_speeds).shape[0]

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
        self.lattice_mesh = np.zeros((self.num_widths, self.Ny, self.Nx), dtype=np.int)
        self.speed_mesh = np.ones((self.num_widths, self.Ny, self.Nx), dtype=np.double)

        self.initialize_meshes()

        self.travel_times = None
        self.run_travel_times()

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

    def run_travel_times(self):
        t_list = []
        for i in range(self.lattice_mesh.shape[0]):
            cur_lattice = self.lattice_mesh[i]
            cur_speed = self.speed_mesh[i]

            print cur_lattice.shape
            print cur_speed.shape
            print self.dx

            t = fmm.travel_time(cur_lattice, cur_speed, float(self.dx))
            t_list.append(t)

        self.travel_times = np.array(t_list)