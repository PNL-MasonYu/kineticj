# %%
from netCDF4 import Dataset
import libconf
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator, interp1d
import os, shutil, io
from subprocess import call
import time

KJPATH = os.environ['KINETICJ_ROOT']
KJDATAPATH = os.environ['KINETICJ_DATA']

# Number of axial coordinates to use (number of r in KineticJ runs? Should be >> nXGrid)
n_z = 5000

# r coordinates list with refinement near the surface
#r_n = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 58, 60, 62, 65, 67, 70, 72, 75, 78, 80, 82, 84, 86, 88, 89, 90]
r_n = [1, 5, 7, 10, 12, 15, 17, 20, 23, 25, 28, 30, 35, 40, 45, 50, 55, 58, 60, 62, 65, 67, 70, 72, 75]

# r coordinates list with linear 5 mm spacing
#r_n = np.linspace(0, 75, 16, dtype=int)

class COMSOL_data:

    freq = 26e6
    
    def __init__(self, comsol_output_dir :str):
        # RF E-field and B-fields
        self.r_coords, self.z_coords, self.Er_real = self.read_comsol_field(comsol_output_dir+"/Er_real.txt")
        self.Er_imag = self.read_comsol_field(comsol_output_dir+"/Er_imag.txt")[2]
        self.Ephi_real = self.read_comsol_field(comsol_output_dir+"/Ephi_real.txt")[2]
        self.Ephi_imag = self.read_comsol_field(comsol_output_dir+"/Ephi_imag.txt")[2]
        self.Ez_real = self.read_comsol_field(comsol_output_dir+"/Ez_real.txt")[2]
        self.Ez_imag = self.read_comsol_field(comsol_output_dir+"/Ez_imag.txt")[2]
        self.Br_real = self.read_comsol_field(comsol_output_dir+"/Br_real.txt")[2]
        self.Br_imag = self.read_comsol_field(comsol_output_dir+"/Br_imag.txt")[2]
        self.Bphi_real = self.read_comsol_field(comsol_output_dir+"/Bphi_real.txt")[2]
        self.Bphi_imag = self.read_comsol_field(comsol_output_dir+"/Bphi_imag.txt")[2]
        self.Bz_real = self.read_comsol_field(comsol_output_dir+"/Bz_real.txt")[2]
        self.Bz_imag = self.read_comsol_field(comsol_output_dir+"/Bz_imag.txt")[2]

        self.Br = self.read_comsol_field(comsol_output_dir+"/static_Br.txt")[2]
        self.Bz = self.read_comsol_field(comsol_output_dir+"/static_Bz.txt")[2]
        self.ne = self.read_comsol_field(comsol_output_dir+"/ne.txt")[2]


    def read_comsol_field(self, path):
        """
        Read in COMSOL output of regular grid data text files (fields) as a 2D array, with r and z coordinates
        Read the r and z coordinate vectors first
        Then iterate through the data line by line
        Imaginary and Real parts are read separately
        """
        with open(path) as f:
            for n in range(9):
                f.readline()
            r_coords = np.float64(f.readline().split())
            z_coords = np.float64(f.readline().split())
            f.readline()
            f.readline()
            data = np.zeros((len(r_coords), len(z_coords)), np.float64)
            for z in range(len(z_coords)):
                line_data = np.float64(f.readline().split())
                np.nan_to_num(line_data, copy=False, nan=0)
                data[:, z] = line_data
            
        return r_coords, z_coords, data

    def plot_comsol_field(self, data_list:list):
        """
        Plot the field data read from COMSOL
        """
        for data_name in data_list:
            data = getattr(self, data_name)
            fig, ax = plt.subplots(figsize=(12, 6))
            rmin, rmax = min(self.r_coords), max(self.r_coords)
            zmin, zmax = min(self.z_coords), max(self.z_coords)
            im = ax.imshow(data, cmap="viridis", interpolation="nearest", extent=(zmin, zmax, rmax, rmin))
            fig.colorbar(im, orientation="horizontal")
            ax.set_xlabel("z (m)")
            ax.set_ylabel("r (m)")
            ax.set_title(data_name)
            fig.tight_layout()
            plt.show()
        return

    def export_to_dispersionnering(self, Te=600, r=0):
        """
        Export data to dispersionnering inputs in columns of R(m), Z(m), Psi(normalized to 1 at LCFS), ne (m^-3), Te (eV) and Bz(T)
        """
        npt = len(self.r_coords)
        print(npt)
        data = np.zeros((npt, 6))
        for n in range(npt):
            data[n, 0] = r
            z = self.z_coords[n]
            data[n, 1] = z
            data[n, 2] = r
            data[n, 3] = self.ne[r, n]
            data[n, 4] = Te
            data[n, 5] = self.Bz[r, n]
        np.savetxt("./WHAM_profiles_1Daxial_COMSOL.csv", data, "%.5f", delimiter=",", header="R(m), Z(m), Psi(normalized to 1 at LCFS), ne (m^-3), Te (eV), Bz(T)")
        return

class KineticJ_1D:
    def __init__(self, run_dir, min_z=-0.8):
        data_dict = {}

        # Try to run KJ again if the output file was not found
        if not os.path.exists(run_dir + "/" + dir + "/output/jP2.nc"):
            run_kineticj(run_dir + "/" + dir)
        nc = Dataset(run_dir + "/" + dir + "/output/jP2.nc", "r", format="NETCDF4")
        z_coords = np.add(nc.variables['x'][:], min_z)
                
        self.jx_re=np.zeros(len(z_coords))
        self.jx_im=np.zeros(len(z_coords))
        self.jy_re=np.zeros(len(z_coords))
        self.jy_im=np.zeros(len(z_coords))
        self.jz_re=np.zeros(len(z_coords))
        self.jz_im=np.zeros(len(z_coords))

        self.jx_re[:] = nc.variables['j1xc_re'][:]
        self.jx_im[:] = nc.variables['j1xc_im'][:]
        self.jy_re[:] = nc.variables['j1yc_re'][:]
        self.jy_im[:] = nc.variables['j1yc_im'][:]
        self.jz_re[:] = nc.variables['j1zc_re'][:]
        self.jz_im[:] = nc.variables['j1zc_im'][:]

        nc.close()
        self.jx_re_interp = interp1d(z_coords, self.jx_re)
        self.jx_im_interp = interp1d(z_coords, self.jx_im)
        self.jy_re_interp = interp1d(z_coords, self.jy_re)
        self.jy_im_interp = interp1d(z_coords, self.jy_im)
        self.jz_re_interp = interp1d(z_coords, self.jz_re)
        self.jz_im_interp = interp1d(z_coords, self.jz_im)
        self.z_coords = z_coords
        self.min_z = min_z
        self.max_z = max(z_coords)
    
    def write_to_file(self, file_dir= KJDATAPATH + "/COMSOL_input", iter=0):
         # Output to COMSOL inputs file
         # It overwrites any outputs from a different simulation, but probably fine because the data is all in the KJ run directories
        z_coords = np.linspace(self.min_z, self.max_z, 2000)
        output = []
        for z in z_coords:
            output.append([z, self.jx_re_interp(z), self.jx_im_interp(z), 
                              self.jy_re_interp(z), self.jy_im_interp(z), 
                              self.jz_re_interp(z), self.jz_im_interp(z)])

        np.savetxt(file_dir + '/kj_output_iter' + str(iter) +'.txt', output)
        np.savetxt(file_dir + '/kj_output_iter_latest.txt', output)

class KineticJ_result:

    def __init__(self, run_dir):
        r_coords = []
        data_dict = {}
        for dir in os.listdir(run_dir):
            if dir.split("_")[0] == "r":
                r_coord = float(dir.split("_")[1]) * 1e-3
                r_coords.append(r_coord)
                # Try to run KJ again if the output file was not found
                if not os.path.exists(run_dir + "/" + dir + "/output/jP2.nc"):
                    run_kineticj(run_dir + "/" + dir)
                nc = Dataset(run_dir + "/" + dir + "/output/jP2.nc", "r", format="NETCDF4")
                z_coords = np.add(nc.variables['x'][:], -0.8)

                # Maybe should compute jr, jphi here directly?
                # Should use Pandas to do this honestly
                data_dict[r_coord] = {}
                data_dict[r_coord]["jx_re"] = nc.variables['j1xc_re'][:]
                data_dict[r_coord]["jx_im"] = nc.variables['j1xc_im'][:]
                data_dict[r_coord]["jy_re"] = nc.variables['j1yc_re'][:]
                data_dict[r_coord]["jy_im"] = nc.variables['j1yc_im'][:]
                data_dict[r_coord]["jz_re"] = nc.variables['j1zc_re'][:]
                data_dict[r_coord]["jz_im"] = nc.variables['j1zc_im'][:]
                nc.close()
        print(r_coords)
        r_coords.sort()
        self.jx_re=np.zeros((len(r_coords), len(z_coords)))
        self.jx_im=np.zeros((len(r_coords), len(z_coords)))
        self.jy_re=np.zeros((len(r_coords), len(z_coords)))
        self.jy_im=np.zeros((len(r_coords), len(z_coords)))
        self.jz_re=np.zeros((len(r_coords), len(z_coords)))
        self.jz_im=np.zeros((len(r_coords), len(z_coords)))
        for n in range(len(r_coords)):
            r_coord = r_coords[n]
            self.jx_re[n][:] = data_dict[r_coord]["jx_re"] + 1e-10
            self.jx_im[n][:] = data_dict[r_coord]["jx_im"] + 1e-10
            self.jy_re[n][:] = data_dict[r_coord]["jy_re"]
            self.jy_im[n][:] = data_dict[r_coord]["jy_im"]
            self.jz_re[n][:] = data_dict[r_coord]["jz_re"]
            self.jz_im[n][:] = data_dict[r_coord]["jz_im"]
        self.jx_re_interp = RegularGridInterpolator([r_coords, z_coords], self.jx_re, bounds_error=False, fill_value=None)
        self.jx_im_interp = RegularGridInterpolator([r_coords, z_coords], self.jx_im, bounds_error=False, fill_value=None)
        self.jy_re_interp = RegularGridInterpolator([r_coords, z_coords], self.jy_re, bounds_error=False, fill_value=None)
        self.jy_im_interp = RegularGridInterpolator([r_coords, z_coords], self.jy_im, bounds_error=False, fill_value=None)
        self.jz_re_interp = RegularGridInterpolator([r_coords, z_coords], self.jz_re, bounds_error=False, fill_value=None)
        self.jz_im_interp = RegularGridInterpolator([r_coords, z_coords], self.jz_im, bounds_error=False, fill_value=None)
        self.r_coords = r_coords
        self.z_coords = z_coords

    def plot(self):
        fig, (ax0,ax1,ax2,ax3,ax4,ax5) = plt.subplots(nrows=6,figsize=(7, 7))
        z, r = np.linspace(min(self.z_coords), max(self.z_coords), 400), np.linspace(0, 0.075, 50)
        zz, rr = np.meshgrid(r, z, indexing='ij')
        ax0.imshow(self.jx_re_interp((zz, rr)), extent=(min(z), max(z), max(r), min(r)))
        ax0.set_title("Jx_re")
        ax1.imshow(self.jx_im_interp((zz, rr)), extent=(min(z), max(z), max(r), min(r)))
        ax1.set_title("Jx_im")
        ax2.imshow(self.jy_re_interp((zz, rr)), extent=(min(z), max(z), max(r), min(r)))
        ax2.set_title("Jy_re")
        ax3.imshow(self.jy_im_interp((zz, rr)), extent=(min(z), max(z), max(r), min(r)))
        ax3.set_title("Jy_im")
        ax4.imshow(self.jz_re_interp((zz, rr)), extent=(min(z), max(z), max(r), min(r)))
        ax4.set_title("Jz_re")
        ax5.imshow(self.jz_im_interp((zz, rr)), extent=(min(z), max(z), max(r), min(r)))
        ax5.set_title("Jz_im")

        plt.tight_layout()
        plt.show(block=False)

    def write_to_file(self, file_dir= KJDATAPATH + "/COMSOL_input", iter=0):
         # Output to COMSOL inputs file
         # It overwrites any outputs from a different simulation, but probably fine because the data is all in the KJ run directories
        z_coords, r_coords = np.linspace(-0.8, 0.8, 2000), np.linspace(0, 0.075, 300)
        output = []
        for z in z_coords:
            for r in r_coords:
                output.append([r, z, self.jx_re_interp((z, r)), self.jx_im_interp((z, r)), 
                                     self.jy_re_interp((z, r)), self.jy_im_interp((z, r)), 
                                     self.jz_re_interp((z, r)), self.jz_im_interp((z, r))])

        np.savetxt(file_dir + '/kj_output_iter' + str(iter) +'.txt', output)
        np.savetxt(file_dir + '/kj_output_iter_latest.txt', output)

def make_run_directory(run_dir, nr:int, n_z, comsol:COMSOL_data, rmdir=True):
    """
    Create the KineticJ run directory structure and put in the input nc                                                   
    """
    if nr > 0:
        nr_rundir = run_dir + "/r_" + str(nr)
    else:
        nr_rundir = run_dir
    if os.path.exists(nr_rundir) & rmdir:
        shutil.rmtree(nr_rundir)
    os.mkdir(nr_rundir)
    os.mkdir(nr_rundir + "/input")
    os.mkdir(nr_rundir + "/output")
    shutil.copy(KJPATH + "/WHAM/kj.cfg", nr_rundir + "/kj.cfg")
    """
    with io.open(nr_dir + str(nr) + "/kj.cfg") as f:
        config = libconf.load(f)
    """
    
    nc_dir = nr_rundir + "/input/input-data.nc"
    write_nc(nc_dir, comsol, nr*1e-3, n_z)

    return nr_rundir

def read_input_nc(nc_dir):
    ncIn = Dataset(nc_dir, 'r', format="NETCDF4")
    print(ncIn.variables["density_m3"][:])
    ncIn.close()
    return

def interp_output(r_coords, z_coords, data, r, start, end, n_pts):
    """
    Interpolate the data to output into an uniformly spaced array of size n_pts suitable for writing to nc at coordinate r; 
    The start and end points are in z_coordinates
    """
    if len(r_coords) == 1:
        interpolator = interp1d(z_coords, data)
    else:
        interpolator = RegularGridInterpolator([z_coords, r_coords], data, bounds_error=True, fill_value=None)
    output = []
    for n in range(n_pts):
        z = (end - start) / n_pts * n + start
        output.append(np.float32(interpolator([z, r])))

    return output

def write_nc(nc_dir, comsol_data:COMSOL_data, r_coord, n_z:int, min_z=-0.8, max_z=0.8):
    # Make the nc file containing input data
    if os.path.exists(nc_dir):
        os.remove(nc_dir)
    
    ncIn = Dataset(nc_dir, 'w', format="NETCDF4")
    ncIn.createDimension("nR", n_z)
    ncIn.createDimension("nSpec", 1)
    ncIn.createDimension("scalar", 1)
    
    freq = ncIn.createVariable("freq", "f4", ("scalar", ))
    freq[0] = comsol_data.freq
    r = ncIn.createVariable("r", "f4", ("nR", ))
    r[:] = np.linspace(0, max_z-min_z, n_z)
    print("Writing input nc file")
    r_coords, z_coords = comsol_data.r_coords, comsol_data.z_coords

    B0_r = ncIn.createVariable("B0_r", "f4", ("nR", ))
    #B0_r[:] = interp_output(r_coords, z_coords, comsol_data.Br, r_coord, min_z, max_z, n_z)
    B0_r[:] = np.zeros(n_z)
    B0_phi = ncIn.createVariable("B0_p", "f4", ("nR", ))
    B0_phi[:] = np.zeros(n_z)
    B0_z = ncIn.createVariable("B0_z", "f4", ("nR", ))
    B0_z[:] = interp_output(r_coords, z_coords, comsol_data.Bz, r_coord, min_z, max_z, n_z)

    e_r_re = ncIn.createVariable("e_r_re", "f4", ("nR", ))
    e_r_re[:] = interp_output(r_coords, z_coords, comsol_data.Er_real, r_coord, min_z, max_z, n_z)
    e_r_im = ncIn.createVariable("e_r_im", "f4", ("nR", ))
    e_r_im[:] = interp_output(r_coords, z_coords, comsol_data.Er_imag, r_coord, min_z, max_z, n_z)
    e_p_re = ncIn.createVariable("e_p_re", "f4", ("nR", ))
    e_p_re[:] = interp_output(r_coords, z_coords, comsol_data.Ephi_real, r_coord, min_z, max_z, n_z)
    e_p_im = ncIn.createVariable("e_p_im", "f4", ("nR", ))
    e_p_im[:] = interp_output(r_coords, z_coords, comsol_data.Ephi_imag, r_coord, min_z, max_z, n_z)
    e_z_re = ncIn.createVariable("e_z_re", "f4", ("nR", ))
    e_z_re[:] = interp_output(r_coords, z_coords, comsol_data.Ez_real, r_coord, min_z, max_z, n_z)
    e_z_im = ncIn.createVariable("e_z_im", "f4", ("nR", ))
    e_z_im[:] = interp_output(r_coords, z_coords, comsol_data.Ez_imag, r_coord, min_z, max_z, n_z)

    b_r_re = ncIn.createVariable("b_r_re", "f4", ("nR", ))
    b_r_re[:] = interp_output(r_coords, z_coords, comsol_data.Br_real, r_coord, min_z, max_z, n_z)
    b_r_im = ncIn.createVariable("b_r_im", "f4", ("nR", ))
    b_r_im[:] = interp_output(r_coords, z_coords, comsol_data.Br_imag, r_coord, min_z, max_z, n_z)
    b_p_re = ncIn.createVariable("b_p_re", "f4", ("nR", ))
    b_p_re[:] = interp_output(r_coords, z_coords, comsol_data.Bphi_real, r_coord, min_z, max_z, n_z)
    b_p_im = ncIn.createVariable("b_p_im", "f4", ("nR", ))
    b_p_im[:] = interp_output(r_coords, z_coords, comsol_data.Bphi_imag, r_coord, min_z, max_z, n_z)
    b_z_re = ncIn.createVariable("b_z_re", "f4", ("nR", ))
    b_z_re[:] = interp_output(r_coords, z_coords, comsol_data.Bz_real, r_coord, min_z, max_z, n_z)
    b_z_im = ncIn.createVariable("b_z_im", "f4", ("nR", ))
    b_z_im[:] = interp_output(r_coords, z_coords, comsol_data.Bz_imag, r_coord, min_z, max_z, n_z)

    density_m3 = ncIn.createVariable("density_m3", "f4", ("nSpec", "nR"))
    density_m3[:] = interp_output(r_coords, z_coords, comsol_data.ne, r_coord, min_z, max_z, n_z)
    ncIn.close()
    return 

def run_kineticj(run_dir):
    # Run KineticJ now
    cmd = os.path.expanduser(KJPATH+"/bin/kineticj")
    args = ""
    #cwd = os.getcwd()
    startTime = time.time()
    os.chdir(run_dir)
    if os.path.exists(run_dir + "/output/jP2.nc"):
        os.remove(run_dir + "/output/jP2.nc")
    print("Starting KineticJ")
    call([cmd,args]) 
    endTime = time.time()
    print('time taken: %.5f'%(endTime-startTime) + ' seconds')
    print('%.2f'%((endTime-startTime)/3600) + ' hours')
    #os.chdir(cwd)
    return

def run_comsol(mph_dir="/home/mason/WHAM_COMSOL"):
    startTime = time.time()
    os.chdir(mph_dir)
    print("starting COMSOL run")
    cmd = "comsol batch"
    args = " -inputfile WHAM_RF_1D.mph -methodcall run_kj_7D70 -nosave"
    call(cmd + args, shell=True)
    endTime = time.time()
    print('COMSOL took: %.5f'%(endTime-startTime) + ' seconds')
    return

def run_iterations_2D(iter_start_n:int, n_iter:int, r_n:list, restart=True):
    for n in range(iter_start_n, iter_start_n+n_iter):
        run_dir = "/home/mason/kineticj/WHAM_KJ/1e19ne_26e6f/iter_" + str(n)
        if not os.path.exists(run_dir):
            os.mkdir(run_dir)
        # Directory containing the COMSOL electric field output files from last iteration
        comsol_output_dir = KJDATAPATH + "/"
        # First run kineticJ with existing COMSOL data, if iter_start_n = 0, this should be prepared
        comsol = COMSOL_data(comsol_output_dir)
        #comsol.plot_comsol_field(["Ephi_real", "Er_real", "Ez_real", "Ephi_imag", "Er_imag", "Ez_imag"])
        for run_n in r_n:
            if os.path.exists(run_dir + "/r_" + str(run_n) + "/output/jP2.nc") and restart:
                print("Skipping r=" + str(run_n))
                continue
            nr_rundir = make_run_directory(run_dir, run_n, n_z, comsol, rmdir=True)
            run_kineticj(nr_rundir)
        
        iter_kj = KineticJ_result(run_dir)
        iter_kj.plot()
        iter_kj.write_to_file(iter=n, file_dir="/mnt/COMSOL/COMSOL_input")
        run_comsol()
        source_dir = KJDATAPATH + "/"
        # Directory containing the COMSOL electric field output files for new iteration
        comsol_next_dir = "/home/mason/kineticj/WHAM_KJ/1e19ne_26e6f/iter_" + str(n+1)
        if not os.path.exists(comsol_next_dir):
            os.mkdir(comsol_next_dir)
        for file_name in os.listdir(source_dir):
            source = source_dir + file_name
            destination = comsol_next_dir + "/" + file_name
            if os.path.isfile(source):
                shutil.copy(source, destination)
        print("finished copying files to " + comsol_next_dir)
    return
    
run_iterations_2D(15, 15, r_n, True)

#iter_data = COMSOL_data(KJDATAPATH)
#iter_data.plot_comsol_field(["ne"])
#make_run_directory(KJDATAPATH + "/iter_0", 1, n_z, iter_data)

#iter_data.export_to_dispersionnering(600, 0)
#read_input_nc("/home/mason/kineticj/WHAM/high_collision_gpu/iter_0/r_0/input/input-data.nc")
# %%