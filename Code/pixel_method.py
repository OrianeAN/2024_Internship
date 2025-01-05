#%%
import numpy as np
from scipy import fft
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
import os
import imageio.v3 as iio
import time


fftfreq = np.fft.fftfreq
fft2 = fft.fft2
ifft2 = fft.ifft2
fftshift = fft.fftshift

def calculate_convergence(epsilon, *argv):
    """Vérifie si les erreurs données en arguments sont inférieures à une tolérance spécifiée, epsilon.
    Retourne un booléen qui indique si les erreurs respectent ou non ce critère de convergence
    """
    if len(argv) == 2:
        error_cs1, error_cs2 = argv
        return error_cs1 < epsilon and error_cs2 < epsilon
    elif len(argv) == 1:
        error = argv[0]
        return error < epsilon

# Yang (2018) criteria
def calculate_intensity_proximity_curr_prev(intensity_curr, intensity_prev):
    """Calcule l'erreur moyenne relative entre deux matrices d'intensité : intensity_curr (intensité actuelle) et 
    intensity_prev (intensité précédente). Ne tient compte que des positions où l'intensité actuelle n'est pas égale à zéro, afin d'éviter les divisions par zéro
    """
    mask = intensity_curr != 0
    error = np.abs((intensity_curr[mask] - intensity_prev[mask]) / intensity_curr[mask])
    average_error = np.mean(error)
    
    return average_error

# Huang (2011) criteria
def calculate_intensity_proximity_calc_meas(intensity_calc, intensity_meas):
    """Calcule l'erreur relative entre une intensité calculée (intensity_calc) et une intensité mesurée (intensity_meas). 
    Utilise l'erreur quadratique moyenne normalisée par la moyenne de l'intensité mesurée pour estimer 
    la proximité entre les deux ensembles de données.
    """
    mask = intensity_calc != 0
    E = np.sqrt(np.mean((intensity_calc[mask] - intensity_meas[mask])**2)) / np.mean(intensity_meas[mask])
    return E

def calculate_mean_square_error(a, b, norm=1):
    """Mean square error between a and b, normalized by norm."""
    diff = a - b
    mse = np.sum(diff**2)
    return mse / norm

def calculate_phase_perfect(X, Y, wavenumber, curvature_convex_side, refractive_index):
    """Calcule la phase optique d'une onde lumineuse en fonction de la position dans un plan bidimensionnel 
    (X,Y) du nombre d'onde, de la courbure d'une lentille, et de l'indice de réfraction du matériau.
    """
    rlens = np.sqrt(X**2 + Y**2)**2
    return wavenumber * rlens / (2*curvature_convex_side/(refractive_index-1))


def plot_intensity_amplitude_phase(complex_field, title, mask_size, save_results, directory):
    """Trace trois aspects d'un champ complexe (complex_field) : l'intensité, l'amplitude et la phase.
    """
    fig, ax = plt.subplots(nrows=1, ncols=3, sharex=True, sharey=True,
                           figsize=(8, 3))
    plt.suptitle(title)
    # check order of axes
    a_labels = np.array([-mask_size[1]/2, mask_size[1]/2, -mask_size[0]/2, mask_size[0]/2])
    a_labels = list(a_labels * 1000)  # put axes labels in mm
    ax[0].set_title('Intensity')
    ax[0].imshow(np.abs(complex_field)**2, cmap='viridis',
                 extent=a_labels)
    #fig.colorbar(ax[2])
    ax[1].set_title('field amplitude')
    ax[1].imshow(np.abs(complex_field), cmap='viridis',
                 extent=a_labels)
    ax[2].set_title('field phase')
    pp = ax[2].imshow(np.angle(complex_field), cmap='twilight',
                      extent=a_labels)
    if save_results:
        name =  title + ".png"
        plt.savefig(os.path.join(directory, name))
    else:
        plt.show()

def plot_proximity_progress(delta_z, save_results, directory, error_info):
    """Trace l'évolution de l'erreur ou de la proximité entre des profils d'intensité au cours de plusieurs itérations.
    """
    fig, ax = plt.subplots(nrows=1, ncols=1, sharex=True, sharey=True, figsize=(8, 3))
    #plt.suptitle("Progress of the proximity error")

    title = None

    if len(error_info) == 2:
        title = f"Proximity progress between the curr. and prev. intensity profiles in both CS ({delta_z =})"
        plt.suptitle(title)
        list_error_cs1 = error_info[0]
        list_error_cs2 = error_info[1]
        ax.plot(list_error_cs1, label='cs1')
        ax.plot(list_error_cs2, label='cs2')
        ax.set_ylabel('Proximity')
    elif len(error_info) == 1:
        title = f"Proximity progress between the calc. and meas. intensity profiles in CS1 ({delta_z =})"
        plt.suptitle(title)
        list_error = error_info[0]
        ax.plot(list_error, label='error')
        ax.set_ylabel('Error')

    ax.set_xlabel('Iteration')
    ax.legend()
    if save_results:
        name =  title + ".png"
        plt.savefig(os.path.join(directory, name))
    else:
        plt.show()

def plot_initial_intensities(I1, I2, z1, z2, dimensions, save_results, directory=""):
    fig, ax = plt.subplots(nrows=1, ncols=2, sharex=True, sharey=True, figsize=(8, 3))
    plt.suptitle("Input")
    a_labels = np.array([-dimensions[1]/2, dimensions[1]/2, -dimensions[0]/2, dimensions[0]/2]) * 1000

    modified_I1 = I1.copy()
    
    ax[0].set_title(f'Intensity1 at position z = {z1*1e3:.1f} mm')
    im0 = ax[0].imshow(modified_I1, cmap='gist_stern', extent=a_labels, vmax=0.00010)
    fig.colorbar(im0, ax=ax[0])
    fig.tight_layout()
    ax[1].set_title(f'Intensity2 at position z = {z2*1e3:.1f} mm')
    img1 = ax[1].imshow(I2, cmap='gist_stern', extent=a_labels)
    fig.colorbar(img1, ax=ax[1])
    fig.tight_layout()
    if save_results:
        plt.savefig(os.path.join(directory, 'initial_intensities.png'))
    else:
        plt.show()

def plot_full_image(phi1_im, phase_type, delta_z, dimensions, u_retrieved, computer_created, save_fig=False, directory="None"):
    u_retrieved_phase = np.angle(u_retrieved)
    #filtrage du résultat
    u_retrieved_phase = gaussian_filter(u_retrieved_phase, 2)


    if computer_created:
        n_images = 2
    else:
        n_images = 1

    fig, ax = plt.subplots(nrows=1, ncols=n_images, sharex=True, sharey=True)
    a_labels = np.array([-dimensions[1]/2, dimensions[1]/2, -dimensions[0]/2, dimensions[0]/2]) * 1000

    #vmin = -np.pi
    #vmax = np.pi

    title = f"{phase_type} phase using distance = {delta_z*1e3:.1f} mm"

    if computer_created:

        ax[0].set_title('Expected phase')
        img0 = ax[0].imshow(phi1_im, cmap='twilight', extent=a_labels)      #, vmin=vmin, vmax=vmax)
        fig.colorbar(img0, ax=ax[0], fraction=0.046, pad=0.04)

        ax[1].set_title('Retrieved phase')
        img1 = ax[1].imshow(u_retrieved_phase, cmap='twilight', extent=a_labels)    #, vmin=vmin, vmax=vmax)
        fig.colorbar(img1, ax=ax[1], fraction=0.046, pad=0.04)

        title = "Comparation " + title

    else:     
        ax.set_title('Retrieved phase after 100 iterations')
        img = ax.imshow(u_retrieved_phase, cmap='twilight', vmin=-1.7, vmax=1.9, extent=a_labels)  #, vmin=vmin, vmax=vmax)
        fig.colorbar(img, ax=ax, fraction=0.046, pad=0.04)

    #plt.suptitle(title)

    fig.tight_layout()
    if save_fig:
        plt.savefig(os. path.join(directory, f'full_image_{phase_type}.png'))
    else:
        plt.show()

    """
    plt.figure('3')
    plt.suptitle('Cut of the phase')
    plt.plot(u_retrieved_phase[256])
    """
    
    list_result.append(u_retrieved_phase)

def propagate_transfer_function_fresnel(origin_field, H):
    """Effectue la propagation du laser en utilisant la méthode de la transformée de Fourier 
    et la fonction de transfert de Fresnel.
    """
    ft_u_origin = fft2(origin_field)
    destination_field = ifft2(ft_u_origin * H)

    return destination_field

#%%
class IterativeAlgorithm:
    """Base class for iterative phase retrieval algorithms."""

    def __init__(self, abs_u1_m, abs_u2_m, guessed_phase, delta_z, shape_in_m, image_set):
        # check if i dont have to do fft.ifftshift, for the next 3 var
        # Initial amplitude for the two cross sections    
        self.abs_u1_m = abs_u1_m
        self.abs_u2_m = abs_u2_m
        self.u1 = None
        self.u2 = None

        self.guessed_phase = guessed_phase
        self.delta_z = delta_z
        self.shape_in_m = shape_in_m

        self.iter = 0
        self.tf_forward = None
        self.tf_backward = None

        self.image_set = image_set

        self.data = {} # to store the data to show

        self.list_error = []

        self._init_iterative()

    def _init_iterative(self):
        """Initialize an instance of an iterative transform type algorithm."""

        self.u1 = self.abs_u1_m * np.exp(1j*self.guessed_phase)


    def _propagate_beam(self, option, origin_field, delta_z):
        if option == "transfert function":
            if delta_z > 0:
                destination_field = propagate_transfer_function_fresnel(origin_field, self.tf_forward)
            elif delta_z < 0:
                destination_field = propagate_transfer_function_fresnel(origin_field, self.tf_backward)

        return destination_field
    
    def _calculate_cost_function(self, u1_retrieved):
        u2_retrieved = self._propagate_beam("transfert function", u1_retrieved, self.delta_z)
        # check calculus 
        residuals = (np.abs(u2_retrieved)**2 - (self.abs_u2_m)**2).flatten()
        cost = np.sum(residuals**2)

        return cost, u2_retrieved

    def set_transfer_functions(self, tf_forward, tf_backward):
        self.tf_forward = tf_forward
        self.tf_backward = tf_backward

    def step(self, print_each_iter=False):
        """Advance the algorithm one iteration."""
        raise NotImplementedError("Subclasses should implement this method.")

    def loop(self, niter, epsilon, print_each_iter):

        # just to see both first intensities, CHECK
        self.data["initial_u1"] = self.u1.copy()

        for _ in range(niter):
            solution_found = self.step(epsilon, print_each_iter)

            if not print_each_iter and self.iter == 0:
                self.data["u1_after_first_iter"] = self.u1.copy()

            self.iter += 1

            if solution_found:
                break

        self.data["cost"], self.data["u2"] = self._calculate_cost_function(self.u1)
        self.data["retrieved_u1"] = self.u1.copy()
        # self.data["error_info"] = [self.list_error_cs1, self.list_error_cs2]
        self.data["error_info"] = [self.list_error]

        return self.data

# Biblio: Yang, Jawla, Huang (2011)
class FresnelAlgorithm(IterativeAlgorithm):
    def __init__(self, abs_u1_m, abs_u2_m, guessed_phase, delta_z, shape_in_m, image_set):
        super().__init__(abs_u1_m, abs_u2_m, guessed_phase, delta_z, shape_in_m, image_set)
        # choose which one to use, one general error or one for each cross section

    def step(self, epsilon, print_each_iter):
        """Advance the Yang algorithm one iteration."""

        # 1. Forward propagation
        self.u2 = self._propagate_beam("transfert function", self.u1, self.delta_z)
        u2_step_1 = self.u2.copy()

        # 2. Replace amplitude and keep phase of u2
        self.u2 = self.abs_u2_m * np.exp(1j * np.angle(self.u2))
        u2_step_2 = self.u2.copy()
            
        # 3. Backward propagation
        self.u1 = self._propagate_beam("transfert function", self.u2, -self.delta_z)
        u1_step_3 = self.u1.copy()

        # 4. Replace amplitude and keep phase of u_1
        self.u1 = self.abs_u1_m * np.exp(1j * np.angle(self.u1))
        u1_step_4 = self.u1.copy()

        if print_each_iter:
            plot_intensity_amplitude_phase(u2_step_1, f'1. u2 prop: {self.iter = }', self.shape_in_m)
            plot_intensity_amplitude_phase(u2_step_2, f'2. u2 replaced: {self.iter = }', self.shape_in_m)
            plot_intensity_amplitude_phase(u1_step_3, f'3. u1 back: {self.iter = }', self.shape_in_m)
            plot_intensity_amplitude_phase(u1_step_4, f'4. u1 replaced: {self.iter = }', self.shape_in_m)

        # Calculate errors
        if self.iter != 0:

            error = calculate_intensity_proximity_calc_meas(np.abs(self.u1)**2, np.abs(self.abs_u1_m)**2)
            self.list_error.append(error)

            # Check for convergence
            if calculate_convergence(epsilon, error):
                return True
            else:
                return False
        else:
            return False
        
    def step_test_propagation_tf(self):

        plot_intensity_amplitude_phase(self.u1, f'u1 before:', self.shape_in_m)

        # Forward propagation
        self.u2 = self._propagate_beam("transfert function", self.u1, self.delta_z)

        plot_intensity_amplitude_phase(self.u2, f'u2 forward prop:', self.shape_in_m)

        # Backward propagation
        self.u1 = self._propagate_beam("transfert function", self.u2, -self.delta_z)
        
        plot_intensity_amplitude_phase(self.u1, f'u1 backward prop:', self.shape_in_m)

#%%
def remove_bkg_cadre(im, rel_frame_width):
        """
        im : 2D np array with image data
        rel_frame_width : float [0, 1] is the fraction of the smaller 
        image side used for defining the background level"""
        if type(im) is not np.ndarray or rel_frame_width < 0 or rel_frame_width > 1:
            print('--> wrong usage of the function remove_bkg')
            return np.nan  # wrong usage of the function
        if im.dtype is not float:
            im = im.astype(float)
        ymax, xmax = im.shape
        small_side = min(ymax, xmax)
        frame_width = int(np.round(small_side * rel_frame_width))
        yfrom = frame_width
        yto = ymax - frame_width
        xfrom = frame_width
        xto = xmax - frame_width
        ma = np.ones(im.shape, dtype=bool)  
        ma[yfrom:yto, xfrom:xto] = np.zeros((yto - yfrom, xto - xfrom), dtype=bool)
        backg = im[ma].mean()
        im = im - backg
        #im = gaussian_filter(im, 2)
        im=np.abs(im)
        return im

def remove_bkg_zone(im, top_left_coord,bottom_left_coord=None):
        """
        im : 2D np array with image data
        top_left_coord: top left coordonante of the area
        bottom_left_coord: bottom left coordonate of the area"""
        if type(im) is not np.ndarray:
            print('--> wrong usage of the function remove_bkg')
            return np.nan  # wrong usage of the function
        if im.dtype is not float:
            im = im.astype(float)
        x_zone, y_zone = top_left_coord
        zone = im[y_zone:, x_zone:]
        backg = zone.mean()
        im = im - backg
        im = gaussian_filter(im, 3)
        im=np.abs(im)
        return im

def centroid_position(im):
    XX, YY = np.meshgrid(range(im.shape[1]), range(im.shape[0]))
    xc = int(np.round((XX * im).sum() / im.sum()))
    yc = int(np.round((YY * im).sum() / im.sum()))
    return [xc, yc]
        
def recentrage(im, cpt=0) :
    ymax, xmax = im.shape
    lxc = [xmax//2-1, xmax//2, xmax//2+1]
    lyc = [ymax//2-1, ymax//2, ymax//2+1]
    xc, yc = centroid_position(im)
    if ((xc in lxc) and (yc in lyc)) or cpt ==4:
        return im
    
    if (xc not in lxc) and cpt !=4: #si l'image est décalé selon les x
        dist_centres = np.abs(xc - xmax//2)
        if xc < xmax//2 : # si centre_im < centre_xy
            im = np.hstack((im[:, -dist_centres:], im[:, :-dist_centres]))          
        else : # si centre_im < centre_xy
            im = np.hstack((im[:, dist_centres:], im[:, :dist_centres]))
        return recentrage(im, cpt+1)
        
    if (yc not in lyc) and cpt !=4: #si l'image est décalé selon les y
        dist_centres = np.abs(yc - ymax//2)
        if yc < ymax//2 : #si centre_im < centre_xy
            im = np.vstack((im[-dist_centres:], im[:-dist_centres]))
        else: #si centre_im > centre_xy
            im = np.vstack((im[dist_centres:], im[:dist_centres]))
        return recentrage(im, cpt+1)
    return im
    
    
#%%
class PhaseRetriever():
    def __init__(self, list_z, image_set, computer_created, crop, niter=10, epsilon=1e-3, real_im=False):
        self.options = {
            "path"      : None,
            "pixel_size": None,
            "shape"       : None, # Tuple of the dimensions of the image -> (dim_x, dim_y)
            "niter"     : niter,
            "epsilon"       : epsilon, # Small value used as convergence criterion
            # Beam parameters
            "wavelength": None, # lambda
            "wavenumber": None,
            "beam_radius": None # w0
        }

        self.lens = {
            # in m
            "curvature_convex_side": 155e-3, # R
            "refractive_index": 1.5 # ri
        }

        self.real_im = real_im

        self.crop = crop
        
        self.I1 = None
        self.I2 = None

        self.cropped_I1 = None
        self.cropped_I2 = None

        self.origin_I1 = None
        self.origin_I2 = None

        self.cropped_origin_I1 = None
        self.cropped_origin_I2 = None     

        if self.real_im :
            self.z1, self.num1 = list_z[0].split('_')[0], list_z[0].split('_')[1]
            self.z2, self.num2 = list_z[1].split('_')[0], list_z[1].split('_')[1]
        else:
            self.z1 = list_z[0]
            self.z2 = list_z[1]

        self.delta_z = None

        self.computer_created = computer_created
        self.image_set = image_set

        self.min_relevant_intensity_value = 1e-5 # 1e-6 (ideal lens), 1e-4 (eye phase)
        # self.mask = None # erase if not used
        self.mask_shape = None
        self.mask_vertex_set_I1 = None
        self.mask_vertex_set_I2 = None


    # check if it is necessary
    def __getitem__(self, key):
        return self.options[key]

    def __setitem__(self, key, value):
        self.config(**{key:value})

    def config(self, **options):
        for option in options:
            if option in self.options:
                self.options[option] = options[option]
            else:
                raise KeyError(f"Option {option} does not exist.")
            

    def _get_mask_limit_values(self, intensity_field):

        shape = intensity_field.shape

        right_limit_coord = 0
        left_limit_coord = 0
        top_limit_coord = 0
        bottom_limit_coord = 0 

        list_row_sum = np.sum(intensity_field, axis=1)
        inside_mask = False
        i = 0

        while i < shape[0] and not inside_mask:
            row_sum = list_row_sum[i]
            if row_sum >= self.min_relevant_intensity_value:
                inside_mask = True
                top_limit_coord = i
            i += 1

        inside_mask = False
        i = 0

        while i < shape[0] and not inside_mask:
            j = shape[0] - 1 - i
            row_sum = list_row_sum[j]
            if row_sum >= self.min_relevant_intensity_value:
                inside_mask = True
                bottom_limit_coord = j
            i += 1

        list_column_sum = np.sum(intensity_field, axis=0)
        inside_mask = False
        i = 0

        while i < shape[1] and not inside_mask:
            column_sum = list_column_sum[i]
            if column_sum >= self.min_relevant_intensity_value:
                inside_mask = True
                left_limit_coord = i
            i += 1
        
        inside_mask = False
        i = 0

        while i < shape[1] and not inside_mask:
            j = shape[1] - 1 - i
            column_sum = list_column_sum[j]
            if column_sum >= self.min_relevant_intensity_value:
                inside_mask = True
                right_limit_coord = j
            i += 1

        return top_limit_coord, bottom_limit_coord, left_limit_coord, right_limit_coord


    # Set values related to the region with highest intensity (mask)
    def _set_mask_values(self):

        # Search for the z with the image with widest region of interest
        # to define the mask for both 
        top_limit_coord_1, bottom_limit_coord_1, left_limit_coord_1, right_limit_coord_1 = self._get_mask_limit_values(self.I1)
        mask_shape_I1 = (bottom_limit_coord_1 - top_limit_coord_1, right_limit_coord_1 - left_limit_coord_1)
        self.mask_vertex_set_I1 = [(bottom_limit_coord_1, left_limit_coord_1),
                                    (bottom_limit_coord_1, right_limit_coord_1),
                                    (top_limit_coord_1, right_limit_coord_1),
                                    (top_limit_coord_1, left_limit_coord_1)]

        top_limit_coord_2, bottom_limit_coord_2, left_limit_coord_2, right_limit_coord_2 = self._get_mask_limit_values(self.I2)
        mask_shape_I2 = (bottom_limit_coord_2 - top_limit_coord_2, right_limit_coord_2 - left_limit_coord_2)
        self.mask_vertex_set_I2 = [(bottom_limit_coord_2, left_limit_coord_2),
                                    (bottom_limit_coord_2, right_limit_coord_2),
                                    (top_limit_coord_2, right_limit_coord_2),
                                    (top_limit_coord_2, left_limit_coord_2)]

        area_I1 = mask_shape_I1[0] * mask_shape_I1[1]
        area_I2 = mask_shape_I2[0] * mask_shape_I2[1]

        if area_I1 > area_I2:
            top_limit_coord = top_limit_coord_1
            bottom_limit_coord = bottom_limit_coord_1
            left_limit_coord = left_limit_coord_1
            right_limit_coord = right_limit_coord_1
        else:
            top_limit_coord = top_limit_coord_2
            bottom_limit_coord = bottom_limit_coord_2
            left_limit_coord = left_limit_coord_2
            right_limit_coord = right_limit_coord_2

        self.mask_shape = (bottom_limit_coord - top_limit_coord, right_limit_coord - left_limit_coord)
        self.mask_vertex_set = [(bottom_limit_coord, left_limit_coord),
                                    (bottom_limit_coord, right_limit_coord),
                                    (top_limit_coord, right_limit_coord),
                                    (top_limit_coord, left_limit_coord)]
        

    def _crop_image(self, option):
        # CHECK IF PROBLEMS: both mask vertex set don't have the same 
        if option == "I1":
            image = self.I1
        elif option == "I2":
            image = self.I2

        #mask_vertex_set = self.mask_vertex_set_I1

        cropped_image = image[self.mask_vertex_set[2][0]:self.mask_vertex_set[0][0],
                                self.mask_vertex_set[0][1]:self.mask_vertex_set[2][1]]
        
        return cropped_image
    
    def _center_and_crop_images(self):
        self.cropped_I1 = self._crop_image("I1")
        self.cropped_I2 = self._crop_image("I2")

        # normalize the cropped images
        self.cropped_I1 = self.cropped_I1 / self.cropped_I1.sum()
        self.cropped_I2 = self.cropped_I2 / self.cropped_I2.sum()
        
        self.cropped_origin_I1 = centroid_position(self.cropped_I1)
        self.cropped_origin_I2 = centroid_position(self.cropped_I2)

    def load_data(self, path):
        self.options["path"] = path

        self.delta_z = abs(round(float(self.z2) - float(self.z1), 3))

        for i in range(2):
            z = 0
            if i == 0:
                z = self.z1
            else:
                z = self.z2

            if self.real_im and self.image_set == "images reelle":
                if i == 0:
                    image_path = rf"{path}\z{z}_{self.num1}_I.tif"   
                if i == 1:
                    image_path = rf"{path}\z{z}_{self.num2}_I.tif"
            elif self.real_im and self.image_set == "Laser_UV_f500mm":
                image_path = rf"{path}\z{z}_{self.num1}.tif"
            else:
                if self.computer_created:
                    image_path = f"{path}\z{z}_I.tif"
                else:
                    image_path = f"{path}\z{z}_I.tif"

            im = iio.imread(image_path)
            y_z, x_z = im.shape
            y_z, x_z = y_z - y_z//2, x_z - x_z//4 
            # Eliminate the background
            im = remove_bkg_zone(im, [x_z, y_z])
            #im = remove_bkg_cadre(im, 0.02)

            # Normalization
            im = im / im.sum()

            # Refocusing
            im = recentrage(im)
            
            profile_I = im

            if i == 0:
                self.I1 = profile_I
                self.options["shape"] = profile_I.shape
                self.origin_I1 = centroid_position(self.I1)
            elif i == 1:
                self.I2 = profile_I
                self.origin_I2 = centroid_position(self.I2)

    def _calculate_guessed_phase(self, option):
        shape = None
        if self.crop:
            shape = self.mask_shape
        else:
            shape = self.options["shape"]

        if option == "random":
            if self.crop:
                return np.random.rand(shape)
        elif option == "zeros":
            return np.zeros(shape)
        elif option == "zeros complex":
            return np.zeros(shape, dtype=np.complex128)

    def _calculate_transfer_function(self, distance):
        nux = None
        nuy = None

        if self.crop:
            nux = fftfreq(self.mask_shape[1], self.options["pixel_size"] )   
            nuy = fftfreq(self.mask_shape[0], self.options["pixel_size"] )    
        else:
            nux = fftfreq(self.options["shape"][1], self.options["pixel_size"] )    
            nuy = fftfreq(self.options["shape"][0], self.options["pixel_size"] )    

        nuX, nuY = np.meshgrid(nux, nuy)

        H = np.exp(2*np.pi*1j * np.sqrt(1/self.options["wavelength"]**2 - nuX**2 - nuY**2) * distance)

        return H
        
    def retrieve_phase(self, print_each_iter, save_results): 
        self._set_mask_values() 

        guessed_phase = self._calculate_guessed_phase("zeros complex")

        if self.crop:
            self._center_and_crop_images()
            #mask shape in meters
            mask_shape_in_m = tuple(element * self.options["pixel_size"] for element in self.mask_shape)

            self.algorithm = FresnelAlgorithm(np.sqrt(self.cropped_I1), np.sqrt(self.cropped_I2), guessed_phase, self.delta_z, mask_shape_in_m, self.image_set)

        else:
            shape_in_m = tuple(element * self.options["pixel_size"] for element in self.options["shape"])   #* 1e-6 a été mis direcetment dans pixel_size
            self.algorithm = FresnelAlgorithm(np.sqrt(self.I1), np.sqrt(self.I2), guessed_phase, self.delta_z, shape_in_m, self.image_set)

        self.algorithm.set_transfer_functions(self._calculate_transfer_function(-self.delta_z),
                                              self._calculate_transfer_function(self.delta_z))
        
        start_time = time.time()

        data = self.algorithm.loop(self.options["niter"], self.options["epsilon"], print_each_iter)

        end_time = time.time()

        duration = end_time - start_time

        current_dir = os.getcwd()
        #current_dir = os.path.dirname(os.path.abspath(__file__))
        
        parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
        cropped_or_not = "cropped" if self.crop else "not_cropped"
        directory = os.path.join(parent_dir, "results", self.image_set, cropped_or_not)
        os.makedirs(directory, exist_ok=True)

        self.show_data_results(data, duration, save_results, directory)
        self.show_graphic_results(data, save_results, directory)

    def retrieve_phase_test_propagation_tf(self):
        self.origin_I1 = centroid_position(self.I1)
        self.origin_I2 = centroid_position(self.I2)

        self._set_mask_values() 

        if self.crop:
            self._center_and_crop_images()

            mask_shape_in_m = tuple(element * self.options["pixel_size"] for element in self.mask_shape)

            x_pts = np.linspace(-mask_shape_in_m[1]/2, mask_shape_in_m[1]/2, num=self.mask_shape[1])
            y_pts = np.linspace(-mask_shape_in_m[0]/2, mask_shape_in_m[0]/2, num=self.mask_shape[0])

            X, Y = np.meshgrid(x_pts, y_pts)

            guessed_phase = calculate_phase_perfect(X, Y, self.options["wavenumber"],  
                                                        self.lens["curvature_convex_side"], self.lens["refractive_index"])
            
            self.algorithm = FresnelAlgorithm(np.sqrt(self.cropped_I1), np.sqrt(self.cropped_I2), guessed_phase, self.delta_z, mask_shape_in_m, self.image_set)
        else:
            shape_in_m = tuple(element * self.options["pixel_size"] for element in self.options["shape"])   #* 1e-6 a été mis direcetment dans pixel_size

            x_pts = np.linspace(-shape_in_m[1]/2, shape_in_m[1]/2, num=self.options["shape"][1])
            y_pts = np.linspace(-shape_in_m[0]/2, shape_in_m[0]/2, num=self.options["shape"][0])

            X, Y = np.meshgrid(x_pts, y_pts)

            guessed_phase = calculate_phase_perfect(X, Y, self.options["wavenumber"],
                                                        self.lens["curvature_convex_side"], self.lens["refractive_index"])

            # check bot cases: with self.I1 and self.I1_theorical, should be the same (now they are =, checked)
            self.algorithm = FresnelAlgorithm(np.sqrt(self.I1), np.sqrt(self.I2), guessed_phase, self.delta_z, shape_in_m, self.image_set)

        # CHECK: for some reason, it works with the transfer functions inverted
        self.algorithm.set_transfer_functions(self._calculate_transfer_function(-self.delta_z),
                                              self._calculate_transfer_function(self.delta_z))
        
        self.algorithm.step_test_propagation_tf()

    def show_data_results(self, data, duration, save_results, directory):
        if save_results:
            with open(f"{directory}/optimization_results.txt", "w") as file:
                file.write(f"*** Execution time: {round(duration, 4)} s ***\n")
                
                file.write("### CONFIGURATION ###\n")
                file.write(f"Image set: {self.image_set}\n")
                file.write(f"Maximum number of iterations: {self.options['niter']}\n")
                file.write(f"Epsilon: {self.options['epsilon']}\n")
                file.write(f"Pixel size: {self.options['pixel_size']}\n")
                file.write(f"Shape: {self.options['shape']}\n")
                file.write(f"Wavelength: {self.options['wavelength']}\n")
                file.write(f"Distance z1: {self.z1}\n")
                file.write(f"Distance z2: {self.z2}\n")
                file.write(f"Distance between the two planes: {self.delta_z}\n")

                file.write("### OTHERS ###\n")
                file.write(f"Origin of the region with highest intensity at z1: {self.origin_I1}\n")
                file.write(f"Origin of the region with highest intensity at z2: {self.origin_I2}\n")
                file.write(f"Mask vertex set:\n")
                
                file.write(f"Mask shape: {self.mask_shape}\n")
                file.write(f"Cropped: {self.crop}\n")
                file.write(f"Min relevant intensity value: {self.min_relevant_intensity_value}\n")


                file.write("### RESULTS ###\n")
                file.write(f"Objective Function value: {data['cost']}\n")
                file.write(f"Final number of iterations: {self.algorithm.iter}\n")

        else:
            print(f"*** Execution time: {round(duration, 4)} s ***")
            print("### CONFIGURATION ###")
            print(f"Image set: {self.image_set}")
            print(f"Maximum number of iterations: {self.options['niter']}")
            print(f"Epsilon: {self.options['epsilon']}")
            print(f"Pixel size: {self.options['pixel_size']}")
            print(f"Shape: {self.options['shape']}")
            print(f"Wavelength: {self.options['wavelength']}")
            print(f"Distance z1: {self.z1}")
            print(f"Distance z2: {self.z2}")
            print(f"Distance between the two planes: {self.delta_z}")

            print("### OTHERS ###")
            print(f"Origin of the region with highest intensity at z1: {self.origin_I1}")
            print(f"Origin of the region with highest intensity at z2: {self.origin_I2}")
            print(f"Mask shape: {self.mask_shape}")
            print(f"Cropped: {self.crop}")
            print(f"Min relevant intensity value: {self.min_relevant_intensity_value}")

            print("### RESULTS ###")
            print(f"Objective Function value: {data['cost']}")
            print(f"Final number of iterations: {self.algorithm.iter}")

    def show_graphic_results(self, data, save_results, directory):
        shape_in_m = None

        if self.crop:
            shape_in_m = tuple(element * self.options["pixel_size"] for element in self.mask_shape)
        else:
            shape_in_m = tuple(element * self.options["pixel_size"] for element in self.options["shape"])

        plot_initial_intensities(self.I1, self.I2, float(self.z1), float(self.z2), shape_in_m,
                                    save_results, directory)
        
        if self.crop:
            plot_initial_intensities(self.cropped_I1, self.cropped_I2, float(self.z1), float(self.z2), shape_in_m,
                                        save_results, directory)

        plot_intensity_amplitude_phase(data["initial_u1"], f"Initial u1 (z1 = {self.z1}, z2 = {self.z2})", shape_in_m,
                                          save_results, directory)
        
        plot_intensity_amplitude_phase(data["u1_after_first_iter"], f"After first iter u1 (z1 = {self.z1}, z2 = {self.z2})", shape_in_m,
                                          save_results, directory)
        
        plot_intensity_amplitude_phase(data["retrieved_u1"], f"Final u1 (z1 = {self.z1}, z2 = {self.z2})", shape_in_m,
                                          save_results, directory)
        
        if self.computer_created:
            base_path = os.path.abspath(os.path.join(os.getcwd(), ".."))
            #base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
            file_path = os.path.join(base_path, "input_images", self.image_set, f"z{self.z1}_phi.png")
            expected_phase_image = iio.imread(file_path)
            if self.crop:
                top_limit_coord = int(self.options["shape"][0])//2 - self.mask_shape[0]//2
                bottom_limit_coord = int(self.options["shape"][0])//2 + self.mask_shape[0]//2
                left_limit_coord = int(self.options["shape"][1])//2 - self.mask_shape[1]//2
                right_limit_coord = int(self.options["shape"][1])//2 + self.mask_shape[1]//2

                expected_phase_image = expected_phase_image[top_limit_coord:bottom_limit_coord, left_limit_coord:right_limit_coord]
        else:
            expected_phase_image = None
        
        phase_type = "retrieved"
        
        plot_full_image(expected_phase_image, phase_type, self.delta_z, shape_in_m, data["retrieved_u1"],
                          self.computer_created, save_results, directory)
        
        plot_proximity_progress(self.delta_z, save_results, directory, error_info = data["error_info"])


#%%
#Pour afficher les message OK et FAIL en couleur (verte:OK, rouge:FAIL)
OK = "\033[0;32mOK\033[0;0m"
FAIL = "\033[91mFAIL\033[0;0m"

#chemin d'accès où se trouvent les images d'entrée
base_path = r"C:\Users\orian\OneDrive\Bureau\Stage 2024-25\Pixel method\Oriane\input_images"

def load_data(retriever, path):   
    """Charge des données à partir de path
    L'ajustement de l'intensité de l'image (normalisation, soustraction de l'arrière-plan) est fait.
    """
    print("Dataset load... ", end="")
    try:
        retriever.load_data(path)
        print(OK)
    except:
        print(FAIL)

def set_values(retriever, pixel_size, wavelength, wavenumber, beam_radius):
    """définit les paramètres du laser
    """
    print("Pixel size set... ", end="")
    try:
        retriever.config(pixel_size=pixel_size)
        if retriever.options["pixel_size"] != pixel_size:
            raise ValueError()
        print(OK)
    except Exception as e:
        print(FAIL, f"Expected {pixel_size} and got {retriever.options['pixel_size']}")

# Beam
    print("Wavelength set... ", end="")
    try:
        retriever.config(wavelength=wavelength)
        if retriever.options["wavelength"] != wavelength:
            raise ValueError()
        print(OK)
    except:
        print(FAIL, f"Expected {wavelength} and got {retriever.options['wavelength']}")

    print("Wavenumber set... ", end="")
    try:
        retriever.config(wavenumber=wavenumber)
        if retriever.options["wavenumber"] != wavenumber:
            raise ValueError()
        print(OK)
    except:
        print(FAIL, f"Expected {wavenumber} and got {retriever.options['wavenumber']}")

    print("Beam radius set... ", end="")
    try:
        retriever.config(beam_radius=beam_radius)
        if retriever.options["beam_radius"] != beam_radius:
            raise ValueError()
        print(OK)
    except:
        print(FAIL, f"Expected {beam_radius} and got {retriever.options['beam_radius']}")

def test_ideal_lens(crop=False, print_each_iter=False, save_results=False):
    """ Utilise le jeu d'images appelé "foc lent idéale".
    """
    image_set = "foc lent ideale"
    path = os.path.join(base_path, image_set)

    niter = 100
    epsilon = 0 # 4.45e-16
    computer_created = True

    retriever = PhaseRetriever(['0.000', '0.310'], image_set, computer_created, crop, niter, epsilon)
    
    pixel_size = 11.71875e-6
    wavelength = 1064e-9
    wavenumber = 2 * np.pi / wavelength
    beam_radius = 2e-3 / 2
    
    load_data(retriever, path)

    set_values(retriever, pixel_size, wavelength, wavenumber, beam_radius)

    retriever.retrieve_phase(print_each_iter, save_results)

def test_real_images(couple_im, crop=False, print_each_iter=False, save_results=False):
    """ Utilise le jeu d'images appelé "images reelle".
    """
    image_set = "images reelle"
    path = os.path.join(base_path, image_set)

    niter = 20
    epsilon = 0 # 4.45e-16
    computer_created = False

    retriever = PhaseRetriever(couple_im, image_set, computer_created, crop, niter, epsilon, real_im=True)
    
    pixel_size = 6.45e-6    #11.71875e-6
    wavelength = 1064e-9
    wavenumber = 2 * np.pi / wavelength
    beam_radius = 2e-3 / 2
    

    load_data(retriever, path)

    set_values(retriever, pixel_size, wavelength, wavenumber, beam_radius)

    retriever.retrieve_phase(print_each_iter, save_results)

def test_Laser_UV_f500mm(couple_im, crop=False, print_each_iter=False, save_results=False):
    """ Utilise le jeu d'images appelé "Laser_UV_f500mm".
    """
    image_set = "Laser_UV_f500mm"
    path = os.path.join(base_path, image_set)

    niter = 100
    epsilon = 0 # 4.45e-16
    computer_created = False

    retriever = PhaseRetriever(couple_im, image_set, computer_created, crop, niter, epsilon, real_im=True)
    
    pixel_size = 12.9e-6
    wavelength = 343e-9
    wavenumber = 2 * np.pi / wavelength
    beam_radius = 2e-3 / 2
    

    load_data(retriever, path)

    set_values(retriever, pixel_size, wavelength, wavenumber, beam_radius)

    retriever.retrieve_phase(print_each_iter, save_results)


#%%
list_result = []

#test_Laser_UV_f500mm(['0.20_0avg', '0.35_0avg'], crop=True, print_each_iter=False, save_results=False)
test_real_images(['0.005_3', '-0.010_5'], crop=True, print_each_iter=False, save_results=False)
#test_ideal_lens(crop=False, print_each_iter=False, save_results=False)

# %% Calcule de la moyenne et de la variance
"""
list_image = [['0.005_3', '-0.010_5'], ['0.005_6', '-0.010_7'], ['0.005_8', '-0.010_8'], ['0.005_11', '-0.010_11'],  ['0.005_13', '-0.010_14']]
list_result = []
for element in list_image :
    test_real_images(element, crop=True, print_each_iter=False, save_results=False)
moyenne = (np.array(list_result).sum(axis=0)) / len(list_result)
variance = 0
ecarttype = 0
for element in list_result:
    ecarttype += (element-moyenne)
    variance += ecarttype**2
esm = ecarttype/moyenne
variance = variance/len(list_result)

plt.figure('moyenne')
plt.suptitle('Mean')
plt.imshow(moyenne, cmap='twilight')
plt.colorbar()

plt.figure('variance')
plt.suptitle('Standard error of mean')
plt.imshow(variance, cmap='viridis',vmin=0,vmax=0.5)
plt.colorbar()
"""
"""
plt.figure('esm')
plt.suptitle('ESM')
plt.imshow(esm, cmap='viridis')
plt.colorbar()
"""
# %%

plt.show()
