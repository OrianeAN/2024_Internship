import imageio.v3 as iio
import numpy as np
from scipy.optimize import leastsq
import matplotlib.pyplot as plt
from scipy.fft import fft2, ifft2, fftfreq

#%% Fonction de base et input
# Fonction pour enlever le background de l'image. Utilise la méthode du coin.
# Return l'image sans le background
def remove_bkg(im, rel_frame_width):
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
    return im

# Fonction pour retrouver le barycentre de la photo
# Retourne (yc, xc)
def centroid_position(im):
    XX, YY = np.meshgrid(range(im.shape[1]), range(im.shape[0]))
    xc = int(np.round((XX * im).sum() / im.sum()))
    yc = int(np.round((YY * im).sum() / im.sum()))
    return xc, yc

# Focntion qui va servir pour la propagation du laser
# Retourne la fonction de transfert qu'on utilise pour la méthode de la transformer de Fourier
def aSCH(nux, nuy, z2, la):  # without Fresnel approximation
    return np.exp(2*np.pi*1j * np.sqrt(1/la**2 - nux**2 - nuy**2) * z2)

# Fonction de la propagation du laser
# retourne le signal après popagation au point(nuX, nuY)
def propag(u1, dz, nuX, nuY, la):
    ft_u1 = fft2(u1)  # puts zero first
    u2 = ifft2(ft_u1 * aSCH(nuX, nuY, dz, la))
    return u2

## importation des photos
file_name_1 = "z0.000_I.tif"  # should contain info on z-position in meters
file_name_2 = "z0.150_I.tif"  # should contain info on z-position in meters
file_name_3 = "z0.310_I.tif"  # should contain info on z-position in meters
file_name_4 = "z0.400_I.tif"  # should contain info on z-position in meters
file_name_0 = "z0.000_phi.png"  # the known phase used at for the generation of the images

"""file_name_2 = r"z0.005_aI.tif"  # should contain info on z-position in meters
file_name_3 = r"z0.005_bI.tif"  # should contain info on z-position in meters
file_name_4 = r"z-0.010_I.tif"  # should contain info on z-position in meters

real_im = True"""

list_file = [file_name_1, file_name_2, file_name_3, file_name_4]
nb_im = 4

pisi = 11.71875e-6  # pixel size in meters
la = 1064e-9  # wavelength in meters

#%% Mise en place des variables
#list_im = [iio.imread(file_name_1), iio.imread(file_name_2),
#           iio.imread(file_name_3), iio.imread(file_name_4)]
list_im = []
list_I = []
list_coord_c = []
list_z = []
list_dz = []
for i in range(nb_im):
    list_im += [iio.imread(list_file[i])]
    list_im[i] = remove_bkg(list_im[i], 0.02)
    list_I += [list_im[i] / list_im[i].sum()]
    list_coord_c += [centroid_position(list_im[i])]
    list_z += [list_file[i].split('_')[0]]
    list_z[i] = float(list_z[i][1:])
    if i != 0:
        list_dz += [list_z[i] - list_z[0]]

"""if real_im:
    list_I[0] = list_I[0][:508, :508]"""

for i in range(nb_im):
    print(f"im{i+1} energy = {list_I[i].sum()}")
    print(f"coordonnées centre im{i+1} xc = {list_coord_c[i][0]}, yc = {list_coord_c[i][1]}")
    print(f"z im{i+1} {list_z[i] = }")    
    if i != 0:
        print(f"distance im1 et im{i+1} {list_dz[i-1] } m")

#taille en mètre des axes
y_extend = (list_I[0].shape[0]-1) * pisi
x_extend = (list_I[0].shape[1]-1) * pisi
#axes centrés en 0,0
y_pts = np.linspace(-y_extend/2, y_extend/2, num=list_I[0].shape[0])
x_pts = np.linspace(-x_extend/2, x_extend/2, num=list_I[0].shape[1])
#matrice des axes
X, Y = np.meshgrid(x_pts, y_pts)    

# %% Prépare la recherche de la phase à la lentille
## prepare starting field
phi1 = np.zeros(list_I[0].shape)
u1 = np.sqrt(list_I[0]) * np.exp(1j * phi1)

## Prepare propagation
nux = fftfreq(u1.shape[1], pisi)  # puts zero first
nuy = fftfreq(u1.shape[0], pisi)  # puts zero first
nuX, nuY = np.meshgrid(nux, nuy)  
# u2 = propag(u1, dz, nuX, nuY, la)

# %% Retrouve les paramètres de la phase à la lentille
## Define the functions for the fit
def phase_1p(p, XX, YY):
    """a centered parabola with rotational symmetry:
    p[0] * er**2"""
    center_x = (XX.max() - XX.min())/2 + XX.min()
    # print(f"{center_x = }")
    center_y = (YY.max() - YY.min())/2 + YY.min()
    # print(f"{center_y = }")
    er = np.sqrt( (XX - center_x)**2 + (YY - center_y)**2)
    ph = p[0] * er**2
    return ph

def phase_2p(p, XX, YY):
    """a centered parabola with rotational symmetry:
    p[0] * er + p[1] * er**2"""
    center_x = (XX.max() - XX.min())/2 + XX.min()
    # print(f"{center_x = }")
    center_y = (YY.max() - YY.min())/2 + YY.min()
    # print(f"{center_y = }")
    er = np.sqrt( (XX - center_x)**2 + (YY - center_y)**2)
    ph = p[0] * er + p[1] * er**2
    return ph

def phase_3p(p, XX, YY):
    """a centered parabola with rotational symmetry:
    p[0] + p[1] * er + p[2] * er**2"""
    center_x = (XX.max() - XX.min())/2 + XX.min()
    # print(f"{center_x = }")
    center_y = (YY.max() - YY.min())/2 + YY.min()
    # print(f"{center_y = }")
    er = np.sqrt( (XX - center_x)**2 + (YY - center_y)**2)
    ph = p[0] + p[1] * er + p[2] * er**2
    return ph

def phase_5p(p, XX, YY):
    """a centered parabola with rotational symmetry:
    p[0] + p[1] * er + p[2] * er**2  and 
    p[3] = center_x and p[4] = center_y"""
    center_x = p[3]
    # print(f"{center_x = }")
    center_y = p[4]
    # print(f"{center_y = }")
    er = np.sqrt( (XX - center_x)**2 + (YY - center_y)**2)
    ph = p[0] + p[1] * er + p[2] * er**2
    return ph

mod_phase = phase_2p

# Change the fit models by changing phase to phase2 or phase3 and adapting the parameters in guess_vals
def residuals(pars, sqrtI1, X_mat, Y_mat, I2_measured, I3_measured, I4_measured):  # one could also pass modelfct
    """The difference between simulated ans measured intensity"""
    u1_assumed = sqrtI1 * np.exp(1j * mod_phase(pars, X_mat, Y_mat))
    u2_simulated =propag(u1_assumed, list_dz[0], nuX, nuY, la)  
    u3_simulated =propag(u1_assumed, list_dz[1], nuX, nuY, la)
    u4_simulated =propag(u1_assumed, list_dz[2], nuX, nuY, la) 
    debut_array = np.abs(u2_simulated)**2 - I2_measured
    milieu_array = np.abs(u3_simulated)**2 - I3_measured
    fin_array = np.abs(u4_simulated)**2 - I4_measured
    return np.vstack((debut_array, milieu_array, fin_array)).flatten()

if mod_phase == phase_3p :
    guess_vals = [0,0,0]
elif mod_phase == phase_1p :
    guess_vals = [0]
elif mod_phase == phase_2p :
    guess_vals = [0,0]
else :
    guess_vals = [0,0,0,0.0005,-0.0005]

#leastsq a besoin de la fonction residuals et d'un point de départ pour sa recherche 
#(Mais plus tard on veut modifier leastsq et utiliser minimize plutot)
#fpars1:les paramètres, pcov:la covariance mais nous on s'en sert pas, errmsg2D: message d'erreur si on en a besoin, success2D:on sait qu'on a réussit en fonction de la valeur

fpars,_, infodict, errmsg2D, success2D = leastsq(residuals, 
    guess_vals, args=(u1, X, Y, list_I[1], list_I[2], list_I[3]), full_output=True, 
    epsfcn=0.0001)
if success2D not in [1, 2, 3, 4]:  # If success is equal to 1, 2, 3 or 4, the solution was found.
    print(errmsg2D)
else:
    print(fpars)
    print(f" Nombre d'itération : {infodict['nfev']}")


# %%Compare graphiquement la phase à la lentille
## compare result to input used for simulation of I2_image
phi1_im = iio.imread(file_name_0)


fig, ax = plt.subplots(nrows=1, ncols=2, sharex=True, sharey=True,
                           figsize=(8, 3))
#plt.suptitle(f"Comparison to ground truth at z = {list_z[0]*1e3:.1f} mm")
ax[0].set_title(f'Ground truth of phase')
ax[0].imshow(phi1_im)
ax[1].set_title(f'Phase recovered')

if mod_phase == phase_5p:
    center_x = fpars[-2]
    center_y = fpars[-1]
    inter_x = np.absolute(X[0]-center_x)
    inter_y = np.absolute(Y[:,0]-center_y)
    pos_cent_x = int(inter_x.argmin())
    pos_cent_y = int(inter_y.argmin())
else:
    pos_cent_x = int(X.shape[1]/2)
    pos_cent_y = int(X.shape[0]/2)
        
mod_p = (mod_phase(fpars, X, Y) - mod_phase(fpars, X, Y)[pos_cent_x][pos_cent_y])
         
""" mod_phase(fpars, np.array([pos_cent_x]), np.array([pos_cent_y])))"""

phi_trouver = np.angle(np.exp(1j * mod_p))
cax = ax[1].imshow(phi_trouver, cmap='twilight', 
                   vmin=-np.pi, vmax=np.pi)
fig.colorbar(cax, ax=ax[0])
fig.tight_layout()

# %%
# Que fais cette fonction ?
def phase_perfect(rlens):
    return 9524595.723956442 * rlens**2

#phase créer par la lentille d'après les données constructeur
phi_known = -np.angle(np.exp(1j * phase_perfect(np.sqrt(X**2 + Y**2)))) # why this sign correction?
#phase retrouver

idx_small = 200
idx_large = 300

fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(10, 3))
plt.suptitle(f'from im{i+2}')
ax[0].set_title(f'Difference to ground truth at z = {list_z[0]*1e3:.1f} mm (zoom)')
cax = ax[0].imshow((phi_known[idx_small:idx_large,idx_small:idx_large] - 
                    phi_trouver[idx_small:idx_large,idx_small:idx_large]), 
                    cmap='twilight', vmin=-np.pi/10, vmax=np.pi/10)
fig.colorbar(cax, ax=ax[0])
ax[1].set_title(f'I1 at position z = {list_z[0]*1e3:.1f} mm')
ax[1].imshow(list_I[0][idx_small:idx_large,idx_small:idx_large], 
            cmap='viridis')
ax[2].set_title(f'Difference to ground truth at z = {list_z[0]*1e3:.1f} mm')
cax = ax[2].imshow((phi_known - phi_trouver), 
                    cmap='twilight', vmin=-np.pi/10, vmax=np.pi/10)
fig.colorbar(cax, ax=ax[2])
fig.tight_layout()


plt.show()
# %%
