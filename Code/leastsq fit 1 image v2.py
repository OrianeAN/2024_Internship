import imageio.v3 as iio
import time as ti
import numpy as np
from scipy.optimize import leastsq
import matplotlib.pyplot as plt
from scipy.fft import fft2, ifft2, fftfreq

#%% input
# Function to remove the background of the image. Use the frame of the image.
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
# Retourne (xc, yc)
def centroid_position(im):
    XX, YY = np.meshgrid(range(im.shape[1]), range(im.shape[0]))
    xc = int(np.round((XX * im).sum() / im.sum()))
    yc = int(np.round((YY * im).sum() / im.sum()))
    return [xc, yc]

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
file_name_1 = r"input_images\foc lent ideale\z0.000_I.tif"  # should contain info on z-position in meters
file_name_2 = r"input_images\foc lent ideale\z0.150_I.tif"  # should contain info on z-position in meters
file_name_3 = r"input_images\foc lent ideale\z0.310_I.tif"  # should contain info on z-position in meters
file_name_4 = r"input_images\foc lent ideale\z0.400_I.tif"  # should contain info on z-position in meters
file_name_0 = r"input_images\foc lent ideale\z0.000_phi.png"  # the known phase used at for the generation of the images

list_file = [file_name_1, file_name_2, file_name_3, file_name_4]
nb_im = len(list_file)

pisi = 11.71875e-6  # pixel size in meters
la = 1064e-9  # wavelength in meters

#%% Mise en place des variables
list_im = []
list_I = []
list_coord_c = []
list_z = []
list_dz = []
for i in range(nb_im):
    list_im += [iio.imread(list_file[i])]
    list_im[i] = remove_bkg(list_im[i], 0.02)
    if i==1: #image qu'on veut décaler
        #Création du décalage selon y sur l'image 2 centre_im < centre_xy
        #list_im[1] = np.vstack((list_im[1][150:], list_im[1][:150]))

        #Création du décalage selon y sur l'image 2 centre_im > centre_xy
        list_im[1] = np.vstack((list_im[1][-150:], list_im[1][:-150]))

        #Création du décalage selon x sur l'image 2 centre_im < centre_xy
        #list_im[1] = np.hstack((list_im[1][:,150:], list_im[1][:,:150]))

        #Création du décalage selon x sur l'image 2 centre_im < centre_xy
        list_im[1] = np.hstack((list_im[1][:,-150:], list_im[1][:,:-150]))
    list_I += [list_im[i] / list_im[i].sum()]
    list_coord_c += [centroid_position(list_im[i])]
    list_z += [list_file[i].split('.tif')[0]]
    list_z[i] = float(list_z[i][-7:-2])
    if i != 0:
        list_dz += [list_z[i] - list_z[0]]

#Affichage des infos importantes
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

#%%
#Recentrage des intensités (les images ont les modifient pas)
def recentrage(im, i) :
    if list_coord_c[i][0] != len(x_pts)//2 : #si l'image est décalé selon les x
        dist_centres = np.abs(list_coord_c[i][0] - len(x_pts)//2)
        if list_coord_c[i][0] < len(x_pts)//2 : # si centre_im < centre_xy
            im = np.hstack((im[:, -dist_centres:], im[:, :-dist_centres]))
            list_coord_c[i][0] = centroid_position(im)[0]
            list_I[i]=im
            return recentrage(im, i)
        if list_coord_c[i][0] > len(x_pts)//2 : # si centre_im < centre_xy
            im = np.hstack((im[:, dist_centres:], im[:, :dist_centres]))
            list_coord_c[i][0] = centroid_position(im)[0]
            list_I[i]=im
            return recentrage(im, i)

    if list_coord_c[i][1] != len(y_pts)//2 : #si l'image est décalé selon les y
        dist_centres = np.abs(list_coord_c[i][1] - len(y_pts)//2)
        if list_coord_c[i][1] < len(x_pts)//2 : #si centre_im < centre_xy
            im = np.vstack((im[-dist_centres:], im[:-dist_centres]))
            list_coord_c[i][1] = centroid_position(im)[1]
            list_I[i]=im
            return recentrage(im, i)
        if list_coord_c[i][1] > len(x_pts)//2 : #si centre_im > centre_xy
            im = np.vstack((im[dist_centres:], im[:dist_centres]))
            list_coord_c[i][1] = centroid_position(im)[1]
            list_I[i]=im
            return recentrage(im, i)
    if list_coord_c[i] == [len(x_pts)//2, len(y_pts)//2]:
        return True

for i in range(len(list_I)):
    while recentrage(list_I[i], i) != True:
        recentrage(list_I[i], i)

# %%Affiche image des intensités aux positions
fig, ax = plt.subplots(nrows=2, ncols=2, sharex=True, sharey=True,
                           figsize=(8, 3))
plt.suptitle("Input")
a_labels = np.array([-x_extend/2, x_extend/2, -y_extend/2, y_extend/2])
a_labels = list(a_labels * 1000)  # put axes labels in mm
ax[0][0].set_title(f'I1 at position z = {list_z[0]*1e3:.1f} mm')
ax[0][0].imshow(list_I[0], cmap='viridis', extent=a_labels)
ax[0][1].set_title(f'I2 at position z = {list_z[1]*1e3:.1f} mm')
ax[0][1].imshow(list_I[1], cmap='viridis', extent=a_labels)
ax[1][0].set_title(f'I3 at position z = {list_z[2]*1e3:.1f} mm')
ax[1][0].imshow(list_I[2], cmap='viridis', extent=a_labels)
ax[1][1].set_title(f'I4 at position z = {list_z[3]*1e3:.1f} mm')
ax[1][1].imshow(list_I[3], cmap='viridis', extent=a_labels)
fig.tight_layout()
for i in range(nb_im):
    print(f"I{i+1} = {list_I[i].max()}")



# %% Prépare la recherche de la phase à la lentille
## prepare starting field
phi1 = np.zeros(list_I[0].shape)
u1 = np.sqrt(list_I[0]) * np.exp(1j * phi1)

## Prepare propagation
nux = fftfreq(u1.shape[1], pisi)  # puts zero first
nuy = fftfreq(u1.shape[0], pisi)  # puts zero first
nuX, nuY = np.meshgrid(nux, nuy)  


# %% Retrouve les paramètres de la phase à la lentille
## Define the functions for the fit
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
def residuals(pars, sqrtI1, X_mat, Y_mat, I2_measured, dz):  # one could also pass modelfct
    """The difference between simulated ans measured intensity"""
    u1_assumed = sqrtI1 * np.exp(1j * mod_phase(pars, X_mat, Y_mat))
    u2_simulated =propag(u1_assumed, dz, nuX, nuY, la)
    return (np.abs(u2_simulated)**2 - I2_measured).flatten()  # not sure if flatten is necessary

if mod_phase == phase_3p :
    guess_vals = [0, 0, 0]
elif mod_phase == phase_1p :
    guess_vals = [0]
elif mod_phase == phase_2p :
    guess_vals = [0, 0]
else :
    guess_vals = [500, 500, 500, 0.0005, -0.0005]


#fpars1:les paramètres, pcov:la covariance mais nous on s'en sert pas, errmsg2D: message d'erreur si on en a besoin, success2D:on sait qu'on a réussit en fonction de la valeur
list_fpars = []
for i in range(nb_im-1):
    fpars,_, infodict, errmsg2D, success2D = leastsq(residuals, 
        guess_vals, args=(u1, X, Y, list_I[i+1], list_dz[i]), full_output=True, 
        epsfcn=0.0001)
    if success2D not in [1, 2, 3, 4]:  # If success is equal to 1, 2, 3 or 4, the solution was found.
        print(errmsg2D)
    else:
        print(fpars)
        list_fpars += [fpars]
        print(f" Nombre d'itération : {infodict['nfev']}")


# %%Compare graphiquement la phase à la lentille
## compare result to input used for simulation of I2_image
phi1_im = iio.imread(file_name_0)

fig, ax = plt.subplots(nrows=2, ncols=2, sharex=True, sharey=True,
                           figsize=(8, 3))
plt.suptitle(f"Comparison to ground truth at z = {list_z[0]*1e3:.1f} mm")
ax[0][0].set_title(f'ground truth of phase')
ax[0][0].imshow(phi1_im)


if mod_phase == phase_5p :
    ax[0][1].set_title(f'result of fit from im2')
    center_x_im2 = list_fpars[0][-2]
    center_y_im2 = list_fpars[0][-1]
    inter_x_im2 = np.absolute(X[0]-center_x_im2)
    inter_y_im2 = np.absolute(Y[:,0]-center_y_im2)
    pos_cent_x_im2 = int(inter_x_im2.argmin())
    pos_cent_y_im2 = int(inter_y_im2.argmin())
    
    mod_p_im2 = (mod_phase(list_fpars[0], X, Y) - mod_phase(list_fpars[0], X, Y)[pos_cent_x_im2][pos_cent_y_im2])
    """
    mod_p_im2 = (mod_phase(list_fpars[0], X, Y) - 
            mod_phase(list_fpars[0], np.array([center_x_im2]), np.array([center_y_im2])))
    """
    phi_trouver_im2 = np.angle(np.exp(1j * mod_p_im2))
    cax = ax[0][1].imshow(phi_trouver_im2, cmap='twilight', 
                    vmin=-np.pi, vmax=np.pi)
    fig.colorbar(cax, ax=ax[0][1])

    ax[1][0].set_title(f'result of fit from im3')
    center_x_im3 = list_fpars[1][-2]
    center_y_im3 = list_fpars[1][-1]
    inter_x_im3 = np.absolute(X[0]-center_x_im3)
    inter_y_im3 = np.absolute(Y[:,0]-center_y_im3)
    pos_cent_x_im3 = int(inter_x_im3.argmin())
    pos_cent_y_im3 = int(inter_y_im3.argmin())

    mod_p_im3 = (mod_phase(list_fpars[1], X, Y) - mod_phase(list_fpars[1], X, Y)[pos_cent_x_im3][pos_cent_y_im3])
    """    
    mod_p_im3 = (mod_phase(list_fpars[1], X, Y) - 
                 mod_phase(list_fpars[1], np.array([center_x_im3]), np.array([center_y_im3])))
    """
    phi_trouver_im3 = np.angle(np.exp(1j * mod_p_im3))
    cax = ax[1][0].imshow(phi_trouver_im3, cmap='twilight', 
                    vmin=-np.pi, vmax=np.pi)
    fig.colorbar(cax, ax=ax[1][0])

    ax[1][1].set_title(f'result of fit from im4')
    center_x_im4 = list_fpars[2][-2]
    center_y_im4 = list_fpars[2][-1]
    inter_x_im4 = np.absolute(X[0]-center_x_im4)
    inter_y_im4 = np.absolute(Y[:,0]-center_y_im4)
    pos_cent_x_im4 = int(inter_x_im4.argmin())
    pos_cent_y_im4 = int(inter_y_im4.argmin())

    mod_p_im4 = (mod_phase(list_fpars[2], X, Y) - mod_phase(list_fpars[2], X, Y)[pos_cent_x_im4][pos_cent_y_im4])
    """    
    mod_p_im4 = (mod_phase(list_fpars[2], X, Y) - 
            mod_phase(list_fpars[2], np.array([center_x_im4]), np.array([center_y_im4])))
    """

    phi_trouver_im4 = np.angle(np.exp(1j * mod_p_im4))
    cax = ax[1][1].imshow(phi_trouver_im4, cmap='twilight', 
                    vmin=-np.pi, vmax=np.pi)
    fig.colorbar(cax, ax=ax[1][1])
    fig.tight_layout()
else:
    pos_cent_x = int(X.shape[1]/2)
    pos_cent_y = int(X.shape[0]/2)

    ax[0][1].set_title(f'result of fit from im2')
    
    mod_p_im2 = (mod_phase(list_fpars[0], X, Y) - 
            mod_phase(list_fpars[0], np.array([pos_cent_x]), np.array([pos_cent_y])))
    
    phi_trouver_im2 = np.angle(np.exp(1j * mod_p_im2))
    cax = ax[0][1].imshow(phi_trouver_im2, cmap='twilight', 
                    vmin=-np.pi, vmax=np.pi)
    fig.colorbar(cax, ax=ax[0][1])

    ax[1][0].set_title(f'result of fit from im3')

    mod_p_im3 = (mod_phase(list_fpars[1], X, Y) - 
            mod_phase(list_fpars[1], np.array([pos_cent_x]), np.array([pos_cent_y])))
    
    phi_trouver_im3 = np.angle(np.exp(1j * mod_p_im3))
    cax = ax[1][0].imshow(phi_trouver_im3, cmap='twilight', 
                    vmin=-np.pi, vmax=np.pi)
    fig.colorbar(cax, ax=ax[1][0])
    ax[1][1].set_title(f'result of fit from im4')

    mod_p_im4 = (mod_phase(list_fpars[2], X, Y) - 
            mod_phase(list_fpars[2], np.array([pos_cent_x]), np.array([pos_cent_y])))
    
    phi_trouver_im4 = np.angle(np.exp(1j * mod_p_im4))
    cax = ax[1][1].imshow(phi_trouver_im4, cmap='twilight', 
                    vmin=-np.pi, vmax=np.pi)
    fig.colorbar(cax, ax=ax[1][1])
    fig.tight_layout()

# %%
# Que fais cette fonction ?
def phase_perfect(rlens):
    return 9524595.723956442 * rlens**2

#phase créer par la lentille d'après les données constructeur
phi_known = -np.angle(np.exp(1j * phase_perfect(np.sqrt(X**2 + Y**2)))) ############# why this sign correction?
#phase retrouver
list_phi_fitted = [phi_trouver_im2, phi_trouver_im3, phi_trouver_im4]


idx_small = 200
idx_large = 300


for i in range(nb_im-1):
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(8, 3))
    if i == 0:
        text = 'before the focale point (150mm)'
    elif i == 1:
        text = 'at the focale point (310mm)'
    else:
        text = 'after the focale point (400mm)'
    plt.suptitle(f'Image used : at the lens and '+text)
    ax[0].set_title(f'The ground truth of phase')
    cax = ax[0].imshow(phi_known, cmap='twilight', vmin=-np.pi, vmax=np.pi)
    fig.colorbar(cax, ax=ax[0])
    ax[1].set_title(f'Phase recovered')
    cax = ax[1].imshow(list_phi_fitted[i], 
                        cmap='twilight', vmin=-np.pi, vmax=np.pi)
    fig.colorbar(cax, ax=ax[1])
    fig.tight_layout()
fig.tight_layout()


# %% Afficher qi² cas à 2 paramètres
"""
Pour les 3 prends 1min30
"""
"""
x_qi = np.linspace(-3000,5000,20) #paramètres a dans : a*er + b*er**2
y_qi = np.linspace(-51*10**6, -3*10**6, 20) #paramètres b dans : a*er + b*er**2
X_qi, Y_qi = np.meshgrid(x_qi[i], y_qi[i]) 



for i in range(len(list_im)-1):
    start = ti.time()

    Qi_c = np.zeros((np.size(y_qi), np.size(x_qi)))
    for x in range(np.size(x_qi)):
        for y in range(np.size(y_qi)):
            Qi_c[y, x] = (residuals(np.array([x_qi[x], y_qi[y]]), u1, X, Y, list_I[i+1], list_dz[i])**2).sum()

    end = ti.time()
    print(end - start)
    if i == 0:
        inter_x_Qi = np.absolute(x_qi-(-1.29004621e-01))
        inter_y_Qi = np.absolute(y_qi-(-9.51415729e+06))
    elif i == 1:
        inter_x_Qi = np.absolute(x_qi-(4.45704394e+02))
        inter_y_Qi = np.absolute(y_qi-(-9.74506931e+06))
    else:
        inter_x_Qi = np.absolute(x_qi-(-1.86239937e+00))
        inter_y_Qi = np.absolute(y_qi-(-5.23847075e+06))
    pos_x_Qi = int(inter_x_Qi.argmin())
    pos_y_Qi = int(inter_y_Qi.argmin())
    Qi_c[pos_y_Qi, pos_x_Qi] = 0

    fig, ax = plt.subplots()
    img = ax.imshow(Qi_c, cmap='viridis', norm='log' ,extent=[x_qi[0], x_qi[-1], y_qi[0], y_qi[-1]], aspect='auto', origin='lower')
    fig.colorbar(img)
    
    plt.xlabel('linear coefficient:b (f(r)=b*r+c*r²)')
    plt.ylabel('quadratic coefficient:c (f(r)=b*r+c*r²)')
"""
plt.show()
# %%
