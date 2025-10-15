import math
import numpy as np
import random

from utils import pad, unpad

'''
 !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
 !!! NE MODIFIEZ PAS LE CODE EN DEHORS DES BLOCS TODO. !!!
 !!!  L'EVALUATEUR AUTOMATIQUE SERA TRES MECHANT AVEC  !!!
 !!!            VOUS SI VOUS LE FAITES !               !!!
 !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
'''

def fit_transform_matrix(p0, p1):
    """ Calcul la matrice de transformation H tel que p0 * H.T = p1

    Indication importante :
        Vous pouvez utiliser la fonction "np.linalg.lstsq" ou
        la fonction "np.linalg.svd" pour résoudre le problème.

    Entrées :
        p0 : un tableau numpy de dimension (M, 2) contenant
             les coordonnées des points à transformer
        p1 : un tableau numpy de dimension (M, 2) contenant
             les coordonnées des points destination

             Chaque coordonnée [x,y] dans p0 ou p1 indique 
             la position d'un point-clé [col, ligne] dans 
             l'image associée. C-à-d.  p0[i,:] = [x_i, y_i] 
             et  p1[j,:] = [x'_j, y'_j]

    Sortie :
        H  : Tableau numpy de dimension (3,3) représentant la 
             matrice de transformation d'homographie.
    """

    assert (p1.shape[0] == p0.shape[0]),\
        'Nombre différent de points en p1 et p2'

    H = None
    
    #TODO 1 : Calculez la matrice de transformation H.
    # TODO-BLOC-DEBUT    
    M = p0.shape[0]  # Nombre de points
    A = []

    for i in range(M):
        x0, y0 = p0[i, 0], p0[i, 1]
        x1, y1 = p1[i, 0], p1[i, 1]

        A.append([x0, y0, 1, 0, 0, 0, -x0 * x1, -y0 * x1,- x1])
        A.append([0, 0, 0, x0, y0, 1, -x0 * y1, -y0 * y1, -y1])

    A = np.array(A)  # Convertir en np array
    
    # Décomposition SVD
    U, S, Vt = np.linalg.svd(A)

    # La solution est le dernier vecteur singulier (colonne de Vt correspondant à la plus petite valeur singulière)
    H = Vt[-1].reshape(3, 3)
    # TODO-BLOC-FIN

    return H

def ransac(keypoints1, keypoints2, matches, n_iters=500, threshold=1):
    """
    Utilisez RANSAC pour trouver une transformation projective robuste

        1. Sélectionnez un ensemble aléatoire de correspondances
        2. Calculez la matrice de transformation H
        3. Calculez les bonnes correspondances (inliers)
        4. Gardez le plus grand ensemble de bonnes correspondances
        5. En final, recalculez la matrice de transformation H sur 
           tout l'ensemble des bonnes correspondances

    Entrées :
        keypoints1 -- matrice M1 x 2, chaque rangée contient les coordonnées 
                      d'un point-clé (x_i,y_i) dans image1
        keypoints2 -- matrice M2 x 2, chaque rangée contient les coordonnées 
                      d'un point-clé (x'_i,y'_i) dans image2
        matches  -- matrice N x 2, chaque rangée représente une correspondance
                    [indice_dans_keypoints1, indice_dans_keypoints2]
        n_iters -- le nombre d'itérations à effectuer pour RANSAC
        threshold -- le seuil pour sélectionner des bonnes correspondances

    Sorties :
        H -- une estimation robuste de la matrice de transformation H
        matches[max_inliers] -- matrice (max_inliers x 2) des bonnes correspondances 
    """
    # indices des bonnes correspondances dans le tableau 'matches' 
    max_inliers = []
    
    # matrice de transformation Homographique
    H = None
    
    # Initialisation du générateur de nombres aléatoires
    # fixé le seed pour pouvoir comparer le résultat retourné par 
    # cette fonction par rapport à la solution référence
    random.seed(131)
    
    #TODO 2 : Implémentez ici la méthode RANSAC pour trouver une transformation robuste
    # entre deux images image1 et image2.
    # TODO-BLOC-DEBUT    
    for _ in range(n_iters):
        # 4 paires de correspondances au hasard
        s_indices = random.sample(range(len(matches)), 4)
        s_matches = matches[s_indices]
        
        p0 = np.array([keypoints1[i] for i, _ in s_matches])
        p1 = np.array([keypoints2[j] for _, j in s_matches])
        
        # Calculer la matrice de transformation H avec les 4 points
        H_s = fit_transform_matrix(p0, p1)
        
        # Trouver les inliers for all the matches
        inliers = []
        for i, (idx1, idx2) in enumerate(matches):
            x, y = keypoints1[idx1]
            x_prime, y_prime = keypoints2[idx2]

            projected = H_s @ np.array([x, y, 1])
            projected /= projected[2]  # Normalisation

            
            dst = np.array([x_prime, y_prime])

            dist = np.sqrt((projected[0] - dst[0])**2 + (projected[1] - dst[1])**2)
            if dist < threshold:
                inliers.append(i)

        # Vérifier si on a trouvé plus d'inliers and updating them and H
        if len(inliers) > len(max_inliers):
            max_inliers = inliers
            H = H_s


    # Recalculer H avec tous les inliers trouvés
    P0 =[]
    P1 = []
    for i in max_inliers:
        P0.append(keypoints1[matches[i][0]])
        P1.append(keypoints2[matches[i][1]])
    P0 = np.array(P0)
    P1 = np.array(P1)
    if len(max_inliers) > 0:
        H = fit_transform_matrix(P0, P1)
    H = H * -1

    # TODO-BLOC-FIN  
    
    return H, matches[max_inliers]


def get_output_space(imgs, transforms):
    """
    Ceci est une fonction auxiliaire qui prend en entrée une liste d'images et
    des transformations associées et calcule en sortie le cadre englobant
    les images transformées.

    Entrées :
        imgs -- liste des images à transformer
        transforms -- liste des matrices de transformation.

    Sorties :
        output_shape (tuple) -- cadre englobant les images transformées.
        offset -- un tableau numpy contenant les coordonnées du coin (0,0) du cadre
    """

    assert (len(imgs) == len(transforms)),\
        'le nombre d\'images et le nombre de transformations associées ne concordent pas'

    output_shape = None
    offset = None

    # liste pour récupérer les coordonnées de tous les coins dans toutes les images
    all_corners = []

    for img, H in zip(imgs, transforms):
        # coordonnées du coin organisées en (x,y)
        r, c, _ = img.shape        
        corners = np.array([[0, 0], [0, r], [c, 0], [c, r]])

        # transformation homographique des coins          
        warped_corners = pad(corners.astype(float)).dot(H.T).T        
        all_corners.append( unpad( np.divide(warped_corners, warped_corners[2,:] ).T ) )
                          
    # Trouver l'étendue des cadres transformées
    # La forme globale du cadre sera max - min
    all_corners = np.vstack(all_corners)

    corner_min = np.min(all_corners, axis=0)
    corner_max = np.max(all_corners, axis=0)
    
    # dimension (largeur, longueur) de la zone d'affichage retournée
    output_shape = corner_max - corner_min
    
    # Conversion en nombres entiers avec np.ceil et dtype
    output_shape = tuple( np.ceil(output_shape).astype(int) )
    
    # Calcul de l'offset (horz, vert) du coin inférieur du cadre par rapport à l'origine (0,0).
    offset = corner_min

    return output_shape, offset


def warp_image(img, H, output_shape, offset, method=None):
    """
    Déforme l'image img grace à la transformation H. L'image déformée
    est copiée dans une image cible de dimensions 'output_shape'.

    Cette fonction calcule également les coefficients alpha de l'image
    déformée pour un fusionnement ultérieur avec d'autres images.

    Entrées :
        img -- l'image à transformée
        H -- matrice de transformation
        output_shape -- dimensions (largeur, hauteur) de l'image transformée 
        offset --  position (horz, vert) du coin du cadre transformé.
        method -- paramètre de sélection de la méthode de calcul des coefficients alpha.
                  'hlinear' -- le alpha varie linéairement de 1.0 à 0.0
                              en horizontal à partir du centre jusqu'au
                              bord de l'image
                  'vlinear' -- le alpha varie linéairement de 1.0 à 0.0
                              en vertical à partir du centre jusqu'au
                              bord de l'image
                  'linear' -- le alpha varie linéairement de 1.0 à 0.0
                              en horizontal et en vertical à partir du
                              centre jusqu'au bord de l'image
                   None -- le alpha des pixels est égale à 1.0

    Sorties :
        img_warped (np.float32) -- l'image transformée de dimensions = output_shape.
                                   Les valeurs des pixels doivent être dans la
                                   plage [0..1] pour pouvoir visualiser les
                                   résultats avec plt.show(...)

        mask -- tableau numpy de booléens (même dimension que img_warped) indiquant 
                les pixels valides dans l'image de sortie "img_warped"
    """

    image_warped = None
    mask = None
    
    #TODO 3 et 5 : Dans un premier temps (TODO 3), implémentez ici la méthode 
    # qui déforme une image img en applicant dessus la matrice de transformation H. 
    # Vous devez utiliser la projection inverse pour votre implémentation.
    # Pour cela, commencez d'abord par translater les coordonnées de l'image 
    # destination  avec "offset" avant d'appliquer la transformation
    # inverse pour retrouver vos coordonnées dans l'image source.
    
    # TODO 5 : Dans un deuxième temps, implémentez la partie du code dans cette
    # fonction (contrôlée avec le paramètre 'method' donné ci-dessus) qui calcule 
    # les coefficients du canal alpha de l'image transformée.
    # TODO-BLOC-DEBUT 
    if img.dtype == np.uint8:
        img = img.astype(np.float32) / 255.0
    else:
        img = img.astype(np.float32)

    height_dst, width_dst = output_shape[1], output_shape[0]
    height_src, width_src = img.shape[:2]

    # Initialize outputs (with 4 channels  RGB + alpha)
    image_warped = np.zeros((height_dst, width_dst, 4), dtype=np.float32)
    mask = np.zeros((height_dst, width_dst), dtype=bool)

    
    # 1 First creating the alpha map for source image
    
    if method is None:
        alpha_src = np.ones((height_src, width_src), dtype=np.float32)
    else:
        # Create coordinate grids
        x = np.arange(width_src + 1)
        y = np.arange(height_src + 1)
        xPrime, yPrime = np.meshgrid(x, y)
        
        # Center coordinates
        xCenter = width_src  / 2
        yCenter = height_src / 2
        
        # Calculate normalized distances
        dist_x = np.abs(xPrime - xCenter) / xCenter
        dist_y = np.abs(yPrime - yCenter) / yCenter
        
        # Calculate alpha components
        alpha_x = 1 - dist_x
        alpha_y = 1 - dist_y
        
        # Combine based on method
        if method == 'hlinear':
            alpha_src = alpha_x
        elif method == 'vlinear':
            alpha_src = alpha_y
        elif method == 'linear':
            alpha_src = alpha_x * alpha_y
        else:
            alpha_src = np.ones_like(alpha_x)

    # Compute inverse homography
    H_inv = np.linalg.inv(H)

    # 2 Warp both image and alpha using the original loop structure

    for y_dst in range(height_dst):
        for x_dst in range(width_dst):
            # Add offset to destination coordinates
            x = x_dst + offset[0]
            y = y_dst + offset[1]

            # Apply inverse homography
            p_dst = np.array([x, y, 1])
            p_src = H_inv @ p_dst
            p_src /= p_src[2]  # Normalize

            u, v = p_src[0], p_src[1]

            # Check bounds
            if 0 <= u < width_src - 1 and 0 <= v < height_src - 1:
                x0, y0 = int(np.floor(u)), int(np.floor(v))
                x1, y1 = x0 + 1, y0 + 1
                dx, dy = u - x0, v - y0

                # Interpolate RGB channels
                for c in range(3):
                    top_left     = img[y0, x0, c]
                    top_right    = img[y0, x1, c]
                    bottom_left  = img[y1, x0, c]
                    bottom_right = img[y1, x1, c]

                    interpolated_value = (
                        (1 - dx) * (1 - dy) * top_left +
                        dx * (1 - dy) * top_right +
                        (1 - dx) * dy * bottom_left +
                        dx * dy * bottom_right
                    )
                    image_warped[y_dst, x_dst, c] = interpolated_value

                # Interpolate alpha channel from precomputed alpha_src
                a_top_left     = alpha_src[y0, x0]
                a_top_right    = alpha_src[y0, x1]
                a_bottom_left  = alpha_src[y1, x0]
                a_bottom_right = alpha_src[y1, x1]

                alpha_value = (
                    (1 - dx) * (1 - dy) * a_top_left +
                    dx * (1 - dy) * a_top_right +
                    (1 - dx) * dy * a_bottom_left +
                    dx * dy * a_bottom_right
                )
                image_warped[y_dst, x_dst, 3] = alpha_value
                mask[y_dst, x_dst] = True
    # TODO-BLOC-FIN
    
    return image_warped, mask


def naive_fusion(img1_warped, img2_warped):
    """
    fusionne deux images selon la formule :
         merged[i,j] = ( image1[i,j] + image2[i,j] ) / (alpha1[i,j]+ alpha2[i,j])
         
    Entrées :
        img1_warped -- Première image RGBA de dimension (Largeur, Heuteur, 4). 
        img2_warped -- Deuxième image RGBA de dimension (Largeur, Heuteur, 4). 
        
    Sorties :
        merged -- image panoramique RGB de dimension (Largeur, Heuteur, 3). 
    """
    
    assert(img1_warped.shape[0] == img2_warped.shape[0] and img1_warped.shape[1] == img2_warped.shape[1] ), \
                 'les images doivent avoir les mêmes dimensions'

    assert(img1_warped.shape[2] == 4 and img2_warped.shape[2] == 4 ), \
                 'les images doivent avoir 4 canaux : R, G, B et A'

    merged = None
    
    #TODO 4 : Implémentez ici la méthode naïve de fusion de deux images en un panorama
    # TODO-BLOC-DEBUT    
    # Separating les canaux RGB and Alpha
    rgb1 = img1_warped[:, :, :3]
    alpha1 = img1_warped[:, :, 3:]

    rgb2 = img2_warped[:, :, :3]
    alpha2 = img2_warped[:, :, 3:]

    # Somme pondérée des pixels RGB
    numerator = rgb1 * alpha1 + rgb2 * alpha2
    denominator = alpha1 + alpha2

    # avoiding div par 0
    # Si alpha1 + alpha2 == 0, we put la valeur RGB à 0 (no contribution)

    merged = np.where(denominator != 0, numerator / denominator, 0.0).astype(np.float32)

    # TODO-BLOC-FIN

    return merged


def fusion(img1_warped, m1, img2_warped, m2):
    """
    fusionne deux images selon la formule :
         merged[i,j] = ( alpha1[i,j] * image1[i,j] + alpha2[i,j] * image2[i,j] ) / (alpha1[i,j]+ alpha2[i,j])
         
    Entrées :
        img1_warped -- Première image RGBA de dimension (Largeur, Heuteur, 4). 
        m1 -- tableau numpy de booléens de dimension (Largeur, Hauteur) indiquant 
                les pixels valides dans l'image img1_warped.
        img2_warped -- Deuxième image RGBA de dimension (Largeur, Heuteur, 4). 
        m2 -- tableau numpy de booléens de dimension (Largeur, Hauteur) indiquant 
                les pixels valides dans l'image img2_warped.

    Sorties :
        merged -- image panoramique RGB de dimension (Largeur, Heuteur, 3). 
    """
        
    assert(img1_warped.shape[0] == img2_warped.shape[0] and img1_warped.shape[1] == img2_warped.shape[1] ), \
                 'les images doivent avoir les mêmes dimensions'

    assert(img1_warped.shape[2] == 4 and img2_warped.shape[2] == 4 ), \
                 'les images doivent avoir 4 canaux : R, G, B et A'
    
    assert(img1_warped.shape[0] == m1.shape[0] and img1_warped.shape[1] == m1.shape[1] ), \
                 'img1_warped et la carte m1 doivent avoir les mêmes dimensions'

    assert(img2_warped.shape[0] == m2.shape[0] and img2_warped.shape[1] == m2.shape[1] ), \
                 'img2_warped et la carte m2 doivent avoir les mêmes dimensions'
    
    merged = None

    #TODO 6 : Implémentez ici la méthode de pondération pour la fusion de deux images en un panorama
    # TODO-BLOC-DEBUT    
    
    # Initialize output image
    merged = np.zeros((img1_warped.shape[0], img1_warped.shape[1], 3), dtype=np.float32)
    
    # Extract alpha channels
    alpha1 = img1_warped[:,:,3]
    alpha2 = img2_warped[:,:,3]
    
    # Combine masks to know where we have valid pixels
    valid1 = m1
    valid2 = m2
    valid_both = valid1 & valid2
    valid_only1 = valid1 & ~valid2
    valid_only2 = valid2 & ~valid1
    
    # Case 1: Both images have valid pixels at this location
    if np.any(valid_both):
        total_alpha = alpha1[valid_both] + alpha2[valid_both]
        for c in range(3):  # For each RGB channel
            merged[valid_both, c] = (
                alpha1[valid_both] * img1_warped[valid_both, c] + 
                alpha2[valid_both] * img2_warped[valid_both, c]
            ) / total_alpha
    
    # Case 2: Only image1 has valid pixels
    if np.any(valid_only1):
        for c in range(3):
            merged[valid_only1, c] = img1_warped[valid_only1, c]
    
    # Case 3: Only image2 has valid pixels
    if np.any(valid_only2):
        for c in range(3):
            merged[valid_only2, c] = img2_warped[valid_only2, c]
    
    # Clip values to [0,1] range
    merged = np.clip(merged, 0.0, 1.0)
    # TODO-BLOC-FIN

    return merged


def stitch_multiple_images(imgs_list, keypoints_list, matches_list, imgref=0, blend=None):
    """
    Assemble une liste ordonnée d'images.

    Entrées :
        imgs_list -- Liste d'images à assembler
        keypoints_list -- Liste des tableaux de points-clés. Chaque tableau de points-clés
                est une matrice Mi x 2 de points-clés (x_k,y_k) dans imgs_list[i]. (0 <= k < Mi)
        matches_list -- Liste des tableaux de correspondances. Chaque tableau de correspondances 
                est une matrice N x 2, où chaque rangée représente une correspondance
                [indice_dans_keypoints1, indice_dans_keypoints2] entre les images adjacentes 
                i et i+1 dans imgs_list.
        imgref  -- indice de l'image de référence dans imgs_list.
        blend -- paramètre de sélection de la méthode de calcul des coefficients alpha.
                  'hlinear' -- le alpha varie linéairement de 1.0 à 0.0
                              en horizontal à partir du centre jusqu'au
                              bord de l'image
                  'vlinear' -- le alpha varie linéairement de 1.0 à 0.0
                              en vertical à partir du centre jusqu'au
                              bord de l'image
                  'linear' -- le alpha varie linéairement de 1.0 à 0.0
                              en horizontal et en vertical à partir du
                              centre jusqu'au bord de l'image
                   None -- le alpha des pixels est égale à 1.0

    Sorties :
        panorama : Image panoramique finale. 
    """

    assert ( len(imgs_list) > 1 ), \
        'Nombre d\'images à assembler >= 2'
    
    assert ( len(matches_list) == len(imgs_list) - 1 ), \
        'Nombre des tableaux de correspondances doit être égale à len(imgs_list) - 1'

    assert ( 0 <= imgref and imgref < len(imgs_list) ), \
        'L\'indice de l\'image référence doit être inférieur à len(imgs_list)' 


    panorama = None
    
    #TODO BONUS : Votre implémentation ici
    # TODO-BLOC-DEBUT    
    raise NotImplementedError("TODO BONUS : dans panorama.py non implémenté")    
    # TODO-BLOC-FIN
    
    return panorama
