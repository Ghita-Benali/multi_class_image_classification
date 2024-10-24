import numpy as np
import os #la lecture de fichiers et la manipulation de chemins.

#On va nommer le dossier par le nom de la classe

train_path = '../TP4 4/dataset/train/entrain' #chemin de data de train
training_names = os.listdir(train_path) #liste qui contient les dossiers des image qui se trouve dans chemin
image_paths = []
image_classes = []
class_id = 0
def imglist(path):
    return [os.path.join(path, f) for f in os.listdir(path)]#renvoie une liste contenant les chemins d'accès complets des fichiers dans ce répertoire  images/image1.jpg.
for training_name in training_names:
    dir = os.path.join(train_path, training_name) #ici cad faire dataset/train/city dataset/train/green etc
    class_path = imglist(dir)# parcourir la liste des images qui sont dans chaque dossier
    image_paths += class_path
    image_classes += [class_id]*len(class_path)
    class_id += 1


from skimage.feature import SIFT
from PIL import Image

des_list = []	# Créer une liste où tous les descripteurs seront stockés
descriptor_extractor = SIFT()
for image_path in image_paths:
    im = np.array(Image.open(image_path).convert('L').resize((128,128)))
    descriptor_extractor.detect_and_extract(im)
    kpts = descriptor_extractor.keypoints
    des = descriptor_extractor.descriptors
    des_list.append((image_path, des))

# Empiler tous les descripteurs verticalement dans un tableau numpy
descriptors = des_list[0][1]
for image_path, descriptor in des_list[1:]:
    descriptors = np.vstack((descriptors, descriptor))


#kmeans ne fonctionne que sur les types float
descriptors_float = descriptors.astype(float)
# Effectuer le clustering k-means et la quantification vectorielle
from scipy.cluster.vq import kmeans, vq

k = 400  #k-means avec 200 clusters
codebook, variance = kmeans(descriptors_float, k, 1)#Applique l'algorithme K-means aux descripteurs pour obtenir un codebook (ensemble de centres de cluster) et une variance.
# Calculer l'histogramme des caractéristiques et les représenter sous forme de vecteur
#vq Attribue des codes du codebook à des observations.
im_features = np.zeros((len(image_paths), k), "float32")
for i in range(len(image_paths)):
    words, distance = vq(des_list[i][1], codebook)#Quantifie les descripteurs de chaque image en utilisant le codebook.
    for w in words:#parcourir chaque cluster
        im_features[i][w] += 1 #incremente l'histo pour l'image en cours en fonctions des clusers
#Normaliser les features en supprimant la moyenne et en mettant à l'échelle la variance unitaire


from sklearn.preprocessing import StandardScaler #sert à normaliser les caracteristisues des clusters

stdSlr = StandardScaler().fit(im_features)
im_features = stdSlr.transform(im_features)


from sklearn.svm import LinearSVC
clf = LinearSVC(max_iter=2000)  #Par défaut 100 itérations
clf.fit(im_features, np.array(image_classes))
# Enregistrer le modèle SVM
#Joblib vide l'objet Python dans un seul fichier
import joblib

joblib.dump((clf, training_names, stdSlr, k, codebook), "bof.pkl", compress=3)
print("votre modele est pret à predire")



#, SVM est utilisé pour la classification finale,
# tandis que k-means est utilisé pour réduire la dimensionnalité des données et extraire des caractéristiques discriminatives des images.
