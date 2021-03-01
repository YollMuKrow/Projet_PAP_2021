# Projet Master 1 PAP 2020-2021
Ce projet porte sur la simulation parallèle du jeu de la vie, inventé par J. Conway en 1970,
qui consiste en un ensemble de régles qui régissent l’évolution d’un automate cellulaire.
Cette simulation s’effectuera à l’aide de l’environnement EASYPAP, dans lequel une version séquentielle de la simulation vous est fournie.

# 1 Cadre du projet
Ce projet est à faire de préférence en binôme. La date prévu pour le rendu est le 20 mai. Votre
rapport, sous forme de fichier au format PDF, contiendra les parties principales de votre code, la
justification des optimisations réalisées, les conditions expérimentales et les graphiques obtenus
accompagnés chacun d’un commentaire le décrivant et l’analysant.

# 2 Description du modèle
Le jeu de la vie simule une forme très basique d’évolution d’un ensemble de cellules en partant
de deux principes simples : pour pouvoir vivre, il faut avoir des voisins ; mais quand il y en a trop,
on étouffe. Le monde est ici un grand damier de cellules vivantes ou mortes, chaque cellule étant
entourée par huit voisines. Pour faire évoluer le monde on découpe le temps en étapes discrètes
et, pour passer d’une étape à la suivante, on compte pour chaque cellule le nombre de cellules
vivantes parmi ses huit voisines puis on applique les règles suivantes :
— Une cellule morte devient vivante si elle a exactement 3 cellules voisines vivantes — autrement elle reste morte.
— Une cellule vivante reste vivante si elle est entourée de 2 ou 3 cellules vivantes — autrement
elle meurt.
Cet ensemble de règles est communément appelé « B3/S23 » (BIRTH if 3 neighbors / SURVIVE
if 2 or 3 neighbors). La simulation est synchrone : à chaque étape, l’état d’une cellule dépend uniquement des états de ses voisines à l’étape précédente.
