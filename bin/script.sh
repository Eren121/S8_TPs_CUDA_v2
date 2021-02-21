#!/bin/bash
# Pour générer de multiples types
# d'image de test à partir de orig.jpeg
# Pour être plus précis dans la mesure des performances,
# on génerera par exemple des images de grandes tailles
# car certains kernels sont extrêment rapides
# ou applicables uniquement sur des images d'une certaine taille

# se mettre dans le repertoire data
cd "${0%/*}"/../data

convert orig.jpeg test.tga # Dans le code C, on ne gère que les images TGA

#convert orig.jpeg -resize 1024x1024 test_1024.tga # Certains codes ne marchent que pour une dim. < nb threads = 1024