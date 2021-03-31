###################
# Variables constantes
# nvcc -dc: relocatable device code: pour pouvoir appeler depuis un kernel des fonctions de la librarie

CC = nvcc
AR = ar
CFLAGS = -g -dc -I $(INCLUDE_DIR)
LDFLAGS = -g
LIBS =

BUILD_DIR = build
SRC_DIR = src
INCLUDE_DIR = include

###################
# Partie à modifier
# Si TARGET ne contient qu'un seul fichier .a: créer une librairie statique avec toutes les sources
# Sinon: TARGET contient 1 ou plus fichier .out: chaque exécutable .out correspond au fichier .cu associé (linké avec lib.a)

TARGETS = add2vec blur bw farhenheit matmul
SOURCES =
LIBS = lib/lib.a

###################
# Target

OBJECTS = $(addprefix $(BUILD_DIR)/, $(SOURCES:.cu=.o))
.DEFAULT_GOAL = $(firstword $(TARGETS))

LIB_INCLUDE_DIRS = $(addsuffix $(INCLUDE_DIR), $(dir $(LIBS)))

.SUFFIXES:
.PHONY: $(TARGETS)
$(TARGETS): %: %.out

$(addsuffix .out, $(TARGETS)): %.out: $(BUILD_DIR)/%.o $(LIBS)
	$(CC) $(LDFLAGS) -I $(LIB_INCLUDE_DIRS) -o $@ $^

$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cu
	@mkdir -p $(BUILD_DIR)
	$(CC) $(CFLAGS) -I $(LIB_INCLUDE_DIRS) -c -o $@ $<

###################
# Nettoyage

.PHONY: clean
clean:
	rm -rf $(BUILD_DIR)

.PHONY: mrproper
mrproper: clean
	rm -f $(addsuffix .out, $(TARGETS))

.PHONY: all
all: $(TARGETS)

.PHONY: rebuild
rebuild: mrproper all
