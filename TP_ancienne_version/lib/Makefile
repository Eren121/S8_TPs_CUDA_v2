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

TARGETS = lib.a
SOURCES = Logging.cu Array2D.cu Image.cu Time.cu

###################
# Target

OBJECTS = $(addprefix $(BUILD_DIR)/, $(SOURCES:.cu=.o))
.DEFAULT_GOAL = $(firstword $(TARGETS))

$(TARGETS):
%.a: $(OBJECTS)
	$(AR) -rcs -o $@ $^

%.out: $(BUILD_DIR)/%.o
	$(CC) $(LDFLAGS) -o $@ $^ $(LIBS)

$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cu $(INCLUDE_DIR)/%.h
	@mkdir -p $(BUILD_DIR)
	$(CC) $(CFLAGS) -c -o $@ $<

###################
# Nettoyage

.PHONY: clean
clean:
	rm -rf $(BUILD_DIR)

.PHONY: mrproper
mrproper: clean
	rm -f $(TARGETS)

.PHONY: all
all: $(TARGETS)

.PHONY: rebuild
rebuild: mrproper all