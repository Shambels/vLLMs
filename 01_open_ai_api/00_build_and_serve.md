# Llama.cpp

## 1. build llama.cpp

Le build n'est pas obligatoire et des images docker sont disponnibles. Cette partie vise à montrer comment build llama.

### Outils de compilation

```sh
sudo apt update
sudo apt install -y build-essential cmake git wget libcurl4-openssl-dev linux-headers-$(uname -r)
```

### Driver et dépôt NVIDIA
```sh
distro=$(. /etc/os-release && echo "ubuntu${VERSION_ID/./}")   # -> ubuntu2604
wget https://developer.download.nvidia.com/compute/cuda/repos/${distro}/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt update

# Skipable si nvidia-smi est déjà disponnible
sudo apt install -y nvidia-open
sudo reboot
```

### Cuda Toolkit

Le **CUDA Toolkit** est le kit de développement de NVIDIA pour écrire et compiler des programmes qui s'exécutent sur GPU. Il ne faut pas le confondre avec le **driver** installé à l'étape précédente :

- Le **driver** (`nvidia-open`, `nvidia-smi`) est la couche bas niveau qui pilote la carte. Il suffit pour *exécuter* un binaire déjà compilé avec CUDA.
- Le **toolkit** apporte ce qu'il faut pour *compiler* du code GPU : le compilateur `nvcc`, les headers CUDA, et les bibliothèques d'accélération comme cuBLAS (algèbre linéaire) dont llama.cpp se sert pour les multiplications de matrices.

Lorsqu'on build llama.cpp avec `-DGGML_CUDA=ON`, cmake a besoin de `nvcc` pour compiler les kernels GPU de ggml. C'est pour cela qu'on ajoute `/usr/local/cuda/bin` au `PATH` et `/usr/local/cuda/lib64` au `LD_LIBRARY_PATH` : le premier permet de trouver `nvcc`, le second permet aux binaires de trouver les bibliothèques CUDA au moment de l'exécution.

```sh
sudo apt install -y cuda-toolkit
echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
nvcc --version
```

### Build
```sh
cd ~
git clone https://github.com/ggml-org/llama.cpp
cd llama.cpp

cmake -B build -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release -j$(nproc)
```

### Installation
```sh
sudo cmake --install build --prefix /usr/local
```

### Sanity check
```sh
llama-server --list-devices
```

## 2. Servir son premier modèle
```sh
llama-server \
  -m /chemin/vers/le/modele.gguf \
  -c 65536 \
  -np 1 \
  -fa on \
  -b 2048 -ub 512 \
  -ngl 99 \
  --host 127.0.0.1 --port 8080
```

### Explication des flags

| Flag | Forme longue | Rôle |
|------|--------------|------|
| `-m` | `--model` | Chemin vers le fichier de poids au format GGUF. |
| `-c 65536` | `--ctx-size` | Taille du contexte en tokens (prompt + réponse). Elle détermine la taille du KV cache, donc la mémoire GPU consommée. `0` reprend la valeur d'entraînement du modèle. |
| `-np 1` | `--parallel` | Nombre de requêtes (slots) traitées simultanément. Le contexte `-c` est partagé entre les slots : avec `-np 4` et `-c 65536`, chaque requête dispose de 16384 tokens. |
| `-fa on` | `--flash-attn` | Active Flash Attention, une implémentation de l'attention plus économe en mémoire et plus rapide, surtout sur les longs contextes. Valeurs : `on`, `off`, `auto`. |
| `-b 2048` | `--batch-size` | Taille logique du batch : nombre maximum de tokens soumis au modèle en un appel. Influence surtout la vitesse de traitement du prompt (prefill). |
| `-ub 512` | `--ubatch-size` | Taille physique du micro-batch : le batch logique est découpé en morceaux de cette taille réellement envoyés au GPU. Plus petit = moins de mémoire, plus grand = meilleur débit. `-ub` doit être inférieur ou égal à `-b`. |
| `-ngl 99` | `--n-gpu-layers` | Nombre de couches du modèle déchargées sur le GPU. `99` est une valeur volontairement supérieure au nombre de couches pour tout mettre sur le GPU. Réduire cette valeur permet de faire tourner un modèle qui ne tient pas entièrement en VRAM, au prix de la vitesse. |
| `--host 127.0.0.1` | | Adresse d'écoute du serveur HTTP. `127.0.0.1` limite l'accès à la machine locale ; `0.0.0.0` expose le serveur sur le réseau. |
| `--port 8080` | | Port d'écoute. L'API compatible OpenAI est alors disponible sur `http://127.0.0.1:8080/v1`. |
