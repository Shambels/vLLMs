# vLLM : faire avec vLLM ce que nous avons fait avec llama-server

Jusqu'ici, nous avons tout fait avec `llama-server` : servir un modèle, l'interroger avec le SDK OpenAI, régler l'échantillonnage, envoyer des images, contraindre la sortie, optimiser la mémoire et la vitesse, mesurer. Ce chapitre reprend chacune de ces étapes avec **vLLM**, met les options en correspondance et montre ce qui change vraiment.

Les deux serveurs exposent la même API compatible OpenAI. Le code client des notebooks fonctionne donc presque tel quel : ce sont surtout la façon de lancer le serveur, le format des modèles et la gestion de la mémoire qui diffèrent.

> Versions de référence : vLLM 0.30 (septembre 2026). Les options de vLLM évoluent vite et plusieurs ont été renommées ou retirées ces derniers mois : en cas de doute, `vllm serve --help` fait foi.

## 1. Deux philosophies

| | llama.cpp / llama-server | vLLM |
|---|---|---|
| Langage | C/C++, sans dépendance | Python, PyTorch, kernels CUDA |
| Cible | un utilisateur ou quelques-uns, poste de travail, machines modestes | beaucoup d'utilisateurs simultanés, serveurs de production |
| Matériel | CPU, GPU NVIDIA, AMD, Intel, Apple, mélange CPU + GPU | Linux avec GPU NVIDIA (compute capability ≥ 7.5, soit RTX 20xx et plus récents), AMD ROCm, CPU en support secondaire. Pas de Windows natif (WSL) ni de GPU Apple |
| Format des modèles | un fichier GGUF, quantisé à la conversion | le dépôt Hugging Face d'origine (safetensors), éventuellement quantisé (FP8, AWQ, GPTQ, NVFP4) |
| Mémoire | alloue ce qui est demandé : poids + KV cache de `-c` tokens | réserve d'emblée 92 % de la VRAM et la découpe en blocs de KV cache partagés entre toutes les requêtes |
| Débordement en RAM | natif et fin (`-ngl`, `-ncmoe`) | possible mais coûteux (`--cpu-offload-gb`) |
| Démarrage | quelques secondes | compilation et capture de graphes CUDA : souvent une minute ou plus au premier lancement |
| Point fort | simplicité, portabilité, faire tenir un gros modèle sur peu de VRAM | débit sous forte charge (*continuous batching*, *PagedAttention*), multi-GPU, observabilité |

Le cœur de vLLM est **PagedAttention** : le KV cache est découpé en blocs de 16 tokens, alloués à la demande comme les pages de la mémoire virtuelle d'un système d'exploitation. Aucun espace n'est réservé à l'avance pour une requête, et des requêtes qui partagent le même début de prompt partagent physiquement les mêmes blocs. Combiné au *continuous batching*, qui fait entrer et sortir les requêtes du batch à chaque itération, c'est ce qui permet à vLLM de servir des centaines de requêtes en parallèle sur un seul GPU.

## 2. Installer

Avec llama.cpp, nous avons compilé le projet avec le CUDA Toolkit (voir `00_build_and_serve.md`). vLLM s'installe comme un paquet Python : les wheels embarquent déjà les bibliothèques CUDA nécessaires. Seul le **driver** NVIDIA est requis sur la machine, pas le toolkit.

```sh
# Dans un environnement dédié : vLLM impose sa propre version de PyTorch
uv venv .venv-vllm --python 3.13
source .venv-vllm/bin/activate
uv pip install vllm --torch-backend=auto
vllm --version
```

Gardez vLLM dans un environnement séparé de celui des notebooks : il pèse plusieurs Go et fixe des versions précises de PyTorch et de nombreuses dépendances. Les notebooks n'ont besoin que du client `openai`.

L'équivalent des images Docker de llama.cpp :

```sh
docker run --runtime nvidia --gpus all \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -v vllm-cache:/root/.cache/vllm \
  --env "HF_TOKEN=$HF_TOKEN" \
  -p 8000:8000 --ipc=host \
  vllm/vllm-openai:latest \
  Qwen/Qwen3-8B --max-model-len 32768
```

- `--ipc=host` est nécessaire : PyTorch utilise la mémoire partagée entre processus.
- Le volume `vllm-cache` conserve le cache de compilation d'un lancement à l'autre (voir la section 3).
- Tout ce qui suit le nom de l'image est passé à `vllm serve`.

## 3. Servir son premier modèle

### La même commande dans les deux mondes

La commande du chapitre 00 :

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

Son équivalent avec vLLM, ici avec Qwen3-8B (contexte natif de 32 768 tokens) :

```sh
vllm serve Qwen/Qwen3-8B \
  --max-model-len 32768 \
  --max-num-seqs 1 \
  --max-num-batched-tokens 2048 \
  --gpu-memory-utilization 0.92 \
  --host 127.0.0.1 --port 8080
```

| llama-server | vLLM | Ce qui change |
|---|---|---|
| `-m fichier.gguf` | premier argument : dépôt Hugging Face (`Qwen/Qwen3-8B`) ou dossier local | vLLM télécharge le dépôt dans `~/.cache/huggingface`, comme `llama-server -hf` |
| `-c 65536` | `--max-model-len 32768` | `-c` est le contexte **total**, partagé entre les slots. `--max-model-len` est la longueur maximale **d'une** requête, prompt et réponse compris |
| `-np 1` | `--max-num-seqs 1` | nombre maximal de requêtes traitées ensemble. Le défaut de vLLM est 256 : le limiter à 1 n'a de sens que pour une comparaison |
| `-fa on` | rien à faire (`--attention-backend` pour forcer) | FlashAttention ou FlashInfer est choisi automatiquement |
| `-b 2048 -ub 512` | `--max-num-batched-tokens 2048` | un seul budget de tokens par itération (voir la section 7) |
| `-ngl 99` | rien à faire | vLLM met tout sur le GPU, il n'y a pas de notion de couches sur CPU |
| *(rien)* | `--gpu-memory-utilization 0.92` | part de la VRAM **totale** que vLLM s'approprie au démarrage (défaut 0,92) |
| `--host 127.0.0.1` | `--host 127.0.0.1` | sans `--host`, llama-server n'écoute qu'en local, alors que vLLM écoute sur **toutes** les interfaces |
| `--port 8080` | `--port 8080` | port par défaut : 8080 pour llama-server, **8000** pour vLLM |

### Ce qui se passe au démarrage

Là où llama-server est prêt en quelques secondes, vLLM enchaîne plusieurs étapes :

1. Chargement des poids : `Model loading took X GiB and Y seconds`.
2. Compilation du modèle avec `torch.compile`.
3. Profilage de la mémoire avec un batch factice, pour mesurer ce que prennent les activations.
4. Allocation du KV cache avec toute la mémoire restante.
5. Capture des graphes CUDA, qui accélèrent le decoding.
6. `Starting vLLM server on http://127.0.0.1:8080`.

Le premier lancement prend souvent une minute ou plus. Les suivants sont plus rapides, car la compilation est mise en cache dans `~/.cache/vllm`. Pour développer ou tester rapidement, `--enforce-eager` saute la compilation et les graphes CUDA : le démarrage est bien plus court, mais le decoding plus lent.

### Lire les logs : la mémoire selon vLLM

Deux lignes résument tout le budget mémoire (valeurs d'exemple, pour Qwen3-8B en BF16) :

```
Available KV cache memory: 5.12 GiB
GPU KV cache size: 37,264 tokens, Maximum concurrency for 32,768 tokens per request: 1.14x
```

- La première donne la place restée libre pour le KV cache, une fois les poids, les activations et les graphes CUDA déduits des 92 %.
- La seconde la traduit en tokens, puis en nombre de requêtes de longueur maximale qui tiennent **en même temps**.

C'est la différence de fond avec llama-server. Avec llama-server, `-c` fixe la taille du KV cache, et la VRAM consommée en découle. Avec vLLM, c'est l'inverse : la VRAM disponible fixe la taille du KV cache, et `--max-model-len` ne fait que plafonner chaque requête. Conséquences :

- `nvidia-smi` affiche environ 22 Go occupés sur une carte de 24 Go, même avec un petit modèle et sans aucune requête. C'est normal.
- Si le KV cache ne peut pas contenir une seule requête de `--max-model-len` tokens, vLLM refuse de démarrer et indique la longueur maximale possible. Il faut alors baisser `--max-model-len`, quantiser le KV cache ou prendre un modèle plus léger.
- Une requête dont le prompt plus `max_tokens` dépasse `--max-model-len` est refusée avec une erreur HTTP 400. Il n'y a pas d'équivalent du context shift.

**Exemple avec Qwen3-8B en BF16 sur 24 Go** (ordres de grandeur, les logs donnent les vraies valeurs) :

| Poste | Taille |
|---|---:|
| Budget : 24 Gio × 0,92 | ~22,1 Gio |
| Poids en BF16 (8,2 milliards × 2 octets) | ~15,3 Gio |
| Activations et graphes CUDA | ~1,5 Gio |
| Reste pour le KV cache | ~5,3 Gio |
| KV cache par token (36 couches × 8 têtes KV × 128 × 2 × 2 octets) | 144 Kio |
| Capacité | ~38 000 tokens |

Le même modèle en Q4_K_M avec llama-server ne pèse que 5 Go. C'est la grande différence pratique sur une carte grand public : sans checkpoint quantisé, vLLM laisse peu de place au KV cache (voir la section 4).

## 4. Le format du modèle : Hugging Face au lieu de GGUF

### Ce que contient un dépôt Hugging Face

Là où un GGUF met tout dans un seul fichier (voir `04_gguf.md`), vLLM lit le dépôt d'origine du modèle :

| Fichier | Rôle | Équivalent GGUF |
|---|---|---|
| `config.json` | architecture : couches, têtes, contexte | métadonnées `{arch}.*` |
| `*.safetensors` et `model.safetensors.index.json` | les poids, souvent découpés en plusieurs fichiers | données des tenseurs |
| `tokenizer.json`, `tokenizer_config.json` | tokenizer et tokens spéciaux | `tokenizer.ggml.*` |
| `chat_template.jinja` ou champ du `tokenizer_config.json` | le chat template | `tokenizer.chat_template` |
| `generation_config.json` | paramètres d'échantillonnage recommandés par l'éditeur | pas d'équivalent systématique |
| `preprocessor_config.json` | prétraitement des images, pour les modèles vision | le fichier `mmproj` |

Pour un modèle vision, l'encodeur d'images fait partie du dépôt : il n'y a **pas** de fichier `mmproj` séparé ni d'option `--mmproj`.

### Lire `config.json` comme nous lisions le GGUF

```python
import json
from huggingface_hub import hf_hub_download   # installé avec vLLM, sinon pip install huggingface_hub

cfg = json.load(open(hf_hub_download("Qwen/Qwen3-8B", "config.json")))
cfg = cfg.get("text_config", cfg)   # les modèles multimodaux rangent la partie texte à part

print("Couches   :", cfg["num_hidden_layers"])
print("Têtes KV  :", cfg["num_key_value_heads"])
print("head_dim  :", cfg.get("head_dim", cfg["hidden_size"] // cfg["num_attention_heads"]))
print("Contexte  :", cfg["max_position_embeddings"])
```

Seul `config.json` est téléchargé, pas les poids. La formule du KV cache du chapitre 03 s'applique telle quelle.

### La quantisation

Avec llama.cpp, nous choisissons un **fichier** quantisé (Q4_K_M, Q8_0…). Avec vLLM, nous choisissons un **dépôt** déjà quantisé, par exemple `Qwen/Qwen3-8B`, `Qwen/Qwen3-8B-FP8` ou `Qwen/Qwen3-8B-AWQ`. vLLM détecte la méthode tout seul à partir du `config.json`.

| llama.cpp | vLLM | Bits | Remarque |
|---|---|---:|---|
| F16 / BF16 | dépôt d'origine | 16 | |
| Q8_0 | FP8 | 8 | accéléré matériellement à partir des RTX 40xx (Ada) ; moins rapide sur les cartes plus anciennes |
| Q4_K_M, Q4_0 | AWQ, GPTQ | 4 | à télécharger déjà quantisé |
| *(aucun)* | NVFP4, MXFP4 | 4 | formats 4 bits natifs des GPU Blackwell |
| Q2_K à Q6_K, IQ* | GGUF via `vllm-gguf-plugin` | 2 à 6 | expérimental, sorti de vLLM depuis la version 0.24 |
| *(conversion préalable)* | `--quantization fp8` sur un dépôt BF16 | 8 | quantisation à la volée au chargement |

Deux conséquences pour notre GPU de 24 Go :

- **Qwen3-8B** tient en BF16, mais ne laisse que ~5 Gio de KV cache. En FP8 (~9 Gio de poids), il en laisse plus du double.
- **Qwen3.8-27B**, le modèle du chapitre 03, pèse ~55 Go en BF16 et ~27 Go en FP8 : il ne tient pas. Il faut un checkpoint 4 bits (AWQ, GPTQ ou NVFP4, ~15 à 17 Go) et un KV cache en FP8. Avec llama.cpp, le GGUF UD-Q4_K_XL nous donnait le même résultat en un seul fichier.

> **GGUF dans vLLM.** Servir un GGUF est possible avec le plugin (`vllm serve ./modele.gguf --tokenizer Qwen/Qwen3-8B`), mais la documentation le décrit comme très expérimental et peu optimisé. Les GGUF représentent une part infime de l'usage de vLLM : pour un GGUF, llama-server reste le bon outil.

## 5. L'API OpenAI (chapitre 01)

### Côté client, presque rien ne change

```python
from openai import OpenAI

BASE_URL = "http://127.0.0.1:8000"      # 8000 par défaut avec vLLM
MODEL = "Qwen/Qwen3-8B"                 # doit correspondre au modèle servi

client = OpenAI(base_url=f"{BASE_URL}/v1", api_key="pas-besoin")
print([m.id for m in client.models.list().data])
```

Les appels `chat.completions.create`, le streaming, la gestion de l'historique et `usage` fonctionnent exactement comme dans le chapitre 01.

**Le paramètre `model` compte.** llama-server l'ignore, ce qui nous permettait d'écrire `model='local'`. vLLM vérifie qu'il correspond au nom servi, et renvoie sinon une erreur 404 *« The model `local` does not exist »*. Deux solutions :

- utiliser le nom renvoyé par `client.models.list()` ;
- lancer le serveur avec `--served-model-name local` pour garder le code des notebooks tel quel.

### Les endpoints

| llama-server | vLLM | Remarque |
|---|---|---|
| `GET /health` | `GET /health` | identique |
| `GET /v1/models` | `GET /v1/models` | vLLM y indique aussi `max_model_len` |
| `POST /v1/chat/completions` | `POST /v1/chat/completions` | identique |
| `POST /tokenize` avec `with_pieces` | `POST /tokenize` avec `return_token_strs: true` | vLLM accepte aussi `messages` : il applique alors le chat template avant de tokeniser |
| `POST /detokenize` | `POST /detokenize` | identique |
| `POST /apply-template` | *(aucun)* | `/tokenize` avec `messages`, puis `/detokenize` |
| `GET /props` | *(aucun)* | les réglages sont dans les logs de démarrage ; `/v1/models` et `/version` en donnent une partie |
| `GET /slots` | `GET /metrics` | métriques Prometheus : requêtes en cours, en attente, remplissage du KV cache |

Pour voir le prompt réellement envoyé au modèle, comme avec `/apply-template` :

```python
import requests

messages = [{"role": "system", "content": "Tu es un robot assistant."},
            {"role": "user", "content": "Bonjour"}]

tok = requests.post(f"{BASE_URL}/tokenize",
                    json={"model": MODEL, "messages": messages, "return_token_strs": True}).json()
print(tok["count"], "tokens :", tok["token_strs"][:15])

prompt = requests.post(f"{BASE_URL}/detokenize", json={"model": MODEL, "tokens": tok["tokens"]}).json()["prompt"]
print(prompt)
```

### Mesurer : pas de `timings`

llama-server renvoie ses mesures dans `model_extra['timings']` à chaque réponse. vLLM ne renvoie rien de tel par défaut. Deux options de lancement s'en approchent :

- `--enable-per-request-metrics` ajoute un champ `metrics` à chaque réponse ;
- `--enable-prompt-tokens-details` ajoute `usage.prompt_tokens_details.cached_tokens`, le nombre de tokens repris du cache.

| llama-server (`timings`) | vLLM | Disponible avec |
|---|---|---|
| `cache_n` | `usage.prompt_tokens_details.cached_tokens` | `--enable-prompt-tokens-details` |
| `prompt_n` | `usage.prompt_tokens` − `cached_tokens` | idem |
| `prompt_ms` | `metrics.time_to_first_token_ms` (inclut l'attente éventuelle dans la file, détaillée dans `queue_time_ms`) | `--enable-per-request-metrics` |
| `predicted_n` | `usage.completion_tokens` | toujours |
| `predicted_ms` | `metrics.generation_time_ms` | `--enable-per-request-metrics` |
| `predicted_per_token_ms` | `metrics.mean_itl_ms` | idem |
| `predicted_per_second` | `metrics.tokens_per_second` | idem |
| `draft_n`, `draft_n_accepted` | `metrics.speculative_decoding` | `--per-request-spec-decode-metrics` |

```python
r = client.chat.completions.create(model=MODEL, messages=messages, max_tokens=512)
print(r.usage)                         # prompt_tokens_details rempli si l'option est active
print(r.model_extra.get("metrics"))    # None si --enable-per-request-metrics n'est pas actif
```

En streaming, `usage` n'est envoyé dans le dernier chunk que si la requête le demande :

```python
stream = client.chat.completions.create(model=MODEL, messages=messages, stream=True,
                                        stream_options={"include_usage": True})
```

Pour une vue globale du serveur, l'endpoint `/metrics` expose des compteurs au format Prometheus. C'est l'outil de supervision standard en production :

```python
texte = requests.get(f"{BASE_URL}/metrics").text
for ligne in texte.splitlines():
    if ligne.startswith(("vllm:num_requests_running", "vllm:num_requests_waiting",
                         "vllm:kv_cache_usage_perc", "vllm:prefix_cache_hits_total",
                         "vllm:prefix_cache_queries_total")):
        print(ligne)
```

| Métrique | Contenu |
|---|---|
| `vllm:time_to_first_token_seconds` | histogramme du temps avant le premier token |
| `vllm:inter_token_latency_seconds` | histogramme du temps entre deux tokens |
| `vllm:e2e_request_latency_seconds` | histogramme de la durée totale des requêtes |
| `vllm:num_requests_running`, `vllm:num_requests_waiting` | requêtes en cours et en attente |
| `vllm:kv_cache_usage_perc` | remplissage du KV cache, de 0 à 1 |
| `vllm:prefix_cache_hits_total` / `vllm:prefix_cache_queries_total` | tokens trouvés dans le cache / tokens cherchés : leur rapport donne le taux de réussite |
| `vllm:num_preemptions_total` | requêtes interrompues faute de place dans le KV cache |

Le serveur écrit aussi toutes les 10 secondes une ligne de synthèse dans ses logs : débit de prefill et de génération, requêtes en cours et en attente, remplissage du KV cache, taux de réussite du cache de prompt.

## 6. Options de génération (chapitre 02)

### Les paramètres d'échantillonnage

| Paramètre | llama-server | vLLM | Attention |
|---|---|---|---|
| `temperature` | standard, défaut 0,8 | standard, défaut tiré du modèle, sinon 1,0 | voir les défauts ci-dessous |
| `top_p` | standard, défaut 0,95 | standard, défaut tiré du modèle, sinon 1,0 | |
| `top_k` | `extra_body`, défaut 40 | `extra_body`, défaut tiré du modèle, sinon désactivé | |
| `min_p` | `extra_body`, défaut 0,05 | `extra_body`, défaut tiré du modèle, sinon 0 | |
| `seed` | standard | standard | |
| `max_tokens` | défaut : jusqu'à la fin du contexte | défaut : `max_model_len` − prompt | sur `/v1/completions`, le défaut de vLLM est **16 tokens** |
| `stop` | standard | standard | |
| pénalité de répétition | `extra_body={"repeat_penalty": ...}` | `extra_body={"repetition_penalty": ...}` | **le nom change** |
| `frequency_penalty`, `presence_penalty` | standard | standard | |
| `dry_multiplier` | `extra_body` | *(aucun)* | DRY n'existe qu'en exemple de *logits processor* personnalisé |
| `typical_p`, `mirostat` | `extra_body` | *(aucun)* | |
| `logprobs`, `top_logprobs` | standard | standard, 20 au maximum (`--max-logprobs`) | |
| `n` | non supporté | supporté | plusieurs réponses alternatives en un seul appel |
| `cache_prompt` | `extra_body`, défaut `true` | *(aucun)* | le cache de prompt est automatique (section 7) |
| `chat_template_kwargs` | `extra_body` | `extra_body` | identique, par exemple `{"enable_thinking": False}` |

**Piège n° 1 : les paramètres inconnus sont ignorés en silence.** vLLM accepte tout champ supplémentaire sans erreur. Un `repeat_penalty` copié d'un notebook llama-server, un `dry_multiplier` ou un `typical_p` n'ont donc aucun effet, sans le moindre avertissement. Vérifiez les noms dans le tableau.

**Piège n° 2 : les défauts viennent du modèle.** Par défaut (`--generation-config auto`), vLLM lit le `generation_config.json` du dépôt et applique ses valeurs de `temperature`, `top_p`, `top_k`, `min_p` et `repetition_penalty`. Le log de démarrage le signale : *« Default vLLM sampling parameters have been overridden by the model's generation_config.json »*. Le même prompt sans paramètre ne donne donc pas les mêmes réglages sur les deux serveurs. Pour comparer équitablement :

- passez explicitement tous les paramètres dans la requête ;
- ou lancez vLLM avec `--generation-config vllm`, qui ignore le fichier du modèle.

À l'inverse, `--override-generation-config '{"temperature": 0.7}'` impose des défauts côté serveur.

### Les sorties structurées

`response_format` fonctionne à l'identique, avec `json_object` comme avec `json_schema`. Le code pydantic du chapitre 02 (`model_json_schema()`, `model_validate_json()`, `client.chat.completions.parse`) s'utilise sans modification.

Pour les autres contraintes, vLLM regroupe tout sous `structured_outputs`, et en propose davantage que llama-server :

```python
# Une grammaire (équivalent du paramètre grammar de llama-server)
extra_body={"structured_outputs": {"grammar": grammaire}}

# Une expression régulière
extra_body={"structured_outputs": {"regex": r"\d{4}-\d{2}-\d{2}"}}

# Un choix parmi des valeurs
extra_body={"structured_outputs": {"choice": ["positif", "négatif", "neutre"]}}
```

- Le moteur par défaut, xgrammar, accepte une grammaire au format EBNF de type `root ::= ...`, très proche du GBNF de llama.cpp. La grammaire du chapitre 02 est un bon point de départ, mais certaines constructions peuvent différer : testez-la.
- Les anciens paramètres `guided_json`, `guided_regex`, `guided_choice` et `guided_grammar`, encore très présents dans les tutoriels, ont été retirés en version 0.12. Ils sont **ignorés** : la sortie n'est plus contrainte, avec un simple avertissement dans les logs du serveur.

### Les images

| | llama-server | vLLM |
|---|---|---|
| Fichiers | modèle GGUF + `--mmproj` | le dépôt du modèle vision (par exemple `Qwen/Qwen3-VL-8B-Instruct`) |
| Nombre d'images par requête | pas de limite dédiée | `--limit-mm-per-prompt.image 2` (999 par défaut) |
| Résolution, donc nombre de tokens par image | dépend du projecteur | `--mm-processor-kwargs`, par exemple `max_pixels` pour les modèles Qwen-VL |
| Format dans la requête | `image_url` avec une data URL en base64 | identique, et vLLM accepte aussi les URL `http(s)://` |

Le code d'envoi d'images du chapitre 02 et de l'exercice 04 fonctionne sans modification.

### Le mode raisonnement

| | llama-server | vLLM |
|---|---|---|
| Désactiver par requête | `extra_body={"chat_template_kwargs": {"enable_thinking": False}}` | identique |
| Désactiver par défaut | `--reasoning off` | `--default-chat-template-kwargs '{"enable_thinking": false}'` |
| Séparer le raisonnement de la réponse | automatique | `--reasoning-parser qwen3` |
| Champ du raisonnement dans la réponse | `message.reasoning_content` | `message.reasoning` |

Sans `--reasoning-parser`, vLLM laisse le bloc `<think>...</think>` dans `message.content`.

## 7. Optimisations (chapitre 03)

### Correspondance des options

| llama-server | vLLM | Ce qui change |
|---|---|---|
| `-ngl N` | `--cpu-offload-gb X` | vLLM ne calcule jamais sur le CPU. Il garde X Gio de poids en RAM et les transfère vers le GPU à **chaque** passe : c'est très lent, à réserver aux cas désespérés |
| `-c` | `--max-model-len` | plafond par requête ; la taille du KV cache dépend de `--gpu-memory-utilization` |
| `-np` | `--max-num-seqs` (défaut 256) | pas de slots réservés : toutes les requêtes puisent dans le même réservoir de blocs |
| *(aucun)* | `--gpu-memory-utilization` (défaut 0,92) | part de la VRAM totale réservée par vLLM |
| *(aucun)* | `--kv-cache-memory-bytes` | taille exacte du KV cache, à la place du pourcentage |
| `-ctk`, `-ctv` | `--kv-cache-dtype fp8` | un seul type pour K et V. FP8 est l'équivalent de `q8_0`. Des formats 4 bits existent, mais seulement avec certains backends et certains GPU |
| `-fa` | `--attention-backend` (auto) | |
| `-b`, `-ub` | `--max-num-batched-tokens` (2048 sur GPU grand public) | voir ci-dessous |
| `-kvo`, `-nkvo` | `--kv-offloading-size N` | rien à voir avec `-nkvo` : vLLM copie en RAM les blocs terminés pour **étendre le cache de prompt**, l'attention reste sur le GPU. C'est l'équivalent de `--cache-ram` |
| `-cmoe`, `-ncmoe` | *(aucun)* | pas de placement des experts en RAM ; avec plusieurs GPU, `--enable-expert-parallel` répartit les experts |
| `--context-shift` | *(aucun)* | une requête trop longue est refusée (HTTP 400) |
| `-sm`, `-ts` (plusieurs GPU) | `--tensor-parallel-size`, `--pipeline-parallel-size` | le tensor parallel est le point fort de vLLM |
| `cache_prompt`, `--cache-reuse` | `--enable-prefix-caching` (actif par défaut) | voir ci-dessous |
| `--ctx-checkpoints` | *(aucun)* | les modèles hybrides sont gérés automatiquement |
| `--spec-type`, `--spec-draft-n-max`, `-md` | `--speculative-config '{...}'` | voir ci-dessous |

### Batch : un budget de tokens au lieu de -b et -ub

llama-server découpe le prompt en batchs (`-b`) puis en micro-batchs (`-ub`). vLLM raisonne avec un seul **budget de tokens par itération**, `--max-num-batched-tokens`, et traite le prefill par morceaux (*chunked prefill*, actif par défaut).

À chaque itération, l'ordonnanceur sert d'abord les requêtes en decoding, un token chacune, puis complète le budget avec un morceau du prompt d'une requête en prefill. Un long prompt n'immobilise donc jamais le serveur : les utilisateurs déjà en train de recevoir leur réponse continuent à la recevoir pendant qu'un nouveau document est ingéré.

- **Budget plus grand** (4096, 8192) : prefill plus rapide, meilleur débit global, mais la génération des autres requêtes ralentit pendant les prefills.
- **Budget plus petit** (512, 1024) : génération plus régulière pour tout le monde, prefill plus lent.

### Le cache de prompt

| | llama-server | vLLM |
|---|---|---|
| Activation | `cache_prompt: true` (défaut) | `--enable-prefix-caching` (défaut) |
| Portée | le slot : la requête précédente traitée par ce slot | **global** : toute requête qui commence par les mêmes tokens, quel que soit l'utilisateur |
| Granularité | au token près | par blocs complets de 16 tokens |
| Réutiliser un passage au milieu du prompt | `--cache-reuse N` | non |
| Cache en RAM | `--cache-ram` | `--kv-offloading-size` |
| Isoler des utilisateurs | un slot par utilisateur | `cache_salt` dans la requête |

Le cache global de vLLM est redoutable en production : un system prompt de 2 000 tokens partagé par mille utilisateurs n'est calculé qu'une seule fois. Les conseils du chapitre 03 restent valables : la partie stable du prompt au début, la partie variable à la fin.

### Le speculative decoding

Les commandes du chapitre 03 et leur traduction :

| llama-server | vLLM |
|---|---|
| `--spec-type draft-mtp --spec-draft-n-max 2` | `--speculative-config '{"method": "mtp", "num_speculative_tokens": 2}'` |
| `--spec-type draft-dflash --spec-draft-n-max 7 -md drafter.gguf` | `--speculative-config '{"method": "dflash", "model": "chemin/vers/le/drafter", "num_speculative_tokens": 7}'` |
| `--spec-type draft-eagle3 -md drafter.gguf` | `--speculative-config '{"method": "eagle3", "model": "chemin/vers/le/drafter", "num_speculative_tokens": 3}'` |
| `--spec-type ngram-mod` | `--speculative-config '{"method": "ngram", "num_speculative_tokens": 4}'` |
| `-md petit_modele.gguf` | `--speculative-config '{"method": "draft_model", "model": "Qwen/Qwen3-0.6B", "num_speculative_tokens": 3}'` |

Les principes du chapitre 03 s'appliquent : MTP avec 2 ou 3 propositions, DFlash avec la taille de bloc de son drafter, gain fort à faible charge et sur du texte prévisible. Deux différences :

- **Sous forte charge**, la vérification coûte batch × k tokens et finit par ralentir le serveur. vLLM propose un speculative decoding *dynamique* (`num_speculative_tokens_per_batch_size`) qui réduit k quand le nombre de requêtes augmente, jusqu'à le couper.
- **Les mesures** passent par une ligne de log dédiée (`SpecDecoding metrics: Mean acceptance length ..., Avg Draft acceptance rate ...`) et par les compteurs `vllm:spec_decode_num_draft_tokens_total` et `vllm:spec_decode_num_accepted_tokens_total` de `/metrics`.

Pour MTP, vérifiez que le checkpoint quantisé conserve bien les couches MTP : certains dépôts quantisés les suppriment.

### Exemple : servir Qwen3-8B au mieux sur 24 Go

| Configuration | Poids | KV cache par token | Capacité approximative | Requêtes de 12 288 tokens en parallèle |
|---|---:|---:|---:|---:|
| BF16, KV cache en BF16 | ~15,3 Gio | 144 Kio | ~38 000 tokens | ~3 |
| FP8, KV cache en FP8 | ~9 Gio | 72 Kio | ~170 000 tokens | ~14 |

```sh
vllm serve Qwen/Qwen3-8B-FP8 \
  --max-model-len 12288 \
  --kv-cache-dtype fp8 \
  --default-chat-template-kwargs '{"enable_thinking": false}'
```

Les valeurs exactes se lisent dans la ligne `Maximum concurrency for 12,288 tokens per request` au démarrage.

## 8. Mesurer : llama-bench et vllm bench

`vllm bench` remplace llama-bench, avec une différence de taille : `vllm bench serve` interroge un **vrai serveur** par son API, avec plusieurs utilisateurs simultanés. Il mesure donc tout ce qui échappait à llama-bench : la montée en charge, le speculative decoding, le cache de prompt.

```sh
pip install "vllm[bench]"

vllm bench serve \
  --backend openai-chat --endpoint /v1/chat/completions \
  --base-url http://127.0.0.1:8000 --model Qwen/Qwen3-8B \
  --dataset-name random --random-input-len 4096 --random-output-len 512 \
  --num-prompts 200 --max-concurrency 8
```

- Avec `--backend openai-chat`, l'option `--endpoint /v1/chat/completions` est **obligatoire**.
- `--max-concurrency` fixe le nombre d'utilisateurs simultanés ; `--request-rate` fixe plutôt un nombre de requêtes par seconde.
- `--goodput ttft:500 tpot:50` compte les requêtes qui respectent un objectif de qualité de service : premier token en moins de 500 ms, puis un token toutes les 50 ms au plus.
- `--save-result` enregistre les résultats en JSON, à analyser ensuite avec polars.

Les mesures affichées :

| Mesure | Signification | Équivalent llama-server |
|---|---|---|
| TTFT (*time to first token*) | temps avant le premier token, file d'attente comprise | `prompt_ms` |
| TPOT (*time per output token*) | temps moyen par token généré, hors premier token | `predicted_per_token_ms` |
| ITL (*inter-token latency*) | temps entre deux tokens consécutifs | |
| Request throughput | requêtes terminées par seconde | |
| Output token throughput | tokens générés par seconde, tous utilisateurs confondus | |

Chaque mesure est donnée en moyenne, en médiane et au 99e centile. Le 99e centile compte autant que la moyenne : c'est l'expérience des utilisateurs les moins bien servis.

**Le même benchmark sur les deux serveurs.** llama-server parle la même API : `vllm bench serve` fonctionne aussi contre lui, en changeant simplement `--base-url` (`http://127.0.0.1:8080`). C'est la façon la plus honnête de comparer les deux serveurs sur notre GPU.

| | llama-bench | vllm bench serve |
|---|---|---|
| Teste | le moteur de calcul | le serveur complet, par l'API |
| Plusieurs utilisateurs | non | oui (`--max-concurrency`, `--request-rate`) |
| Speculative decoding, cache de prompt | non | oui, avec un jeu de données réaliste (`sharegpt`, `hf`) plutôt que `random` |
| Grille de paramètres serveur en un seul appel | oui (`-ub 256,512,1024`) | non : il faut relancer le serveur pour chaque réglage |
| Fonctionne avec | llama.cpp uniquement | tout serveur compatible OpenAI |

Pour des mesures hors serveur, `vllm bench latency` (latence d'un batch) et `vllm bench throughput` (débit maximal) chargent le modèle directement.

## 9. Les exercices avec vLLM

| Exercice | llama-server | vLLM |
|---|---|---|
| 01 – Comparer deux modèles | relancer le serveur avec un autre `-m` | relancer avec un autre dépôt ; garder `--served-model-name local` pour ne pas changer le code |
| 03 – Deux modèles sur un GPU | deux serveurs, chacun avec son `-c` | deux serveurs avec `--gpu-memory-utilization 0.45` chacun, sinon le second ne démarre pas faute de mémoire libre |
| 04 – Extraction de contrats | `-m Qwen3VL-8B-Instruct-Q4_K_M.gguf --mmproj ... -c 16384` | `vllm serve Qwen/Qwen3-VL-8B-Instruct --max-model-len 16384 --limit-mm-per-prompt.image 2` |
| 05 – Montée en charge | `-c 49152 -np 4 --reasoning off` | `--max-model-len 12288 --default-chat-template-kwargs '{"enable_thinking": false}'` |

L'exercice 05 est celui où vLLM a le plus à montrer. Relancez-le avec 4, 8 puis 16 clients sur les deux serveurs. Avec llama-server, les clients au-delà de `-np` attendent leur tour. Avec vLLM, tous entrent dans le batch tant que le KV cache a de la place, et `vllm:num_requests_waiting` dans `/metrics` montre quand ce n'est plus le cas.

## 10. Lequel choisir ?

| Situation | Plutôt |
|---|---|
| Un seul utilisateur, un poste de travail, un GPU grand public | llama-server |
| Modèle plus gros que la VRAM, avec une partie en RAM | llama-server |
| Mac, CPU seul, Windows sans WSL | llama-server |
| Quantisations fines (Q3, Q5, Q6, imatrix) pour ajuster au Go près | llama-server |
| Démarrage rapide, prototypage, changements fréquents de modèle | llama-server |
| Beaucoup d'utilisateurs simultanés, API partagée en production | vLLM |
| Plusieurs GPU sur un même modèle | vLLM |
| Supervision (Prometheus), benchmarks de charge, objectifs de latence | vLLM |
| Même system prompt ou mêmes documents pour de nombreux utilisateurs | vLLM (cache de prompt global) |

Les deux ne s'opposent pas : un usage courant consiste à prototyper avec llama-server sur un poste de travail, puis à déployer avec vLLM sur un serveur. Le code client, lui, ne change presque pas.

## Aide-mémoire

| Je veux… | llama-server | vLLM |
|---|---|---|
| Servir un modèle | `llama-server -m modele.gguf` | `vllm serve org/modele` |
| Télécharger depuis Hugging Face | `-hf org/modele-GGUF:Q4_K_M` | `vllm serve org/modele` |
| Fixer le contexte | `-c 32768` (total) | `--max-model-len 32768` (par requête) |
| Servir plusieurs utilisateurs | `-np 4` | rien à faire (256 par défaut) |
| Limiter la mémoire utilisée | réduire `-c`, `-ngl` | `--gpu-memory-utilization 0.6` |
| Quantiser le KV cache | `-ctk q8_0 -ctv q8_0` | `--kv-cache-dtype fp8` |
| Modèle vision | `--mmproj fichier.gguf` | rien à faire |
| Couper le raisonnement | `--reasoning off` | `--default-chat-template-kwargs '{"enable_thinking": false}'` |
| Nom du modèle dans les requêtes | ignoré | `--served-model-name local` |
| Protéger l'API | `--api-key cle` | `--api-key cle` (ne protège que `/v1` : `/metrics` et `/tokenize` restent ouverts) |
| Speculative decoding MTP | `--spec-type draft-mtp --spec-draft-n-max 2` | `--speculative-config '{"method": "mtp", "num_speculative_tokens": 2}'` |
| Voir le prompt formaté | `POST /apply-template` | `POST /tokenize` avec `messages`, puis `/detokenize` |
| Mesurer une requête | `timings` dans la réponse | `--enable-per-request-metrics`, champ `metrics` |
| Superviser le serveur | `/slots` | `/metrics` |
| Benchmark | `llama-bench` | `vllm bench serve` |
| Démarrer vite pour tester | *(déjà rapide)* | `--enforce-eager` |
