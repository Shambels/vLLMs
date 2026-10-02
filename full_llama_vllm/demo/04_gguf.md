# La librairie `gguf` : lire un modèle sans le charger

## Ce qu'il y a dans un fichier GGUF

Un GGUF est un conteneur unique qui contient tout ce dont llama.cpp a besoin :

```
┌──────────────────────────────┐
│ En-tête                      │  version, nombre de tenseurs, nombre de métadonnées
├──────────────────────────────┤
│ Métadonnées (clé → valeur)   │  architecture, nombre de couches, contexte,
│                              │  tokenizer, chat template, quantisation…
├──────────────────────────────┤
│ Descriptions des tenseurs    │  nom, forme, type de quantisation, position
├──────────────────────────────┤
│ Données des tenseurs         │  les poids eux-mêmes (l'essentiel du fichier)
└──────────────────────────────┘
```

Les trois premières parties font quelques Mo. La librairie `gguf` les lit sans charger les poids : on inspecte un modèle de 16 Go en une fraction de seconde, sans GPU et sans lancer de serveur.

C'est utile pour répondre avant de lancer `llama-server` à des questions comme : combien de couches ? quel contexte d'entraînement ? combien pèsera le KV cache à 32k tokens ? ce GGUF contient-il des têtes MTP ? quels tenseurs sont en Q4, lesquels en Q6 ?

## Installation et outil en ligne de commande

```bash
pip install gguf
```

La librairie est maintenue dans le dépôt llama.cpp. Elle installe aussi quelques commandes :

```bash
gguf-dump --no-tensors modele.gguf        # toutes les métadonnées
gguf-dump modele.gguf | less              # métadonnées + liste des tenseurs
gguf-dump --json modele.gguf > meta.json  # la même chose en JSON
gguf-set-metadata modele.gguf general.name "Mon modèle"   # modifie une valeur simple en place
gguf-new-metadata ...                     # copie le fichier avec des métadonnées modifiées
```

## Lire les métadonnées en Python

```python
from gguf import GGUFReader

reader = GGUFReader("/chemin/vers/modele.gguf")

def meta(reader, cle, defaut=None):
    """Valeur d'une métadonnée, ou `defaut` si la clé n'existe pas."""
    champ = reader.fields.get(cle)
    return champ.contents() if champ is not None else defaut

arch = meta(reader, "general.architecture")
print("Architecture :", arch)
print("Nom          :", meta(reader, "general.name"))
print("Couches      :", meta(reader, f"{arch}.block_count"))
print("Contexte     :", meta(reader, f"{arch}.context_length"))
print("Vocabulaire  :", len(meta(reader, "tokenizer.ggml.tokens", [])))
```

`reader.fields` est un dictionnaire `nom → champ`, et `champ.contents()` renvoie la valeur décodée (entier, chaîne, liste…).

Pour tout lister, en sautant les listes énormes du tokenizer :

```python
for nom, champ in reader.fields.items():
    if nom.startswith(("GGUF.", "tokenizer.ggml.")):
        continue
    print(f"{nom:45s} {str(champ.contents())[:60]}")
```

### Les clés à connaître

Les clés propres au modèle sont préfixées par le nom de l'architecture : pour un Qwen3 c'est `qwen3.block_count`, pour un Llama `llama.block_count`. D'où le `f"{arch}.…"`.

| Clé | Contenu |
|---|---|
| `general.architecture` | famille du modèle (`qwen3`, `llama`, `gemma3`, `clip` pour un mmproj…) |
| `general.name` | nom lisible |
| `general.file_type` | quantisation principale, en code numérique (voir plus bas) |
| `{arch}.block_count` | nombre de couches |
| `{arch}.context_length` | contexte d'entraînement |
| `{arch}.embedding_length` | dimension des vecteurs internes |
| `{arch}.attention.head_count` | nombre de têtes d'attention |
| `{arch}.attention.head_count_kv` | nombre de têtes K/V (plus petit que le précédent avec GQA) |
| `{arch}.attention.key_length`, `value_length` | dimension d'une tête ; si absente : `embedding_length / head_count` |
| `{arch}.rope.*` | paramètres de position (fréquence, scaling intégré) |
| `{arch}.expert_count`, `expert_used_count` | nombre d'experts et experts actifs par token (modèles MoE) |
| `tokenizer.ggml.model`, `tokenizer.ggml.tokens` | type de tokenizer et vocabulaire |
| `tokenizer.chat_template` | le template Jinja vu dans le cours |
| `split.count` | nombre de fichiers si le modèle est découpé (`-00001-of-00003.gguf`) |

Le code de quantisation se traduit avec l'énumération fournie par la librairie :

```python
import gguf
print(gguf.LlamaFileType(meta(reader, "general.file_type")).name)   # MOSTLY_Q4_K_M
```

## Lire la liste des tenseurs

```python
for t in reader.tensors[:12]:
    forme = [int(x) for x in t.shape]
    print(f"{t.name:32s} {t.tensor_type.name:6s} {str(forme):>16s} {t.n_bytes / 1e6:9.1f} Mo")
```

Chaque tenseur a un `name`, un `tensor_type` (F16, Q8_0, Q4_K…), une `shape`, un nombre d'éléments `n_elements` et une taille en octets `n_bytes`. Attention, la forme est donnée dans l'ordre de ggml, **inversé** par rapport à PyTorch.

Les noms suivent une convention commune à toutes les architectures :

| Nom | Rôle |
|---|---|
| `token_embd.weight` | table des embeddings (token → vecteur) |
| `blk.N.attn_q`, `attn_k`, `attn_v`, `attn_output` | attention de la couche N |
| `blk.N.ffn_gate`, `ffn_up`, `ffn_down` | réseau feed-forward de la couche N |
| `blk.N.ffn_*_exps` | experts d'un MoE (ce que `-ncmoe` et `-ot` déplacent en RAM) |
| `blk.N.attn_norm`, `ffn_norm`, `output_norm` | normalisations |
| `output.weight` | projection finale vers le vocabulaire |

### Un Q4_K_M n'est pas « tout en 4 bits »

```python
from collections import Counter

par_type = Counter()
for t in reader.tensors:
    par_type[t.tensor_type.name] += t.n_bytes

total = sum(par_type.values())
for typ, octets in par_type.most_common():
    print(f"{typ:6s} {octets / 1e9:6.2f} Go  {octets / total:6.1%}")

print(f"Paramètres : {sum(int(t.n_elements) for t in reader.tensors) / 1e9:.2f} milliards")
```

On y voit un mélange : Q4_K pour la plupart des poids, Q6_K pour certains tenseurs sensibles, F32 pour les normalisations. Le nom de la quantisation désigne une **recette**, pas un type unique.

## Trois usages pour le cours

### 1. Prédire la taille du KV cache avant de lancer le serveur

```python
OCTETS = {"f32": 4, "f16": 2, "bf16": 2, "q8_0": 34/32, "q5_1": 24/32, "q5_0": 22/32,
          "q4_1": 20/32, "q4_0": 18/32, "iq4_nl": 18/32}

def taille_kv_gio(reader, n_ctx, type_k="f16", type_v="f16"):
    arch = meta(reader, "general.architecture")
    n_layer = meta(reader, f"{arch}.block_count")
    n_head = meta(reader, f"{arch}.attention.head_count")
    n_head_kv = meta(reader, f"{arch}.attention.head_count_kv", n_head)
    d_k = meta(reader, f"{arch}.attention.key_length", meta(reader, f"{arch}.embedding_length") // n_head)
    d_v = meta(reader, f"{arch}.attention.value_length", d_k)
    k = n_layer * n_ctx * n_head_kv * d_k * OCTETS[type_k]
    v = n_layer * n_ctx * n_head_kv * d_v * OCTETS[type_v]
    return (k + v) / 2**30

for ctx in (8192, 32768, 131072):
    print(f"{ctx:>7d} tokens : f16 {taille_kv_gio(reader, ctx):5.2f} Gio"
          f" | q8_0 {taille_kv_gio(reader, ctx, 'q8_0', 'q8_0'):5.2f} Gio"
          f" | q4_0 {taille_kv_gio(reader, ctx, 'q4_0', 'q4_0'):5.2f} Gio")
```

Les facteurs `34/32` et `18/32` viennent du format des blocs : 32 valeurs quantisées plus une échelle en f16.

À comparer avec la ligne du KV cache dans les logs de `llama-server` et avec `nvidia-smi`. Pour les modèles à attention glissante ou hybrides (voir le chapitre optimisation), la formule **surestime** : seules certaines couches ont un cache complet. L'écart entre la prédiction et la mesure est justement ce qui révèle l'architecture.

### 2. Vérifier qu'un GGUF contient des têtes MTP

Avant d'essayer `--spec-type draft-mtp`, on vérifie que le fichier contient bien les couches de prédiction multi-tokens. Elles apparaissent dans les métadonnées (clé du type `{arch}.nextn_predict_layers`) et dans les noms de tenseurs (`nextn` ou `mtp`) :

```python
cles_mtp = [k for k in reader.fields if "nextn" in k or "mtp" in k.lower()]
tenseurs_mtp = [t.name for t in reader.tensors if "nextn" in t.name or "mtp" in t.name]

print("Clés MTP     :", {k: meta(reader, k) for k in cles_mtp})
print("Tenseurs MTP :", len(tenseurs_mtp), tenseurs_mtp[:3])
```

Les conventions de nommage varient selon les architectures : c'est pour ça qu'on cherche des motifs plutôt qu'un nom exact. Si les deux listes sont vides, le serveur refusera le mode MTP.

### 3. Comparer deux quantisations du même modèle

```python
for chemin in ["Qwen3-8B-Q4_K_M.gguf", "Qwen3-8B-Q8_0.gguf"]:
    r = GGUFReader(chemin)
    taille = sum(t.n_bytes for t in r.tensors) / 1e9
    types = Counter(t.tensor_type.name for t in r.tensors)
    print(f"{chemin:28s} {taille:5.2f} Go  {dict(types.most_common(3))}")
```

Même architecture, même nombre de tenseurs, même KV cache : seule la taille des poids change. D'après la règle de pouce du chapitre optimisation, la vitesse de génération devrait varier à peu près dans le rapport inverse des tailles. À vérifier avec `llama-bench`.

## Pièges

- **Fichiers découpés** : un modèle en plusieurs parties (`split.count` > 1) n'a qu'une partie des tenseurs dans chaque fichier. Les métadonnées sont dans le premier, la taille totale est la somme de tous.
- **Le mmproj est un GGUF à part** : `general.architecture` y vaut `clip`, avec ses propres clés. La taille d'un modèle vision = modèle + mmproj.
- **Modifier les métadonnées** : `gguf-set-metadata` ne change que des valeurs simples, sur place, sans sauvegarde. Travailler sur une copie. Pour un essai sans toucher au fichier, `llama-server --override-kv cle=type:valeur` remplace une métadonnée au chargement (et `--chat-template-file` remplace le template).