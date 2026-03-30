# NeuronBox — MVP Spec

**Positionnement actuel :** le dépôt est présenté comme une **pile locale fonctionnelle** (CLI, `neurond`, dashboard, protocole socket, store). Ce document garde la **spécification et la roadmap d’origine** ; les titres « MVP » ci-dessous sont un **cadre historique**, pas une étiquette sur la maturité du produit livré.

> Outil de conteneurisation IA-native pour le développement local sérieux.
> Cible : ML engineers, chercheurs, startups IA qui veulent aller plus loin qu'Ollama sans passer au cloud.

---

## Contexte & positionnement

| Acteur | Force | Angle mort |
|---|---|---|
| Ollama | Inférence locale simple | Pas de training, format proprio |
| Docker Model Runner | Intégration CI/CD | Inférence only, pas de training |
| SageMaker / Vertex | Training à l'échelle | Trop lourd, cloud obligatoire |
| **NeuronBox** | **Training local + contrôle** | **Zone non couverte** |

**Insight clé** : Docker contourne son propre paradigme pour faire de l'IA (Model Runner tourne hors conteneur). Preuve que l'architecture conteneur classique n'est pas adaptée nativement aux workloads IA.

---

## Scope MVP

Le MVP valide une seule hypothèse :

> *Un dev IA peut setup, lancer, et itérer sur un fine-tuning local en moins de 5 minutes, sans configuration manuelle de GPU ou d'environnement.*

Ce qui est **inclus** dans le MVP :

- CLI `neuron` avec les commandes de base
- Gestion GPU automatique (détection + quotas VRAM « soft » côté processus)
- Model store partagé sur disque ; chargement en VRAM par les scripts / worker Python (pas par le daemon Rust)
- Format `neuron.yaml` déclaratif
- Hot-swap de modèles à chaud
- Profiling GPU basique (`neuron stats`)
- Compatibilité Docker (`neuron` = alias `docker` pour les commandes standard)

Ce qui est **exclu** du MVP :

- Interface graphique
- Multi-GPU distribué
- Snapshots mid-training automatiques
- Intégration CI/CD
- Support cloud / remote GPU

---

## Architecture (code réel)

Le dépôt est un **workspace Cargo** : binaire `neuron` + bibliothèque / binaire `neurond`. Pas de crate `sdk/` obligatoire ; des scripts Python d’exemple vivent sous `cli/scripts/`.

```
NeuroBox/
├── cli/                         # Binaire `neuron`
│   ├── src/
│   │   ├── main.rs
│   │   ├── commands/          # run, pull, serve, swap, stats, gpu, host, lock, oci, …
│   │   ├── env_hash.rs        # venv hashé (store/envs/) + index PyTorch (CUDA/ROCm)
│   │   ├── oci/               # Docker / préparation runc
│   │   ├── daemon_client.rs   # Client socket Unix JSON (une requête par connexion)
│   │   └── daemon_spawn.rs
│   ├── scripts/
│   │   ├── serve_worker.py              # Worker long-vivant pour `neuron serve` (swap_signal.json)
│   │   └── neuronbox_daemon_session.py  # Exemple : session / tokens/s (voir specs/daemon-sessions.md)
│   └── Cargo.toml
│
├── runtime/                     # Lib `neuronbox_runtime` + binaire `neurond`
│   ├── src/
│   │   ├── main.rs            # Point d’entrée neurond
│   │   ├── server.rs          # Boucle socket : sessions, stats, swap → fichier signal
│   │   ├── protocol.rs        # DaemonRequest / DaemonResponse
│   │   ├── gpu_manager.rs     # Registre sessions (PID, VRAM estimée, tokens/s optionnel)
│   │   ├── model_loader.rs    # État logique « modèle actif » (pas de poids en Rust)
│   │   ├── host/              # HostProbe / HostSnapshot, parsing nvidia-smi partagé
│   │   ├── gpu.rs             # Facade detect_gpus / soft_vram_check
│   │   └── vram_watch.rs      # Enforcement soft Linux NVIDIA (nvidia-smi + SIGKILL)
│   └── Cargo.toml
│
└── specs/
    ├── neuron.yaml.schema.json
    ├── swap-signal.schema.json
    ├── daemon-sessions.md     # Contrat RegisterSession / tokens/s pour `neuron stats`
    └── examples/
        ├── inference.yaml
        └── finetune.yaml
```

### Rôles des processus

| Composant | Rôle |
|-----------|------|
| **`neuron`** | CLI : résout `neuron.yaml`, prépare venv et store, lance Python ou Docker, parle au daemon pour enregistrer les sessions `run`. |
| **`neurond`** | Daemon : socket `~/.neuronbox/neuron.sock`, registre des sessions, stats agrégées, écrit `swap_signal.json` sur `neuron swap`. |
| **`neuron serve`** | Démarre le daemon si besoin, puis le **même venv** que `neuron run` pour exécuter `serve_worker.py` (inférence + réaction au swap). |
| **`neuron run`** | Lance l’entrypoint du projet avec le venv du store ; enregistre / désenregistre la session auprès du daemon. |

### `neuron host inspect`

Expose un JSON `HostSnapshot` (version de schéma, GPU, `training_backend`, sondes) pour debug et support — même source que la détection utilisée par le CLI.

---

## Format `neuron.yaml`

Fichier déclaratif placé à la racine du projet. Équivalent du `Dockerfile` mais pensé IA.

```yaml
# neuron.yaml
name: mon-projet-llm
version: "1.0"

model:
  name: meta-llama/Meta-Llama-3-8B-Instruct
  source: huggingface          # huggingface | local | s3
  quantization: q4_k_m         # optionnel, défaut: none

runtime:
  python: "3.11"
  cuda: "12.1"                 # auto-détecté si absent
  packages:
    - transformers==4.40.0
    - peft==0.10.0
    - datasets==2.19.0
    - trl==0.8.6

gpu:
  min_vram: 16gb               # Refus de lancer si insuffisant
  strategy: single             # single | pipeline | tensor

entrypoint: train.py           # Script lancé par `neuron run`

env:
  HF_TOKEN: "${HF_TOKEN}"      # Variables d'env, jamais de secrets en dur
  BATCH_SIZE: "4"
```

---

## CLI — Commandes MVP

### `neuron init`

Initialise un projet dans le répertoire courant.

```bash
neuron init
# Crée neuron.yaml interactif
# Détecte le GPU disponible
# Suggère une config compatible
```

### `neuron run`

Lance le projet défini dans `neuron.yaml`, ou un modèle directement.

```bash
neuron run                          # Utilise neuron.yaml local
neuron run llama3:8b                # Lance un modèle directement
neuron run llama3:8b --gpu 1        # Forcer l'index GPU
neuron run llama3:8b --vram 12gb    # Limiter la VRAM allouée
```

Comportement :
1. Lit `neuron.yaml` (ou flags CLI)
2. Vérifie que le GPU a assez de VRAM
3. Télécharge les poids si absents du store (lazy)
4. Installe les deps Python dans un env isolé
5. Lance le script ou le serveur d'inférence

### `neuron pull`

Télécharge un modèle dans le store global. Partagé entre tous les projets.

```bash
neuron pull meta-llama/Meta-Llama-3-8B-Instruct
neuron pull mistral:7b
neuron pull ./mon-modele-local      # Import depuis le filesystem
```

### `neuron swap`

Met à jour l’état logique du daemon et écrit **`~/.neuronbox/swap_signal.json`** (schéma versionné, voir [`specs/swap-signal.schema.json`](specs/swap-signal.schema.json)). Un worker lancé par **`neuron serve`** peut recharger un modèle après lecture de ce fichier.

```bash
neuron swap mistral:7b
neuron swap meta-llama/Meta-Llama-3-70B-Instruct --quantization q4_k_m
```

### `neuron serve`

Démarre le daemon si nécessaire, assure le **même venv** que `neuron run` pour ce `neuron.yaml`, puis exécute `cli/scripts/serve_worker.py` (boucle d’inférence / écoute du swap). Les dépendances listées dans le yaml (ex. `transformers`) doivent donc être installées via `neuron run` ou la création du venv.

### `neuron host inspect`

Affiche un JSON `HostSnapshot` (GPU, plateforme, `training_backend`, état des sondes).

### `neuron stats`

Affiche les sessions enregistrées auprès du daemon et les processus compute NVIDIA (**NVML** si build `--features nvml`, sinon `nvidia-smi`). La colonne **Tokens/s** n’est remplie que si le processus met à jour la session (voir [`specs/daemon-sessions.md`](specs/daemon-sessions.md) et `cli/scripts/neuronbox_daemon_session.py`).

### `neuron dashboard`

TUI locale (**ratatui**) : une session daemon persistante pour les requêtes `stats` en boucle, plus un `HostProbe::snapshot()` côté client pour le résumé GPU / `training_backend`. Aucune API HTTP ni cloud. Quitter avec `q` ou Échap.

```bash
neuron dashboard
```

```bash
neuron stats

# Output exemple :
# ┌─────────────────────────────────────────┐
# │ NeuronBox Stats                         │
# ├──────────────┬───────┬──────────────────┤
# │ Conteneur    │ VRAM  │ Tokens/s         │
# ├──────────────┼───────┼──────────────────┤
# │ mon-projet   │ 14/24 │ 47.3             │
# │ test-mistral │  8/24 │ 91.2             │
# └──────────────┴───────┴──────────────────┘
```

### Pass-through Docker (retiré)

Les commandes `neuron ps` / `neuron stop` / `neuron rm` et le proxy `neuron run` → `docker run` ont été **supprimées** : utiliser directement `docker ps`, `docker stop`, `docker rm`, `docker run`. NeuronBox reste centré ML sur le chemin principal ; Docker n’intervient que pour le mode **OCI** (`neuron run --oci`, `neuron oci prepare`, etc.). Voir [docs/OCI_AND_DOCKER.md](docs/OCI_AND_DOCKER.md).

---

## Model Store

Le store est global, partagé entre tous les projets NeuronBox sur la machine.

**Localisation par défaut :**
- Linux/macOS : `~/.neuronbox/store/`
- Windows : `%APPDATA%\neuronbox\store\`

**Structure :**

```
~/.neuronbox/store/
├── models/
│   ├── meta-llama--Meta-Llama-3-8B-Instruct/
│   │   ├── config.json
│   │   ├── tokenizer.json
│   │   └── model.safetensors      # ou shards
│   └── mistralai--Mistral-7B-v0.1/
│       └── ...
├── envs/
│   ├── py311-cu121-transformers440/   # Hash de la config
│   │   └── lib/
│   └── ...
└── index.json                          # Registre local
```

**Chargement en VRAM** : le daemon NeuronBox ne charge pas les poids ; c’est le script d’entraînement, le worker `serve`, ou votre code Python qui appelle Transformers / torch. Le store ne fait que dédupliquer les fichiers sur disque.

**Déduplication** : si deux projets utilisent le même modèle, un seul exemplaire sur disque.

---

## GPU Manager

Le **GpuManager** (dans `neurond`) garde une table des sessions enregistrées par la CLI (`RegisterSession` / `UnregisterSession`) : nom, PID, VRAM **estimée**, débit **tokens/s** optionnel. Ce n’est pas une allocation matérielle au sens du pilote NVIDIA ; sur Linux, une tâche de fond peut comparer l’usage réel (`nvidia-smi`) à l’estimation et envoyer **SIGKILL** si la limite soft est dépassée.

### Détection automatique (CLI / `neuron gpu list` / `neuron host inspect`)

La détection utilise le module **`host`** (nvidia-smi, rocm-smi, Apple `system_profiler`) :

1. NVIDIA (CUDA) via `nvidia-smi`
2. AMD (ROCm) via `rocm-smi`
3. Apple Silicon (Metal/MPS) via `system_profiler`

```bash
neuron gpu list

# Output :
# GPU 0: NVIDIA RTX 4090 — 24 GB VRAM — CUDA 12.1
# GPU 1: NVIDIA RTX 3080 — 10 GB VRAM — CUDA 12.1
```

### Quotas « soft »

Avant `neuron run`, une vérification optionnelle (`gpu.min_vram` dans `neuron.yaml`) compare la VRAM **totale** estimée du GPU principal à la demande. À l’exécution, la limite enregistrée sert de référence pour la surveillance NVIDIA (voir ci-dessus), pas pour réserver de la mémoire dans le pilote.

```bash
# Erreur explicite :
# [neuronbox] Erreur : mon-projet demande 18 GB VRAM, GPU 0 n'a que 12 GB disponibles.
# Suggestion : neuron run --quantization q4_k_m (réduit à ~8 GB)
```

### Stratégies d'allocation

| Stratégie | Usage | Quand l'utiliser |
|---|---|---|
| `single` | Un modèle sur un GPU | Modèles ≤ 24B |
| `pipeline` | Modèle splitté sur N GPUs | Modèles > 24B |
| `tensor` | Parallélisme tenseur | Training distribué |

MVP : implémenter `single` uniquement. `pipeline` et `tensor` en v1.1.

---

## Environnement Python (venv hashé)

Les conflits `torch + CUDA/ROCm + transformers` sont gérés par un **venv par hash** (`store/envs/py-<hash>/`) : contenu dérivé de `runtime.python`, `runtime.cuda`, `runtime.packages`, plus l’index PyTorch (wheels CUDA vs ROCm) selon la détection **`HostProbe`** (`neuron host inspect` → `training_backend`).

- Si `requirements.lock` est présent dans le venv et que `uv` est disponible : `uv pip sync --frozen`.
- Sinon : `uv pip install` ou `pip install` avec les `packages` du yaml.

L’env résolu est **hashé et réutilisé** dans le store. `neuron run` et **`neuron serve`** utilisent le **même** venv pour un même `neuron.yaml`.

---

## Docker et OCI (positionnement actuel)

- **`neuron pull`** : uniquement modèles ML (HF, alias, chemins locaux). Pas de délégation à `docker pull` pour les tags d’images OCI.
- **`neuron run`** (mode hôte) : venv + script, sans `docker run`.
- **Mode OCI** : `neuron run --oci` / `runtime.mode: oci` utilise **`docker run`** avec les montages NeuronBox ; **`neuron oci prepare`** utilise **`docker pull` / create / export** pour fabriquer un bundle runc.

Référence : [docs/OCI_AND_DOCKER.md](docs/OCI_AND_DOCKER.md).

---

## Stack technique

| Composant | Techno | Raison |
|---|---|---|
| CLI & daemon | Rust | Performance, binaire unique, Tokio pour le socket |
| Isolation optionnelle | Docker / runc (OCI) | Aligné Linux + NVIDIA Container Toolkit pour le mode `oci` |
| API daemon | Unix socket + JSON (NDJSON une ligne par requête) | Latence faible, local |
| Détection GPU | `nvidia-smi` / `rocm-smi` / `system_profiler` | Pas de NVML obligatoire dans le MVP |
| Model store | Fichiers HF / locaux via `hf-hub` + chemins projet | Standard Hugging Face |
| Environnement Python | `venv` + `uv` ou `pip`, hash de config | Reproductibilité avec `neuron lock` |
| Scripts d’exemple | Python 3 (`cli/scripts/`) | Worker serve, client session pour stats |

---

## Roadmap MVP → v1

### MVP (v0.1) — 8 semaines

**Semaine 1-2 : fondations**
- [ ] Repo Rust, CI GitHub Actions
- [ ] Détection GPU (NVIDIA + Apple Silicon)
- [ ] `neuron gpu list`

**Semaine 3-4 : model store**
- [ ] Téléchargement HuggingFace (avec token)
- [ ] Store local avec déduplication
- [ ] `neuron pull`, `neuron model list`

**Semaine 5-6 : runtime**
- [ ] `neuron run` avec env Python isolé
- [ ] Parser `neuron.yaml` basique
- [ ] Allocation VRAM + erreurs explicites

**Semaine 7 : features différenciantes**
- [ ] `neuron swap` (hot-swap modèle)
- [ ] `neuron stats` (VRAM temps réel)

**Semaine 8 : polish**
- [ ] Compatibilité commandes Docker de base
- [ ] Env solver (version simplifiée)
- [ ] Docs + README + demo video

### v1.0 — 3 mois après MVP

- [ ] Hot-swap production-grade
- [ ] Snapshots mid-training automatiques
- [ ] Multi-GPU pipeline parallelism
- [ ] `neuron.yaml` complet (toutes options)
- [ ] SDK Python stable

### v1.1 — 6 mois après MVP

- [ ] Support ROCm (AMD)
- [ ] Intégration CI/CD (GitHub Actions plugin)
- [ ] Dashboard web local
- [ ] Datasets streaming (S3 / HuggingFace Datasets)

---

## Métriques de validation MVP

Le MVP est validé si :

1. **Setup < 5 min** : De zéro à un fine-tuning qui tourne, sur une machine avec GPU NVIDIA, en moins de 5 minutes.
2. **Zéro config GPU** : L'utilisateur ne touche jamais `nvidia-smi`, `LD_LIBRARY_PATH`, ou les flags CUDA.
3. **Hot-swap < 10s** : Changer de modèle sans redémarrer le conteneur en moins de 10 secondes.
4. **Adoption** : 100 devs qui téléchargent et lancent le MVP dans les 30 jours post-lancement (Product Hunt / HN).

---

## Personas cibles

**Jordan, ML engineer freelance**
Passe 30% de son temps à débugger des conflits de dépendances et des configurations GPU. Connaît Docker mais en a marre de la friction pour l'IA. Utiliserait NeuronBox dès qu'il peut faire `neuron run llama3:8b` et que ça marche du premier coup.

**Aisha, chercheuse en NLP**
A accès à 4 GPU on-prem dans son labo. Utilise aujourd'hui Conda + scripts shell artisanaux. Veut reproductibilité et partage facile avec ses collègues. Le `neuron.yaml` commitable dans git est sa killer feature.

**Startup IA early-stage (3 personnes)**
Prototype vite, fine-tune souvent. Cloud trop cher pour l'itération rapide. Veut un outil qui fait le lien entre le laptop du dev et le serveur du labo, avec le même `neuron.yaml`.

---

## Questions ouvertes

- Distribution : binaire statique via `curl | sh` ou package manager (brew, apt) ?
- Monétisation : open source + support enterprise, ou freemium (features avancées payantes) ?
- Priorité plateforme : Linux first, ou macOS (Apple Silicon) en parallèle dès le début ?
- Nom final : NeuronBox est un working title — trop générique ?
