# VRAM, MIG et monitoring NeuronBox

## Contrôle « soft » avant lancement

`neuron run` peut refuser de démarrer si `gpu.min_vram` dans `neuron.yaml` dépasse la VRAM totale estimée du GPU (NVIDIA via **NVML** si compilé avec la feature `nvml` sur Linux, sinon `nvidia-smi` ; AMD via `rocm-smi` quand disponible).

## Surveillance pendant l’exécution (daemon)

`neurond` peut lancer une boucle qui obtient la mémoire utilisée par PID sur GPU NVIDIA (**NVML** si feature `nvml`, sinon `nvidia-smi`). Si un processus dépasse environ **115 %** de la valeur `estimated_vram_mb` enregistrée lors de `RegisterSession`, le daemon envoie **SIGKILL** au PID (Linux).

### Désactiver la surveillance VRAM (SIGKILL)

Pour désactiver cette boucle (par ex. environnements de test, machines sans GPU, ou politique locale) :

```bash
export NEURONBOX_DISABLE_VRAM_WATCH=1
# valeurs reconnues : 1, true, yes (insensible à la casse)
neuron daemon
```

Sans cette variable, le comportement par défaut reste inchangé.

Ce n’est **pas** un plafond matériel : sans **MIG** (Multi-Instance GPU, NVIDIA datacenter) ou partitionnement équivalent, CUDA ne garantit pas un quota VRAM strict par processus comme pour la RAM système.

## MIG (NVIDIA)

Pour un **isolation forte** par instance GPU sur des cartes compatibles (A100, H100, etc.) :

1. Configurer MIG via les outils NVIDIA du système (`nvidia-smi mig`).
2. Exposer uniquement l’instance souhaitée avec `CUDA_VISIBLE_DEVICES` (souvent un UUID d’instance).
3. Documenter dans votre déploiement qu’un job NeuronBox = une instance MIG.

NeuronBox ne configure pas MIG automatiquement ; la doc ici sert de point d’entrée.

## Variables PyTorch injectées

En mode host ou via Docker OCI, NeuronBox peut définir `PYTORCH_CUDA_ALLOC_CONF` (par ex. `max_split_size_mb` dérivé de `gpu.min_vram`) pour limiter la fragmentation. C’est une **heuristique**, pas une garantie de pic VRAM.

## NVML (Linux NVIDIA)

- **Prérequis** : pilote NVIDIA installé (`libnvidia-ml.so` présent au runtime).
- **Build** : `cargo build -p neuronbox-runtime --features nvml` ou `cargo build -p neuronbox-cli --features nvml` (réactive la même feature sur le runtime).
- **`neuron host inspect`** : champ `probes.nvml` indique si la liste GPU NVIDIA a été obtenue via NVML (`true`) ou `nvidia-smi` (`false`).
- En cas d’échec d’init NVML, le code **retombe** sur `nvidia-smi` (comportement inchangé sans feature).

Sur **macOS**, la feature compile mais le chemin NVML n’est pas utilisé (pas de module NVML dans la sonde) ; Apple GPU reste géré à part.

## Multi-GPU et DDP

Voir le guide dédié : [MULTI_GPU.md](MULTI_GPU.md) (`neuron run --gpu`, `torchrun`, champ `gpu.strategy`, feuille de route).
