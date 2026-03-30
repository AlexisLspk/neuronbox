# Contrat daemon : sessions et `neuron stats`

Le daemon `neurond` écoute sur un socket Unix (défaut : `~/.neuronbox/neuron.sock`, surcharge possible avec `NEURONBOX_SOCKET`). Protocole : **une ligne JSON par requête**, une ligne JSON par réponse.

Les types exacts sont définis en Rust dans `runtime/src/protocol.rs` (`DaemonRequest` / `DaemonResponse`), avec `#[serde(tag = "method")]` et `rename_all = "snake_case"`.

## Enregistrer une session (`neuron run`)

`neuron run` envoie après le lancement du processus Python :

```json
{
  "method": "register_session",
  "name": "nom-du-projet",
  "estimated_vram_mb": 8192,
  "pid": 12345,
  "tokens_per_sec": null
}
```

- **`estimated_vram_mb`** : estimation utilisée pour la surveillance soft NVIDIA (Linux), pas une réservation matérielle.
- **`tokens_per_sec`** : optionnel ; peut être omis ou `null` si inconnu.

À la fin du processus, la CLI envoie `unregister_session` avec le même `pid`.

## Mettre à jour les tokens/s depuis votre code

Le registre est indexé par **`pid`** : un nouvel appel `register_session` avec le **même** `pid` **remplace** la ligne existante (même nom, nouvelle estimation, nouveau débit).

Pour afficher une valeur dans `neuron stats`, renvoyez périodiquement par exemple :

```json
{
  "method": "register_session",
  "name": "mon-inference",
  "estimated_vram_mb": 12000,
  "pid": 12345,
  "tokens_per_sec": 47.3
}
```

### Script d’exemple

Le dépôt fournit [`cli/scripts/neuronbox_daemon_session.py`](../cli/scripts/neuronbox_daemon_session.py) (`ping`, `register`) utilisant uniquement la bibliothèque standard Python.

## Autres méthodes utiles

| `method`           | Rôle |
|--------------------|------|
| `ping`             | Santé du daemon → `pong` |
| `list_sessions`    | Liste des `SessionInfo` |
| `stats`            | Sessions + processus compute NVIDIA (NVML si build `nvml`, sinon `nvidia-smi`) |
| `version`          | Négociation de version de protocole (`v` entier) |

## Session persistante (plusieurs requêtes)

Le serveur (`runtime/src/server.rs`) lit des lignes JSON en boucle sur **une même connexion** jusqu’à EOF. Un client peut donc :

1. Ouvrir le socket une fois ;
2. Écrire une ligne JSON (requête) + `\n` ;
3. Lire une ligne JSON (réponse) ;
4. Répéter 2–3 sans reconnecter ;
5. Fermer le socket quand terminé.

La CLI Rust expose cela via `DaemonSession` dans `cli/src/daemon_client.rs` (`connect`, puis `request` en boucle). L’helper `request()` sans session reste disponible pour un aller-retour unique.

Le script Python d’exemple ouvre encore une connexion par commande ; pour une session longue en Python, garder le socket ouvert et enchaîner les lignes comme ci-dessus.

## Limites actuelles

- Pas de chiffrement : trafic local uniquement (socket Unix).
- `tokens_per_sec` n’est pas agrégé dans le temps côté daemon : c’est la **dernière** valeur enregistrée pour ce PID.
