# CLI UX (contributors)

## Default entry

- `neuron` with **no subcommand** prints the welcome screen (banner, short pitch, numbered flow, hint to run `neuron help`).
- `neuron help` / `neuron --help` use Clap `long_about` and `after_help` in English.

## Theming (`cli/src/ui/theme.rs`)

- **TTY + color:** colors apply when stdout is a terminal and `NO_COLOR` is unset.
- **Ratatui** helpers (`rt_*`) style the dashboard (titles, commands, success, warning, error, muted).
- **Crossterm** helpers (`print_*_line`) style the welcome screen on plain stdout.

Semantic intent: primary/title, command/cyan hints, warning/orange tips, error/red, success/green, muted/grey secondary text.

## Welcome vs dashboard

- Welcome: one-shot stdout + crossterm (no alternate screen).
- Dashboard: ratatui full-screen TUI; `q` or `Esc` to quit.
