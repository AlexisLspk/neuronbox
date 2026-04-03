//! Color tokens for CLI welcome (crossterm) and TUI (ratatui). Respects `NO_COLOR` and non-TTY stdout.
//! Dashboard colors stay **light on black** (no DarkGray body text).

use std::io::{self, IsTerminal};

use crossterm::style::{Color as CColor, Stylize};
use ratatui::style::{Color as RColor, Modifier, Style};

/// True when we should emit ANSI colors on stdout.
pub fn colors_enabled() -> bool {
    std::env::var_os("NO_COLOR").is_none() && io::stdout().is_terminal()
}

// --- Ratatui (dashboard) ---

/// Secondary text (readable on black).
pub fn rt_muted() -> Style {
    Style::default().fg(RColor::Rgb(145, 150, 165))
}

pub fn rt_title() -> Style {
    Style::default()
        .fg(RColor::Cyan)
        .add_modifier(Modifier::BOLD)
}

/// Panel titles (borders): plain white, readable on black.
pub fn rt_block_title() -> Style {
    Style::default().fg(RColor::Rgb(255, 255, 255))
}

pub fn rt_command() -> Style {
    Style::default().fg(RColor::LightCyan)
}

pub fn rt_success() -> Style {
    Style::default().fg(RColor::Green)
}

pub fn rt_warning() -> Style {
    Style::default().fg(RColor::Rgb(255, 185, 60))
}

pub fn rt_error() -> Style {
    Style::default().fg(RColor::Rgb(255, 100, 100))
}

/// Footnotes, chart axis lines (visible on black).
pub fn rt_note() -> Style {
    Style::default().fg(RColor::Rgb(165, 175, 195))
}

pub fn rt_header_secondary() -> Style {
    Style::default().fg(RColor::Rgb(200, 205, 220))
}

/// Chart X/Y axes and grid labels.
pub fn rt_axis() -> Style {
    Style::default().fg(RColor::Rgb(150, 165, 190))
}

/// Table column headers (distinct from cell text).
pub fn rt_table_header() -> Style {
    Style::default()
        .fg(RColor::Rgb(240, 245, 255))
        .add_modifier(Modifier::BOLD)
}

/// PID column (high contrast).
pub fn rt_pid() -> Style {
    Style::default()
        .fg(RColor::Rgb(255, 235, 120))
        .add_modifier(Modifier::BOLD)
}

// --- Dashboard metrics (one semantic color per variable family) ---

/// Throughput tokens/s (chart lines, labels).
pub fn rt_metric_tokens() -> Style {
    Style::default()
        .fg(RColor::Rgb(130, 210, 255))
        .add_modifier(Modifier::BOLD)
}

/// Declared VRAM estimate (YAML / RegisterSession).
pub fn rt_metric_vram() -> Style {
    Style::default().fg(RColor::Rgb(200, 170, 255))
}

/// Measured NVIDIA VRAM for PID (compute apps).
pub fn rt_metric_vram_live() -> Style {
    Style::default().fg(RColor::Rgb(255, 160, 230))
}

/// Hot-swap / active model strip.
pub fn rt_metric_swap() -> Style {
    Style::default()
        .fg(RColor::Rgb(255, 140, 210))
        .add_modifier(Modifier::BOLD)
}

/// GPU listing and compute process lines.
pub fn rt_metric_gpu() -> Style {
    Style::default().fg(RColor::Rgb(255, 220, 120))
}

/// Aggregate tok/s sparkline fill.
pub fn rt_sparkline_fill() -> RColor {
    RColor::Rgb(90, 235, 175)
}

/// Aggregate VRAM-used sparkline (registered PIDs).
pub fn rt_sparkline_vram_fill() -> RColor {
    RColor::Rgb(255, 140, 220)
}

/// Distinct series on the multi-line chart (sessions).
pub fn chart_series_color(index: usize) -> RColor {
    const PALETTE: [RColor; 8] = [
        RColor::Rgb(130, 210, 255),
        RColor::Rgb(255, 190, 100),
        RColor::Rgb(140, 255, 180),
        RColor::Rgb(255, 130, 200),
        RColor::Rgb(210, 170, 255),
        RColor::Rgb(120, 235, 235),
        RColor::Rgb(245, 245, 130),
        RColor::Rgb(255, 150, 150),
    ];
    PALETTE[index % PALETTE.len()]
}

// --- Crossterm (welcome) ---

pub fn print_line(s: &str) -> io::Result<()> {
    use std::io::Write;
    writeln!(io::stdout().lock(), "{s}")
}

pub fn print_primary_line(s: &str) -> io::Result<()> {
    use std::io::Write;
    let mut out = io::stdout().lock();
    if colors_enabled() {
        writeln!(out, "{}", s.with(CColor::Cyan))?;
    } else {
        writeln!(out, "{s}")?;
    }
    Ok(())
}

pub fn print_command_line(prefix: &str, cmd: &str) -> io::Result<()> {
    use std::io::Write;
    let mut out = io::stdout().lock();
    if colors_enabled() {
        write!(
            out,
            "{}",
            prefix.with(CColor::Rgb {
                r: 160,
                g: 168,
                b: 185
            })
        )?;
        writeln!(out, " {}", cmd.with(CColor::Cyan))?;
    } else {
        writeln!(out, "{prefix} {cmd}")?;
    }
    Ok(())
}

pub fn print_warning_line(s: &str) -> io::Result<()> {
    use std::io::Write;
    let mut out = io::stdout().lock();
    if colors_enabled() {
        writeln!(
            out,
            "{}",
            s.with(CColor::Rgb {
                r: 255,
                g: 185,
                b: 60
            })
        )?;
    } else {
        writeln!(out, "{s}")?;
    }
    Ok(())
}

pub fn print_success_line(s: &str) -> io::Result<()> {
    use std::io::Write;
    let mut out = io::stdout().lock();
    if colors_enabled() {
        writeln!(out, "{}", s.with(CColor::Green))?;
    } else {
        writeln!(out, "{s}")?;
    }
    Ok(())
}
