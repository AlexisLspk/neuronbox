//! Color tokens for CLI welcome (crossterm) and TUI (ratatui). Respects `NO_COLOR` and non-TTY stdout.

use std::io::{self, IsTerminal};

use crossterm::style::{Color as CColor, Stylize};
use ratatui::style::{Color as RColor, Modifier, Style};

/// True when we should emit ANSI colors on stdout.
pub fn colors_enabled() -> bool {
    std::env::var_os("NO_COLOR").is_none() && io::stdout().is_terminal()
}

// --- Ratatui (dashboard) ---

pub fn rt_muted() -> Style {
    Style::default().fg(RColor::DarkGray)
}

pub fn rt_title() -> Style {
    Style::default()
        .fg(RColor::Cyan)
        .add_modifier(Modifier::BOLD)
}

pub fn rt_block_title() -> Style {
    Style::default().fg(RColor::LightBlue)
}

pub fn rt_command() -> Style {
    Style::default().fg(RColor::LightCyan)
}

pub fn rt_success() -> Style {
    Style::default().fg(RColor::Green)
}

pub fn rt_warning() -> Style {
    Style::default().fg(RColor::Rgb(255, 165, 0))
}

pub fn rt_error() -> Style {
    Style::default().fg(RColor::Red)
}

pub fn rt_note() -> Style {
    Style::default().fg(RColor::DarkGray)
}

pub fn rt_header_secondary() -> Style {
    Style::default().fg(RColor::Gray)
}

// --- Crossterm (welcome) ---

pub fn print_line(s: &str) -> io::Result<()> {
    use std::io::Write;
    let mut out = io::stdout().lock();
    if colors_enabled() {
        writeln!(out, "{}", s)?;
    } else {
        // Strip simple ANSI if any — plain lines here are ASCII
        writeln!(out, "{s}")?;
    }
    Ok(())
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
        write!(out, "{}", prefix.with(CColor::Grey))?;
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
                g: 165,
                b: 0
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
