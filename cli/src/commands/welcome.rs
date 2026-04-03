//! Home screen when `neuron` is run with no subcommand.

use std::io::{self, Write};

use crate::ui::theme;

const BANNER: &str = r#"

'|.   '|'                                           '||''|.                   
 |'|   |    ....  ... ...  ... ..    ...   .. ...    ||   ||    ...   ... ... 
 | '|. |  .|...||  ||  ||   ||' '' .|  '|.  ||  ||   ||'''|.  .|  '|.  '|..'  
 |   |||  ||       ||  ||   ||     ||   ||  ||  ||   ||    || ||   ||   .|.   
.|.   '|   '|...'  '|..'|. .||.     '|..|' .||. ||. .||...|'   '|..|' .|  ||. 
                                                                        
"#;

/// Print welcome banner and getting-started flow (English, themed when TTY).
pub fn run() -> io::Result<()> {
    let mut out = io::stdout().lock();

    if theme::colors_enabled() {
        use crossterm::style::{Color, Stylize};
        writeln!(out, "{}", BANNER.trim_start_matches('\n').with(Color::Cyan))?;
        writeln!(
            out,
            "{}",
            " Local ML projects — declarative neuron.yaml, shared model store, lightweight daemon."
                .with(Color::Grey)
        )?;
    } else {
        writeln!(out, "{}", BANNER.trim_start_matches('\n'))?;
        writeln!(
            out,
            "Local ML projects — declarative neuron.yaml, shared model store, lightweight daemon."
        )?;
    }

    writeln!(out)?;

    theme::print_primary_line("Getting started")?;
    theme::print_command_line(
        " 1.",
        "neuron init              # create neuron.yaml in the current directory",
    )?;
    theme::print_command_line(
        " 2.",
        "neuron pull <model>      # download weights into the global store",
    )?;
    theme::print_command_line(
        " 3.",
        "neuron run               # run your entrypoint (venv + optional GPU checks)",
    )?;
    theme::print_command_line(
        " 4.",
        "neuron dashboard         # live TUI: sessions + host GPU summary",
    )?;
    theme::print_command_line(
        "   ",
        "neuron dashboard --demo  # full mock (Unix): fake sessions + synthetic VRAM overlay",
    )?;
    theme::print_warning_line(" Tip: start the daemon with `neuron daemon` in another terminal if `run` does not spawn it.")?;

    writeln!(out)?;
    theme::print_success_line(
        " Run `neuron help` for all commands. Use `neuron help <command>` for details.",
    )?;
    theme::print_line("")?;

    Ok(())
}
