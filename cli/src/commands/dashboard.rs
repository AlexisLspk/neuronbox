//! Local TUI: daemon sessions + host summary (Unix socket only, no HTTP).

use std::io::stdout;
use std::time::Duration;

use anyhow::{Context, Result};
use crossterm::{
    event::{self, Event, KeyCode},
    execute,
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
};
use neuronbox_runtime::host::HostProbe;
use neuronbox_runtime::protocol::{DaemonRequest, DaemonResponse};
use neuronbox_runtime::HostSnapshot;
use ratatui::layout::{Constraint, Layout, Rect};
use ratatui::style::{Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, Cell, Paragraph, Row, Table, Wrap};

use crate::daemon_client::DaemonSession;
use crate::daemon_spawn;
use crate::ui::theme;

pub async fn run() -> Result<()> {
    daemon_spawn::ensure_daemon_running().await?;

    let mut session = DaemonSession::connect().await?;

    enable_raw_mode()?;
    let stdout = stdout();
    let mut stdout = stdout;
    execute!(stdout, EnterAlternateScreen)?;
    let backend = ratatui::backend::CrosstermBackend::new(stdout);
    let mut terminal = ratatui::Terminal::new(backend)?;

    let res = run_loop(&mut terminal, &mut session).await;

    let _ = disable_raw_mode();
    let _ = execute!(terminal.backend_mut(), LeaveAlternateScreen);

    res
}

async fn run_loop(
    terminal: &mut ratatui::Terminal<ratatui::backend::CrosstermBackend<std::io::Stdout>>,
    session: &mut DaemonSession,
) -> Result<()> {
    loop {
        while event::poll(Duration::ZERO)? {
            if let Event::Key(key) = event::read()? {
                if key.code == KeyCode::Char('q') || key.code == KeyCode::Esc {
                    return Ok(());
                }
            }
        }

        let stats_resp = session.request(DaemonRequest::Stats).await;
        let snap = tokio::task::spawn_blocking(HostProbe::snapshot)
            .await
            .context("host probe join failed")?;

        terminal.draw(|frame| {
            let area = frame.area();
            let chunks = Layout::default()
                .direction(ratatui::layout::Direction::Vertical)
                .constraints([
                    Constraint::Length(3),
                    Constraint::Min(6),
                    Constraint::Length(10),
                    Constraint::Length(2),
                ])
                .split(area);

            render_header(frame, chunks[0]);
            render_sessions(frame, chunks[1], &stats_resp);
            render_host_and_gpu(frame, chunks[2], &snap, &stats_resp);
            render_footer(frame, chunks[3]);
        })?;

        tokio::time::sleep(Duration::from_secs(1)).await;
    }
}

fn block_with_title(title: &str) -> Block<'_> {
    Block::default()
        .borders(Borders::ALL)
        .title(Line::from(vec![Span::styled(
            title,
            theme::rt_block_title(),
        )]))
}

fn render_header(frame: &mut ratatui::Frame<'_>, area: Rect) {
    let title = Paragraph::new(Line::from(vec![
        Span::styled(" NeuronBox ", theme::rt_title()),
        Span::styled(
            "— local dashboard",
            theme::rt_header_secondary().add_modifier(Modifier::BOLD),
        ),
        Span::styled(" (Unix socket + HostProbe)", theme::rt_note()),
    ]))
    .block(block_with_title("neuron dashboard"));
    frame.render_widget(title, area);
}

fn render_sessions(
    frame: &mut ratatui::Frame<'_>,
    area: Rect,
    stats_resp: &Result<DaemonResponse, anyhow::Error>,
) {
    let block = block_with_title("Sessions (daemon)");

    match stats_resp {
        Ok(DaemonResponse::Stats { sessions, .. }) if sessions.is_empty() => {
            let p = Paragraph::new(Line::from(Span::styled(
                "(no registered sessions — run a project with `neuron run`)",
                theme::rt_muted(),
            )))
            .block(block);
            frame.render_widget(p, area);
        }
        Ok(DaemonResponse::Stats { sessions, .. }) => {
            let header =
                Row::new(vec!["Name", "PID", "VRAM est.", "tok/s"]).style(theme::rt_title());
            let rows: Vec<Row> = sessions
                .iter()
                .map(|s| {
                    let tok_style = if s.tokens_per_sec.is_some() {
                        theme::rt_success()
                    } else {
                        theme::rt_muted()
                    };
                    let tok = s
                        .tokens_per_sec
                        .map(|t| format!("{t:.1}"))
                        .unwrap_or_else(|| "—".into());
                    Row::new(vec![
                        Cell::from(Span::styled(s.name.as_str(), theme::rt_command())),
                        Cell::from(Span::styled(
                            s.pid.to_string(),
                            Style::default().fg(ratatui::style::Color::Yellow),
                        )),
                        Cell::from(format!("{} MiB", s.estimated_vram_mb)),
                        Cell::from(Span::styled(tok, tok_style)),
                    ])
                })
                .collect();
            let t = Table::new(
                rows,
                [
                    Constraint::Percentage(35),
                    Constraint::Percentage(15),
                    Constraint::Percentage(25),
                    Constraint::Percentage(25),
                ],
            )
            .header(header)
            .block(block);
            frame.render_widget(t, area);
        }
        Ok(other) => {
            let p = Paragraph::new(Line::from(Span::styled(
                format!("Unexpected daemon response: {other:?}"),
                theme::rt_warning(),
            )))
            .block(block);
            frame.render_widget(p, area);
        }
        Err(e) => {
            let p = Paragraph::new(Line::from(Span::styled(
                format!("Daemon error: {e:#}"),
                theme::rt_error(),
            )))
            .block(block);
            frame.render_widget(p, area);
        }
    }
}

fn render_host_and_gpu(
    frame: &mut ratatui::Frame<'_>,
    area: Rect,
    snap: &HostSnapshot,
    stats_resp: &Result<DaemonResponse, anyhow::Error>,
) {
    let mut lines: Vec<Line> = Vec::new();

    lines.push(Line::from(vec![
        Span::styled("Host: ", theme::rt_title()),
        Span::raw(format!(
            "{} / {} — training backend {:?}",
            snap.platform.os, snap.platform.arch, snap.training_backend
        )),
    ]));

    let nvml_txt = if snap.probes.nvml {
        "yes (GPU list via NVML)"
    } else {
        "no (nvidia-smi or non-NVIDIA)"
    };
    let nvml_style = if snap.probes.nvml {
        theme::rt_success()
    } else {
        theme::rt_warning()
    };
    lines.push(Line::from(vec![
        Span::styled("NVML: ", theme::rt_title()),
        Span::styled(nvml_txt, nvml_style),
    ]));

    for g in &snap.gpus {
        lines.push(Line::from(vec![Span::styled(
            format!(
                "  GPU {} : {} — {} MiB — {}",
                g.index, g.name, g.memory_total_mb, g.backend
            ),
            theme::rt_command(),
        )]));
    }

    if let Ok(DaemonResponse::Stats {
        gpu_lines, note, ..
    }) = stats_resp
    {
        lines.push(Line::from(vec![Span::styled(
            "NVIDIA compute processes",
            theme::rt_title(),
        )]));
        if gpu_lines.is_empty() {
            lines.push(Line::from(Span::styled(
                "(no lines — idle or not NVIDIA)",
                theme::rt_muted(),
            )));
        } else {
            for l in gpu_lines.iter().take(10) {
                lines.push(Line::from(Span::styled(l.as_str(), theme::rt_success())));
            }
        }
        if let Some(n) = note {
            lines.push(Line::from(Span::styled(n.as_str(), theme::rt_warning())));
        }
    }

    let p = Paragraph::new(lines)
        .wrap(Wrap { trim: true })
        .block(block_with_title("GPU / processes"));
    frame.render_widget(p, area);
}

fn render_footer(frame: &mut ratatui::Frame<'_>, area: Rect) {
    let p = Paragraph::new(Line::from(vec![
        Span::styled("q", theme::rt_command()),
        Span::styled(" or ", theme::rt_note()),
        Span::styled("Esc", theme::rt_command()),
        Span::styled(" quit  ·  refresh ~1s", theme::rt_note()),
    ]))
    .block(block_with_title("Shortcuts"));
    frame.render_widget(p, area);
}
