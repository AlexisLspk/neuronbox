//! Local TUI: daemon sessions, live token chart, VRAM gauge, hot-swap strip, host/GPU (Unix socket).
//!
//! GitHub / Mac demo: `NEURONBOX_DEMO_SYNTHETIC_METRICS=1` injects fake NVIDIA MiB + GPU cap
//! (purely local data, not sent to the daemon).

use std::collections::hash_map::DefaultHasher;
use std::collections::{HashMap, HashSet, VecDeque};
use std::hash::{Hash, Hasher};
use std::io::stdout;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use anyhow::{Context, Result};
use crossterm::{
    event::{self, Event, KeyCode},
    execute,
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
};
use neuronbox_runtime::host::HostProbe;
use neuronbox_runtime::protocol::{ActiveModelInfo, DaemonRequest, DaemonResponse, SessionInfo};
use neuronbox_runtime::HostSnapshot;
use ratatui::layout::{Constraint, Layout, Rect};
use ratatui::style::{Color as RColor, Modifier, Style};
use ratatui::symbols;
use ratatui::text::{Line, Span};
use ratatui::widgets::{
    Axis, Block, Borders, Cell, Chart, Dataset, GraphType, Paragraph, Row, Table, Wrap,
};

use super::dashboard_demo;
use crate::daemon_client::DaemonSession;
use crate::daemon_spawn;
use crate::ui::theme;
use tokio::sync::watch;

/// Samples kept for throughput / VRAM sparklines (~30 s wall time at [`STATS_POLL`], ~10 Hz).
const HISTORY_CAP: usize = 300;
/// Recent suffix (samples) for the Y cap: max over the **whole** window + slow `smooth_y`
/// flattened curves after a spike; we track the recent max (~[`AXIS_Y_TAIL`] × 100 ms).
const AXIS_Y_TAIL: usize = 90;
/// Poll `Stats` this often so tok/s updates feel real-time (~10 Hz).
const STATS_POLL: Duration = Duration::from_millis(100);
/// `HostProbe::snapshot` is heavier; refresh about once per second while UI polls at [`STATS_POLL`].
const HOST_PROBE_EVERY_N_TICKS: u64 = 10;
/// Visible rows in the sessions table (excluding the aggregated « Others » row).
const SESSIONS_SUMMARY_MAX: usize = 15;
/// Throughput series (tok/s) in the main chart: top N + aggregate of the rest.
const THROUGHPUT_TOP_N: usize = 5;

static DEMO_CLI_SYNTHETIC: AtomicBool = AtomicBool::new(false);

#[derive(Default)]
struct DashboardUiState {
    sessions_fullscreen: bool,
    throughput_fullscreen: bool,
}

struct CliSyntheticGuard;

impl Drop for CliSyntheticGuard {
    fn drop(&mut self) {
        DEMO_CLI_SYNTHETIC.store(false, Ordering::SeqCst);
    }
}

fn demo_synthetic_metrics() -> bool {
    if DEMO_CLI_SYNTHETIC.load(Ordering::Relaxed) {
        return true;
    }
    std::env::var_os("NEURONBOX_DEMO_SYNTHETIC_METRICS")
        .map(|v| {
            let s = v.to_string_lossy().to_ascii_lowercase();
            matches!(s.as_str(), "1" | "true" | "yes")
        })
        .unwrap_or(false)
}

fn pid_name_hash(pid: u32, name: &str) -> u64 {
    let mut h = DefaultHasher::new();
    pid.hash(&mut h);
    name.hash(&mut h);
    h.finish()
}

/// Stable color per session (PID + name) — outside the « sticky » throughput chart.
fn throughput_series_color(s: &SessionInfo) -> RColor {
    let idx = (pid_name_hash(s.pid, s.name.as_str()) as usize) % 8;
    theme::chart_series_color(idx)
}

/// Throughput chart colors: one hue per slot in [`TokenHistory::chart_sticky_pids`]
/// (0..5 → five always-distinct palette colors), even when tok/s order changes.
fn throughput_chart_series_color(history: &TokenHistory, s: &SessionInfo) -> RColor {
    if let Some(slot) = history.chart_sticky_pids.iter().position(|&p| p == s.pid) {
        return theme::chart_series_color(slot);
    }
    throughput_series_color(s)
}

/// Fake « NVIDIA » MiB to populate the dashboard (demo only).
fn synthetic_vram_mib(pid: u32, name: &str, est: u64) -> u64 {
    let est = est.max(512);
    let x = pid_name_hash(pid, name);
    let frac = 0.14 + (x % 10_000) as f64 / 10_000.0 * 0.78;
    let v = (est as f64 * frac) as u64;
    v.clamp(256, est)
}

fn effective_vram_by_pid(sessions: &[SessionInfo], real: &HashMap<u32, u64>) -> HashMap<u32, u64> {
    if !demo_synthetic_metrics() {
        return real.clone();
    }
    let mut m = real.clone();
    for s in sessions {
        let r = m.get(&s.pid).copied().unwrap_or(0);
        if r < 128 {
            m.insert(
                s.pid,
                synthetic_vram_mib(s.pid, &s.name, s.estimated_vram_mb),
            );
        }
    }
    m
}

#[derive(Clone, Copy)]
enum GpuCapMode {
    /// Sum of HostProbe `memory_total_mb` (reliable).
    Real(u64),
    /// Invented cap for demo (Mac / probe reports 0 MiB).
    Synthetic(u64),
    /// Metal / no total: avoid a misleading ratio.
    Unknown,
}

fn resolve_gpu_cap_mb(snap: &HostSnapshot, total_est: u64, n_sessions: usize) -> GpuCapMode {
    let real: u64 = snap.gpus.iter().map(|g| g.memory_total_mb).sum();
    if demo_synthetic_metrics() {
        // Small real NVIDIA card: keep the real total if the probe is credible.
        if real >= 512 {
            return GpuCapMode::Real(real.max(1));
        }
        let cap = 49152u64
            .max(total_est.saturating_add(20480))
            .max(6144 * n_sessions as u64);
        return GpuCapMode::Synthetic(cap.max(32768));
    }
    if real >= 512 {
        return GpuCapMode::Real(real.max(1));
    }
    GpuCapMode::Unknown
}

pub async fn run(demo: bool) -> Result<()> {
    if demo {
        #[cfg(not(unix))]
        {
            anyhow::bail!(
                "neuron dashboard --demo is only supported on Unix (needs `sleep` for mock PIDs)"
            );
        }
    }

    let _synth_guard = if demo {
        std::env::set_var("NEURONBOX_DISABLE_VRAM_WATCH", "1");
        DEMO_CLI_SYNTHETIC.store(true, Ordering::SeqCst);
        Some(CliSyntheticGuard)
    } else {
        None
    };

    daemon_spawn::ensure_daemon_running().await?;

    let (shutdown_tx, shutdown_rx) = watch::channel(false);
    #[cfg(unix)]
    let demo_task = if demo {
        Some(tokio::spawn(dashboard_demo::run(shutdown_rx)))
    } else {
        None
    };

    if demo {
        tokio::time::sleep(Duration::from_millis(350)).await;
    }

    let mut session = DaemonSession::connect().await?;

    enable_raw_mode()?;
    let stdout = stdout();
    let mut stdout = stdout;
    execute!(stdout, EnterAlternateScreen)?;
    let backend = ratatui::backend::CrosstermBackend::new(stdout);
    let mut terminal = ratatui::Terminal::new(backend)?;

    let mut history = TokenHistory::default();
    let mut demo_tick: u64 = 0;

    let demo_shutdown = if demo {
        Some(shutdown_tx.clone())
    } else {
        None
    };
    let mut ui_state = DashboardUiState::default();
    let res = run_loop(
        &mut terminal,
        &mut session,
        &mut history,
        &mut demo_tick,
        demo_shutdown,
        &mut ui_state,
    )
    .await;

    let _ = disable_raw_mode();
    let _ = execute!(terminal.backend_mut(), LeaveAlternateScreen);

    let _ = shutdown_tx.send(true);
    #[cfg(unix)]
    if let Some(t) = demo_task {
        let _ = tokio::time::timeout(Duration::from_secs(8), t).await;
    }

    res
}

type ThroughputTopSeries = Vec<(SessionInfo, Vec<(f64, f64)>)>;

struct TokenHistory {
    by_pid: HashMap<u32, VecDeque<f64>>,
    aggregate_tok: VecDeque<f64>,
    /// Sum of NVIDIA MiB (`vram_used_by_pid`) for in-session PIDs, one point per [`STATS_POLL`].
    vram_used_registered: VecDeque<f64>,
    /// Fixed cohort (≤5 PIDs) for the throughput chart: no churn while a slot is held
    /// by a still-live session; only display order follows instantaneous tok/s.
    chart_sticky_pids: Vec<u32>,
    /// Displayed Y cap (rises fast, falls slowly) so the axis does not jump every frame.
    throughput_y_axis: f64,
    aggregate_y_axis: f64,
    vram_y_axis: f64,
}

impl Default for TokenHistory {
    fn default() -> Self {
        Self {
            by_pid: HashMap::new(),
            aggregate_tok: VecDeque::new(),
            vram_used_registered: VecDeque::new(),
            chart_sticky_pids: Vec::new(),
            throughput_y_axis: 10.0,
            aggregate_y_axis: 1.0,
            vram_y_axis: 1.0,
        }
    }
}

impl TokenHistory {
    /// One record per PID (last wins): avoids double push per tick if the list is duplicated.
    fn dedup_sessions_last_wins(sessions: &[SessionInfo]) -> Vec<SessionInfo> {
        let mut m: HashMap<u32, SessionInfo> = HashMap::new();
        for s in sessions {
            m.insert(s.pid, s.clone());
        }
        let mut v: Vec<SessionInfo> = m.into_values().collect();
        v.sort_by_key(|s| s.pid);
        v
    }

    fn update(&mut self, sessions: &[SessionInfo], vram_map: &HashMap<u32, u64>) {
        let sessions = Self::dedup_sessions_last_wins(sessions);
        let live: HashSet<u32> = sessions.iter().map(|s| s.pid).collect();
        self.by_pid.retain(|pid, _| live.contains(pid));

        let mut sum = 0.0_f64;
        for s in &sessions {
            let v = s.tokens_per_sec.unwrap_or(0.0);
            sum += v;
            let q = self
                .by_pid
                .entry(s.pid)
                .or_insert_with(|| VecDeque::with_capacity(HISTORY_CAP));
            while q.len() >= HISTORY_CAP {
                q.pop_front();
            }
            q.push_back(v);
        }

        while self.aggregate_tok.len() >= HISTORY_CAP {
            self.aggregate_tok.pop_front();
        }
        self.aggregate_tok.push_back(sum);

        let sum_vram: f64 = sessions
            .iter()
            .map(|s| vram_map.get(&s.pid).copied().unwrap_or(0) as f64)
            .sum();
        while self.vram_used_registered.len() >= HISTORY_CAP {
            self.vram_used_registered.pop_front();
        }
        self.vram_used_registered.push_back(sum_vram);
        self.reconcile_chart_sticky_pids(&sessions);
        self.refresh_display_axis_caps();
    }

    /// Fills the chart with at most [`THROUGHPUT_TOP_N`] PIDs: once chosen, they are replaced
    /// only if the session disappears (unregister). Display order is recomputed elsewhere from live tok/s.
    fn reconcile_chart_sticky_pids(&mut self, sessions: &[SessionInfo]) {
        if sessions.is_empty() {
            self.chart_sticky_pids.clear();
            return;
        }
        let live: HashSet<u32> = sessions.iter().map(|s| s.pid).collect();
        self.chart_sticky_pids.retain(|p| live.contains(p));

        let mut outside: Vec<SessionInfo> = sessions
            .iter()
            .filter(|s| !self.chart_sticky_pids.contains(&s.pid))
            .cloned()
            .collect();
        outside.sort_by(|a, b| {
            let sa = a.tokens_per_sec.unwrap_or(0.0);
            let sb = b.tokens_per_sec.unwrap_or(0.0);
            sb.partial_cmp(&sa)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.pid.cmp(&b.pid))
        });

        while self.chart_sticky_pids.len() < THROUGHPUT_TOP_N {
            let Some(next) = outside.first() else {
                break;
            };
            let pid = next.pid;
            self.chart_sticky_pids.push(pid);
            outside.retain(|x| x.pid != pid);
        }
    }

    /// Same X scale for all series: the latest sample is on the right (`HISTORY_CAP - 1`),
    /// short histories occupy the end of the window (not squashed to the left).
    fn deque_to_aligned_points(dq: &VecDeque<f64>) -> Vec<(f64, f64)> {
        let l = dq.len();
        dq.iter()
            .enumerate()
            .map(|(i, &v)| ((HISTORY_CAP - l + i) as f64, v))
            .collect()
    }

    fn deque_tail_max(dq: &VecDeque<f64>, tail: usize) -> f64 {
        if dq.is_empty() {
            return 0.0;
        }
        let n = dq.len().min(tail.max(1));
        dq.iter().rev().take(n).copied().fold(0.0_f64, f64::max)
    }

    /// Rises fast with the recent peak, falls reasonably fast when values drop (avoids a « frozen » axis).
    fn smooth_y_cap_responsive(field: &mut f64, target: f64) {
        if target > *field {
            *field = target;
        } else {
            *field = *field * 0.88 + target * 0.12;
        }
    }

    fn refresh_display_axis_caps(&mut self) {
        let tail = AXIS_Y_TAIL;
        let mut peak_tok = 1.0_f64;
        if self.chart_sticky_pids.is_empty() {
            for q in self.by_pid.values() {
                peak_tok = peak_tok.max(Self::deque_tail_max(q, tail));
            }
        } else {
            for &pid in &self.chart_sticky_pids {
                if let Some(q) = self.by_pid.get(&pid) {
                    peak_tok = peak_tok.max(Self::deque_tail_max(q, tail));
                }
            }
        }
        let t_tok = (peak_tok * 1.12).max(10.0);
        Self::smooth_y_cap_responsive(&mut self.throughput_y_axis, t_tok);

        let agg_peak = Self::deque_tail_max(&self.aggregate_tok, tail);
        let t_agg = (agg_peak * 1.08).max(1.0);
        Self::smooth_y_cap_responsive(&mut self.aggregate_y_axis, t_agg);

        let v_peak = Self::deque_tail_max(&self.vram_used_registered, tail);
        let t_v = (v_peak * 1.08).max(1.0);
        Self::smooth_y_cap_responsive(&mut self.vram_y_axis, t_v);
    }

    /// Curves = locked cohort `chart_sticky_pids`, sorted by **instantaneous** tok/s (same PIDs, variable order).
    fn chart_points_top_n(&self, sessions: &[SessionInfo], n: usize) -> ThroughputTopSeries {
        let sessions = Self::dedup_sessions_last_wins(sessions);
        let pid_to: HashMap<u32, SessionInfo> =
            sessions.iter().map(|s| (s.pid, s.clone())).collect();

        let mut pids: Vec<u32> = self
            .chart_sticky_pids
            .iter()
            .copied()
            .filter(|p| pid_to.contains_key(p))
            .collect();
        pids.sort_by(|&pa, &pb| {
            let sa = pid_to
                .get(&pa)
                .map(|s| s.tokens_per_sec.unwrap_or(0.0))
                .unwrap_or(0.0);
            let sb = pid_to
                .get(&pb)
                .map(|s| s.tokens_per_sec.unwrap_or(0.0))
                .unwrap_or(0.0);
            sb.partial_cmp(&sa)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| pa.cmp(&pb))
        });
        pids.truncate(n);

        let mut top_series: Vec<(SessionInfo, Vec<(f64, f64)>)> = Vec::new();
        for pid in pids {
            let Some(s) = pid_to.get(&pid) else {
                continue;
            };
            let pts = self
                .by_pid
                .get(&pid)
                .map(Self::deque_to_aligned_points)
                .unwrap_or_default();
            top_series.push((s.clone(), pts));
        }
        top_series
    }

    fn aggregate_line_points(&self) -> Vec<(f64, f64)> {
        Self::deque_to_aligned_points(&self.aggregate_tok)
    }

    fn vram_sum_line_points(&self) -> Vec<(f64, f64)> {
        Self::deque_to_aligned_points(&self.vram_used_registered)
    }
}

async fn run_loop(
    terminal: &mut ratatui::Terminal<ratatui::backend::CrosstermBackend<std::io::Stdout>>,
    session: &mut DaemonSession,
    history: &mut TokenHistory,
    demo_tick: &mut u64,
    demo_shutdown: Option<watch::Sender<bool>>,
    ui_state: &mut DashboardUiState,
) -> Result<()> {
    let mut snap = tokio::task::spawn_blocking(HostProbe::snapshot)
        .await
        .context("host probe join failed")?;
    let mut ui_tick: u64 = 0;

    loop {
        while event::poll(Duration::ZERO)? {
            if let Event::Key(key) = event::read()? {
                match key.code {
                    KeyCode::Char('q') => {
                        if let Some(tx) = demo_shutdown.as_ref() {
                            let _ = tx.send(true);
                        }
                        return Ok(());
                    }
                    KeyCode::Char('x') => {
                        if ui_state.sessions_fullscreen || ui_state.throughput_fullscreen {
                            ui_state.sessions_fullscreen = false;
                            ui_state.throughput_fullscreen = false;
                        }
                    }
                    KeyCode::Esc => {
                        if ui_state.sessions_fullscreen || ui_state.throughput_fullscreen {
                            ui_state.sessions_fullscreen = false;
                            ui_state.throughput_fullscreen = false;
                        } else {
                            if let Some(tx) = demo_shutdown.as_ref() {
                                let _ = tx.send(true);
                            }
                            return Ok(());
                        }
                    }
                    KeyCode::Char('a') => {
                        if !ui_state.sessions_fullscreen && !ui_state.throughput_fullscreen {
                            ui_state.sessions_fullscreen = true;
                            ui_state.throughput_fullscreen = false;
                        }
                    }
                    KeyCode::Char('g') | KeyCode::Char('G') => {
                        if !ui_state.sessions_fullscreen && !ui_state.throughput_fullscreen {
                            ui_state.throughput_fullscreen = true;
                            ui_state.sessions_fullscreen = false;
                        }
                    }
                    _ => {}
                }
            }
        }

        let stats_resp = session.request(DaemonRequest::Stats).await;
        if let Ok(DaemonResponse::Stats {
            sessions,
            vram_used_by_pid,
            ..
        }) = &stats_resp
        {
            let eff = effective_vram_by_pid(sessions, vram_used_by_pid);
            history.update(sessions, &eff);
        }

        if ui_tick > 0 && ui_tick.is_multiple_of(HOST_PROBE_EVERY_N_TICKS) {
            snap = tokio::task::spawn_blocking(HostProbe::snapshot)
                .await
                .context("host probe join failed")?;
        }

        let stats_resp_borrow = stats_resp;
        let synth = demo_synthetic_metrics();
        let tick = *demo_tick;
        terminal.draw(|frame| {
            let area = frame.area();

            if ui_state.sessions_fullscreen {
                render_sessions_fullscreen(frame, area, &stats_resp_borrow);
                return;
            }
            if ui_state.throughput_fullscreen {
                render_throughput_fullscreen(frame, area, &stats_resp_borrow, history);
                return;
            }

            const TOP_H: u16 = 6;
            const FOOTER_H: u16 = 3;
            const CHART_MIN: u16 = 8;
            let desired_triple = sessions_summary_column_height();
            let max_triple = area.height.saturating_sub(TOP_H + FOOTER_H + CHART_MIN);
            let triple_h = desired_triple.min(max_triple).max(5);

            let main_chunks = Layout::default()
                .direction(ratatui::layout::Direction::Vertical)
                .constraints([
                    Constraint::Length(TOP_H),
                    Constraint::Length(triple_h),
                    Constraint::Min(CHART_MIN),
                    Constraint::Length(FOOTER_H),
                ])
                .split(area);

            let top_band = Layout::default()
                .direction(ratatui::layout::Direction::Horizontal)
                .constraints([Constraint::Percentage(50), Constraint::Percentage(50)])
                .split(main_chunks[0]);

            let top_left = Layout::default()
                .direction(ratatui::layout::Direction::Vertical)
                .constraints([Constraint::Length(3), Constraint::Length(3)])
                .split(top_band[0]);
            render_header(frame, top_left[0]);
            render_swap_strip(frame, top_left[1], &stats_resp_borrow);
            render_vram_gauge(frame, top_band[1], &stats_resp_borrow, &snap);

            let triple = Layout::default()
                .direction(ratatui::layout::Direction::Horizontal)
                .constraints([
                    Constraint::Percentage(34),
                    Constraint::Percentage(33),
                    Constraint::Percentage(33),
                ])
                .split(main_chunks[1]);

            render_sessions_summary(frame, triple[0], &stats_resp_borrow);
            if synth {
                render_synthetic_feed(frame, triple[1], tick);
            } else {
                let p = Paragraph::new(vec![
                    Line::from(Span::styled(
                        "Normal mode: only the left column is live daemon data.",
                        theme::rt_command(),
                    )),
                    Line::from(Span::styled(
                        "This panel is idle. Animated sample lines: `neuron dashboard --demo` \
                         or NEURONBOX_DEMO_SYNTHETIC_METRICS=1.",
                        theme::rt_muted(),
                    )),
                ])
                .block(block_with_title("Decorative feed (off)"))
                .wrap(Wrap { trim: true });
                frame.render_widget(p, triple[1]);
            }
            render_host_and_gpu(frame, triple[2], &snap, &stats_resp_borrow, synth);

            let bottom_row = Layout::default()
                .direction(ratatui::layout::Direction::Horizontal)
                .constraints([Constraint::Percentage(67), Constraint::Percentage(33)])
                .split(main_chunks[2]);

            render_throughput_main_chart(frame, bottom_row[0], &stats_resp_borrow, history, false);
            render_sum_charts_column(frame, bottom_row[1], history);

            render_footer(frame, main_chunks[3]);
        })?;

        *demo_tick = tick.wrapping_add(1);
        ui_tick = ui_tick.wrapping_add(1);
        tokio::time::sleep(STATS_POLL).await;
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
    let mut spans = vec![
        Span::styled(" NeuronBox ", theme::rt_title()),
        Span::styled(
            " live dashboard",
            theme::rt_header_secondary().add_modifier(Modifier::BOLD),
        ),
        Span::styled(" · ~10 Hz stats (~100 ms)", theme::rt_note()),
    ];
    if demo_synthetic_metrics() {
        spans.push(Span::styled(" · demo overlay", theme::rt_warning()));
    }
    let title = Paragraph::new(Line::from(spans)).block(block_with_title("neuron dashboard"));
    frame.render_widget(title, area);
}

fn render_swap_strip(
    frame: &mut ratatui::Frame<'_>,
    area: Rect,
    stats_resp: &Result<DaemonResponse, anyhow::Error>,
) {
    let block = block_with_title("Hot-swap (active model)");
    let line = match stats_resp {
        Ok(DaemonResponse::Stats { active_model, .. }) => match active_model {
            Some(ActiveModelInfo {
                model_ref,
                quantization,
            }) => {
                let q = quantization
                    .as_deref()
                    .filter(|s| !s.is_empty())
                    .unwrap_or("default");
                Line::from(vec![
                    Span::styled("model: ", theme::rt_metric_swap()),
                    Span::styled(model_ref.as_str(), theme::rt_command()),
                    Span::styled("  ·  quant: ", theme::rt_metric_swap()),
                    Span::styled(q, theme::rt_success()),
                ])
            }
            None => Line::from(vec![Span::styled(
                "(none yet: use `neuron swap <id>`)",
                theme::rt_muted(),
            )]),
        },
        Err(e) => Line::from(vec![Span::styled(
            format!("daemon: {e:#}"),
            theme::rt_error(),
        )]),
        _ => Line::from(vec![Span::styled(
            "unexpected response",
            theme::rt_warning(),
        )]),
    };
    let p = Paragraph::new(line).block(block);
    frame.render_widget(p, area);
}

fn sessions_sorted_by_tok(sessions: &[SessionInfo]) -> Vec<SessionInfo> {
    let mut v: Vec<SessionInfo> = sessions.to_vec();
    v.sort_by(|a, b| {
        let ta = a.tokens_per_sec.unwrap_or(0.0);
        let tb = b.tokens_per_sec.unwrap_or(0.0);
        tb.partial_cmp(&ta)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.pid.cmp(&b.pid))
    });
    v
}

fn session_table_rows(
    slice: &[SessionInfo],
    v_eff: &HashMap<u32, u64>,
    name_max: usize,
    path_max: usize,
    others_label: Option<(usize, u64, u64, f64)>,
) -> Vec<Row<'static>> {
    let mut rows: Vec<Row<'static>> = slice
        .iter()
        .map(|s| {
            let tok_style = if s.tokens_per_sec.is_some() {
                theme::rt_metric_tokens()
            } else {
                theme::rt_muted()
            };
            let tok = s
                .tokens_per_sec
                .map(|t| format!("{t:.1}"))
                .unwrap_or_else(|| "—".into());
            let nv = v_eff
                .get(&s.pid)
                .map(|m| format!("{m}"))
                .unwrap_or_else(|| "—".into());
            Row::new(vec![
                Cell::from(Span::styled(
                    truncate_name(&s.name, name_max),
                    theme::rt_command(),
                )),
                Cell::from(Span::styled(
                    truncate_name(
                        s.model_dir
                            .as_deref()
                            .filter(|p| !p.is_empty())
                            .unwrap_or("—"),
                        path_max,
                    ),
                    theme::rt_note(),
                )),
                Cell::from(Span::styled(s.pid.to_string(), theme::rt_pid())),
                Cell::from(Span::styled(
                    format!("{}", s.estimated_vram_mb),
                    theme::rt_metric_vram(),
                )),
                Cell::from(Span::styled(nv, theme::rt_metric_vram_live())),
                Cell::from(Span::styled(tok, tok_style)),
            ])
        })
        .collect();
    if let Some((n_rest, sum_est, sum_nv, sum_tok)) = others_label {
        rows.push(Row::new(vec![
            Cell::from(Span::styled(
                format!("Others ({n_rest})"),
                theme::rt_warning(),
            )),
            Cell::from(Span::styled("—", theme::rt_muted())),
            Cell::from(Span::styled("—", theme::rt_muted())),
            Cell::from(Span::styled(format!("{sum_est}"), theme::rt_metric_vram())),
            Cell::from(Span::styled(
                format!("{sum_nv}"),
                theme::rt_metric_vram_live(),
            )),
            Cell::from(Span::styled(
                format!("{sum_tok:.1}"),
                theme::rt_metric_tokens(),
            )),
        ]));
    }
    rows
}

/// Sessions column height: fixed (max rows + « Others ») so the chart area does not jump.
fn sessions_summary_column_height() -> u16 {
    const TITLE_BOTTOM: u16 = 1;
    const BLOCK_FRAME: u16 = 3;
    let body_rows = SESSIONS_SUMMARY_MAX + 1;
    let table_inner = 1u16 + body_rows as u16;
    let raw = BLOCK_FRAME + table_inner + TITLE_BOTTOM;
    raw.saturating_sub(1).max(4)
}

fn sessions_summary_block(title_bottom: Line<'_>) -> Block<'_> {
    Block::default()
        .borders(Borders::ALL)
        .title(Line::from(vec![Span::styled(
            "Sessions (daemon)",
            theme::rt_block_title(),
        )]))
        .title_bottom(title_bottom)
}

fn render_sessions_summary(
    frame: &mut ratatui::Frame<'_>,
    area: Rect,
    stats_resp: &Result<DaemonResponse, anyhow::Error>,
) {
    let hint_line = |sessions_len: Option<usize>| match sessions_len {
        Some(n) if n > SESSIONS_SUMMARY_MAX => Line::from(vec![
            Span::styled("[a] ", theme::rt_command()),
            Span::styled("See all ", theme::rt_command()),
            Span::styled(format!("· {n} sessions "), theme::rt_muted()),
        ]),
        Some(_) | None => Line::from(vec![
            Span::styled("[a] ", theme::rt_command()),
            Span::styled("See all ", theme::rt_command()),
            Span::styled("(full table)", theme::rt_muted()),
        ]),
    };

    let name_max = match area.width {
        w if w < 22 => 5,
        w if w < 30 => 8,
        _ => 14,
    };
    let path_max = match area.width {
        w if w < 40 => 8,
        w if w < 55 => 12,
        _ => 18,
    };

    match stats_resp {
        Ok(DaemonResponse::Stats { sessions, .. }) if sessions.is_empty() => {
            let p = Paragraph::new(Line::from(Span::styled(
                "(no sessions: `neuron run` + RegisterSession)",
                theme::rt_muted(),
            )))
            .block(sessions_summary_block(hint_line(None)))
            .wrap(Wrap { trim: true });
            frame.render_widget(p, area);
        }
        Ok(DaemonResponse::Stats {
            sessions,
            vram_used_by_pid,
            ..
        }) => {
            let v_eff = effective_vram_by_pid(sessions, vram_used_by_pid);
            let ordered = sessions_sorted_by_tok(sessions);

            let (top_slice, others_agg) = if ordered.len() > SESSIONS_SUMMARY_MAX {
                let top: Vec<SessionInfo> =
                    ordered.iter().take(SESSIONS_SUMMARY_MAX).cloned().collect();
                let rest = &ordered[SESSIONS_SUMMARY_MAX..];
                let n_rest = rest.len();
                let sum_est: u64 = rest.iter().map(|s| s.estimated_vram_mb).sum();
                let sum_nv: u64 = rest
                    .iter()
                    .map(|s| v_eff.get(&s.pid).copied().unwrap_or(0))
                    .sum();
                let sum_tok: f64 = rest.iter().map(|s| s.tokens_per_sec.unwrap_or(0.0)).sum();
                (top, Some((n_rest, sum_est, sum_nv, sum_tok)))
            } else {
                (ordered, None)
            };

            let rows = session_table_rows(&top_slice, &v_eff, name_max, path_max, others_agg);
            let header = Row::new(vec![
                Cell::from(Span::styled("Name", theme::rt_command())),
                Cell::from(Span::styled("Model path", theme::rt_note())),
                Cell::from(Span::styled("PID", theme::rt_pid())),
                Cell::from(Span::styled("Est.", theme::rt_metric_vram())),
                Cell::from(Span::styled("NVIDIA", theme::rt_metric_vram_live())),
                Cell::from(Span::styled("tok/s", theme::rt_metric_tokens())),
            ])
            .style(theme::rt_table_header());
            let t = Table::new(
                rows,
                [
                    Constraint::Percentage(22),
                    Constraint::Percentage(24),
                    Constraint::Percentage(10),
                    Constraint::Percentage(12),
                    Constraint::Percentage(14),
                    Constraint::Percentage(18),
                ],
            )
            .header(header)
            .block(sessions_summary_block(hint_line(Some(sessions.len()))));
            frame.render_widget(t, area);
        }
        Ok(other) => {
            let p = Paragraph::new(Line::from(Span::styled(
                format!("Unexpected daemon response: {other:?}"),
                theme::rt_warning(),
            )))
            .block(sessions_summary_block(Line::from(Span::styled(
                "—",
                theme::rt_muted(),
            ))))
            .wrap(Wrap { trim: true });
            frame.render_widget(p, area);
        }
        Err(e) => {
            let p = Paragraph::new(Line::from(Span::styled(
                format!("Daemon error: {e:#}"),
                theme::rt_error(),
            )))
            .block(sessions_summary_block(Line::from(Span::styled(
                "—",
                theme::rt_muted(),
            ))))
            .wrap(Wrap { trim: true });
            frame.render_widget(p, area);
        }
    }
}

fn render_sessions_fullscreen(
    frame: &mut ratatui::Frame<'_>,
    area: Rect,
    stats_resp: &Result<DaemonResponse, anyhow::Error>,
) {
    let parts = Layout::default()
        .direction(ratatui::layout::Direction::Vertical)
        .constraints([Constraint::Length(1), Constraint::Min(0)])
        .split(area);

    let bar = Paragraph::new(Line::from(vec![
        Span::styled(" [x] ", theme::rt_error().add_modifier(Modifier::BOLD)),
        Span::styled("Close  ", theme::rt_note()),
        Span::styled("·  Esc", theme::rt_command()),
        Span::styled("  ·  ", theme::rt_note()),
        Span::styled("q", theme::rt_command()),
        Span::styled(" quit", theme::rt_note()),
    ]));
    frame.render_widget(bar, parts[0]);

    let block = block_with_title("All sessions (daemon)");

    match stats_resp {
        Ok(DaemonResponse::Stats { sessions, .. }) if sessions.is_empty() => {
            let p = Paragraph::new(Line::from(Span::styled("(no sessions)", theme::rt_muted())))
                .block(block);
            frame.render_widget(p, parts[1]);
        }
        Ok(DaemonResponse::Stats {
            sessions,
            vram_used_by_pid,
            ..
        }) => {
            let v_eff = effective_vram_by_pid(sessions, vram_used_by_pid);
            let ordered = sessions_sorted_by_tok(sessions);
            let name_max = match parts[1].width {
                w if w < 28 => 8,
                w if w < 50 => 16,
                _ => 24,
            };
            let path_max = match parts[1].width {
                w if w < 60 => 18,
                w if w < 90 => 30,
                _ => 42,
            };
            let rows = session_table_rows(&ordered, &v_eff, name_max, path_max, None);
            let header = Row::new(vec![
                Cell::from(Span::styled("Name", theme::rt_command())),
                Cell::from(Span::styled("Model path", theme::rt_note())),
                Cell::from(Span::styled("PID", theme::rt_pid())),
                Cell::from(Span::styled("Est.", theme::rt_metric_vram())),
                Cell::from(Span::styled("NVIDIA", theme::rt_metric_vram_live())),
                Cell::from(Span::styled("tok/s", theme::rt_metric_tokens())),
            ])
            .style(theme::rt_table_header());
            let t = Table::new(
                rows,
                [
                    Constraint::Percentage(18),
                    Constraint::Percentage(34),
                    Constraint::Percentage(8),
                    Constraint::Percentage(10),
                    Constraint::Percentage(12),
                    Constraint::Percentage(18),
                ],
            )
            .header(header)
            .block(block);
            frame.render_widget(t, parts[1]);
        }
        Ok(other) => {
            let p = Paragraph::new(Line::from(Span::styled(
                format!("Unexpected: {other:?}"),
                theme::rt_warning(),
            )))
            .block(block);
            frame.render_widget(p, parts[1]);
        }
        Err(e) => {
            let p = Paragraph::new(Line::from(Span::styled(
                format!("Daemon error: {e:#}"),
                theme::rt_error(),
            )))
            .block(block);
            frame.render_widget(p, parts[1]);
        }
    }
}

fn render_mini_line_chart(
    frame: &mut ratatui::Frame<'_>,
    area: Rect,
    title: Line<'_>,
    points: &[(f64, f64)],
    line_color: RColor,
    y_axis_title: &str,
    y_max: f64,
) {
    if points.is_empty() {
        let p = Paragraph::new(Line::from(Span::styled(
            "... awaiting samples",
            theme::rt_muted(),
        )))
        .block(Block::default().borders(Borders::ALL).title(title))
        .wrap(Wrap { trim: true });
        frame.render_widget(p, area);
        return;
    }

    let x_max = HISTORY_CAP as f64;
    let y_max = y_max.max(1.0);
    let dataset = Dataset::default()
        .marker(symbols::Marker::Braille)
        .graph_type(GraphType::Line)
        .style(Style::default().fg(line_color))
        .data(points);
    let chart = Chart::new(vec![dataset])
        .block(Block::default().borders(Borders::ALL).title(title))
        .x_axis(Axis::default().style(theme::rt_axis()).bounds([0.0, x_max]))
        .y_axis(
            Axis::default()
                .title(Span::styled(y_axis_title, theme::rt_block_title()))
                .style(theme::rt_axis())
                .labels([
                    Line::from("0"),
                    Line::from(format!("{:.0}", y_max * 0.5)),
                    Line::from(format!("{y_max:.0}")),
                ])
                .bounds([0.0, y_max]),
        );
    frame.render_widget(chart, area);
}

fn render_throughput_main_chart(
    frame: &mut ratatui::Frame<'_>,
    area: Rect,
    stats_resp: &Result<DaemonResponse, anyhow::Error>,
    history: &TokenHistory,
    fullscreen: bool,
) {
    match stats_resp {
        Ok(DaemonResponse::Stats { sessions, .. }) if sessions.is_empty() => {
            let p = Paragraph::new(Line::from(Span::styled(
                "Chart fills when sessions register.",
                theme::rt_muted(),
            )))
            .block(block_with_title("Throughput (tok/s)"))
            .wrap(Wrap { trim: true });
            frame.render_widget(p, area);
        }
        Ok(DaemonResponse::Stats { sessions, .. }) => {
            let top = history.chart_points_top_n(sessions, THROUGHPUT_TOP_N);
            let y_max = history.throughput_y_axis.max(10.0);
            let x_max = HISTORY_CAP as f64;

            let title_top = if fullscreen {
                Line::from(vec![Span::styled(
                    "[x] close",
                    theme::rt_warning().add_modifier(Modifier::BOLD),
                )])
            } else {
                Line::from(vec![Span::styled("[G] expand", theme::rt_command())])
            };

            let datasets: Vec<Dataset> = top
                .iter()
                .map(|(s, pts)| {
                    Dataset::default()
                        .marker(symbols::Marker::Braille)
                        .style(Style::default().fg(throughput_chart_series_color(history, s)))
                        .graph_type(GraphType::Line)
                        .data(pts)
                })
                .collect();

            let mut legend_spans: Vec<Span> = vec![Span::styled("top tok/s: ", theme::rt_note())];
            for (i, (s, _)) in top.iter().enumerate() {
                if i > 0 {
                    legend_spans.push(Span::raw(" · "));
                }
                let leg = format!("{} ·{}", truncate_name(&s.name, 10), s.pid);
                legend_spans.push(Span::styled(
                    leg,
                    Style::default().fg(throughput_chart_series_color(history, s)),
                ));
            }

            let chart = Chart::new(datasets)
                .block(
                    Block::default()
                        .borders(Borders::ALL)
                        .title(title_top)
                        .title_style(theme::rt_block_title())
                        .title_bottom(Line::from(legend_spans)),
                )
                .x_axis(
                    Axis::default()
                        .title(Span::styled("time →", theme::rt_block_title()))
                        .style(theme::rt_axis())
                        .bounds([0.0, x_max]),
                )
                .y_axis(
                    Axis::default()
                        .title(Span::styled("tok/s", theme::rt_block_title()))
                        .style(theme::rt_axis())
                        .labels([Line::from("0"), Line::from(format!("{y_max:.0}"))])
                        .bounds([0.0, y_max]),
                );
            frame.render_widget(chart, area);
        }
        Err(_) => {
            let p = Paragraph::new("-")
                .block(block_with_title("Throughput"))
                .style(theme::rt_muted());
            frame.render_widget(p, area);
        }
        Ok(_) => {}
    }
}

fn render_throughput_fullscreen(
    frame: &mut ratatui::Frame<'_>,
    area: Rect,
    stats_resp: &Result<DaemonResponse, anyhow::Error>,
    history: &TokenHistory,
) {
    let parts = Layout::default()
        .direction(ratatui::layout::Direction::Vertical)
        .constraints([Constraint::Length(1), Constraint::Min(0)])
        .split(area);

    let bar = Paragraph::new(Line::from(vec![
        Span::styled(" [x] ", theme::rt_error().add_modifier(Modifier::BOLD)),
        Span::styled("Close throughput · ", theme::rt_note()),
        Span::styled("Esc", theme::rt_command()),
    ]));
    frame.render_widget(bar, parts[0]);
    render_throughput_main_chart(frame, parts[1], stats_resp, history, true);
}

fn render_sum_charts_column(frame: &mut ratatui::Frame<'_>, area: Rect, history: &TokenHistory) {
    let col = Layout::default()
        .direction(ratatui::layout::Direction::Vertical)
        .constraints([Constraint::Percentage(50), Constraint::Percentage(50)])
        .split(area);

    let tok_pts = history.aggregate_line_points();
    let vram_pts = history.vram_sum_line_points();
    let vram_title = if demo_synthetic_metrics() {
        "Σ MiB (demo)"
    } else {
        "Σ MiB NVIDIA"
    };

    render_mini_line_chart(
        frame,
        col[0],
        Line::from(vec![Span::styled(
            "Σ tok/s (all sessions)",
            theme::rt_block_title(),
        )]),
        &tok_pts,
        theme::rt_sparkline_fill(),
        "tok/s",
        history.aggregate_y_axis,
    );
    render_mini_line_chart(
        frame,
        col[1],
        Line::from(vec![Span::styled(
            format!("{vram_title} (registered PIDs)"),
            theme::rt_block_title(),
        )]),
        &vram_pts,
        theme::rt_sparkline_vram_fill(),
        "MiB",
        history.vram_y_axis,
    );
}

fn truncate_name(s: &str, max_chars: usize) -> String {
    if s.chars().count() <= max_chars {
        return s.to_string();
    }
    s.chars()
        .take(max_chars.saturating_sub(2))
        .chain("..".chars())
        .collect()
}

/// Wrap text into lines of at most `width` characters (word-aware when possible, else hard break).
fn wrap_text_to_lines(s: &str, width: usize) -> Vec<String> {
    let width = width.max(1);
    let s = s.trim();
    if s.is_empty() {
        return vec![String::new()];
    }
    let mut lines: Vec<String> = Vec::new();
    let mut current = String::new();
    let mut cur_w = 0usize;
    for word in s.split_whitespace() {
        let wl = word.chars().count();
        if wl > width {
            if !current.is_empty() {
                lines.push(std::mem::take(&mut current));
                cur_w = 0;
            }
            let mut buf = String::new();
            let mut n = 0usize;
            for c in word.chars() {
                buf.push(c);
                n += 1;
                if n >= width {
                    lines.push(std::mem::take(&mut buf));
                    n = 0;
                }
            }
            if !buf.is_empty() {
                current = buf;
                cur_w = current.chars().count();
            }
            continue;
        }
        let add = if cur_w == 0 { wl } else { 1 + wl };
        if cur_w + add > width && !current.is_empty() {
            lines.push(std::mem::take(&mut current));
            cur_w = 0;
        }
        if !current.is_empty() {
            current.push(' ');
            cur_w += 1;
        }
        current.push_str(word);
        cur_w += wl;
    }
    if !current.is_empty() {
        lines.push(current);
    }
    lines
}

/// VRAM ratio bar: not Ratatui's `Gauge` widget — with an empty label it still reserves
/// a center column on the middle row (inverted space), leaving a hole / ghost character.
/// Here we fill the rectangle with `█` uniformly on every row.
fn paint_vram_ratio_bar(buf: &mut ratatui::buffer::Buffer, area: Rect, ratio: f64, style: Style) {
    if area.width == 0 || area.height == 0 {
        return;
    }
    let r = ratio.clamp(0.0, 1.0);
    let w = f64::from(area.width);
    let filled_end = area.left().saturating_add((w * r).floor() as u16);
    let fg = style.fg.unwrap_or(RColor::Reset);
    let bg = style.bg.unwrap_or(RColor::Reset);
    for y in area.top()..area.bottom() {
        for x in area.left()..area.right() {
            let c = &mut buf[(x, y)];
            if x < filled_end {
                c.set_symbol(symbols::block::FULL).set_fg(fg).set_bg(bg);
            } else {
                c.set_symbol(" ");
                c.set_fg(RColor::Reset);
                c.set_bg(RColor::Reset);
            }
        }
    }
}

/// Ratio gauge in a frame: caption on the **first line inside** the block (not `Block::title`),
/// to avoid a colored artifact on the top border when the title mixes several `Span`s.
fn render_vram_ratio_gauge_box(
    frame: &mut ratatui::Frame<'_>,
    area: Rect,
    caption: Line<'_>,
    ratio: f64,
    gauge_style: Style,
) {
    let block = Block::default().borders(Borders::ALL);
    let inner = block.inner(area);
    if inner.height < 2 {
        frame.render_widget(
            Paragraph::new(caption).block(Block::default().borders(Borders::ALL)),
            area,
        );
        return;
    }
    let rows = Layout::default()
        .direction(ratatui::layout::Direction::Vertical)
        .constraints([Constraint::Length(1), Constraint::Min(1)])
        .split(inner);
    frame.render_widget(block, area);
    frame.render_widget(Paragraph::new(caption), rows[0]);
    paint_vram_ratio_bar(frame.buffer_mut(), rows[1], ratio, gauge_style);
}

fn render_vram_gauge(
    frame: &mut ratatui::Frame<'_>,
    area: Rect,
    stats_resp: &Result<DaemonResponse, anyhow::Error>,
    snap: &HostSnapshot,
) {
    match stats_resp {
        Ok(DaemonResponse::Stats {
            sessions,
            vram_used_by_pid,
            ..
        }) => {
            let total_est: u64 = sessions.iter().map(|s| s.estimated_vram_mb).sum();
            let v_eff = effective_vram_by_pid(sessions, vram_used_by_pid);
            let used_reg: u64 = sessions
                .iter()
                .map(|s| v_eff.get(&s.pid).copied().unwrap_or(0))
                .sum();
            let cap_mode = resolve_gpu_cap_mb(snap, total_est, sessions.len());

            match cap_mode {
                GpuCapMode::Unknown => {
                    let lines = vec![
                        Line::from(vec![
                            Span::styled("Declared Σ ", theme::rt_metric_vram()),
                            Span::styled(format!("{total_est} MiB"), theme::rt_metric_vram()),
                            Span::styled(
                                " · GPU VRAM total: unknown (e.g. Metal)",
                                theme::rt_warning(),
                            ),
                        ]),
                        Line::from(vec![
                            Span::styled("Driver MiB (PIDs): ", theme::rt_metric_vram_live()),
                            Span::styled(
                                format!("{used_reg} MiB (no VRAM cap ratio on this host)"),
                                theme::rt_note(),
                            ),
                        ]),
                        Line::from(Span::styled(
                            "Tip: NEURONBOX_DEMO_SYNTHETIC_METRICS=1 for full mock gauges + feed.",
                            theme::rt_note(),
                        )),
                    ];
                    let p = Paragraph::new(lines)
                        .block(block_with_title("VRAM (no bogus ratios)"))
                        .wrap(Wrap { trim: true });
                    frame.render_widget(p, area);
                }
                GpuCapMode::Real(total_gpu) => {
                    let split = Layout::default()
                        .direction(ratatui::layout::Direction::Horizontal)
                        .constraints([Constraint::Percentage(50), Constraint::Percentage(50)])
                        .split(area);
                    let r_decl = (total_est as f64 / total_gpu as f64).clamp(0.0, 1.0);
                    let r_used = (used_reg as f64 / total_gpu as f64).clamp(0.0, 1.0);
                    let label_decl =
                        format!("{total_est} / {total_gpu} MiB · {} sess", sessions.len());
                    let label_used = format!("{used_reg} / {total_gpu} MiB");
                    let cap_decl = Line::from(vec![
                        Span::styled("Declared (est.) ", theme::rt_block_title()),
                        Span::styled(label_decl.as_str(), theme::rt_metric_vram()),
                    ]);
                    let cap_used = Line::from(vec![
                        Span::styled("Used (NVIDIA PIDs) ", theme::rt_block_title()),
                        Span::styled(label_used.as_str(), theme::rt_metric_vram_live()),
                    ]);
                    render_vram_ratio_gauge_box(
                        frame,
                        split[0],
                        cap_decl,
                        r_decl,
                        theme::rt_metric_vram(),
                    );
                    render_vram_ratio_gauge_box(
                        frame,
                        split[1],
                        cap_used,
                        r_used,
                        theme::rt_metric_vram_live(),
                    );
                }
                GpuCapMode::Synthetic(total_gpu) => {
                    let split = Layout::default()
                        .direction(ratatui::layout::Direction::Horizontal)
                        .constraints([Constraint::Percentage(50), Constraint::Percentage(50)])
                        .split(area);
                    let r_decl = (total_est as f64 / total_gpu as f64).clamp(0.0, 1.0);
                    let r_used = (used_reg as f64 / total_gpu as f64).clamp(0.0, 1.0);
                    let label_decl =
                        format!("{total_est} / {total_gpu} MiB · {} sess", sessions.len());
                    let label_used = format!("{used_reg} / {total_gpu} MiB");
                    let cap_decl = Line::from(vec![
                        Span::styled("Declared (est.) ", theme::rt_block_title()),
                        Span::styled(label_decl.as_str(), theme::rt_metric_vram()),
                    ]);
                    let cap_used = Line::from(vec![
                        Span::styled("Used (table) ", theme::rt_block_title()),
                        Span::styled(label_used.as_str(), theme::rt_metric_vram_live()),
                    ]);
                    render_vram_ratio_gauge_box(
                        frame,
                        split[0],
                        cap_decl,
                        r_decl,
                        theme::rt_metric_vram(),
                    );
                    render_vram_ratio_gauge_box(
                        frame,
                        split[1],
                        cap_used,
                        r_used,
                        theme::rt_metric_vram_live(),
                    );
                }
            }
        }
        _ => {
            let p = Paragraph::new(Line::from(Span::styled("no stats", theme::rt_muted())))
                .block(block_with_title("VRAM"));
            frame.render_widget(p, area);
        }
    }
}

fn render_synthetic_feed(frame: &mut ratatui::Frame<'_>, area: Rect, tick: u64) {
    const LINES: &[&str] = &[
        "[kv] cache hit 94% · 128 tok batch",
        "[attn] flash sdp · seq 4096 · layer 18/32",
        "[opt] adamw step 18440 · lr 1.8e-5 (cosine)",
        "[ckpt] shard 2/4 written · 1.2 GB",
        "[swap] signal v1 applied · weights mmap warm",
        "[data] prefetch queue 12 batches · workers 4",
        "[mem] allocator peak 41.2 GiB · fragmentation low",
        "[eval] bleu 42.7 · rougeL 0.39 · step 2200",
        "[dist] rank 0 allreduce 3.1 ms · bucket 50 MiB",
        "[prof] forward 118 ms · backward 240 ms",
        "[hf] hub etag match · skip redownload",
        "[sched] next batch ETA 40 ms · backlog 2",
    ];
    let block = Block::default()
        .borders(Borders::ALL)
        .title(Line::from(vec![Span::styled(
            "Synthetic activity (demo · NEURONBOX_DEMO_SYNTHETIC_METRICS)",
            theme::rt_block_title(),
        )]));
    let inner = block.inner(area);
    let n = inner.height.max(1) as usize;
    let mut out: Vec<Line> = Vec::with_capacity(n);
    for i in 0..n {
        let idx = (tick as usize + i) % LINES.len();
        out.push(Line::from(vec![Span::styled(
            LINES[idx],
            Style::default().fg(theme::chart_series_color(idx)),
        )]));
    }
    let p = Paragraph::new(out).block(block);
    frame.render_widget(p, area);
}

fn render_host_and_gpu(
    frame: &mut ratatui::Frame<'_>,
    area: Rect,
    snap: &HostSnapshot,
    stats_resp: &Result<DaemonResponse, anyhow::Error>,
    synthetic_demo: bool,
) {
    let compact = area.width < 46;
    let w = (area.width as usize).saturating_sub(2).max(8);
    // One fewer row on the daemon side to match the usable height of the other columns.
    let gpu_line_cap = if compact { 5 } else { 12 };

    let mut lines: Vec<Line> = Vec::new();

    let host_label = "Host: ";
    let host_detail = format!(
        "{} / {} · {:?}",
        snap.platform.os, snap.platform.arch, snap.training_backend
    );
    let host_wrap_w = w.saturating_sub(host_label.chars().count());
    let host_parts = wrap_text_to_lines(&host_detail, host_wrap_w.max(4));
    for (i, p) in host_parts.iter().enumerate() {
        if i == 0 {
            lines.push(Line::from(vec![
                Span::styled(host_label, theme::rt_title()),
                Span::styled(p.as_str(), theme::rt_command()),
            ]));
        } else {
            let indent = " ".repeat(host_label.chars().count());
            lines.push(Line::from(vec![
                Span::styled(indent, theme::rt_title()),
                Span::styled(p.as_str(), theme::rt_command()),
            ]));
        }
    }

    let nvml_txt = if snap.probes.nvml {
        "NVML yes"
    } else {
        "NVML no"
    };
    let nvml_style = if snap.probes.nvml {
        theme::rt_success()
    } else {
        theme::rt_warning()
    };
    lines.push(Line::from(vec![
        Span::styled("Probe: ", theme::rt_metric_gpu()),
        Span::styled(nvml_txt, nvml_style),
    ]));

    for g in &snap.gpus {
        let line = format!(
            "  GPU{} · {} · {} MiB · {}",
            g.index, g.name, g.memory_total_mb, g.backend
        );
        for chunk in wrap_text_to_lines(&line, w) {
            lines.push(Line::from(Span::styled(chunk, theme::rt_metric_gpu())));
        }
    }

    if synthetic_demo {
        if compact {
            for chunk in wrap_text_to_lines(
                "Demo: RSS ~18 GiB · prefill 6 / decode 14 · ~38 W (illustrative, not measured).",
                w,
            ) {
                lines.push(Line::from(Span::styled(chunk, theme::rt_warning())));
            }
        } else {
            for chunk in wrap_text_to_lines(
                "Demo overlay: lines below are illustrative (not measured).",
                w,
            ) {
                lines.push(Line::from(Span::styled(chunk, theme::rt_warning())));
            }
            for chunk in wrap_text_to_lines("  Pool · unified memory · host RSS ~18.4 GiB", w) {
                lines.push(Line::from(Span::styled(chunk, theme::rt_command())));
            }
            for chunk in wrap_text_to_lines("  Queue · prefill 6 · decode 14 · batch cap 8", w) {
                lines.push(Line::from(Span::styled(chunk, theme::rt_command())));
            }
            for chunk in wrap_text_to_lines("  Power · package ~38 W · throttle none", w) {
                lines.push(Line::from(Span::styled(chunk, theme::rt_success())));
            }
        }
    }

    if let Ok(DaemonResponse::Stats {
        gpu_lines, note, ..
    }) = stats_resp
    {
        let sec_title = if compact {
            "neurond GPU"
        } else {
            "NVIDIA compute (daemon)"
        };
        lines.push(Line::from(Span::styled(sec_title, theme::rt_title())));
        if gpu_lines.is_empty() {
            lines.push(Line::from(Span::styled(
                "(idle / not NVIDIA)",
                theme::rt_muted(),
            )));
        } else {
            for l in gpu_lines.iter().take(gpu_line_cap) {
                for chunk in wrap_text_to_lines(l.as_str(), w) {
                    lines.push(Line::from(Span::styled(chunk, theme::rt_success())));
                }
            }
        }
        if let Some(n) = note {
            for chunk in wrap_text_to_lines(n.as_str(), w) {
                lines.push(Line::from(Span::styled(chunk, theme::rt_warning())));
            }
        }
    }

    let block_title = if compact {
        "Host / GPU"
    } else {
        "GPU / processes"
    };
    let block = block_with_title(block_title);
    let inner = block.inner(area);
    let max_lines = inner.height.max(1) as usize;
    if lines.len() > max_lines {
        lines.truncate(max_lines);
    }
    let p = Paragraph::new(lines).wrap(Wrap { trim: true }).block(block);
    frame.render_widget(p, area);
}

fn render_footer(frame: &mut ratatui::Frame<'_>, area: Rect) {
    let p = Paragraph::new(Line::from(vec![
        Span::styled("q", theme::rt_command()),
        Span::styled(" quit · ", theme::rt_note()),
        Span::styled("Esc", theme::rt_command()),
        Span::styled(" back · ", theme::rt_note()),
        Span::styled("a", theme::rt_command()),
        Span::styled(" sessions · ", theme::rt_note()),
        Span::styled("G", theme::rt_command()),
        Span::styled(" graph · ", theme::rt_note()),
        Span::styled("x", theme::rt_command()),
        Span::styled(" close panel", theme::rt_note()),
    ]))
    .block(block_with_title("Shortcuts"));
    frame.render_widget(p, area);
}
