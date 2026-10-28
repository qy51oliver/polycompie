"""Production-oriented styling for the PolyCompIE NiceGUI workbench."""

STYLES = """
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display&family=DM+Sans:opsz,wght@9..40,400;9..40,500;9..40,600&family=JetBrains+Mono:wght@400;500&display=swap');

:root {
    --canvas: #F3F2F1;
    --paper: #FFFFFF;
    --paper-soft: #F8F9FC;
    --ink: #262626;
    --muted: #666666;
    --navy: #123D7A;
    --navy-dark: #0A2E63;
    --blue: #1469AA;
    --blue-soft: #EDF3F9;
    --line: #E2E6ED;
    --line-strong: #B5B5B5;
    --warning: #94651B;
    --warning-soft: #FFF8E8;
    --danger: #9C3C38;
    --success: #246A4B;
    --shadow: 0 1px 2px rgba(1, 33, 105, 0.035), 0 10px 26px rgba(1, 33, 105, 0.045);
    --radius: 8px;
}

* { box-sizing: border-box; }
html { scroll-behavior: smooth; }
.sr-only { position: absolute !important; width: 1px !important; height: 1px !important; padding: 0 !important; margin: -1px !important; overflow: hidden !important; clip: rect(0, 0, 0, 0) !important; white-space: nowrap !important; border: 0 !important; }
body {
    margin: 0;
    background: var(--canvas);
    color: var(--ink);
    font-family: 'DM Sans', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    -webkit-font-smoothing: antialiased;
}

.q-page,
.q-page-container {
    align-items: stretch !important;
    justify-content: flex-start !important;
    padding: 0 !important;
}
.nicegui-content,
body > .nicegui-content {
    width: 100% !important;
    max-width: none !important;
    padding: 0 !important;
    display: block !important;
}
.app-shell {
    width: min(1240px, calc(100% - 48px));
    margin: 0 auto;
    padding: 0 0 40px;
}

/* Header */
.topbar {
    min-height: 64px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 24px;
    border-bottom: 1px solid var(--line);
}
.brand-lockup { display: flex; align-items: center; gap: 12px; }
.brand-mark {
    width: 40px;
    height: 40px;
    border-radius: 3px;
    background: var(--navy);
    color: #fff;
    display: grid;
    place-items: center;
}
.brand-glyph { width: 31px; height: 31px; overflow: visible; }
.polymer-bond, .data-row { fill: none; stroke: #FFFFFF; stroke-width: 1.7; stroke-linecap: round; stroke-linejoin: round; }
.polymer-node { fill: #FFFFFF; }
.data-cell { fill: #9DC6E7; }
.brand-name { font-family: 'DM Serif Display', Georgia, serif; font-size: 1.25rem; line-height: 1.1; color: var(--navy); }
.brand-subtitle { margin-top: 3px; color: var(--muted); font-size: 0.72rem; letter-spacing: 0.01em; }

/* Workbench */
.workbench-grid { display: grid; grid-template-columns: minmax(0, 1.62fr) minmax(340px, 0.88fr); gap: 18px; align-items: start; margin-top: 18px; }
.workflow-column { display: flex; flex-direction: column; gap: 18px; min-width: 0; }
.panel { background: var(--paper); border: 1px solid var(--line); border-radius: var(--radius); box-shadow: var(--shadow); }
.workflow-panel { padding: 22px; }
.config-panel { padding: 22px; position: sticky; top: 16px; }
.section-heading { display: flex; align-items: flex-start; gap: 12px; margin-bottom: 18px; }
.step-index { flex: 0 0 auto; width: 28px; height: 28px; display: grid; place-items: center; border-radius: 4px; color: var(--blue); background: var(--blue-soft); font: 500 0.72rem 'JetBrains Mono', monospace; }
.section-title { color: var(--navy-dark); font-size: 1.04rem; font-weight: 600; line-height: 1.25; }
.section-description { color: var(--muted); font-size: 0.78rem; line-height: 1.55; margin-top: 4px; max-width: 620px; }

/* Upload */
.upload-zone-wrap { width: 100%; position: relative; }
.drop-zone { min-height: 94px; width: 100%; border: 1.5px dashed #AABDD5; border-radius: 8px; background: var(--paper-soft); display: grid; grid-template-columns: minmax(0, 1fr) auto; align-items: center; gap: 18px; padding: 18px 20px; cursor: pointer; transition: border-color .18s ease, background .18s ease, transform .18s ease; }
.drop-zone:hover { border-color: var(--blue); background: #F4F7FB; transform: translateY(-1px); }
.drop-copy { min-width: 0; }
.drop-title { color: var(--ink); font-size: 0.9rem; font-weight: 600; }
.drop-subtitle { margin-top: 4px; color: var(--muted); font-size: 0.75rem; }
.browse-action { color: var(--blue); border: 1px solid #AABDD5; background: white; border-radius: 5px; padding: 8px 12px; font-size: 0.75rem; font-weight: 600; white-space: nowrap; }
.hidden-uploader { position: absolute !important; width: 1px !important; height: 1px !important; opacity: 0 !important; overflow: hidden !important; pointer-events: none !important; }
.file-list-header { display: flex; align-items: center; justify-content: space-between; margin-top: 20px; }
.file-count { font-size: 0.74rem; color: var(--muted); font-weight: 600; text-transform: uppercase; letter-spacing: .08em; }
.text-action { color: var(--blue); font-size: 0.75rem; cursor: pointer; }
.file-list { display: flex; flex-direction: column; gap: 8px; margin-top: 9px; }
.file-row { display: flex; align-items: center; min-width: 0; gap: 12px; padding: 12px 12px; border: 1px solid var(--line); border-radius: 7px; background: #FCFCFD; }
.file-type { flex: 0 0 auto; padding: 5px 7px; border-radius: 5px; background: #F0E8E7; color: #8B4542; font: 500 0.58rem 'JetBrains Mono', monospace; }
.file-meta { min-width: 0; flex: 1; }
.file-name { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; font-size: 0.8rem; font-weight: 500; }
.file-size { color: var(--muted); font-size: 0.69rem; margin-top: 2px; }
.remove-file { color: var(--muted) !important; min-width: 30px !important; }

/* Properties */
.property-entry { display: grid; grid-template-columns: 1fr auto; gap: 10px; align-items: stretch; }
.property-input, .api-key-input, .ollama-select { width: 100%; }
.q-field__control { min-height: 44px !important; border-radius: 9px !important; }
.q-field--outlined .q-field__control:before { border-color: var(--line-strong) !important; }
.q-field--focused .q-field__control:after { border-color: var(--blue) !important; }
.secondary-btn { min-height: 44px !important; padding: 0 17px !important; border-radius: 9px !important; background: var(--navy) !important; color: white !important; text-transform: none !important; font-size: 0.78rem !important; font-weight: 600 !important; }
.field-caption { margin: 16px 0 8px; color: var(--muted); font-size: 0.7rem; font-weight: 600; text-transform: uppercase; letter-spacing: .08em; }
.suggestion-list { display: flex; flex-wrap: wrap; gap: 7px; }
.suggestion-chip { min-height: 31px !important; padding: 3px 10px !important; border: 1px solid var(--line) !important; border-radius: 999px !important; background: #FBFBFC !important; color: #4F5968 !important; font-size: 0.71rem !important; font-weight: 500 !important; }
.suggestion-chip:hover { border-color: #9EB6D2 !important; color: var(--blue) !important; background: #F2F6FB !important; }
.selected-properties { min-height: 38px; margin-top: 14px; padding-top: 12px; border-top: 1px solid var(--line); display: flex; flex-wrap: wrap; gap: 8px; align-items: flex-start; }
.selected-property { display: flex; align-items: center; gap: 8px; border-radius: 999px; background: var(--navy); color: #fff; padding: 7px 9px 7px 12px; font-size: 0.73rem; }
.chip-remove { width: 18px; height: 18px; border-radius: 50%; background: rgba(255,255,255,.14); display: grid; place-items: center; cursor: pointer; line-height: 1; }
.chip-remove:hover { background: rgba(255,255,255,.25); }
.empty-selection { color: #7A828E; font-size: 0.76rem; font-style: italic; }

/* Configuration */
.field-label { color: var(--ink); font-size: 0.75rem; font-weight: 600; margin: 18px 0 7px; }
.section-heading + .field-label { margin-top: 0; }
.segmented-control { width: 100%; padding: 3px; border: 1px solid var(--line); border-radius: 8px !important; background: #F1F3F7 !important; box-shadow: none !important; }
.segmented-control .q-btn { flex: 1; min-height: 36px; border-radius: 5px !important; color: #5F6774 !important; font-size: 0.74rem !important; font-weight: 600 !important; box-shadow: none !important; }
.segmented-control .q-btn.bg-primary { color: var(--navy-dark) !important; background: #FFF !important; box-shadow: 0 1px 3px rgba(1,33,105,.12) !important; }
.field-help { color: var(--muted); font-size: 0.69rem; line-height: 1.48; margin-top: 7px; }
.provider-panel { margin-top: 11px; padding: 13px; border: 1px solid var(--line); border-radius: 10px; background: var(--paper-soft); }
.input-label { color: #4F5968; font-size: 0.69rem; font-weight: 600; margin-bottom: 7px; }
.provider-status { font-size: 0.72rem; color: var(--muted); line-height: 1.45; margin-bottom: 7px; }
.provider-status-success { color: var(--success); }
.provider-status-error { color: var(--danger); }
.text-button { color: var(--blue) !important; font-size: .7rem !important; padding: 4px 0 !important; min-height: 28px !important; }
.run-summary { margin-top: 18px; padding: 12px; border-left: 3px solid var(--blue); border-radius: 0 6px 6px 0; background: var(--blue-soft); }
.summary-kicker { color: var(--blue); font: 500 .61rem 'JetBrains Mono', monospace; letter-spacing: .1em; }
.run-summary-text { color: #294A72; font-size: .72rem; line-height: 1.5; margin-top: 5px; }
.extract-btn { width: 100%; min-height: 48px !important; margin-top: 18px; padding: 0 18px !important; border-radius: 9px !important; background: var(--navy) !important; color: white !important; text-transform: none !important; font-size: .82rem !important; font-weight: 600 !important; box-shadow: none !important; transition: background .18s ease, transform .18s ease; }
.extract-btn:hover { background: var(--navy-dark) !important; transform: translateY(-1px); }
.extract-btn .q-btn__content { display: flex; justify-content: space-between; width: 100%; }
.cancel-btn { width: 100%; min-height: 42px !important; margin-top: 8px; border: 1px solid #D9B3B1 !important; border-radius: 9px !important; background: #FFF !important; color: var(--danger) !important; text-transform: none !important; font-size: .76rem !important; box-shadow: none !important; }
.spinner-icon { display: inline-block; margin-right: 6px; animation: spin 1s linear infinite; }
@keyframes spin { to { transform: rotate(360deg); } }

/* Activity and results */
.output-panel { margin-top: 18px; padding: 22px; }
.output-heading { display: flex; justify-content: space-between; align-items: flex-start; gap: 20px; margin-bottom: 20px; }
.live-badge, .complete-badge { flex: 0 0 auto; display: inline-flex; align-items: center; gap: 6px; padding: 6px 9px; border-radius: 999px; font: 500 .6rem 'JetBrains Mono', monospace; letter-spacing: .06em; }
.live-badge { color: #6F4E13; background: #FFF5D9; border: 1px solid #F0DAA2; }
.live-badge span { width: 6px; height: 6px; border-radius: 50%; background: #C98A21; animation: pulse 1.6s ease-in-out infinite; }
.complete-badge { color: var(--success); background: #E9F5EE; border: 1px solid #B8D9C6; }
@keyframes pulse { 50% { opacity: .35; } }
.log-box { min-height: 130px; max-height: 340px; overflow: auto; border-radius: 8px; background: #081B3A; color: #D9E6F2; padding: 18px 20px; white-space: pre-wrap; font: 400 .72rem/1.75 'JetBrains Mono', monospace; }
.metrics-grid { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 10px; }
.metric-card { min-width: 0; padding: 15px; border: 1px solid var(--line); border-radius: 10px; background: var(--paper-soft); }
.metric-value { color: var(--navy); font-family: 'DM Serif Display', Georgia, serif; font-size: 1.38rem; line-height: 1.15; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.metric-label { color: var(--muted); font-size: .67rem; margin-top: 5px; }
.partial-warning { margin-top: 14px; padding: 11px 13px; border: 1px solid #EAD6A9; border-radius: 8px; background: var(--warning-soft); color: var(--warning); font-size: .73rem; line-height: 1.5; }
.json-expansion { width: 100%; margin-top: 18px; border: 1px solid var(--line); border-radius: 10px; color: var(--navy); font-size: .78rem; overflow: hidden; }
.json-box { width: 100%; max-height: 440px; overflow: auto; margin: 0; padding: 18px; background: #081B3A; color: #D9E6F2; font: 400 .7rem/1.62 'JetBrains Mono', monospace; white-space: pre; }
.empty-results { margin-top: 18px; padding: 18px; border: 1px dashed var(--line-strong); border-radius: 10px; color: var(--muted); background: var(--paper-soft); font-size: .78rem; line-height: 1.55; }
.export-row { display: flex; align-items: center; justify-content: space-between; gap: 16px; margin-top: 20px; padding-top: 18px; border-top: 1px solid var(--line); }
.export-label { color: var(--ink); font-size: .76rem; font-weight: 600; }
.export-actions { display: flex; flex-wrap: wrap; justify-content: flex-end; gap: 8px; }
.export-btn, .json-btn { min-height: 38px !important; border-radius: 8px !important; padding: 0 13px !important; text-transform: none !important; font-size: .73rem !important; box-shadow: none !important; }
.export-btn { background: var(--blue) !important; color: white !important; }
.json-btn { color: var(--navy) !important; border-color: var(--line-strong) !important; background: white !important; }

/* Footer */
.footer { margin-top: 28px; padding-top: 16px; border-top: 1px solid var(--line); display: flex; justify-content: space-between; gap: 20px; color: var(--muted); font-size: .67rem; }
.footer-note { text-align: right; }

/* Quasar consistency */
.q-btn { letter-spacing: 0 !important; }
.q-focus-helper { opacity: 0 !important; }
.q-field, .q-field__inner, .q-field__control, .q-field__native { max-width: none !important; }

@media (max-width: 980px) {
    .app-shell { width: min(760px, calc(100% - 36px)); }
    .workbench-grid { grid-template-columns: 1fr; }
    .config-panel { position: static; }
}

@media (max-width: 680px) {
    .app-shell { width: calc(100% - 24px); padding-bottom: 34px; }
    .topbar { min-height: 60px; }
    .brand-subtitle { display: none; }
    .workbench-grid { margin-top: 14px; }
    .workflow-panel, .config-panel, .output-panel { padding: 18px; }
    .drop-zone { grid-template-columns: 1fr; gap: 12px; padding: 16px; }
    .browse-action { justify-self: start; }
    .property-entry { grid-template-columns: 1fr; }
    .secondary-btn { width: 100%; }
    .metrics-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
    .export-row, .footer { align-items: flex-start; flex-direction: column; }
    .export-actions { width: 100%; justify-content: flex-start; }
    .footer-note { text-align: left; }
}

@media (prefers-reduced-motion: reduce) {
    *, *::before, *::after { scroll-behavior: auto !important; animation-duration: .01ms !important; animation-iteration-count: 1 !important; transition-duration: .01ms !important; }
}
"""
