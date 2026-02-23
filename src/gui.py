"""
gui.py
======
Tabbed GUI for the CUDA Kernel Analyzer.

Tabs
----
  1. Dashboard  – interactive kernel scorecard (scores, hotspots, detail panel)
  2. Kernels    – register / remove user kernels
  3. Run        – run the analysis pipeline with live log output
  4. Settings   – edit classifier thresholds (written to analyzer_config.json)

Run with:
    python gui.py
"""

import json
import math
import os
import queue
import subprocess
import sys
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, scrolledtext, ttk

# Ensure src/ is on the path when launched as "python src/gui.py" from repo root
sys.path.insert(0, str(Path(__file__).parent))

import kernel_manager as km

CONFIG_FILE = km.BASE_DIR / "analyzer_config.json"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_config() -> dict:
    defaults = {
        "thresh_memory_ratio":    0.30,
        "thresh_branch_ratio":    0.10,
        "thresh_arith_intensity": 3.0,
        "thresh_bw_efficiency":   0.20,
    }
    if CONFIG_FILE.exists():
        with open(CONFIG_FILE, "r") as f:
            data = json.load(f)
        defaults.update(data)
    return defaults


def _save_config(cfg: dict):
    with open(CONFIG_FILE, "w") as f:
        json.dump(cfg, f, indent=2)


# ---------------------------------------------------------------------------
# Main application window
# ---------------------------------------------------------------------------

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("CUDA Kernel Analyzer")
        self.geometry("1000x680")
        self.resizable(True, True)

        notebook = ttk.Notebook(self)
        notebook.pack(fill=tk.BOTH, expand=True, padx=6, pady=6)

        self.tab_dashboard = DashboardTab(notebook)
        self.tab_kernels   = KernelsTab(notebook)
        self.tab_run       = RunTab(notebook)
        self.tab_settings  = SettingsTab(notebook)

        notebook.add(self.tab_dashboard, text="  Dashboard  ")
        notebook.add(self.tab_kernels,   text="  Kernels  ")
        notebook.add(self.tab_run,       text="  Run & Results  ")
        notebook.add(self.tab_settings,  text="  Settings  ")

        # Cross-tab refresh: when kernels change, update Run tab label
        self.tab_kernels.on_change = self.tab_run.refresh_kernel_count

        # When the user switches to Dashboard, auto-refresh if data exists
        notebook.bind("<<NotebookTabChanged>>",
                      lambda e: self._on_tab_change(notebook))

    def _on_tab_change(self, notebook):
        tab = notebook.tab(notebook.select(), "text").strip()
        if tab == "Dashboard":
            self.tab_dashboard.refresh()


# ---------------------------------------------------------------------------
# Tab 1 — Kernels
# ---------------------------------------------------------------------------

class KernelsTab(ttk.Frame):
    def __init__(self, parent):
        super().__init__(parent)
        self.on_change = None   # callback set by App

        # ---- Left: built-in list + user list ----
        left = ttk.Frame(self)
        left.pack(side=tk.LEFT, fill=tk.BOTH, padx=(8, 4), pady=8, expand=False)
        left.config(width=280)

        ttk.Label(left, text="Built-in kernels", font=("", 10, "bold")).pack(anchor="w")
        builtin_frame = ttk.Frame(left, relief=tk.SUNKEN, borderwidth=1)
        builtin_frame.pack(fill=tk.X, pady=(2, 8))
        for k in km.BUILTINS:
            ttk.Label(builtin_frame, text=f"  {k['name']}", foreground="#555").pack(anchor="w")

        ttk.Label(left, text="Active kernels", font=("", 10, "bold")).pack(anchor="w")
        self.active_list = tk.Listbox(left, height=6, selectmode=tk.SINGLE,
                                      font=("Consolas", 9), bg="#efffef")
        self.active_list.pack(fill=tk.BOTH, expand=True)
        self.active_list.bind("<<ListboxSelect>>", lambda e: self._on_select(e, self.active_list))

        act_btn_row = ttk.Frame(left)
        act_btn_row.pack(fill=tk.X, pady=(2, 6))
        ttk.Button(act_btn_row, text="Deactivate",
                   command=self._deactivate_selected).pack(side=tk.LEFT)
        ttk.Button(act_btn_row, text="Remove",
                   command=self._remove_selected).pack(side=tk.LEFT, padx=(6, 0))

        ttk.Label(left, text="Inactive kernels", font=("", 10, "bold")).pack(anchor="w")
        self.inactive_list = tk.Listbox(left, height=6, selectmode=tk.SINGLE,
                                        font=("Consolas", 9), bg="#fff8ee")
        self.inactive_list.pack(fill=tk.BOTH, expand=True)
        self.inactive_list.bind("<<ListboxSelect>>", lambda e: self._on_select(e, self.inactive_list))

        inact_btn_row = ttk.Frame(left)
        inact_btn_row.pack(fill=tk.X, pady=(2, 0))
        ttk.Button(inact_btn_row, text="Activate",
                   command=self._activate_selected).pack(side=tk.LEFT)
        ttk.Button(inact_btn_row, text="Remove",
                   command=self._remove_selected).pack(side=tk.LEFT, padx=(6, 0))

        # ---- Right: add form ----
        right = ttk.LabelFrame(self, text="Add new kernel")
        right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(4, 8), pady=8)

        # Form fields
        form = ttk.Frame(right)
        form.pack(fill=tk.X, padx=8, pady=6)

        def _row(label, col=1, width=20):
            r = form.grid_size()[1]
            ttk.Label(form, text=label).grid(row=r, column=0, sticky="e", padx=(0, 6), pady=3)
            var = tk.StringVar()
            e = ttk.Entry(form, textvariable=var, width=width)
            e.grid(row=r, column=col, sticky="ew", pady=3)
            return var

        form.columnconfigure(1, weight=1)

        self.v_name    = _row("Kernel name:")
        self.v_block   = _row("Block size:")
        self.v_block.set("256")
        self.v_n       = _row("N elements:")
        self.v_n.set("16777216")
        self.v_iters   = _row("Iters (0=memory):")
        self.v_iters.set("0")
        self.v_grid_x  = _row("Grid X (blank=1D auto):")
        self.v_grid_y  = _row("Grid Y (blank=1D auto):")
        self.v_block_x = _row("Block X (2D, def 32):")
        self.v_block_y = _row("Block Y (2D, def 32):")
        self.v_params  = _row("Params (auto):", width=40)

        # File picker
        file_row_idx = form.grid_size()[1]
        ttk.Label(form, text="Source file:").grid(row=file_row_idx, column=0, sticky="e",
                                                   padx=(0, 6), pady=3)
        file_inner = ttk.Frame(form)
        file_inner.grid(row=file_row_idx, column=1, sticky="ew", pady=3)
        file_inner.columnconfigure(0, weight=1)
        self.v_file = tk.StringVar()
        ttk.Entry(file_inner, textvariable=self.v_file).grid(row=0, column=0, sticky="ew")
        ttk.Button(file_inner, text="Browse…", width=8,
                   command=self._browse).grid(row=0, column=1, padx=(4, 0))

        # Code editor
        ttk.Label(right, text="Kernel source (paste or load via Browse):").pack(
            anchor="w", padx=8)
        self.code_text = scrolledtext.ScrolledText(right, height=16,
                                                    font=("Consolas", 9),
                                                    wrap=tk.NONE)
        self.code_text.pack(fill=tk.BOTH, expand=True, padx=8, pady=(0, 4))

        # Bottom buttons
        btn_bar = ttk.Frame(right)
        btn_bar.pack(fill=tk.X, padx=8, pady=(0, 8))
        ttk.Button(btn_bar, text="Add Kernel",
                   command=self._add_kernel).pack(side=tk.LEFT)
        ttk.Button(btn_bar, text="Clear form",
                   command=self._clear_form).pack(side=tk.LEFT, padx=(8, 0))
        self.status_lbl = ttk.Label(btn_bar, text="", foreground="green")
        self.status_lbl.pack(side=tk.LEFT, padx=(12, 0))

        self._refresh_list()

    # ---- internals ----

    def _refresh_list(self):
        self.active_list.delete(0, tk.END)
        self.inactive_list.delete(0, tk.END)
        for k in km.load_registry():
            if k.get("active", True):
                self.active_list.insert(tk.END, k["name"])
            else:
                self.inactive_list.insert(tk.END, k["name"])

    def _on_select(self, _event, listbox):
        sel = listbox.curselection()
        if not sel:
            return
        name = listbox.get(sel[0])
        # Deselect the other list
        other = self.inactive_list if listbox is self.active_list else self.active_list
        other.selection_clear(0, tk.END)
        # Find the kernel in the registry and populate the form
        for k in km.load_registry():
            if k["name"] == name:
                self.v_name.set(k["name"])
                self.v_block.set(str(k.get("block_size", 256)))
                self.v_n.set(str(k.get("n_elements", 16777216)))
                self.v_iters.set(str(k.get("iters", 0)))
                self.v_grid_x.set(str(k["grid_x"]) if k.get("grid_x") is not None else "")
                self.v_grid_y.set(str(k["grid_y"]) if k.get("grid_y") is not None else "")
                self.v_block_x.set(str(k["block_x"]) if k.get("block_x") is not None else "")
                self.v_block_y.set(str(k["block_y"]) if k.get("block_y") is not None else "")
                self.v_params.set(k.get("params", ""))
                self.code_text.delete("1.0", tk.END)
                self.code_text.insert("1.0", k.get("code", ""))
                break

    def _selected_name(self):
        """Return the name of whichever list has a selection, or None."""
        for lb in (self.active_list, self.inactive_list):
            sel = lb.curselection()
            if sel:
                return lb.get(sel[0])
        return None

    def _deactivate_selected(self):
        name = self._selected_name()
        if not name:
            messagebox.showinfo("Nothing selected", "Select an active kernel to deactivate.")
            return
        err = km.set_kernel_active(name, False)
        if err:
            messagebox.showerror("Error", err)
            return
        self.status_lbl.config(text=f"'{name}' deactivated.", foreground="orange")
        self._refresh_list()
        if self.on_change:
            self.on_change()

    def _activate_selected(self):
        name = self._selected_name()
        if not name:
            messagebox.showinfo("Nothing selected", "Select an inactive kernel to activate.")
            return
        err = km.set_kernel_active(name, True)
        if err:
            messagebox.showerror("Error", err)
            return
        self.status_lbl.config(text=f"'{name}' activated.", foreground="green")
        self._refresh_list()
        if self.on_change:
            self.on_change()

    def _browse(self):
        path = filedialog.askopenfilename(
            title="Select kernel .cu file",
            filetypes=[("CUDA source", "*.cu"), ("All files", "*.*")],
        )
        if path:
            self.v_file.set(path)
            try:
                code = Path(path).read_text(encoding="utf-8", errors="replace")
                self.code_text.delete("1.0", tk.END)
                self.code_text.insert("1.0", code)
                # Auto-fill name from filename if blank
                if not self.v_name.get().strip():
                    self.v_name.set(Path(path).stem)
                # Auto-extract params
                name = self.v_name.get().strip()
                if name:
                    params = km.extract_params(code, name)
                    if params:
                        self.v_params.set(params)
            except Exception as exc:
                messagebox.showerror("Read error", str(exc))

    def _add_kernel(self):
        name   = self.v_name.get().strip()
        code   = self.code_text.get("1.0", tk.END).strip()
        params = self.v_params.get().strip()

        if not name:
            messagebox.showerror("Missing", "Kernel name is required.")
            return
        if not code:
            messagebox.showerror("Missing", "Kernel source code is required.")
            return

        # Auto-extract params if not provided
        if not params:
            params = km.extract_params(code, name)
            if not params:
                messagebox.showerror(
                    "Param extraction failed",
                    f"Could not find '__global__ void {name}(...)' in the code.\n"
                    "Please fill in the Params field manually."
                )
                return
            self.v_params.set(params)

        try:
            block = int(self.v_block.get())
            iters = int(self.v_iters.get())
            n_raw = self.v_n.get().strip()
            if n_raw.upper().endswith("M"):
                n_elem = int(float(n_raw[:-1]) * 1024 * 1024)
            elif n_raw.upper().endswith("K"):
                n_elem = int(float(n_raw[:-1]) * 1024)
            else:
                n_elem = int(n_raw)

            def _opt_int(var, label):
                raw = var.get().strip()
                if not raw:
                    return None
                v = int(raw)
                if v <= 0:
                    raise ValueError(f"{label} must be a positive integer.")
                return v

            grid_x  = _opt_int(self.v_grid_x,  "Grid X")
            grid_y  = _opt_int(self.v_grid_y,   "Grid Y")
            block_x = _opt_int(self.v_block_x,  "Block X")
            block_y = _opt_int(self.v_block_y,  "Block Y")
        except ValueError as exc:
            messagebox.showerror("Invalid value", str(exc))
            return

        if (grid_x is None) != (grid_y is None):
            messagebox.showerror(
                "Invalid 2D config",
                "Grid X and Grid Y must both be filled in, or both left blank."
            )
            return

        err = km.add_kernel(name, code, params, block, n_elem, iters,
                            grid_x=grid_x, grid_y=grid_y,
                            block_x=block_x, block_y=block_y)
        if err:
            messagebox.showerror("Error", err)
            return

        self.status_lbl.config(text=f"'{name}' added.", foreground="green")
        self._refresh_list()
        if self.on_change:
            self.on_change()

    def _remove_selected(self):
        name = self._selected_name()
        if not name:
            messagebox.showinfo("Nothing selected", "Select a kernel to remove.")
            return
        if not messagebox.askyesno("Confirm", f"Remove kernel '{name}'?"):
            return
        err = km.remove_kernel(name)
        if err:
            messagebox.showerror("Error", err)
            return
        self.status_lbl.config(text=f"'{name}' removed.", foreground="green")
        self._refresh_list()
        if self.on_change:
            self.on_change()

    def _clear_form(self):
        self.v_name.set("")
        self.v_file.set("")
        self.v_params.set("")
        self.v_block.set("256")
        self.v_n.set("16777216")
        self.v_iters.set("0")
        self.v_grid_x.set("")
        self.v_grid_y.set("")
        self.v_block_x.set("")
        self.v_block_y.set("")
        self.code_text.delete("1.0", tk.END)
        self.status_lbl.config(text="")


# ---------------------------------------------------------------------------
# Tab 2 — Run & Results
# ---------------------------------------------------------------------------

class RunTab(ttk.Frame):
    def __init__(self, parent):
        super().__init__(parent)
        self._log_queue: queue.Queue = queue.Queue()
        self._running = False

        # Top controls
        top = ttk.Frame(self)
        top.pack(fill=tk.X, padx=8, pady=8)

        self.kernel_count_lbl = ttk.Label(top, text="")
        self.kernel_count_lbl.pack(side=tk.LEFT)
        self.refresh_kernel_count()

        # Kernel scope
        scope = ttk.LabelFrame(self, text="Kernel scope")
        scope.pack(fill=tk.X, padx=8, pady=(0, 4))
        self.v_include_builtins = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            scope,
            text="Include built-in kernels  (coalesced_add, strided_add, divergent_add, divergent_compute, compute_ref)",
            variable=self.v_include_builtins,
        ).pack(side=tk.LEFT, padx=8, pady=4)

        # Skip checkboxes
        opts = ttk.LabelFrame(self, text="Pipeline steps")
        opts.pack(fill=tk.X, padx=8, pady=(0, 6))
        self.v_skip_build   = tk.BooleanVar(value=False)
        self.v_skip_bench   = tk.BooleanVar(value=False)
        self.v_skip_ptx     = tk.BooleanVar(value=False)
        self.v_skip_analyze = tk.BooleanVar(value=False)
        ttk.Checkbutton(opts, text="Skip build",   variable=self.v_skip_build).pack(side=tk.LEFT, padx=8)
        ttk.Checkbutton(opts, text="Skip bench",   variable=self.v_skip_bench).pack(side=tk.LEFT)
        ttk.Checkbutton(opts, text="Skip PTX",     variable=self.v_skip_ptx).pack(side=tk.LEFT, padx=8)
        ttk.Checkbutton(opts, text="Skip analyze", variable=self.v_skip_analyze).pack(side=tk.LEFT)

        # Buttons
        btn_row = ttk.Frame(self)
        btn_row.pack(fill=tk.X, padx=8, pady=(0, 6))
        self.run_btn = ttk.Button(btn_row, text="Run Analysis",
                                   command=self._start_run)
        self.run_btn.pack(side=tk.LEFT)
        ttk.Button(btn_row, text="Clear log",
                   command=self._clear_log).pack(side=tk.LEFT, padx=(8, 0))
        ttk.Button(btn_row, text="Open output folder",
                   command=self._open_output).pack(side=tk.LEFT, padx=(8, 0))
        self.progress = ttk.Progressbar(btn_row, mode="indeterminate", length=160)
        self.progress.pack(side=tk.LEFT, padx=(16, 0))

        # Log output
        ttk.Label(self, text="Output:").pack(anchor="w", padx=8)
        self.log = scrolledtext.ScrolledText(self, state=tk.DISABLED,
                                              font=("Consolas", 9),
                                              wrap=tk.NONE, height=30)
        self.log.pack(fill=tk.BOTH, expand=True, padx=8, pady=(0, 8))

    def refresh_kernel_count(self):
        n_user = len(km.load_registry())
        n_builtin = len(km.BUILTINS)
        self.kernel_count_lbl.config(
            text=f"Kernels: {n_builtin} built-in + {n_user} user  "
                 f"= {n_builtin + n_user} total"
        )

    def _log_write(self, msg: str):
        self.log.config(state=tk.NORMAL)
        self.log.insert(tk.END, msg + "\n")
        self.log.see(tk.END)
        self.log.config(state=tk.DISABLED)

    def _clear_log(self):
        self.log.config(state=tk.NORMAL)
        self.log.delete("1.0", tk.END)
        self.log.config(state=tk.DISABLED)

    def _open_output(self):
        out_dir = str(km.BASE_DIR / "output")
        os.makedirs(out_dir, exist_ok=True)
        if os.name == "nt":
            os.startfile(out_dir)
        else:
            subprocess.Popen(["xdg-open", out_dir])

    def _start_run(self):
        if self._running:
            return
        self._running = True
        self.run_btn.config(state=tk.DISABLED)
        self.progress.start(10)

        def worker():
            km.run_pipeline(
                skip_build       = self.v_skip_build.get(),
                skip_bench       = self.v_skip_bench.get(),
                skip_ptx         = self.v_skip_ptx.get(),
                skip_analyze     = self.v_skip_analyze.get(),
                include_builtins = self.v_include_builtins.get(),
                log_cb           = lambda msg: self._log_queue.put(msg),
            )
            self._log_queue.put(None)  # sentinel

        threading.Thread(target=worker, daemon=True).start()
        self._poll_log()

    def _poll_log(self):
        try:
            while True:
                msg = self._log_queue.get_nowait()
                if msg is None:
                    # Done
                    self._running = False
                    self.run_btn.config(state=tk.NORMAL)
                    self.progress.stop()
                    return
                self._log_write(msg)
        except queue.Empty:
            pass
        self.after(100, self._poll_log)


# ---------------------------------------------------------------------------
# Tab 1 — Dashboard
# ---------------------------------------------------------------------------

# Colours ─────────────────────────────────────────────────────────────────────
# Risk level colours: red=high, amber=medium, green=low (same as traffic lights)
_RISK_BG = {"HIGH": "#ffe0e0", "MEDIUM": "#fff7e0", "LOW": "#e6f7e6"}
_RISK_FG = {"HIGH": "#8b0000", "MEDIUM": "#5c3d00", "LOW": "#1a5c1a"}
# Bar fill colours per score band
_BAR_CLR = {"high": "#d94040", "medium": "#cc8800", "low": "#3a9e3a", "none": "#cccccc"}
# Treeview row colours
_TAG_BG  = {"high": "#ffe0e0", "medium": "#fff7e0", "low": "#e6f7e6"}
_TAG_FG  = {"high": "#8b0000", "medium": "#5c3d00", "low": "#1a5c1a"}


def _score_band(score: int) -> str:
    """Map a normalised 0-100 score to a colour band name."""
    if score >= 60:
        return "high"
    if score >= 25:
        return "medium"
    if score > 0:
        return "low"
    return "none"


# Report / CSV paths (must match summary_report.py) ──────────────────────────
_BASE        = km.BASE_DIR
_FRAG_JSON   = _BASE / "output" / "report" / "portability.json"
_DET_JSON    = _BASE / "output" / "report" / "determinism.json"
_RES_JSON    = _BASE / "output" / "report" / "resource.json"
_RUNTIME_CSV = _BASE / "output" / "data"   / "runtimes.csv"
_PTX_CSV     = _BASE / "output" / "data"   / "ptx_stats.csv"

_TABLE_COLS = ("Kernel", "Bottleneck", "Fragility", "Non-Det", "Resource", "Risk")
_TOP_N      = 3
_SEV_LABEL  = {"high": "HI", "medium": "MD", "low": "LO"}


class DashboardTab(ttk.Frame):
    """Interactive kernel scorecard with detail panel and score legend."""

    def __init__(self, parent):
        super().__init__(parent)
        self._summaries: list = []
        self._tooltip         = None

        self._build_ui()

    # ── UI construction ──────────────────────────────────────────────────────

    def _build_ui(self):
        # Use grid on self so the pane can expand freely while the legend
        # and toolbar stay fixed-height at top and bottom respectively.
        self.rowconfigure(1, weight=1)   # row 1 = pane (expands)
        self.columnconfigure(0, weight=1)

        # ── Row 0: top toolbar ────────────────────────────────────────────────
        toolbar = ttk.Frame(self)
        toolbar.grid(row=0, column=0, sticky="ew", padx=8, pady=(8, 0))

        ttk.Label(toolbar, text="Kernel Dashboard",
                  font=("", 12, "bold")).pack(side=tk.LEFT)
        self._status_lbl = ttk.Label(toolbar, text="No data loaded.",
                                      foreground="#666")
        self._status_lbl.pack(side=tk.LEFT, padx=(16, 0))
        ttk.Button(toolbar, text="Refresh",
                   command=self.refresh).pack(side=tk.RIGHT)

        # ── Row 1: vertical pane (expands to fill remaining space) ────────────
        pane = ttk.PanedWindow(self, orient=tk.VERTICAL)
        pane.grid(row=1, column=0, sticky="nsew", padx=8, pady=4)

        # ── Kernel table ──────────────────────────────────────────────────────
        table_frame = ttk.LabelFrame(pane, text="Kernel Summary Table  "
                                     "(click column header to sort)")
        pane.add(table_frame, weight=2)

        self._tree = ttk.Treeview(
            table_frame, columns=_TABLE_COLS,
            show="headings", selectmode="browse", height=8,
        )
        for risk, (bg, fg) in zip(
            ("high", "medium", "low"),
            (("#ffe0e0","#8b0000"), ("#fff7e0","#5c3d00"), ("#e6f7e6","#1a5c1a"))
        ):
            self._tree.tag_configure(risk, background=bg, foreground=fg)

        col_widths = {
            "Kernel": 165, "Bottleneck": 200,
            "Fragility": 85, "Non-Det": 85, "Resource": 85, "Risk": 75,
        }
        for col in _TABLE_COLS:
            self._tree.heading(col, text=col,
                               command=lambda c=col: self._sort_by(c))
            self._tree.column(col, width=col_widths[col],
                              anchor="center",
                              stretch=(col == "Bottleneck"))
        self._tree.column("Kernel",     anchor="w")
        self._tree.column("Bottleneck", anchor="w")

        tree_vsb = ttk.Scrollbar(table_frame, orient="vertical",
                                  command=self._tree.yview)
        self._tree.configure(yscrollcommand=tree_vsb.set)
        tree_vsb.pack(side=tk.RIGHT, fill=tk.Y)
        self._tree.pack(fill=tk.BOTH, expand=True)
        self._tree.bind("<<TreeviewSelect>>", self._on_row_select)
        self._tree.bind("<Motion>", self._on_motion)
        self._tree.bind("<Leave>",  self._hide_tooltip)

        # ── Detail panel (scrollable canvas + inner frame) ────────────────────
        detail_outer = ttk.LabelFrame(pane, text="Kernel Detail")
        pane.add(detail_outer, weight=3)

        self._detail_canvas = tk.Canvas(detail_outer, bg="#fafafa",
                                         highlightthickness=0)
        detail_vsb = ttk.Scrollbar(detail_outer, orient="vertical",
                                    command=self._detail_canvas.yview)
        self._detail_canvas.configure(yscrollcommand=detail_vsb.set)
        detail_vsb.pack(side=tk.RIGHT, fill=tk.Y)
        self._detail_canvas.pack(fill=tk.BOTH, expand=True)

        # inner frame lives inside the canvas window
        self._detail_inner = tk.Frame(self._detail_canvas, bg="#fafafa")
        self._detail_window = self._detail_canvas.create_window(
            (0, 0), window=self._detail_inner, anchor="nw"
        )
        self._detail_inner.bind("<Configure>", self._on_inner_configure)
        self._detail_canvas.bind("<Configure>", self._on_canvas_resize)

        # Mouse-wheel scroll works anywhere inside the detail area
        self._detail_canvas.bind("<Enter>",     self._bind_mousewheel)
        self._detail_canvas.bind("<Leave>",     self._unbind_mousewheel)

        tk.Label(self._detail_inner,
                 text="Select a kernel above to see details.",
                 bg="#fafafa", fg="#888").pack(anchor="w", padx=12, pady=8)

        # ── Row 2: score legend (fixed height, always visible at bottom) ──────
        self._build_legend()

    # ── Score legend ──────────────────────────────────────────────────────────

    def _build_legend(self):
        leg = ttk.LabelFrame(self, text="Score Legend", padding=(8, 5))
        leg.grid(row=2, column=0, sticky="ew", padx=8, pady=(0, 6))
        leg.columnconfigure(1, weight=1)

        # Three score rows
        score_rows = [
            ("Fragility (0–100)",
             "How likely the kernel breaks when ported to a different GPU generation. "
             "Flags warp-size assumptions, arch-specific instructions, alignment issues."),
            ("Non-Determinism (0–100)",
             "How likely the kernel produces inconsistent results. "
             "Flags data races, unsynced shared-mem reads, FP-reduction ordering."),
            ("Resource Pressure (0–100)",
             "How likely the kernel reduces occupancy or monopolizes the SM. "
             "Flags register spill, shared-mem overuse, non-power-of-2 block sizes."),
        ]
        for i, (name, desc) in enumerate(score_rows):
            tk.Label(leg, text=name, font=("", 9, "bold"),
                     anchor="w").grid(row=i, column=0, sticky="nw",
                                      padx=(0, 12), pady=1)
            tk.Label(leg, text=desc, font=("", 9), fg="#444",
                     anchor="w", justify="left").grid(
                row=i, column=1, sticky="ew", pady=1)

        # Risk / threshold row
        thresh_frame = tk.Frame(leg)
        thresh_frame.grid(row=len(score_rows), column=0, columnspan=2,
                          sticky="w", pady=(6, 0))
        tk.Label(thresh_frame, text="Risk Level:",
                 font=("", 9, "bold")).pack(side=tk.LEFT, padx=(0, 8))
        for bg, fg, label in [
            ("#ffe0e0", "#8b0000", "  HIGH  (combined ≥ 60)  "),
            ("#fff7e0", "#5c3d00", "  MEDIUM  (25 – 59)  "),
            ("#e6f7e6", "#1a5c1a", "  LOW  (1 – 24)  "),
            ("#f0f0f0", "#555555", "  None  (0)  "),
        ]:
            tk.Label(thresh_frame, text=label, bg=bg, fg=fg,
                     font=("", 8, "bold"), relief="solid", bd=1,
                     padx=2).pack(side=tk.LEFT, padx=(0, 4))

        calc_row = len(score_rows) + 1
        tk.Label(leg,
                 text="Combined = 40% Fragility + 35% Non-Det + 25% Resource.  "
                      "Scores normalised: 100 = worst kernel this run, 0 = no issues.",
                 font=("", 8), fg="#666", anchor="w").grid(
            row=calc_row, column=0, columnspan=2, sticky="w", pady=(2, 0))

    # ── Data loading ──────────────────────────────────────────────────────────

    def _load_data(self) -> list:
        import csv

        def _jload(path):
            if path.exists():
                with open(path, encoding="utf-8") as f:
                    return json.load(f)
            return {}

        frag = _jload(_FRAG_JSON)
        det  = _jload(_DET_JSON)
        res  = _jload(_RES_JSON)

        runtime: dict = {}
        if _RUNTIME_CSV.exists():
            with open(_RUNTIME_CSV, newline="", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    n = row.get("kernel", "").strip()
                    if n:
                        runtime[n] = row

        def _raw(flags):
            return sum(f.get("score", 0) for f in flags)

        def _norm(pass_data):
            raws = {k: _raw(v) for k, v in pass_data.items()}
            mx   = max(raws.values(), default=1) or 1
            return {k: min(100, round(100 * r / mx)) for k, r in raws.items()}

        frag_n = _norm(frag)
        det_n  = _norm(det)
        res_n  = _norm(res)

        all_kernels = (set(frag) | set(det) | set(res) | set(runtime)) - {"kernels"}

        def _top(flags, n=_TOP_N):
            seen, out = set(), []
            for f in sorted(flags, key=lambda x: -x.get("score", 0)):
                key = f.get("description", "")[:40]
                if key not in seen:
                    seen.add(key)
                    out.append(f)
                if len(out) >= n:
                    break
            return out

        def _sf(v):
            try:
                return float(v)
            except (TypeError, ValueError):
                return float("nan")

        summaries = []
        for name in sorted(all_kernels):
            rt  = runtime.get(name, {})
            f_s = frag_n.get(name, 0)
            d_s = det_n.get(name,  0)
            r_s = res_n.get(name,  0)
            combined = round(0.40 * f_s + 0.35 * d_s + 0.25 * r_s)
            summaries.append({
                "name":        name,
                "bottleneck":  rt.get("bottleneck", "—"),
                "mean_us":     _sf(rt.get("mean_us")),
                "bw_GBs":      _sf(rt.get("approx_bandwidth_GBs")),
                "fragility":   f_s,
                "determinism": d_s,
                "resource":    r_s,
                "combined":    combined,
                "risk":        "HIGH"   if combined >= 60 else
                               "MEDIUM" if combined >= 25 else "LOW",
                "top_frag":    _top(frag.get(name, [])),
                "top_det":     _top(det.get(name,  [])),
                "top_res":     _top(res.get(name,  [])),
            })

        summaries.sort(key=lambda s: -s["combined"])
        return summaries

    # ── Refresh ───────────────────────────────────────────────────────────────

    def refresh(self):
        try:
            self._summaries = self._load_data()
        except Exception as exc:
            self._status_lbl.config(
                text=f"Error loading data: {exc}", foreground="red")
            return

        if not self._summaries:
            self._status_lbl.config(
                text="No analysis output found — run the pipeline first.",
                foreground="#888")
            self._tree.delete(*self._tree.get_children())
            self._clear_detail()
            return

        self._status_lbl.config(
            text=f"Loaded {len(self._summaries)} kernel(s).",
            foreground="#226622")
        self._populate_table(self._summaries)
        self._clear_detail()

    # ── Table ─────────────────────────────────────────────────────────────────

    def _populate_table(self, summaries):
        self._tree.delete(*self._tree.get_children())
        for s in summaries:
            tag = s["risk"].lower()
            self._tree.insert("", tk.END, iid=s["name"],
                values=(
                    s["name"], s["bottleneck"],
                    f"{s['fragility']}/100",
                    f"{s['determinism']}/100",
                    f"{s['resource']}/100",
                    s["risk"],
                ),
                tags=(tag,))

    _sort_state: dict = {}

    def _sort_by(self, col):
        asc = not self._sort_state.get(col, False)
        self._sort_state[col] = asc
        col_map = {
            "Kernel":     lambda s: s["name"],
            "Bottleneck": lambda s: s["bottleneck"],
            "Fragility":  lambda s: s["fragility"],
            "Non-Det":    lambda s: s["determinism"],
            "Resource":   lambda s: s["resource"],
            "Risk":       lambda s: {"HIGH": 2, "MEDIUM": 1, "LOW": 0}[s["risk"]],
        }
        self._summaries.sort(key=col_map.get(col, lambda s: s["name"]),
                             reverse=not asc)
        self._populate_table(self._summaries)

    # ── Row selection → detail panel ─────────────────────────────────────────

    def _on_row_select(self, _event):
        sel = self._tree.selection()
        if not sel:
            return
        s = next((x for x in self._summaries if x["name"] == sel[0]), None)
        if s:
            self._show_detail(s)

    # ── Detail panel ──────────────────────────────────────────────────────────

    def _clear_detail(self):
        for w in self._detail_inner.winfo_children():
            w.destroy()
        tk.Label(self._detail_inner,
                 text="Select a kernel above to see details.",
                 bg="#fafafa", fg="#888").pack(anchor="w", padx=12, pady=8)

    def _show_detail(self, s: dict):
        for w in self._detail_inner.winfo_children():
            w.destroy()

        risk_bg = _RISK_BG.get(s["risk"], "#f0f0f0")
        risk_fg = _RISK_FG.get(s["risk"], "#000")

        # ── Summary card ──────────────────────────────────────────────────────
        card = tk.Frame(self._detail_inner, bg=risk_bg, relief="solid", bd=1)
        card.pack(fill=tk.X, padx=0, pady=(0, 6))

        # Name + risk badge
        title_row = tk.Frame(card, bg=risk_bg)
        title_row.pack(fill=tk.X, padx=10, pady=(8, 2))
        tk.Label(title_row, text=s["name"],
                 font=("Consolas", 11, "bold"),
                 bg=risk_bg, fg=risk_fg).pack(side=tk.LEFT)
        tk.Label(title_row,
                 text=f"   [{s['risk']}]   combined score: {s['combined']}/100",
                 font=("", 10, "bold"), bg=risk_bg, fg=risk_fg).pack(side=tk.LEFT)

        # Metrics line
        def _fmt(v, unit):
            return f"{v:.1f} {unit}" if not math.isnan(v) else "n/a"

        tk.Label(card,
                 text=(f"Bottleneck: {s['bottleneck']}     "
                       f"Runtime: {_fmt(s['mean_us'], 'us')}     "
                       f"Bandwidth: {_fmt(s['bw_GBs'], 'GB/s')}"),
                 font=("", 9), bg=risk_bg, fg=risk_fg,
                 anchor="w").pack(fill=tk.X, padx=10, pady=(0, 4))

        # Score bars — use a grid so columns stay aligned
        bars_frame = tk.Frame(card, bg=risk_bg)
        bars_frame.pack(fill=tk.X, padx=10, pady=(2, 8))
        bars_frame.columnconfigure(2, weight=1)

        BAR_W = 260
        for row_idx, (lbl, score) in enumerate([
            ("Fragility",        s["fragility"]),
            ("Non-Determinism",  s["determinism"]),
            ("Resource Pressure", s["resource"]),
        ]):
            band   = _score_band(score)
            bar_fg = _BAR_CLR[band]
            tk.Label(bars_frame, text=lbl, font=("", 9), bg=risk_bg,
                     fg="#333", anchor="w", width=18).grid(
                row=row_idx, column=0, sticky="w", padx=(0, 6), pady=2)
            tk.Label(bars_frame, text=f"{score:>3}/100",
                     font=("Consolas", 9, "bold"), bg=risk_bg,
                     fg=bar_fg, width=7, anchor="e").grid(
                row=row_idx, column=1, sticky="e", padx=(0, 8), pady=2)
            bar_cvs = tk.Canvas(bars_frame, width=BAR_W, height=13,
                                bg="#e0e0e0", highlightthickness=0)
            bar_cvs.grid(row=row_idx, column=2, sticky="ew", pady=2)
            if score > 0:
                # Draw filled region
                bar_cvs.create_rectangle(
                    0, 0, int(BAR_W * score / 100), 13,
                    fill=bar_fg, outline="")
                # Threshold markers at 25 and 60
                for thresh, clr in ((25, "#cc8800"), (60, "#d94040")):
                    x = int(BAR_W * thresh / 100)
                    bar_cvs.create_line(x, 0, x, 13, fill=clr,
                                        width=1, dash=(3, 2))

        # ── Next action ───────────────────────────────────────────────────────
        action   = self._action(s["bottleneck"], s["risk"])
        act_bg   = "#f5f5f5"
        act_frame = tk.Frame(self._detail_inner, bg=act_bg)
        act_frame.pack(fill=tk.X, padx=0, pady=(0, 6))
        act_frame.columnconfigure(1, weight=1)
        tk.Label(act_frame, text="Next action:", font=("", 9, "bold"),
                 bg=act_bg, fg="#333", anchor="w").grid(
            row=0, column=0, sticky="w", padx=(10, 6), pady=6)
        tk.Label(act_frame, text=action, font=("", 9),
                 bg=act_bg, fg="#114411", anchor="w",
                 justify="left", wraplength=700).grid(
            row=0, column=1, sticky="ew", padx=(0, 10), pady=6)

        # ── Per-pass flag sections ─────────────────────────────────────────────
        for pass_label, flags, hdr_clr in [
            ("Fragility",        s["top_frag"],  "#c0603a"),
            ("Non-Determinism",  s["top_det"],   "#3a6aaa"),
            ("Resource Pressure", s["top_res"],  "#3a8a3a"),
        ]:
            self._flags_section(pass_label, flags, hdr_clr)

    def _flags_section(self, label: str, flags: list, hdr_clr: str):
        if not flags:
            return

        # Coloured header bar
        hdr = tk.Frame(self._detail_inner, bg=hdr_clr)
        hdr.pack(fill=tk.X, pady=(6, 0))
        tk.Label(hdr, text=f"  {label}  —  top {len(flags)} flag(s)",
                 font=("", 9, "bold"), bg=hdr_clr, fg="white",
                 padx=6, pady=3).pack(anchor="w")

        # Use a grid-based layout inside a container so columns stay aligned
        container = tk.Frame(self._detail_inner, bg="#fafafa")
        container.pack(fill=tk.X)
        container.columnconfigure(2, weight=1)   # description column expands

        for i, f in enumerate(flags):
            row_bg = "#f8f8f8" if i % 2 == 0 else "#efefef"
            sev    = _SEV_LABEL.get(f.get("severity", "").lower(), "??")
            cat    = f.get("category", "unknown")
            loc    = f.get("location", "")
            desc   = f.get("description", "")
            sev_fg = {"HI": "#bb2222", "MD": "#995500",
                      "LO": "#228822"}.get(sev, "#555")

            # Severity badge
            tk.Label(container,
                     text=f" [{sev}] ",
                     font=("Consolas", 8, "bold"),
                     bg=row_bg, fg=sev_fg, anchor="center").grid(
                row=i, column=0, sticky="nsew", padx=(8, 2), pady=3)

            # Category + location stacked
            meta = tk.Frame(container, bg=row_bg)
            meta.grid(row=i, column=1, sticky="nsw", padx=(2, 8), pady=3)
            tk.Label(meta, text=cat,
                     font=("Consolas", 8, "bold"),
                     bg=row_bg, fg=sev_fg, anchor="w",
                     justify="left", wraplength=200).pack(anchor="w")
            if loc:
                tk.Label(meta, text=loc,
                         font=("Consolas", 8), bg=row_bg,
                         fg="#777", anchor="w").pack(anchor="w")

            # Full description — wraps naturally, no fixed width needed
            tk.Label(container, text=desc,
                     font=("", 9), bg=row_bg, fg="#222",
                     anchor="w", justify="left", wraplength=550).grid(
                row=i, column=2, sticky="ew", padx=(0, 10), pady=3)

            # Row background spans all columns
            for col in (0, 1, 2):
                container.grid_columnconfigure(col, minsize=0)

    # ── Canvas / scroll helpers ───────────────────────────────────────────────

    def _on_inner_configure(self, _event):
        self._detail_canvas.configure(
            scrollregion=self._detail_canvas.bbox("all"))

    def _on_canvas_resize(self, event):
        self._detail_canvas.itemconfig(self._detail_window, width=event.width)

    def _bind_mousewheel(self, _event):
        self._detail_canvas.bind_all("<MouseWheel>",   self._on_mousewheel)
        self._detail_canvas.bind_all("<Button-4>",     self._on_mousewheel)
        self._detail_canvas.bind_all("<Button-5>",     self._on_mousewheel)

    def _unbind_mousewheel(self, _event):
        self._detail_canvas.unbind_all("<MouseWheel>")
        self._detail_canvas.unbind_all("<Button-4>")
        self._detail_canvas.unbind_all("<Button-5>")

    def _on_mousewheel(self, event):
        if event.num == 4:
            self._detail_canvas.yview_scroll(-1, "units")
        elif event.num == 5:
            self._detail_canvas.yview_scroll(1, "units")
        else:
            self._detail_canvas.yview_scroll(
                int(-1 * (event.delta / 120)), "units")

    # ── Tooltip ───────────────────────────────────────────────────────────────

    def _on_motion(self, event):
        item = self._tree.identify_row(event.y)
        if not item:
            self._hide_tooltip()
            return
        s = next((x for x in self._summaries if x["name"] == item), None)
        if not s:
            self._hide_tooltip()
            return
        lines = [f"Kernel: {s['name']}  [{s['risk']}]  combined: {s['combined']}/100"]
        for lbl, flags in [("Frag", s["top_frag"]),
                            ("Det",  s["top_det"]),
                            ("Res",  s["top_res"])]:
            for f in flags[:2]:
                sev = _SEV_LABEL.get(f.get("severity", "").lower(), "??")
                cat = f.get("category", "")[:32]
                lines.append(f"  [{lbl}/{sev}] {cat}")
        if len(lines) == 1:
            lines.append("  (no flags)")
        self._show_tooltip("\n".join(lines), event)

    def _show_tooltip(self, text: str, event):
        self._hide_tooltip()
        tip = tk.Toplevel(self)
        tip.wm_overrideredirect(True)
        tip.wm_geometry(f"+{event.x_root + 16}+{event.y_root + 10}")
        tk.Label(tip, text=text, justify=tk.LEFT, font=("Consolas", 8),
                 bg="#ffffcc", fg="#222", relief="solid", bd=1,
                 padx=6, pady=4).pack()
        self._tooltip = tip

    def _hide_tooltip(self, _event=None):
        if self._tooltip:
            try:
                self._tooltip.destroy()
            except Exception:
                pass
            self._tooltip = None

    # ── Action helper ─────────────────────────────────────────────────────────

    @staticmethod
    def _action(bottleneck: str, risk: str) -> str:
        prefix = {"HIGH": "URGENT: ", "MEDIUM": "Consider: ", "LOW": ""}.get(risk, "")
        actions = {
            "Warp Divergence":
                "Eliminate per-thread branch with branchless arithmetic.",
            "Memory Bound (Poor Coalescing)":
                "Restructure access pattern for coalesced reads (SoA layout).",
            "Memory Bound":
                "Tile into shared memory to reduce global traffic.",
            "Shared Memory Bound":
                "Tune block size or replace smem reduction with warp shuffles.",
            "Bandwidth Limited":
                "Fuse kernels or use half-precision to cut bandwidth demand.",
            "Compute Balanced":
                "Profile register pressure / occupancy in Nsight Compute.",
            "Compute Bound (Reference)":
                "Reference kernel — compare against divergent variant.",
        }
        return prefix + actions.get(bottleneck, "Inspect flags below.")


# ---------------------------------------------------------------------------
# Tab 4 — Settings
# ---------------------------------------------------------------------------

class SettingsTab(ttk.Frame):
    def __init__(self, parent):
        super().__init__(parent)

        ttk.Label(self,
                  text="Classifier thresholds — saved to analyzer_config.json\n"
                       "and loaded automatically by analyze.py on next run.",
                  justify=tk.LEFT).pack(anchor="w", padx=12, pady=(12, 4))

        form = ttk.LabelFrame(self, text="Thresholds")
        form.pack(fill=tk.X, padx=12, pady=8)
        form.columnconfigure(1, weight=0)
        form.columnconfigure(2, weight=1)

        cfg = _load_config()

        def _thresh_row(label, key, description, row):
            ttk.Label(form, text=label).grid(row=row, column=0, sticky="e",
                                              padx=(8, 6), pady=6)
            var = tk.StringVar(value=str(cfg.get(key, "")))
            ttk.Entry(form, textvariable=var, width=10).grid(row=row, column=1,
                                                              sticky="w", pady=6)
            ttk.Label(form, text=description, foreground="#555").grid(
                row=row, column=2, sticky="w", padx=(8, 8))
            return var, key

        self._fields = [
            _thresh_row("Memory ratio >",    "thresh_memory_ratio",
                        ">N% global mem ops → Memory Bound", 0),
            _thresh_row("Branch ratio >",    "thresh_branch_ratio",
                        ">N% branch instructions → Warp Divergence", 1),
            _thresh_row("Arith intensity <", "thresh_arith_intensity",
                        "<N arith/mem → bandwidth-limited (not compute)", 2),
            _thresh_row("BW efficiency <",   "thresh_bw_efficiency",
                        "<N% of peak BW → Poor Coalescing", 3),
        ]

        btn_row = ttk.Frame(self)
        btn_row.pack(fill=tk.X, padx=12, pady=4)
        ttk.Button(btn_row, text="Save settings",
                   command=self._save).pack(side=tk.LEFT)
        ttk.Button(btn_row, text="Reset to defaults",
                   command=self._reset).pack(side=tk.LEFT, padx=(8, 0))
        self.status = ttk.Label(btn_row, text="", foreground="green")
        self.status.pack(side=tk.LEFT, padx=(12, 0))

        ttk.Separator(self, orient=tk.HORIZONTAL).pack(fill=tk.X, padx=12, pady=12)
        ttk.Label(self,
                  text="Bottleneck labels and their meanings:\n\n"
                       "  Bandwidth Limited        — low arithmetic intensity, near-peak BW\n"
                       "  Memory Bound             — high global mem ratio, low intensity\n"
                       "  Memory Bound (Poor Coalescing) — low BW efficiency + low intensity\n"
                       "  Warp Divergence          — elevated branch ratio\n"
                       "  Compute Bound (Reference)— hardcoded for compute_ref kernel\n"
                       "  Compute Balanced         — no dominant bottleneck",
                  justify=tk.LEFT,
                  foreground="#444").pack(anchor="w", padx=12)

    def _save(self):
        cfg = {}
        for var, key in self._fields:
            try:
                cfg[key] = float(var.get())
            except ValueError:
                messagebox.showerror("Invalid value",
                                     f"'{var.get()}' is not a valid number for '{key}'.")
                return
        _save_config(cfg)
        self.status.config(text="Saved.", foreground="green")

    def _reset(self):
        defaults = {
            "thresh_memory_ratio":    0.30,
            "thresh_branch_ratio":    0.10,
            "thresh_arith_intensity": 3.0,
            "thresh_bw_efficiency":   0.20,
        }
        for var, key in self._fields:
            var.set(str(defaults[key]))
        self.status.config(text="Reset to defaults (not yet saved).", foreground="orange")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    app = App()
    app.mainloop()
