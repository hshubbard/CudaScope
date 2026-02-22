"""
gui.py
======
Tabbed GUI for the CUDA Kernel Analyzer.

Tabs
----
  1. Kernels  – register / remove user kernels
  2. Run      – run the analysis pipeline with live log output
  3. Settings – edit classifier thresholds (written to analyzer_config.json)

Run with:
    python gui.py
"""

import json
import os
import queue
import subprocess
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, scrolledtext, ttk

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

        self.tab_kernels  = KernelsTab(notebook)
        self.tab_run      = RunTab(notebook)
        self.tab_settings = SettingsTab(notebook)

        notebook.add(self.tab_kernels,  text="  Kernels  ")
        notebook.add(self.tab_run,      text="  Run & Results  ")
        notebook.add(self.tab_settings, text="  Settings  ")

        # Cross-tab refresh: when kernels change, update Run tab label
        self.tab_kernels.on_change = self.tab_run.refresh_kernel_count


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
# Tab 3 — Settings
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
