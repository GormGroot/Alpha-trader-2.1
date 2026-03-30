#!/usr/bin/env python3
"""Generate PDF report for the GPU crash investigation and fix (March 28, 2026)."""

from fpdf import FPDF
from datetime import datetime


class BugfixPDF(FPDF):
    def header(self):
        self.set_font("Helvetica", "B", 11)
        self.set_text_color(180, 40, 40)
        self.cell(0, 8, "Bug Fix Report - GPU Crash Investigation", align="L")
        self.set_font("Helvetica", "", 8)
        self.set_text_color(100, 100, 100)
        self.cell(0, 8, "March 28, 2026", align="R", new_x="LMARGIN", new_y="NEXT")
        self.set_draw_color(180, 40, 40)
        self.line(10, self.get_y(), 200, self.get_y())
        self.ln(4)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(128, 128, 128)
        self.cell(0, 10, f"Page {self.page_no()}/{{nb}}", align="C")

    def section(self, title):
        self.set_font("Helvetica", "B", 13)
        self.set_text_color(30, 30, 40)
        self.ln(3)
        self.cell(0, 8, title, new_x="LMARGIN", new_y="NEXT")
        self.set_draw_color(180, 40, 40)
        self.line(10, self.get_y(), 200, self.get_y())
        self.ln(3)

    def sub(self, title):
        self.set_font("Helvetica", "B", 10)
        self.set_text_color(50, 50, 70)
        self.ln(2)
        self.cell(0, 6, title, new_x="LMARGIN", new_y="NEXT")
        self.ln(1)

    def body(self, text):
        self.set_font("Helvetica", "", 9)
        self.set_text_color(40, 40, 40)
        self.set_x(self.l_margin)
        self.multi_cell(0, 4.5, text)
        self.ln(1)

    def mono(self, text):
        self.set_font("Courier", "", 8)
        self.set_text_color(0, 80, 60)
        self.set_x(self.l_margin)
        self.multi_cell(0, 4, text)
        self.ln(1)

    def bullet(self, text):
        self.set_font("Helvetica", "", 9)
        self.set_text_color(40, 40, 40)
        self.set_x(self.l_margin)
        self.multi_cell(0, 4.5, "   - " + text)

    def label_value(self, label, value):
        self.set_font("Helvetica", "B", 9)
        self.set_text_color(80, 80, 80)
        self.cell(55, 5, label)
        self.set_font("Helvetica", "", 9)
        self.set_text_color(40, 40, 40)
        self.cell(0, 5, value, new_x="LMARGIN", new_y="NEXT")

    def timeline_entry(self, time, event, severity="normal"):
        if severity == "critical":
            self.set_text_color(180, 30, 30)
        elif severity == "warning":
            self.set_text_color(200, 140, 0)
        else:
            self.set_text_color(40, 40, 40)
        self.set_font("Courier", "B", 8)
        self.cell(18, 4.5, time)
        self.set_font("Helvetica", "", 8.5)
        if severity == "critical":
            self.set_text_color(180, 30, 30)
        elif severity == "warning":
            self.set_text_color(200, 140, 0)
        else:
            self.set_text_color(40, 40, 40)
        self.set_x(self.l_margin + 18)
        self.multi_cell(0, 4.5, event)
        self.ln(0.5)


def build_pdf():
    pdf = BugfixPDF()
    pdf.alias_nb_pages()
    pdf.set_auto_page_break(auto=True, margin=20)
    pdf.add_page()

    # ── TITLE PAGE ──
    pdf.ln(15)
    pdf.set_font("Helvetica", "B", 26)
    pdf.set_text_color(180, 40, 40)
    pdf.cell(0, 12, "GPU Crash Investigation", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 12, "& Resolution Report", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(5)
    pdf.set_font("Helvetica", "", 14)
    pdf.set_text_color(80, 80, 80)
    pdf.cell(0, 8, "Rock 5B (RK3588) - Chromium / Mali GPU Incompatibility", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(3)
    pdf.set_draw_color(180, 40, 40)
    pdf.line(60, pdf.get_y(), 150, pdf.get_y())
    pdf.ln(10)

    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(60, 60, 60)
    info = [
        ("Date:", "March 28, 2026"),
        ("System:", "Rock 5B (RK3588, aarch64, Debian 12, 16 GB RAM)"),
        ("Software:", "Chromium browser with EGL GPU backend on KDE Plasma/KWin"),
        ("Symptom:", "Complete system hang requiring hard reboot"),
        ("Crash time:", "04:16:19 UTC, after ~12 hours uptime"),
        ("Root cause:", "Chromium GPU process failure rendering a Dash/Plotly web application"),
        ("Resolution:", "Launch Chromium with --disable-gpu flag"),
    ]
    for label, value in info:
        pdf.set_font("Helvetica", "B", 10)
        pdf.cell(30, 6, label)
        pdf.set_font("Helvetica", "", 10)
        pdf.cell(0, 6, value, new_x="LMARGIN", new_y="NEXT")

    # ── USER REPORT ──
    pdf.add_page()
    pdf.section("1. User Report")
    pdf.body(
        'The user reported a system crash after rebooting the Rock 5B. The system had been '
        'running a Dash/Plotly web application (served on port 8050 and viewed in Chromium) '
        'and a Claude Code terminal session. The user requested a post-reboot system analysis '
        'to determine the cause of the crash.'
    )

    pdf.sub("Initial System State After Reboot")
    entries = [
        ("Uptime:", "3 minutes (fresh boot)"),
        ("RAM:", "1.7 GB / 16 GB used (13 GB free)"),
        ("Swap:", "0 B / 16 GB used"),
        ("Disk:", "23 GB / 470 GB used (5%)"),
        ("Temperatures:", "39-40 C across all thermal zones"),
        ("CPU Load:", "0.47 (very light)"),
        ("Failed services:", "None"),
        ("Kernel errors:", "None (current boot)"),
        ("Network:", "Ethernet up, 192.168.1.34"),
    ]
    for label, value in entries:
        pdf.label_value(label, value)

    pdf.ln(2)
    pdf.body(
        "Post-reboot diagnostics showed a clean system. No OOM events, no kernel panics, "
        "no failed services. The previous boot's only kernel error was a benign Rockchip MPP "
        "(media processing pipeline) cleanup message: 'mpp hal_bufs_deinit invalid NULL input'."
    )

    # ── INVESTIGATION ──
    pdf.add_page()
    pdf.section("2. Investigation")

    pdf.sub("Boot History Analysis")
    pdf.body(
        "The previous boot (boot -1) ran from March 27, 15:54:08 to March 28, 04:16:19 "
        "- approximately 12 hours. The journal ended abruptly at 04:16:19 with NO shutdown "
        "sequence recorded. No systemd service stop messages, no reboot command logged. "
        "The 'last' command showed no shutdown record - only reboot entries. This confirms "
        "a hard lockup, not a controlled shutdown or reboot."
    )

    pdf.sub("Ruling Out Common Causes")
    pdf.body("The following potential causes were investigated and ruled out:")
    ruled_out = [
        "OOM Killer: No OOM events in kernel log or journald. Memory was not exhausted.",
        "Kernel Panic: No panic, oops, or BUG messages in dmesg. No coredumps in /var/lib/systemd/coredump/.",
        "Thermal Shutdown: No thermal trip events, no throttling messages. Temps were normal (39-40 C at reboot).",
        "Disk Failure: No I/O errors, filesystem clean, eMMC health normal.",
        "Power Supply: No undervoltage or brownout warnings in PMIC/regulator logs.",
        "Application Memory Leak: Comprehensive code analysis of the running application showed proper "
        "memory management - gc.collect() cycles, bounded caches, database pruning.",
    ]
    for item in ruled_out:
        pdf.bullet(item)

    pdf.sub("The 7-Hour Log Gap")
    pdf.body(
        "A critical observation: between 21:36 and 04:08 there were almost no log entries "
        "(only periodic desktop vsync messages). This 7-hour gap is significant - it means "
        "the system appeared idle but was actually in a degraded state."
    )

    # ── TIMELINE ──
    pdf.add_page()
    pdf.section("3. Crash Timeline - Second by Second")

    pdf.body(
        "Detailed reconstruction of events from journalctl -b -1, correlating plasmashell, "
        "Chromium, KWin, and KScreen log entries:"
    )
    pdf.ln(2)

    timeline = [
        ("15:54:02", "System boots. KDE Plasma desktop starts.", "normal"),
        ("15:55:39", "Konsole terminal opened.", "normal"),
        ("15:55:59", "Chromium launched: chromium-bin --use-gl=egl --no-sandbox", "warning"),
        ("", "Opened to view a Dash/Plotly web application at http://0.0.0.0:8050", "normal"),
        ("15:56:00", "Chromium GPU process (PID 2145) attempts EGL initialization.", "normal"),
        ("", "FAILURE: 'Passthrough is not supported, GL is egl'", "critical"),
        ("", "GPU passthrough mode fails on Mali GPU driver.", "critical"),
        ("15:56:01", "16 consecutive 'Failed to export buffer to dma_buf' errors.", "critical"),
        ("", "Mali GPU cannot share DMA buffers with Chromium's compositor.", "critical"),
        ("15:56:02", "'Failed to query video capabilities: Inappropriate ioctl'", "warning"),
        ("", "V4L2 hardware video decode also fails.", "warning"),
        ("15:56:03", "'*** stack smashing detected ***: terminated'", "critical"),
        ("", "Chromium GPU subprocess crashes with stack buffer overflow.", "critical"),
        ("15:56:03", "First 'process_memory_range.cc: read out of range' error.", "critical"),
        ("", "Crash handler tries to read corrupted GPU process memory.", "critical"),
        ("16:16:48", "'GPU process exited unexpectedly: exit_code=5'", "critical"),
        ("", "Chromium's GPU process fully terminates.", "critical"),
        ("15:56-", "Every 5-15 min: 'process_memory_range: read out of range'", "warning"),
        ("21:36", "repeated 59 times total over 6 hours.", "warning"),
        ("", "Each page auto-refresh triggers a failed GPU render attempt.", "warning"),
        ("", "'blink.mojom.WidgetHost' connection rejected 17 times.", "warning"),
        ("21:36-", "7-hour apparent silence. System in degraded GPU state.", "warning"),
        ("04:08", "", "normal"),
        ("04:08:32", "TRIGGER EVENT: Monitor disconnects then reconnects.", "critical"),
        ("", "HDMI output bounces (likely DPMS wake / Samsung monitor sleep cycle).", "critical"),
        ("", "KScreen attempts display reconfiguration (EDID re-read, output 66).", "warning"),
        ("", "KWin reports 'BadWindow' errors during reconfiguration.", "warning"),
        ("04:14:30", "plasmashell vsync errors become sporadic with large counter gaps.", "warning"),
        ("", "Thousands of missed frames between log entries.", "warning"),
        ("04:16:19", "LAST LOG ENTRY. Journal abruptly ends.", "critical"),
        ("", "No shutdown sequence. No systemd stop messages. Hard lockup.", "critical"),
    ]

    for time, event, severity in timeline:
        if event:
            pdf.timeline_entry(time, event, severity)

    # ── ROOT CAUSE ──
    pdf.add_page()
    pdf.section("4. Root Cause Analysis")

    pdf.sub("The Causal Chain")
    pdf.body(
        "The crash was caused by a chain of failures originating from Chromium's GPU process "
        "being unable to handle the RK3588's Mali GPU driver when rendering a Dash/Plotly "
        "web application. The complete chain:"
    )

    chain = [
        "1. ORIGIN: A Dash/Plotly web application serves dynamic charts at localhost:8050. "
        "Chromium was opened to view this application using the default --use-gl=egl GPU backend.",

        "2. GPU FAILURE: The RK3588's Mali GPU driver cannot properly export DMA-BUF buffers "
        "for Chromium's GPU compositor. 16 consecutive DMA-BUF export failures occurred in "
        "the first second, followed by a stack buffer overflow (stack smashing detected) in "
        "Chromium's GPU subprocess.",

        "3. DEGRADED STATE: Chromium's GPU process crashed (exit code 5), but the main browser "
        "process (PID 2111) continued running. The web page remained open in the browser tab.",

        "4. REPEATED FAILURES: The web application's auto-refresh intervals (15s to 60s on "
        "various pages) kept pushing new chart renders to the browser. Each render attempt hit "
        "the dead GPU process, generating 'process_memory_range: read out of range' errors and "
        "'blink.mojom.WidgetHost' connection rejections. This pattern repeated 59 + 17 = 76 "
        "times over 6 hours.",

        "5. TRIGGER: At 04:08:32, the monitor's DPMS sleep/wake cycle caused an HDMI "
        "disconnect/reconnect event. KScreen attempted to reconfigure the display output.",

        "6. LOCKUP: The combination of a corrupted GPU compositor (6 hours of memory-range "
        "errors), a dead GPU process, and a display hotplug reconfiguration event overwhelmed "
        "the RK3588's VOP2 display controller. The VOP2 shares the memory bus with the CPU "
        "cores, causing a full SoC hard hang. No further log entries were written.",
    ]
    for step in chain:
        pdf.body(step)

    pdf.sub("Why the Web Application's Auto-Refresh Contributed")
    pdf.body(
        "While the web application code itself has no bugs causing this crash, its "
        "auto-refresh intervals were the mechanism that kept hammering the dead "
        "GPU process. Without the continuous chart updates every 15-60 seconds, "
        "Chromium's main process would have been idle and the monitor DPMS wake at 04:08 "
        "would likely not have caused a system hang.\n\n"
        "The auto-refresh intervals in the web application:\n"
        "   - Page 1: 15 seconds\n"
        "   - Page 2: 30 seconds\n"
        "   - Page 3: 30 seconds\n"
        "   - Page 4: 60 seconds\n"
        "   - Page 5: 60 seconds"
    )

    # ── TECHNICAL DETAIL ──
    pdf.add_page()
    pdf.section("5. Technical Details")

    pdf.sub("Chromium Launch Command (from journald)")
    pdf.mono("chromium-bin --use-gl=egl --no-sandbox")

    pdf.sub("GPU Process Error Sequence (15:56:00-15:56:03)")
    errors = [
        "Passthrough is not supported, GL is egl",
        "Failed to export buffer to dma_buf  (x16)",
        "Failed to query video capabilities: Inappropriate ioctl for device",
        "*** stack smashing detected ***: terminated",
        "process_memory_range.cc: read out of range  (x59 over 6 hours)",
        "GPU process exited unexpectedly: exit_code=5",
    ]
    for e in errors:
        pdf.mono("  " + e)

    pdf.sub("Display Reconfiguration Event (04:08:32)")
    pdf.mono('  Connection: "Disconnected" -> "Connected"')
    pdf.mono("  KScreen: EDID re-read, output 66 reconfiguration")
    pdf.mono('  KWin: BadWindow (invalid Window parameter)')
    pdf.mono('  KWin: Could not create scene graph context for backend "rhi"')

    pdf.sub("plasmashell Error Statistics (Previous Boot)")
    stats = [
        ("process_memory_range read errors:", "59 occurrences"),
        ("blink.mojom.WidgetHost rejections:", "17 occurrences"),
        ("DMA-BUF export failures:", "16 occurrences"),
        ("Duration of degraded state:", "~6 hours (15:56 to 21:36 active, then silent)"),
        ("Total boot duration:", "12h 22m (15:54 to 04:16)"),
    ]
    for label, value in stats:
        pdf.label_value(label, value)

    # ── RESOLUTION ──
    pdf.add_page()
    pdf.section("6. Resolution")

    pdf.sub("Fix Applied: Chromium --disable-gpu Flag")
    pdf.body(
        "The fix is to launch Chromium with GPU hardware acceleration disabled when viewing "
        "the web application. This prevents the Mali GPU / DMA-BUF incompatibility entirely."
    )

    pdf.sub("Desktop Shortcut Created")
    pdf.body("File: /home/rock/Desktop/AlphaDashboard.desktop")
    pdf.mono(
        "  [Desktop Entry]\n"
        "  Name=Dashboard\n"
        "  Comment=Open web dashboard (GPU disabled for stability)\n"
        "  Exec=chromium --disable-gpu\n"
        "       --disable-software-rasterizer-for-gpu-compositing\n"
        "       http://0.0.0.0:8050\n"
        "  Icon=chromium\n"
        "  Terminal=false\n"
        "  Type=Application"
    )

    pdf.sub("Performance Impact Assessment")
    pdf.body(
        "Disabling GPU acceleration has negligible performance impact for this use case:\n\n"
        "   - Dash/Plotly renders charts as SVG, not WebGL\n"
        "   - Only Plotly 3D charts (scatter3d) use WebGL - this application uses 2D charts\n"
        "   - CPU-based software rasterization is handled by the RK3588's 4x A76 big cores\n"
        "   - The application is a data display tool, not a graphics-intensive application\n"
        "   - Slight delay on initial page loads with complex charts, but not perceptible\n"
        "     during normal 15-60 second auto-refresh cycles"
    )

    pdf.sub("Alternative Mitigations Considered")
    alternatives = [
        "Access the web application from a different machine on the network (it binds to 0.0.0.0:8050) "
        "- viable but less convenient",
        "Run the application in headless mode without the web dashboard - loses real-time monitoring",
        "Use Firefox instead of Chromium - handles Mali GPU fallback more gracefully, but untested "
        "with this specific application",
        "Disable DPMS/screen blanking (xset -dpms; xset s off) - prevents the trigger event but "
        "doesn't fix the underlying GPU process crash",
        "Switch from KDE Plasma/KWin to a lighter window manager (Openbox) - reduces GPU compositor "
        "complexity but is a major desktop change",
    ]
    for alt in alternatives:
        pdf.bullet(alt)

    # ── PREVENTION ──
    pdf.add_page()
    pdf.section("7. Prevention")

    pdf.sub("Immediate Actions Taken")
    pdf.bullet(
        "Created desktop shortcut with --disable-gpu flag"
    )
    pdf.bullet(
        "Documented the Chromium/Mali GPU incompatibility for future reference"
    )

    pdf.ln(3)
    pdf.sub("Recommended Future Actions")
    pdf.bullet(
        "Monitor for updated Mali GPU drivers (libmali) or mesa packages for the RK3588 "
        "that may resolve the DMA-BUF export issue"
    )
    pdf.bullet(
        "Consider disabling DPMS on the monitor as a secondary safeguard: xset -dpms; xset s off"
    )
    pdf.bullet(
        "If running applications overnight unattended, use headless mode to avoid any "
        "browser-related issues entirely"
    )
    pdf.bullet(
        "Check Chromium version and update if a newer release has improved ARM64/Mali GPU support"
    )

    pdf.ln(5)
    pdf.set_draw_color(180, 40, 40)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(3)
    pdf.set_font("Helvetica", "I", 9)
    pdf.set_text_color(100, 100, 100)
    pdf.cell(0, 5, "Report generated by Claude Code on March 28, 2026.", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 5, "Investigation performed on Rock 5B system post-reboot.", align="C")

    output_path = "/home/rock/reports/gpu_crash_bugfix_report.pdf"
    pdf.output(output_path)
    print(f"PDF saved to: {output_path}")


if __name__ == "__main__":
    build_pdf()
