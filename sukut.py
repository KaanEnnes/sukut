"""Sükût — gürültü maskeleme + DeepFilterNet3 mikrofon temizleyici (Windows)."""
import queue
import threading
import time
import tkinter as tk
from tkinter import ttk

import numpy as np
import sounddevice as sd

SR = 48000


def find_device(name_part, kind, hostapi="WASAPI"):
    for i, d in enumerate(sd.query_devices()):
        ha = sd.query_hostapis(d["hostapi"])["name"]
        ch = d["max_input_channels"] if kind == "in" else d["max_output_channels"]
        if ch > 0 and hostapi in ha and name_part.lower() in d["name"].lower():
            return i
    return None


def device_names(kind):
    out = []
    for i, d in enumerate(sd.query_devices()):
        ha = sd.query_hostapis(d["hostapi"])["name"]
        ch = d["max_input_channels"] if kind == "in" else d["max_output_channels"]
        if ch > 0 and "WASAPI" in ha:
            out.append(f"{i}: {d['name']}")
    return out


WASAPI = sd.WasapiSettings(auto_convert=True)


# ---------------------------------------------------------------- Maskeleme
class NoiseGen:
    """Yumuşak, cızırtısız maskeleme sesleri: hepsi alçak geçiren filtreden geçer, stereo."""

    def __init__(self):
        from scipy.signal import butter, lfilter, lfilter_zi
        self.lfilter = lfilter
        self.rng = np.random.default_rng()
        self.filters = {}
        def mk(name, b, a):
            self.filters[name] = (b, a, np.zeros((2, max(len(a), len(b)) - 1)))
        mk("white", *butter(2, 7000, fs=SR))           # beyaz: tiz uç törpülü
        # pembe: -3 dB/oktav (Kellet katsayılarına eşdeğer IIR) + 5 kHz alçak geçiren
        mk("pink", [0.049922035, -0.095993537, 0.050612699, -0.004408786],
           [1, -2.494956002, 2.017265875, -0.522189400])
        mk("pink_lp", *butter(2, 5000, fs=SR))
        mk("brown", *butter(1, 120, fs=SR))             # kahverengi: derin uğultu
        mk("rain", *butter(2, [400, 3500], btype="band", fs=SR))
        mk("rain_low", *butter(2, 900, fs=SR))
        mk("drop", *butter(2, 2500, fs=SR))
        self.mod = 0.0

    def f(self, name, x):
        b, a, zi = self.filters[name]
        y = np.empty_like(x)
        for ch in range(2):
            y[ch], zi[ch] = self.lfilter(b, a, x[ch], zi=zi[ch])
        return y

    def make(self, kind, n):
        w = self.rng.standard_normal((2, n))
        if kind == "Beyaz":
            out = self.f("white", w) * 0.32
        elif kind == "Pembe":
            out = self.f("pink_lp", self.f("pink", w)) * 2.3
        elif kind == "Kahverengi":
            out = self.f("brown", w) * 2.0
        else:  # Yağmur: hışırtı bandı + derin taban + yumuşak, filtrelenmiş damlalar
            out = self.f("rain", w) * 0.39 + self.f("rain_low", self.rng.standard_normal((2, n))) * 0.56
            drops = np.zeros((2, n))
            for p in self.rng.integers(0, n, self.rng.poisson(n / SR * 60)):
                L = min(int(SR * 0.012), n - p)
                env = np.exp(-np.arange(L) / (SR * 0.003))
                drops[self.rng.integers(0, 2), p:p + L] += self.rng.standard_normal(L) * env * self.rng.uniform(0.05, 0.18)
            out += self.f("drop", drops)
            # yağmur şiddeti yavaşça dalgalansın
            self.mod += 0.02 * (self.rng.uniform(0.7, 1.15) - self.mod - 0.0) if self.mod else 1.0
            out *= self.mod
        return out


class Masker:
    def __init__(self):
        self.kind = "Pembe"
        self.base_db = -30.0
        self.auto = True
        self.on = False
        self.gen = NoiseGen()
        self.gain_db = self.base_db
        self.ambient_db = -60.0
        self.quiet_ref = None
        self.out_dev = None
        self.in_dev = None
        self.streams = []

    def target_db(self):
        if not self.auto or self.quiet_ref is None:
            return self.base_db
        extra = (self.ambient_db - self.quiet_ref) * 0.8  # ortam her 1 dB artınca 0.8 dB yükselt
        return float(np.clip(self.base_db + max(0.0, extra), -60, -6))

    def _in_cb(self, data, frames, t, status):
        rms = np.sqrt(np.mean(data[:, 0] ** 2) + 1e-12)
        db = 20 * np.log10(rms)
        a = 0.3 if db > self.ambient_db else 0.02  # hızlı yüksel, yavaş in
        self.ambient_db += a * (db - self.ambient_db)
        if db < -100:
            return
        if self.quiet_ref is None:
            self.ambient_db = self.quiet_ref = db
        # sessiz referansı ortamın en düşük seviyesini yavaşça izler
        self.quiet_ref = min(self.quiet_ref + 0.0005, self.ambient_db)

    def _out_cb(self, out, frames, t, status):
        tgt = self.target_db()
        step = frames / SR
        rate = 6.0 if tgt > self.gain_db else 2.0  # dB/sn yumuşak geçiş
        d = np.clip(tgt - self.gain_db, -rate * step, rate * step)
        g0 = 10 ** (self.gain_db / 20)
        self.gain_db += d
        g1 = 10 ** (self.gain_db / 20)
        sig = self.gen.make(self.kind, frames) * np.linspace(g0, g1, frames)
        out[:] = np.clip(sig.T, -1, 1).astype(np.float32)

    def start(self):
        self.stop()
        self.quiet_ref = None
        o = sd.OutputStream(SR, blocksize=1024, device=self.out_dev, channels=2, dtype="float32",
                            callback=self._out_cb, extra_settings=WASAPI)
        self.streams = [o]
        if self.auto:
            i = sd.InputStream(SR, blocksize=2048, device=self.in_dev, channels=1, dtype="float32",
                               callback=self._in_cb, extra_settings=WASAPI)
            self.streams.append(i)
        for s in self.streams:
            s.start()
        self.on = True

    def stop(self):
        for s in self.streams:
            s.stop(); s.close()
        self.streams = []
        self.on = False


# ---------------------------------------------------------------- Temizleyici
class Cleaner:
    """Gerçek mikrofon -> DeepFilterNet3 (kayan pencere) -> CABLE Input."""

    BLOCK = 480         # 10 ms
    CONTEXT = 4800      # 100 ms geçmiş bağlam (model durumu için)
    FADE = 96           # 2 ms blok geçişi

    def __init__(self):
        import dfshim  # noqa: F401
        import torch
        from df.enhance import enhance, init_df
        torch.set_num_threads(2)
        self.torch = torch
        self.enhance = enhance
        import os, sys
        base = getattr(sys, "_MEIPASS", None)  # tek exe: model paketin içinde
        mdir = os.path.join(base, "DeepFilterNet3") if base else None
        self.model, self.state, _ = init_df(mdir, log_level="ERROR", log_file=None)
        self.enabled = True
        self.strength = 80  # 0..100
        self.in_dev = None
        self.out_dev = None
        self.running = False
        self.proc_ms = 0.0

    def atten_db(self):
        # %100 = sınırsız bastırma, %0 = sadece 3 dB
        return None if self.strength >= 100 else 3 + self.strength * 0.6

    def process_block(self, hist):
        x = self.torch.from_numpy(hist[None, :].copy())
        y = self.enhance(self.model, self.state, x, atten_lim_db=self.atten_db())
        return y[0].numpy()

    def _worker(self):
        hist = np.zeros(self.CONTEXT, np.float32)
        prev_tail = None
        pend = np.zeros(0, np.float32)
        while self.running:
            try:
                chunk = self.q_in.get(timeout=0.2)
            except queue.Empty:
                continue
            pend = np.concatenate([pend, chunk])
            while len(pend) >= self.BLOCK:
                blk, pend = pend[:self.BLOCK], pend[self.BLOCK:]
                hist = np.concatenate([hist[self.BLOCK:], blk])
                if not self.enabled:
                    self.q_out.put(blk.copy()); prev_tail = None; continue
                t = time.perf_counter()
                y = self.process_block(hist)
                self.proc_ms = 0.9 * self.proc_ms + 0.1 * (time.perf_counter() - t) * 1000
                # FADE kadar geride kal; onceki pencerenin son FADE ornegiyle capraz gecis
                cur = y[-self.BLOCK - self.FADE:-self.FADE].copy()
                if prev_tail is not None:
                    r = np.linspace(0, 1, self.FADE, dtype=np.float32)
                    cur[:self.FADE] = prev_tail * (1 - r) + cur[:self.FADE] * r
                prev_tail = y[-self.FADE:].copy()
                self.q_out.put(cur.astype(np.float32))

    def start(self):
        self.stop()
        self.q_in, self.q_out = queue.Queue(), queue.Queue()
        self.outbuf = np.zeros(0, np.float32)
        self.running = True
        threading.Thread(target=self._worker, daemon=True).start()

        def in_cb(data, frames, t, st):
            self.q_in.put(data[:, 0].copy())

        def out_cb(out, frames, t, st):
            while len(self.outbuf) < frames:
                try:
                    self.outbuf = np.concatenate([self.outbuf, self.q_out.get_nowait()])
                except queue.Empty:
                    break
            n = min(frames, len(self.outbuf))
            out[:n, 0] = self.outbuf[:n]; out[n:, 0] = 0
            if out.shape[1] > 1:
                out[:, 1] = out[:, 0]
            self.outbuf = self.outbuf[n:]
            if len(self.outbuf) > self.BLOCK * 3:  # gecikme birikmesin
                self.outbuf = self.outbuf[-self.BLOCK:]

        def open_streams(extra):
            si = sd.InputStream(SR, blocksize=480, device=self.in_dev, channels=1, dtype="float32",
                                callback=in_cb, latency=0.005, extra_settings=extra)
            so = sd.OutputStream(SR, blocksize=480, device=self.out_dev, channels=2, dtype="float32",
                                 callback=out_cb, latency=0.005, extra_settings=extra)
            return si, so
        try:  # WASAPI exclusive = en düşük gecikme; olmazsa paylaşımlı moda düş
            self.si, self.so = open_streams(sd.WasapiSettings(exclusive=True))
        except Exception:
            self.si, self.so = open_streams(WASAPI)
        self.si.start(); self.so.start()

    def stop(self):
        if self.running:
            self.running = False
            for s in (self.si, self.so):
                s.stop(); s.close()


# ---------------------------------------------------------------- Arayüz
class App:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Sükût")
        self.root.geometry("420x430")
        self.masker = Masker()
        self.cleaner = None
        nb = ttk.Notebook(self.root); nb.pack(fill="both", expand=True, padx=8, pady=8)
        self._mask_tab(nb); self._clean_tab(nb)
        self.root.protocol("WM_DELETE_WINDOW", self.hide)
        self._tray()
        self._tick()

    # -- maskeleme
    def _mask_tab(self, nb):
        f = ttk.Frame(nb, padding=10); nb.add(f, text="Gürültü Maskeleme")
        ttk.Label(f, text="Ses tipi").pack(anchor="w")
        self.kind = tk.StringVar(value="Pembe")
        for k in ("Beyaz", "Pembe", "Kahverengi", "Yağmur"):
            ttk.Radiobutton(f, text=k, value=k, variable=self.kind,
                            command=lambda: setattr(self.masker, "kind", self.kind.get())).pack(anchor="w")
        ttk.Label(f, text="Temel seviye").pack(anchor="w", pady=(8, 0))
        self.base = tk.DoubleVar(value=-30)
        ttk.Scale(f, from_=-60, to=-10, variable=self.base,
                  command=lambda v: setattr(self.masker, "base_db", float(v))).pack(fill="x")
        self.auto = tk.BooleanVar(value=True)
        ttk.Checkbutton(f, text="Otomatik ayar (ortam gürültüsüne göre)", variable=self.auto,
                        command=self._mask_restart).pack(anchor="w", pady=6)
        ttk.Label(f, text="Ortamı dinleyen mikrofon").pack(anchor="w")
        self.mmic = ttk.Combobox(f, values=device_names("in"), state="readonly"); self.mmic.pack(fill="x")
        ttk.Label(f, text="Çıkış (kulaklık)").pack(anchor="w")
        self.mout = ttk.Combobox(f, values=device_names("out"), state="readonly"); self.mout.pack(fill="x")
        self._preselect(self.mmic, find_device("H2039BG", "in")); self._preselect(self.mout, find_device("H2039BG", "out"))
        self.mbtn = ttk.Button(f, text="Başlat", command=self._mask_toggle); self.mbtn.pack(pady=8)
        self.mstat = ttk.Label(f, text=""); self.mstat.pack()

    def _preselect(self, cb, idx):
        for v in cb["values"]:
            if idx is not None and v.startswith(f"{idx}:"):
                cb.set(v); return
        if cb["values"]:
            cb.current(0)

    @staticmethod
    def _idx(cb):
        return int(cb.get().split(":")[0]) if cb.get() else None

    def _mask_toggle(self):
        if self.masker.on:
            self.masker.stop(); self.mbtn.config(text="Başlat")
        else:
            self._mask_restart(force=True)

    def _mask_restart(self, force=False):
        self.masker.auto = self.auto.get()
        if self.masker.on or force:
            self.masker.in_dev, self.masker.out_dev = self._idx(self.mmic), self._idx(self.mout)
            self.masker.start(); self.mbtn.config(text="Durdur")

    # -- temizleyici
    def _clean_tab(self, nb):
        f = ttk.Frame(nb, padding=10); nb.add(f, text="Mikrofon Temizleyici")
        ttk.Label(f, text="Gerçek mikrofon").pack(anchor="w")
        self.cin = ttk.Combobox(f, values=device_names("in"), state="readonly"); self.cin.pack(fill="x")
        ttk.Label(f, text="Sanal mikrofona gönder (CABLE Input)").pack(anchor="w")
        self.cout = ttk.Combobox(f, values=device_names("out"), state="readonly"); self.cout.pack(fill="x")
        self._preselect(self.cin, find_device("H2039BG", "in"))
        self._preselect(self.cout, find_device("CABLE Input", "out"))
        self.cen = tk.BooleanVar(value=True)
        ttk.Checkbutton(f, text="Temizleme açık", variable=self.cen,
                        command=lambda: self.cleaner and setattr(self.cleaner, "enabled", self.cen.get())).pack(anchor="w", pady=6)
        ttk.Label(f, text="Güç").pack(anchor="w")
        self.cstr = tk.IntVar(value=80)
        ttk.Scale(f, from_=0, to=100, variable=self.cstr,
                  command=lambda v: self.cleaner and setattr(self.cleaner, "strength", int(float(v)))).pack(fill="x")
        self.cbtn = ttk.Button(f, text="Başlat", command=self._clean_toggle); self.cbtn.pack(pady=8)
        self.cstat = ttk.Label(f, text="Discord/oyunda mikrofon olarak \"CABLE Output\" seç.", wraplength=380)
        self.cstat.pack()

    def _clean_toggle(self):
        if self.cleaner and self.cleaner.running:
            self.cleaner.stop(); self.cbtn.config(text="Başlat"); return
        if self.cleaner is None:
            self.cstat.config(text="Model yükleniyor…"); self.root.update()
            self.cleaner = Cleaner()
        self.cleaner.enabled, self.cleaner.strength = self.cen.get(), self.cstr.get()
        self.cleaner.in_dev, self.cleaner.out_dev = self._idx(self.cin), self._idx(self.cout)
        self.cleaner.start(); self.cbtn.config(text="Durdur")

    def _tick(self):
        if self.masker.on:
            self.mstat.config(text=f"Ortam {self.masker.ambient_db:.0f} dB · maske {self.masker.gain_db:.0f} dB")
        if self.cleaner and self.cleaner.running:
            c = self.cleaner
            tot = (c.si.latency + c.so.latency) * 1000 + (c.BLOCK + c.FADE) / 48 + c.proc_ms
            self.cstat.config(text=f"Çalışıyor · toplam gecikme ≈ {tot:.0f} ms (işlem {c.proc_ms:.1f} ms)\n"
                                   "Discord/oyunda mikrofon: \"CABLE Output\"")
        self.root.after(300, self._tick)

    # -- tepsi
    def _tray(self):
        import pystray
        from PIL import Image, ImageDraw
        img = Image.new("RGBA", (64, 64), (0, 0, 0, 0)); d = ImageDraw.Draw(img)
        d.ellipse((4, 4, 60, 60), fill=(90, 110, 150)); d.rectangle((26, 16, 38, 48), fill="white")
        self.icon = pystray.Icon("sukut", img, "Sükût", pystray.Menu(
            pystray.MenuItem("Göster", lambda: self.root.after(0, self.show), default=True),
            pystray.MenuItem("Çıkış", lambda: self.root.after(0, self.quit))))
        threading.Thread(target=self.icon.run, daemon=True).start()

    def hide(self):
        self.root.withdraw()

    def show(self):
        self.root.deiconify(); self.root.lift()

    def quit(self):
        self.masker.stop()
        if self.cleaner:
            self.cleaner.stop()
        self.icon.stop(); self.root.destroy()


if __name__ == "__main__":
    import sys
    if "--selftest" in sys.argv:  # paket testi: modeli yükle, 1 sn işle, sonucu yaz
        c = Cleaner(); y = c.process_block(np.random.randn(SR).astype(np.float32) * 0.05)
        open(sys.argv[-1], "w").write(f"ok {len(y)} {len(sd.query_devices())}")
        sys.exit(0)
    App().root.mainloop()
