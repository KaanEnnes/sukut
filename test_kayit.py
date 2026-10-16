"""Gercek mikrofondan kayit al, uygulamadaki ayni blok-blok yolla temizle, oncesi/sonrasi yaz."""
import sys, time, numpy as np, sounddevice as sd, soundfile as sf
import sukut
sec = float(sys.argv[1]) if len(sys.argv) > 1 else 10
dev = sukut.find_device("H2039BG", "in")
print("Kayit", sec, "sn, cihaz", dev, flush=True)
x = sd.rec(int(sec * sukut.SR), sukut.SR, channels=1, device=dev, dtype="float32",
           extra_settings=sukut.WASAPI); sd.wait(); x = x[:, 0]
c = sukut.Cleaner(); c.strength = 100
B, C, F = c.BLOCK, c.CONTEXT, c.FADE
hist = np.zeros(C, np.float32); out = []; prev = None; ts = []
for i in range(0, len(x) - B + 1, B):
    hist = np.concatenate([hist[B:], x[i:i + B]])
    t = time.perf_counter(); y = c.process_block(hist); ts.append((time.perf_counter() - t) * 1000)
    cur = y[-B - F:-F].copy()
    if prev is not None:
        r = np.linspace(0, 1, F); cur[:F] = prev * (1 - r) + cur[:F] * r
    prev = y[-F:].copy(); out.append(cur)
y = np.concatenate(out)
sf.write("ornekler/oncesi.wav", x, sukut.SR); sf.write("ornekler/sonrasi.wav", y, sukut.SR)
print(f"islem ort {np.mean(ts):.1f} ms, p95 {np.percentile(ts,95):.1f} ms (blok 20 ms)")
print("algoritmik gecikme ~", (B + F) / 48, "ms + cihaz tamponlari")
rms = lambda a: 20*np.log10(np.sqrt(np.mean(a**2))+1e-9)
print(f"seviye oncesi {rms(x):.1f} dBFS, sonrasi {rms(y):.1f} dBFS")
