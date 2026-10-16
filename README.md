# Sükût — Gürültü maskeleme + Mikrofon temizleyici (Windows)

## Kullanım
`Sukut.exe`'yi indir ve çift tıkla, hepsi bu. Python gerekmez, model içinde. Açılış 10-20 sn sürebilir.

Kaynaktan çalıştırmak için: `kur_ve_calistir.bat`
Exe'yi yeniden üretmek için: `.venv\Scripts\pyinstaller` komutu (bkz. build notu aşağıda).
Pencereyi kapatınca sistem tepsisine iner; tepsideki ikondan "Göster" / "Çıkış".

## 1) Gürültü Maskeleme
- Ses tipi: Beyaz / Pembe / Kahverengi / Yağmur. Temel seviye kaydırıcısı.
- **Otomatik ayar**: PC mikrofonuyla ortamı ölçer; ortam gürültülenince maskeyi yumuşakça yükseltir (6 dB/sn), sessizleşince indirir (2 dB/sn).

## 2) Mikrofon Temizleyici (DeepFilterNet3)
Akış: **Havit H2039BG mikrofonu → DeepFilterNet3 (gerçek zamanlı) → VB-Audio CABLE Input → (Discord/oyun) "CABLE Output"**

### VB-Audio Virtual Cable kurulumu (bir kere, kendin yap)
1. https://vb-audio.com/Cable/ → "Download" → `VBCABLE_Driver_Pack` zip'ini indir.
2. Zip'i bir klasöre aç, `VBCABLE_Setup_x64.exe`'ye sağ tık → **Yönetici olarak çalıştır** → *Install Driver*.
3. Bilgisayarı yeniden başlat.
4. Sükût → "Mikrofon Temizleyici" sekmesi: Gerçek mikrofon = **Mikrofon (H2039BG)**, Çıkış = **CABLE Input** → Başlat.
5. **Discord / oyunda giriş (mikrofon) cihazı olarak "CABLE Output" seç.** (Discord'un kendi Krisp'ini kapat.)

- "Temizleme açık" kutusu: kapatırsan ses ham geçer.
- "Güç": %100 = tam bastırma, düşürdükçe arka plan biraz duyulur (daha doğal ses).
- Gecikme: 10 ms blok + ~7 ms işlem + WASAPI exclusive tamponlar ≈ **45–50 ms** toplam (ölçülen). Exclusive açılamazsa paylaşımlı moda düşer (~65 ms).

## Test
`.venv\Scripts\python test_kayit.py 10` → gerçek mikrofondan 10 sn kaydeder, `ornekler\oncesi.wav` / `sonrasi.wav` üretir.

## Notlar
- DeepFilterNet 0.5.6 yeni torchaudio ile uyumsuz; `dfshim.py` bunu yamalar.
- `wheels\` içindeki DeepFilterLib, Windows/Python 3.12 için önceden derlenmiş (PyPI'de hazır wheel yok).
- RNNoise yedeğine gerek kalmadı: DeepFilterNet3 CPU'da gerçek zamanın ~%70'inde çalışıyor.
