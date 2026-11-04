# 05 — Kode

- [src](src/): modul pipeline, classification, paired/power, diagnostics dan runner.
- [tests](tests/): seluruh unit tests existing.
- [run.ps1](run.ps1): launcher dengan Python existing dan cwd runtime kompatibel.

Contoh dari root proyek:

```powershell
.\05_kode\run.ps1 classification_research.py --help
.\05_kode\run.ps1 -m unittest test_stage0_research_plan test_protocol_addendum -v
.\05_kode\run.ps1 -Tests
.\05_kode\run.ps1 -Notebook
```

Jalankan melalui launcher, bukan langsung dari folder src/tests: source beku mengasumsikan cwd dan
file pendamping pada satu root. Launcher mempertahankan asumsi itu tanpa mengubah byte/hash kode.
Source di section dan `.compat_runtime/` adalah hard links ke file yang sama, bukan dua versi kode.
Mengedit source mengubah keduanya dan akan terdeteksi lock pada run lama.
Tidak ada training yang otomatis dijalankan oleh reorganisasi.

Launcher memeriksa hard links sebelum menjalankan Python. Bila editor melakukan atomic-save dan memisahkan inode, launcher berhenti serta mempertahankan kedua versi untuk direkonsiliasi; tidak melakukan sinkronisasi yang dapat membuang hasil edit.
