# Organisasi lengkap — 4 November 2026

58 file root masuk section kode/notebook/dokumen/review; train/test masuk dataset.
Kode/protokol/notebook/input lama tidak diedit. `.compat_runtime/` menyediakan filename/cwd historis
lewat hard links;2.800 video memakai inode yang sama, bukan salinan data.
Path48 folder eksperimen sebelumnya tetap kompatibel melalui junction tersembunyi di root.

README dan gitignore root sekarang khusus navigasi/layout baru. Versi original keduanya tetap diarsipkan
dan dipakai runtime untuk mempertahankan snapshot historis. Metadata Git tetap di root.

- [Manifest](manifest.json): lokasi sebelum/sesudah, hash, file IDs dan daftar dataset.
- [Skrip migrasi](organize_all.ps1): sudah dijalankan, bukan untuk diulang.
- [Rollback](rollback_all.ps1): mengembalikan layout sebelum segmentasi lengkap, tanpa mengubah isi data.

Cek rollback tanpa mengubah file:

```powershell
.\07_dokumen\organisasi_20261104_lengkap\rollback_all.ps1 -CheckOnly
```

Rollback lengkap ini harus dilakukan sebelum rollback organisasi48 folder tahap sebelumnya.
Tidak menghapus runtime/alias/hard links secara manual. File kode/dataset di section dan runtime adalah
file fisik yang sama; edit pada salah satu akses memengaruhi keduanya.
Tidak ada training atau Tahap2 yang dijalankan.
