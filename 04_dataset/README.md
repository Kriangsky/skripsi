# 04 — Dataset

- [train](train/): 2.400 video internal,300/kelas. Frozen split menentukan train/validation.
- [test](test/): 400 video external,50/kelas. Tidak digunakan untuk tuning.

Nama kelas, isi video dan split tidak diubah. Runtime lama mengakses file yang sama melalui hard links;
video tidak disalin dua kali secara fisik. Jangan mengedit/menghapus sumber dari salah satu akses.
[Protokol](../07_dokumen/protokol/PREREGISTRATION.md) · [Hasil](../01_hasil/README.md).
