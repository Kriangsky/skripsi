# 09 — Environment

- [python_env](python_env/): akses ke environment `.venv-bisindo` existing.

Environment tidak dipindahkan karena interpreter dan entry point menyimpan path absolut.
Folder aslinya tetap tersembunyi di root; junction ini hanya akses tersegmentasi, bukan instalasi kedua.
Gunakan [launcher](../05_kode/run.ps1). Tidak ada dependency yang diinstall/diupgrade.
