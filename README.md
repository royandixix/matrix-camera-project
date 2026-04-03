AI Hand & Face Detection – Realtime Detection System

Sistem deteksi tangan dan wajah secara real-time menggunakan MediaPipe, mendukung multi orang, dengan visualisasi langsung di kamera. Sistem ini dapat mendeteksi:

Tangan & jari
Mata
Mulut (luar & dalam)
Pipi
Perkiraan area lidah
📦 Fitur
Objek	Warna Visual di Kamera	Keterangan
Tangan & jari	Hijau	Deteksi posisi jari naik/turun, skeleton tangan
Mata	Kuning	Landmark mata kiri & kanan
Mulut luar	Ungu	Landmark titik mulut luar
Mulut dalam / lidah (perkiraan)	Merah	Landmark titik mulut dalam
Pipi	Orange	Landmark pipi kiri & kanan
Multi orang	–	Bisa mendeteksi lebih dari satu wajah sekaligus
🔧 Persiapan

Install Python 3.11
Direkomendasikan menggunakan Homebrew di macOS:

brew install python@3.11

Buat virtual environment (opsional tapi disarankan)

python3.11 -m venv venv
source venv/bin/activate

Install dependencies

pip install opencv-python mediapipe
🚀 Jalankan Project
Pastikan kamera tersambung.

Jalankan script utama:

python3 hand_cam.py
Tekan ESC untuk keluar dari aplikasi.
⚙️ Struktur Folder
matrix_camera_project/
├── hand_cam.py           # Script utama
├── hand_landmarker.task
├── face_landmarker.task
└── README.md