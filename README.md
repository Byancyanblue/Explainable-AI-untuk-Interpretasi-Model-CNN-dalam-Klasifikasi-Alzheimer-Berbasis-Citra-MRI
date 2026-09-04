# Explainable AI untuk Interpretasi Model CNN dalam Klasifikasi Alzheimer Berbasis Citra MRI

---

## 1. Latar Belakang

Deteksi dini penyakit Alzheimer sangat krusial karena intervensi pada tahap awal dapat memperlambat perkembangan penyakit dan meningkatkan efektivitas terapi. Model klasifikasi berbasis *deep learning* berpotensi mempermudah deteksi dini, namun sifat *black box* dari model tersebut membuat keandalannya diragukan oleh tenaga medis maupun pasien sebagai pengguna akhir.

*Explainable AI* (XAI) hadir sebagai jembatan untuk menginterpretasikan bagaimana model menghasilkan keputusan, sehingga sifat *black box* dapat dipahami oleh pengguna. Penelitian ini membangun model klasifikasi Alzheimer berbasis *pre-trained* CNN yang tidak hanya akurat, tetapi juga dapat dijelaskan proses pengambilan keputusannya, sekaligus melakukan **evaluasi objektif** terhadap beberapa metode XAI karena masing-masing memiliki karakteristik kerja dan keluaran yang berbeda.

Penelitian ini **tidak** bertujuan mengunggulkan satu metode XAI tertentu, melainkan memberikan wawasan lebih dalam mengenai karakteristik tiap metode berdasarkan hasil evaluasinya — baik secara kuantitatif (metrik XAI) maupun kualitatif (kesesuaian dengan literatur medis terkait area otak yang relevan dengan tiap tingkat keparahan Alzheimer).

## 2. Tujuan Penelitian

1. Membangun model klasifikasi tingkat keparahan Alzheimer dari citra MRI otak menggunakan tiga arsitektur *pre-trained* CNN (VGG19, ResNet50, EfficientNetB2).
2. Menerapkan tiga metode XAI (Grad-CAM, LIME, SHAP) untuk menginterpretasikan hasil klasifikasi model.
3. Mengevaluasi metode XAI secara objektif menggunakan metrik kuantitatif (*deletion-insertion*, *robustness*, *sparsity*, *computational cost*).
4. Menganalisis kesesuaian area atensi model (hasil XAI) dengan literatur medis terkait tingkat keparahan Alzheimer, baik pada sampel prediksi benar maupun salah.

## 3. Model & Metode

| Komponen | Pilihan |
|---|---|
| Arsitektur CNN | VGG19, ResNet50, EfficientNetB2 (*pre-trained*, *fine-tuned*) |
| Metode XAI | Grad-CAM, LIME, SHAP |
| Metrik evaluasi model | Accuracy, Precision, Recall, F1-score, ROC-AUC |
| Metrik evaluasi XAI | Deletion-Insertion, Robustness, Sparsity, Computational Cost |
| Penentuan threshold confidence | Otsu / K-Means |

## 4. Struktur Repository

Berikut struktur repository **sesuai kondisi yang tersedia saat ini**. Folder `src/`, `configs/`, `docs/`, beserta `requirements.txt`/`environment.yml` akan ditambahkan menyusul seiring pengembangan pipeline training dan XAI dipindahkan dari notebook ke modul terpisah.

```
alzheimer-cnn-xai/
│
├── README.md
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_training_vgg19.ipynb
│   ├── 03_training_resnet50.ipynb
│   ├── 04_training_efficientnetb2.ipynb
│   ├── 05_model_evaluation.ipynb
│   ├── 06_xai_sample_selection.ipynb
│   ├── 07_xai_implementation.ipynb
│   └── 08_xai_quantitative_eval.ipynb
│
├── results/
│   ├── model_scores/
│   │   ├── vgg19_scenarios.csv
│   │   ├── resnet50_scenarios.csv
│   │   └── efficientnetb2_scenarios.csv
│   ├── xai_scores/
│   │   ├── vgg19_xai_metrics.csv
│   │   ├── resnet50_xai_metrics.csv
│   │   └── efficientnetb2_xai_metrics.csv
│   ├── sample_indices/
│   │   ├── fixed_indices_vgg19.json
│   │   ├── fixed_indices_resnet50.json
│   │   └── fixed_indices_efficientnetb2.json
│   └── figures/
│       ├── confusion_matrices/
│       ├── loss_accuracy_curves/
│       ├── xai_visualizations/
│       └── confidence_histograms/
│
└── models/
    ├── vgg19/
    ├── resnet50/
    └── efficientnetb2/
```

> Struktur di atas akan berkembang menjadi bentuk modular (`src/`, `configs/`, `docs/`) seiring progres

## 5. Alur Kerja (Pipeline)

1. **Eksplorasi Data** (`01_data_exploration.ipynb`) — memuat dataset MRI Alzheimer dari Hugging Face, distribusi kelas, pengecekan imbalance.
2. **Training Model** (`02`–`04`) — pelatihan 3 arsitektur *pre-trained* CNN dengan 24 skenario (kombinasi *fine-tuning*, *optimizer*, *scheduler*, penanganan imbalance) per arsitektur.
3. **Evaluasi Model** (`05_model_evaluation.ipynb`) — perhitungan Accuracy, Precision, Recall, F1, ROC-AUC, confusion matrix, dan penentuan threshold confidence.
4. **Pemilihan Sampel XAI** (`06_xai_sample_selection.ipynb`) — *stratified sampling* (60 sampel eksplorasi awal, 21 sampel representatif final) dengan indeks yang di-*fix* agar reproducible.
5. **Implementasi XAI** (`07_xai_implementation.ipynb`) — penerapan Grad-CAM, LIME, dan SHAP pada sampel terpilih (7 skenario per model).
6. **Evaluasi Kuantitatif XAI** (`08_xai_quantitative_eval.ipynb`) — perhitungan metrik Deletion-Insertion, Robustness, Sparsity, dan Computational Cost, serta analisis kesesuaian dengan literatur medis.

## 6. Dataset

**[Falah/Alzheimer_MRI Disease Classification](https://huggingface.co/datasets/Falah/Alzheimer_MRI)** — dataset citra MRI otak untuk klasifikasi tingkat keparahan Alzheimer, tersedia di Hugging Face Datasets. Dataset ini menjadi sumber data utama bagi riset klasifikasi Alzheimer maupun aplikasi kesehatan medis lainnya.

**Kelas:**

| Label | Kelas |
|---|---|
| 0 | Mild_Demented |
| 1 | Moderate_Demented |
| 2 | Non_Demented |
| 3 | Very_Mild_Demented |

**Informasi dataset:**

| Split | Jumlah Sampel | Ukuran (bytes) |
|---|---|---|
| Train | 5.120 | 22.560.791,2 |
| Test | 1.280 | 5.637.447,08 |

- Download size: 28.289.848 bytes
- Dataset size: 28.198.238,28 bytes

**Preprocessing:** resize, normalisasi, augmentasi.
**Imbalance handling:** class weighting dan/atau *weighted random sampler* (dataset memiliki distribusi kelas yang tidak seimbang, khususnya kelas `Moderate_Demented`).

**Sitasi:**

```bibtex
@dataset{alzheimer_mri_dataset,
  author = {Falah.G.Salieh},
  title = {Alzheimer MRI Dataset},
  year = {2023},
  publisher = {Hugging Face},
  version = {1.0},
  url = {https://huggingface.co/datasets/Falah/Alzheimer_MRI}
}
```

## 7. Hasil (Ringkasan)

Detail lengkap skor model per skenario tersedia di `results/model_scores/`, dan skor evaluasi XAI di `results/xai_scores/`. Visualisasi (confusion matrix, kurva loss/akurasi, heatmap XAI, histogram confidence) tersedia di `results/figures/`.

| Model | Skenario Terbaik | Accuracy | Precision | Recall | F1-score | ROC-AUC | Catatan |
|---|---|---|---|---|---|---|---|
| VGG19 | S-23 | 96,72 | 97,16 | 96,84 | 97,5 | 99,82 | Sampler, Full Tuning, AdamW, ReduceLR |
| ResNet50 | S-10 | 98,75 | 99,1 | 99,27 | 98,93 | 99,93 | Class Weight, Full Tuning, Adam, Cosine |
| EfficientNetB2 | S-24 | 99,14 | 99,34 | 99,48 | 99,2 | 99,98 | Sampler, Full Tuning, AdamW, Cosine |

## 8. Lisensi & Etika Penggunaan

- Dataset MRI yang digunakan bersifat publik/anonim dan digunakan hanya untuk kepentingan riset akademik.
- Model dan hasil pada repository ini **bukan alat diagnosis medis** dan tidak dimaksudkan untuk menggantikan penilaian tenaga medis profesional.
