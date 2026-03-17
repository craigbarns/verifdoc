# VerifDoc — AI-Powered Document Fraud Detection

**Détection automatisée de falsification de documents pour PME/TPE**

VerifDoc analyse vos documents (bulletins de paie, avis d'imposition, RIB, factures) et détecte les falsifications en quelques secondes grâce à un pipeline d'analyse forensique multi-couches.

## 🔍 Pipeline de détection

| Couche | Technique | Ce qu'elle détecte |
|--------|-----------|-------------------|
| 1. ELA | Error Level Analysis | Zones modifiées (Photoshop, éditeur PDF) |
| 2. Bruit | Analyse wavelet | Incohérences de texture/compression |
| 3. Copy-Move | ORB + RANSAC | Zones dupliquées/clonées |
| 4. Métadonnées | Analyse PDF/EXIF | Logiciel de création, timestamps suspects |
| 5. OCR + Validation | EasyOCR + règles métier | Cohérence des montants, formats, SIRET |

## 🚀 Démarrage rapide

```bash
# Installation
pip install -r requirements.txt

# Lancer l'API
uvicorn verifdoc.api.main:app --reload --port 8000

# Lancer le dashboard (UI premium, pipeline parallèle, rapport HTML exportable)
streamlit run dashboard.py

# Tester
curl -X POST http://localhost:8000/api/v1/analyze \
  -F "file=@mon_document.pdf"
```

## 📁 Structure du projet

```
verifdoc/
├── analyzers/          # Modules d'analyse
│   ├── ela.py          # Error Level Analysis
│   ├── noise.py        # Analyse wavelet du bruit
│   ├── copy_move.py    # Détection copy-move (ORB+RANSAC)
│   ├── metadata.py     # Analyse métadonnées PDF/image
│   ├── ocr.py          # Extraction OCR
│   └── cross_check.py  # Validation croisée métier
├── api/
│   └── main.py         # API FastAPI
├── utils/
│   └── pdf_handler.py  # Conversion PDF → images
├── pipeline.py         # Orchestrateur principal
└── scoring.py          # Moteur de scoring
dashboard.py            # Interface Streamlit
```

## 📊 Modèle économique

- **Gratuit** : 20 vérifications/mois
- **Pro** (99€/mois) : 200 vérifications
- **Business** (299€/mois) : illimité + API

## 📜 Licence

MIT — Fork, modifie, commercialise.
