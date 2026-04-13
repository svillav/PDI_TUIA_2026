# PDI - Procesamiento Digital de Imágenes
## TUIA — Universidad Nacional de Rosario

---

## Estructura del repositorio

```
PDI/
├── README.md
├── requirements.txt
├── PDI_TUIA_2026/
│   ├── informe.pdf
│   ├── PROBLEMA1.py
│   ├── PROBLEMA2.py
│   ├── examen_1.png
│   ├── examen_2.png
│   ├── examen_3.png
│   ├── examen_4.png
│   ├── examen_5.png
```

---

## Requisitos

```bash
pip install -r requirements.txt
```

`requirements.txt`:
```
opencv-python
numpy
matplotlib
```

---

## Cómo ejecutar

### Problema 1

```bash
cd PDI_TUIA_2026
python PROBLEMA1.py
```
------------------


### Problema 2 

```bash
cd PDI_TUIA_2026
python PROBLEMA2.py
```

Imprime para cada examen:
```
--- examen_1.png ---
  Pregunta  1: OK  (marcó: C, correcta: C)
  Pregunta  2: MAL (marcó: vacío, correcta: B)
  ...
  Correctas: 6/10 -> APROBADO
```

Para ver el detalle interno de la detección:

```python
corregir_examen('examen_2.png', debug=True)
```

---

## Entorno virtual

```bash
python -m venv venv
source venv/bin/activate      # Linux/Mac
venv\Scripts\activate         # Windows
pip install -r requirements.txt
```
