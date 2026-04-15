import cv2
import numpy as np
import matplotlib.pyplot as plt
 
# =====================================================================
# PROBLEMA 2 - Partes A y C
# =====================================================================
 
ANSWER_KEY = {1:'C', 2:'B', 3:'A', 4:'D', 5:'B',
              6:'B', 7:'A', 8:'B', 9:'D', 10:'D'}
 
 
def encontrar_lineas_1d(arr, umbral):
    centros = []
    en_linea = False
    inicio = 0
    for i, val in enumerate(arr):
        if val and not en_linea:
            en_linea = True
            inicio = i
        elif not val and en_linea:
            en_linea = False
            centros.append((inicio + i) // 2)
    if en_linea:
        centros.append((inicio + len(arr)) // 2)
    return centros
 
 
def detectar_grilla(img):
    img_bin    = (img < 100).astype(np.uint8)
    suma_filas = np.sum(img_bin, axis=1)
    suma_cols  = np.sum(img_bin, axis=0)
    filas = encontrar_lineas_1d(suma_filas > 170, 170)
    cols  = encontrar_lineas_1d(suma_cols  > 400, 400)
    return filas, cols
 
 
def encontrar_grupos(suma, umbral):
    grupos = []
    en_grupo = False
    inicio = 0
    for i, v in enumerate(suma):
        if v > umbral and not en_grupo:
            en_grupo = True
            inicio = i
        elif v <= umbral and en_grupo:
            en_grupo = False
            grupos.append((inicio, i))
    if en_grupo:
        grupos.append((inicio, len(suma)))
    return grupos
 
 
def identificar_letra(bin_zona, stat):
    """
    Recorta la letra, re-binariza localmente y clasifica:
      B -> 2 agujeros internos (contornos hijos)
      C -> 0 agujeros
      A -> 1 agujero, fila del medio con travesaño (rellena)
      D -> 1 agujero, fila del medio con hueco (vacía en el centro)
    """
    x, y, w, h, area = stat
    pad = 2
    y1 = max(0, y - pad)
    y2 = min(bin_zona.shape[0], y + h + pad)
    x1 = max(0, x - pad)
    x2 = min(bin_zona.shape[1], x + w + pad)
    letra_crop = bin_zona[y1:y2, x1:x2]
 
    if letra_crop.size == 0:
        return 'C'
 
    # Re-binarizamos con Otsu para mejor contraste local
    _, letra_bin = cv2.threshold(letra_crop, 0, 255,
                                 cv2.THRESH_BINARY + cv2.THRESH_OTSU)
 
    # --- Contornos para detectar B (2 agujeros) ---
    contornos, jerarquia = cv2.findContours(letra_bin, cv2.RETR_TREE,
                                            cv2.CHAIN_APPROX_SIMPLE)
    n_hijos = 0
    if jerarquia is not None and len(contornos) > 0:
        cont_validos = [(i, c) for i, c in enumerate(contornos)
                        if cv2.contourArea(c) > 3]
        n_hijos = sum(1 for i, _ in cont_validos if jerarquia[0][i][3] != -1)
 
    if n_hijos >= 2:
        return 'B'
    if n_hijos == 0:
        return 'C'
 
    # --- 1 agujero: distinguir A de D ---
    # Recortamos al bounding box real de los píxeles (sin el padding)
    filas_con_px = np.any(letra_bin > 0, axis=1)
    cols_con_px  = np.any(letra_bin > 0, axis=0)
    if not np.any(filas_con_px) or not np.any(cols_con_px):
        return 'C'
 
    r_min, r_max = np.where(filas_con_px)[0][[0, -1]]
    c_min, c_max = np.where(cols_con_px)[0][[0, -1]]
    solo_letra = letra_bin[r_min:r_max+1, c_min:c_max+1]
 
    if solo_letra.size == 0:
        return 'C'
 
    # Normalizamos a 20x20 para análisis consistente
    norm = cv2.resize(solo_letra, (20, 20), interpolation=cv2.INTER_NEAREST)
 
    # Criterio: comparar la fila más ancha de la mitad sup vs la inferior
    # A: travesaño en la mitad baja -> max_inf >> max_sup
    # D: curva uniforme -> max_inf similar a max_sup
    fills   = [int(np.sum(norm[i] > 0)) for i in range(20)]
    max_sup = max(fills[:10]) if fills[:10] else 1
    max_inf = max(fills[10:]) if fills[10:] else 1
 
    if max_inf / max(max_sup, 1) > 1.5:
        return 'A'
    else:
        return 'D'
 
 
def extraer_respuesta_celda(img, y1, y2, x1, x2, debug=False):
    """
    1. Gap más grande -> separa enunciado de opciones
    2. Línea de guiones (_____) en el enunciado
    3. Letra con base a máx 5px del guión, solapando horizontalmente
    4. Clasifica recortando el bounding box de la letra
    """
    margen = 3
    celda  = img[y1+margen : y2-margen, x1+margen : x2-margen]
    if celda.size == 0:
        return None
 
    c_bin  = (celda < 128).astype(np.uint8)
    c_suma = np.sum(c_bin, axis=1)
 
    grupos = encontrar_grupos(c_suma, umbral=3)
    if len(grupos) < 2:
        return None
 
    gaps = [(grupos[i+1][0] - grupos[i][1], i) for i in range(len(grupos) - 1)]
    gap_max_tam, gap_max_idx = max(gaps, key=lambda x: x[0])
 
    if gap_max_tam < 10:
        return None
 
    fin_enunciado = grupos[gap_max_idx][1]
    zona_enun = celda[:fin_enunciado, :]
 
    _, bin_enun = cv2.threshold(zona_enun, 160, 255, cv2.THRESH_BINARY_INV)
    _, _, stats, _ = cv2.connectedComponentsWithStats(bin_enun, 8, cv2.CV_32S)
 
    # Línea de guiones: h<=2 y mayor área
    linea_guion = None
    for s in stats[1:]:
        x, y, w, h, area = s
        if h <= 2 and area > 40:
            if linea_guion is None or area > linea_guion[4]:
                linea_guion = s
 
    if linea_guion is None:
        if debug:
            print("    No se encontró línea de guiones")
        return None
 
    gx, gy, gw, gh, ga = linea_guion
    if debug:
        print(f"    Línea guiones: x={gx} y={gy} w={gw} area={ga}")
 
    candidatas = []
    for s in stats[1:]:
        x, y, w, h, area = s
        if h <= 2 or area < 5:
            continue
        solapa  = x < gx + gw and x + w > gx
        base_ok = gy - 5 <= y + h <= gy + 3
        if solapa and base_ok:
            candidatas.append(s)
 
    if debug:
        print(f"    Candidatas: {len(candidatas)}")
        for s in sorted(candidatas, key=lambda x: x[4], reverse=True):
            print(f"      x={s[0]} y={s[1]} w={s[2]} h={s[3]} area={s[4]}")
 
    if len(candidatas) == 0:
        return None
 
    candidatas_ord = sorted(candidatas, key=lambda s: s[4], reverse=True)
 
    if len(candidatas_ord) > 1:
        ratio = candidatas_ord[1][4] / candidatas_ord[0][4]
        if ratio > 0.6:
            if debug:
                print(f"    Múltiple (ratio={ratio:.2f})")
            return 'MULTIPLE'
 
    letra_stat = candidatas_ord[0]
    if debug:
        print(f"    Stat letra: {letra_stat}")
 
    return identificar_letra(bin_enun, letra_stat)
 
 
def corregir_examen(ruta_imagen, verbose=True, debug=False):
    img = cv2.imread(ruta_imagen, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Error: no se pudo leer '{ruta_imagen}'")
        return None
 
    filas, cols = detectar_grilla(img)
 
    if len(filas) < 6 or len(cols) < 4:
        print(f"Error: grilla no detectada en '{ruta_imagen}'")
        return None
 
    COLUMNAS = [(cols[0], cols[1]), (cols[2], cols[3])]
    OFFSET   = [0, 5]
 
    respuestas = {}
    for col_idx, (x1, x2) in enumerate(COLUMNAS):
        for fila_idx in range(1, len(filas) - 1):
            num_p = fila_idx + OFFSET[col_idx]
            if num_p > 10:
                break
            y1, y2 = filas[fila_idx], filas[fila_idx + 1]
            if debug:
                print(f"\n  P{num_p}:")
            respuestas[num_p] = extraer_respuesta_celda(
                img, y1, y2, x1, x2, debug=debug
            )
 
    correctas = 0
    resultado = {}
 
    if verbose:
        print(f"\n--- {ruta_imagen} ---")
 
    for num in range(1, 11):
        alumno   = respuestas.get(num)
        correcta = ANSWER_KEY[num]
        ok       = (alumno == correcta) and alumno is not None
 
        if ok:
            correctas += 1
 
        resultado[num] = {'alumno': alumno, 'correcta': correcta, 'ok': ok}
 
        if verbose:
            if alumno is None:
                marcado = 'vacío'
            elif alumno == 'MULTIPLE':
                marcado = 'múltiple'
            else:
                marcado = alumno
            print(f"  Pregunta {num:2d}: {'OK' if ok else 'MAL'}  "
                  f"(marcó: {marcado}, correcta: {correcta})")
 
    if verbose:
        print(f"  Correctas: {correctas}/10 -> "
              f"{'APROBADO' if correctas >= 6 else 'DESAPROBADO'}")
 
    return {
        'respuestas': resultado,
        'correctas':  correctas,
        'aprobado':   correctas >= 6
    }

#corregir_examen('examen_2.png', debug=True)
for ruta in ['examen_1.png','examen_2.png','examen_3.png','examen_4.png','examen_5.png']:
    corregir_examen(ruta, verbose=True)




def validar_punto_b(img):
    """
    Segmenta el encabezado en tres areas (Name, Date, Class) usando coordenadas fijas.
    Utiliza 'encontrar_grupos' para contabilizar caracteres y medir espacios entre ellos.
    Verifica restricciones de cantidad (Name <= 25, Date = 8, Class = 1) y separación de palabras.
    """

    alto_h = 25
    config = [
        {'nombre': 'Name',  'x': 60,  'y': 5, 'w': 180},
        {'nombre': 'Date',  'x': 295, 'y': 5, 'w': 75},
        {'nombre': 'Class', 'x': 420, 'y': 6, 'w': 35}
    ]

    for campo in config:
        x, y, w = campo['x'], campo['y'], campo['w']
        crop = img[y:y+alto_h, x:x+w]
        crop_bin = (crop < 120).astype(np.uint8)
        suma_cols = np.sum(crop_bin, axis=0)
        
        # Usamos un umbral bajo (2) porque las letras son finas
        caracteres = encontrar_grupos(suma_cols, umbral=2)
        
        cantidad = len(caracteres)
        res = "MAL"

        if campo['nombre'] == 'Name':
            # Buscamos espacios grandes entre grupos de letras
            espacios_grandes = 0
            for i in range(len(caracteres) - 1):
                gap = caracteres[i+1][0] - caracteres[i][1]
                if gap > 7: # Umbral de píxeles para considerar "espacio"
                    espacios_grandes += 1
            
            if espacios_grandes >= 1 and 2 <= cantidad <= 25:
                res = "OK"

        elif campo['nombre'] == 'Date':
            if cantidad == 8:
                res = "OK"

        elif campo['nombre'] == 'Class':
            if cantidad == 1:
                res = "OK"

        print(f"{campo['nombre']}: {res} ")


archivos = ['examen_1.png', 'examen_2.png', 'examen_3.png', 'examen_4.png', 'examen_5.png' ]
for nombre_archivo in archivos:
    img_test = cv2.imread(nombre_archivo, cv2.IMREAD_GRAYSCALE)
    print(f"\nResultados para {nombre_archivo}:")
    validar_punto_b(img_test)


