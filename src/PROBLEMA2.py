import cv2
import numpy as np
import matplotlib.pyplot as plt


ANSWER_KEY = {1:'C', 2:'B', 3:'A', 4:'D', 5:'B',
              6:'B', 7:'A', 8:'B', 9:'D', 10:'D'}


# Parte A ----------------
def encontrar_lineas_1d(arr):
    '''
    Analiza un array unidimensional (como una fila o columna de pixeles)
    para encontrar donde hay segmentos de pixeles 
    y determinar el centro exacto de cada uno.
    '''
    # guardamos los puntos centrales en una lista
    centros = []
    en_linea = False
    inicio = 0
    # iniciamos un bucle donde analizamos cada elemento del array, 
    for i, val in enumerate(arr):
        # verifica si hay un pixel y si no estamos en una linea, detectando el inicio de un segmento
        if val and not en_linea:
            en_linea = True
            inicio = i
        # detectamos el final del segmento
        elif not val and en_linea:
            en_linea = False
            # calcula el punto emdio del inicio y el final y agrega a la lista de centros
            centros.append((inicio + i) // 2)
    # si el valor sigue siendo True, significa que la ultima linea llegaba hasta el final
    if en_linea:
        centros.append((inicio + len(arr)) // 2)
    return centros


def encontrar_grupos(arr):
    '''
    Devuelve las coordenadas de inicio y fin de cada segmento encontrado.
    Identifica el "ancho" o la "altura" de los elementos en la imagen.
    '''
    # creamos una lista para almacenar los inicio-fin
    # un bool para detectar si estamos dentro de un segmento de pixeles 
    # y una variable que tiene el indice donde comienza el grupo
    grupos = []
    en_grupo = False
    inicio = 0
    # recorremos el array ingresado y si encontramos un pixel pero no estabamos en un grupo 
    # marca el comienzo de uno nuevo
    for i, v in enumerate(arr):
        if v and not en_grupo:
            en_grupo = True
            inicio = i
        # si el pixel es false indica fin de un segmento y guarda la coord para indicar el fin
        elif not v and en_grupo:
            en_grupo = False
            grupos.append((inicio, i))
    # si sigue en true es porque es el ultimo grupo 
    if en_grupo:
        grupos.append((inicio, len(arr)))
    return grupos


def detectar_grilla(img_bin):
    '''
    Se encarga de localizar la estructura de la tabla en el examen,
    eliminando todo el ruido para quedarse solo con las lineas de la grilla.
    Recibe una imagen ya binarizada.
    '''
    h_img, w_img = img_bin.shape # alto y ancho de la imagen

    # longitud para un filtro horizontal (un 30% del ancho de la imagen)
    largo_h = int(w_img * 0.3) 
    # definimos un rectangulo de 1px de alto para detectar lineas horiz
    kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (largo_h, 1))
    # aplicamos erosion que elimina cualquier objeto mas pequenio que el molde
    lineas_h = cv2.erode(img_bin, kernel_h)

    # hacemos lo mismo para el ancho usando el alto 
    largo_v = int(h_img * 0.3)
    kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, largo_v))
    lineas_v = cv2.erode(img_bin, kernel_v)

    # sumamos los piceles para obtener el perfil de intensidad (vertical y horizontal)
    suma_filas = np.sum(lineas_h, axis=1)
    suma_cols  = np.sum(lineas_v, axis=0)

    # llamamos a la funcion que creamos para encontrar los centros de las lineas
    filas = encontrar_lineas_1d(suma_filas > 0)
    cols  = encontrar_lineas_1d(suma_cols  > 0)
    return filas, cols


def identificar_letra(letra_bin, w, h):
    '''
    Reconoce que letra fue escrita utilizando geometria y topologia para decidir
    '''
    # busca las formas en la imagen, RETR_TREE logra detectar si hay contornos dentro de otros
    contornos, jerarquia = cv2.findContours(letra_bin, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # primer control de C, si no encuentra forma devuelve una C 
    if jerarquia is None: 
        return 'C'
    
    # guardamos un contador de huecos internos que tiene una letra
    # si un contorno tiene un padre es porque es un agujero
    n_agujeros = sum(1 for h_info in jerarquia[0] if h_info[3] != -1)

    # si tiene dos, es una B
    if n_agujeros >= 2:
        return 'B'
    # segundo control de C, si no tiene ninguno es una C
    if n_agujeros == 0:
        return 'C'
    
    # si tiene 1 hueco puede ser una A o una D entonces hacemos un analisis mas profundo
    # encontramos las filas y columnas con pixeles blancos 
    rows = np.any(letra_bin > 0, axis=1)
    cols_px = np.any(letra_bin > 0, axis=0)
    # tercer control de C
    if not np.any(rows) or not np.any(cols_px):
        return 'C'
    
    # encontramos los limites exactos para eliminar el espacio alrededor de la letra y la guardamos
    r_min, r_max = np.where(rows)[0][[0, -1]]
    c_min, c_max = np.where(cols_px)[0][[0, -1]]
    solo_letra = letra_bin[r_min:r_max+1, c_min:c_max+1]

    # comparamos la anchura superior 
    # encontrando a fila mas ancha de mitad para arriba y la de mitad para abajo
    alto = solo_letra.shape[0]
    mitad = alto // 2
    max_sup = np.max(np.sum(solo_letra[:mitad] > 0, axis=1))
    max_inf = np.max(np.sum(solo_letra[mitad:] > 0, axis=1))

    # si el ancho de abajo es mucho mas ancha que la de arriba (un 1.5 veces mas)
    # entonces es una A, sino por descarte es una D
    if max_inf / max(max_sup, 1) > 1.5:
        return 'A'
    return 'D'


def extraer_respuesta_celda(celda_bin):
    '''
    Aisla el trazo de la letra elegida dentro de un recuadro de la grilla, 
    ignorando el texto del enunciado y el guion de respuesta
    '''
    # verificamos que no sea una imagen vacia
    if celda_bin.size == 0:
        return None

    # sumamos los px horizontal para detectar bandas de texto
    # identificamos los bloques y nos fijamos la cantidad de ellas
    suma_filas = np.sum(celda_bin, axis=1)
    bandas = encontrar_grupos(suma_filas > 0)
    # si hay menos de 2 bandas es porque no hay respuesta (siempre debe haber enunciado y linea resp)
    if len(bandas) < 2:
        return None

    # calculamos el espacio vacio entre bandas y 
    # buscamos el espacio mas grande que separa el area de respuesta
    gaps = [(bandas[i+1][0] - bandas[i][1], i) for i in range(len(bandas)-1)]
    _, idx_gap_max = max(gaps, key=lambda x: x[0])
    fin_enunciado = bandas[idx_gap_max][1]
    # guardamos un corte donde esta la respuesta y la linea de resp
    zona_enun = celda_bin[:fin_enunciado, :]

    # etiquetamos los objetos para obtener estadisticas
    n, _, stats, _ = cv2.connectedComponentsWithStats(zona_enun)
    alto_zona  = zona_enun.shape[0]
    ancho_zona = zona_enun.shape[1]

    # recorre los objetos para encontrar el que parece un la linea de respuesta
    linea_guion = None
    for i in range(1, n):
        h = stats[i, cv2.CC_STAT_HEIGHT]
        w = stats[i, cv2.CC_STAT_WIDTH]
        # con un criterio bajo (menos del 10% de alto) y suficientemente ancho (mas de 15% de ancho)
        if h <= alto_zona * 0.1 and w >= ancho_zona * 0.15:
            if linea_guion is None or stats[i, cv2.CC_STAT_AREA] > linea_guion[4]:
                linea_guion = stats[i]
    if linea_guion is None:
        return None

    gx, gy, gw, gh, _ = linea_guion # guardamos las coord de la linea 
    margen_vertical = int(alto_zona * 0.2) # que tan arriba de la linea buscar 
    candidatas = [] 

    # recorremos para encontrar la letra
    for i in range(1, n):
        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        # cualquier cosa que mida menos del 10% de alto lo ignoramos
        if h <= alto_zona * 0.1:
            continue
        # verificamos alineacion horizontal con el guion
        solapa = x < gx + gw and x + w > gx
        base_cerca = gy - margen_vertical <= y + h <= gy + 3
        # si cumple lo tomamos como posible respuesta
        if solapa and base_cerca:
            candidatas.append((stats[i], i))

    if len(candidatas) == 0:
        return None
    
    # ordenamso de mayor a menor
    candidatas.sort(key=lambda c: c[0][cv2.CC_STAT_AREA], reverse=True)
    # si hay mas de una candidata y la 2da es al menos un 60% del tamanio de la primera
    # usamos el criterio de multiple respuesta
    if len(candidatas) > 1:
        area_1 = candidatas[0][0][cv2.CC_STAT_AREA]
        area_2 = candidatas[1][0][cv2.CC_STAT_AREA]
        if area_2 / area_1 > 0.6:
            return 'MULTIPLE'

    s = candidatas[0][0]
    #guardamos las coordenadas de la candidata 1 mediante comandos opencv
    x = s[cv2.CC_STAT_LEFT]
    y = s[cv2.CC_STAT_TOP]
    w = s[cv2.CC_STAT_WIDTH]
    h = s[cv2.CC_STAT_HEIGHT]
    # hace un recorte de la letra candidata
    letra = zona_enun[y:y+h, x:x+w]
    # analizamos que letra es con la funcion anterior
    return identificar_letra(letra, w, h)




def corregir_examen(ruta_imagen, verbose=True):
    '''
    Coorina todo el proceso de abrir la imagen hasta evaluar las respuestas del alumno
    Ingresa una ruta de imagen y determina un booleano en true 
    '''
    # cargamos la imagen en escala de grises
    img = cv2.imread(ruta_imagen, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Error: no se pudo leer '{ruta_imagen}'")
        return None
    # pasamos a byn con otsu para separar el fondo del frente y 
    # el BINARY_INV para invertir los colores (contenido = blanco, fondo = negro)
    _, img_bin = cv2.threshold(img, 0, 1, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # detectamos coordenadas de las lineas 
    filas, cols = detectar_grilla(img_bin)
    # validacion de si no encuentra tantas lineas como para hacer grilla de 10 preguntas se detiene
    if len(filas) < 6 or len(cols) < 4:
        print(f"Error: grilla no detectada en '{ruta_imagen}'")
        return None
    
    #define limite izquierdo y derecho de las dos columas 
    columnas_preg = [(cols[0], cols[1]), (cols[2], cols[3])]
    respuestas = {}
    #recorremos las celdas, itera por las dos columnas de la grilla
    for col_idx, (x1, x2) in enumerate(columnas_preg):
        # itera por las filas salteando el encabezado
        for fila_idx in range(1, len(filas)-1):
            # calcula el numero de pregunta segun posicion en la tabla
            num_p = fila_idx + col_idx * 5
            if num_p > 10:
                break
            y1, y2 = filas[fila_idx], filas[fila_idx+1]
            # determina el recorte de la celda con un margen de 3px para adentro 
            celda = img_bin[y1+3:y2-3, x1+3:x2-3]
            # realizamos el recorte
            respuestas[num_p] = extraer_respuesta_celda(celda)

    correctas = 0
    resultado = {}
    
    # nos imprime en terminal todo el analisis, desde cuantas respuestas hizo bien 
    # el alumno hasta las dimensiones de la imagen
    if verbose:
        print(f"\n--- {ruta_imagen} ---")
    # un rang
    for num in range(1, 11):
        alumno   = respuestas.get(num)
        correcta = ANSWER_KEY[num]
        ok       = (alumno == correcta)
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
        'aprobado':   correctas >= 6,
        'img_bin':    img_bin,
        'filas':      filas,
        'cols':       cols
    }


# --------------------------------
def validar_encabezado(ruta_imagen):
    '''
    Validar que el alumno haya completado correctamente los datos personales (Nombre, Fecha y Clase)
    en la parte superior del examen.  
    '''
    # utilizamos la funcion que corrige los examenes para quedarnos con sus retornos
    res = corregir_examen(ruta_imagen, verbose=False)
    if res is None:
        return None
    img_bin = res['img_bin']
    filas   = res['filas']
    cols    = res['cols']

    # identificamos la priemra linea horizontal para usarla de limite inferior
    # recortamos usando el inicio de la imagen hasta esta primera linea.
    y_linea_relleno = filas[0]
    encabezado = img_bin[:y_linea_relleno+1, :]
    h_enc, w_enc = encabezado.shape # coordenadas del encabezado
    suma_cols_enc = np.sum(encabezado, axis=0)

    lineas_largas = []
    for fila in range(h_enc):
        # analizamos cada fila para encontrar segmentos horizontales de px
        grupos_fila = encontrar_grupos(encabezado[fila, :] > 0)
        # filtramos solo las lineas que tengan un 5% de ancho, descartando resto 
        for ini, fin in grupos_fila:
            ancho = fin - ini
            if ancho >= w_enc * 0.05:
                # guardamos la info relevante del encabezado 
                lineas_largas.append((fila, ini, fin, ancho))

    if not lineas_largas:
        return None

    # 
    fila_max = max(l[0] for l in lineas_largas)
    # intentamos agrupar las lineas a una altura similar
    lineas_relleno = [l for l in lineas_largas if l[0] >= fila_max - 2]
    # ordena las lineas de izq a der para asignarles el campo que corresponda
    lineas_relleno.sort(key=lambda l: l[1])

    # si hay menos de 3 lineas alineadas cambia de plan y toma las 3 mas largas y ordena primero por ancho 
    # luego ordena por mayor longitud de izq a der
    if len(lineas_relleno) < 3:
        lineas_relleno = sorted(lineas_largas, key=lambda l: -l[3])[:3]
        lineas_relleno.sort(key=lambda l: l[1])

    # si luego de intentar obtener las 3 mas largas, sigue sin haber al menos 3, pasa de largo
    if len(lineas_relleno) < 3:
        return None


    nombres_campos = ['Name', 'Date', 'Class']
    resultado = {}
    # itera en simultaneo los nombres y las lineas en orden 
    # se hace una busqueda del 80% del encabezadom mirando sobre la linea y recortamos 
    # si el recort esta vacio o no tiene px blancos lo toma como mal 
    for nombre, (y_l, x_ini, x_fin, _) in zip(nombres_campos, lineas_relleno[:3]):
        alto_recorte = int(h_enc * 0.8)
        y_top_recorte = max(0, y_l - alto_recorte)
        y_bot_recorte = max(0, y_l - 2)
        zona_texto = encabezado[y_top_recorte:y_bot_recorte, x_ini:x_fin]
        if zona_texto.size == 0 or zona_texto.sum() == 0:
            resultado[nombre] = 'MAL'
            continue

        # detectamos los caracteres de la zona recortada
        n, _, stats_c, _ = cv2.connectedComponentsWithStats(zona_texto)
        cantidad = n - 1
        ancho_zona = zona_texto.shape[1]

        # para el campo name verificamos que haya 2 componentes y hacemos un conteo de caracteres
        # para que haya mas de 2 pero menos de 25
        if nombre == 'Name':
            if cantidad < 2:
                resultado[nombre] = 'MAL'
                continue
            ranges_x = sorted([(stats_c[i, cv2.CC_STAT_LEFT],
                                stats_c[i, cv2.CC_STAT_LEFT] + stats_c[i, cv2.CC_STAT_WIDTH])
                                for i in range(1, n)], key=lambda r: r[0])
            espacios = [ranges_x[i+1][0] - ranges_x[i][1]
                        for i in range(len(ranges_x)-1)]
            espacios_palabras = sum(1 for e in espacios if e > ancho_zona * 0.03)
            es_ok = espacios_palabras >= 1 and 2 <= cantidad <= 25
            resultado[nombre] = 'OK' if es_ok else 'MAL'
            
        # colocamos el nombre = ok porque necesitamos que haya 3 lineas de guia siempre, 
        # para que el codigo siga y no falle por un desplazamiento de coordenadas
        #para la fecha tiene que haber justo 8 digitos 
        elif nombre == 'Date':
            resultado[nombre] = 'OK' if cantidad == 8 else 'MAL'
        #para la clase debe ser 1 solo caracter
        elif nombre == 'Class':
            resultado[nombre] = 'OK' if cantidad == 1 else 'MAL'

    # mostramos resultados en terminal
    print(f"\nEncabezado de {ruta_imagen}:")
    for k, v in resultado.items():
        print(f"  {k}: {v}")

    return resultado

#  ----------------------------------------
def generar_imagen_resultados(rutas_examenes):
    
    # creamos una ventana de visualizacion con una fila para cada alumno
    fig, axes = plt.subplots(len(rutas_examenes), 1, figsize=(6, 7))
    fig.suptitle("RESULTADOS FINALES", fontsize=12, fontweight='bold')

    # ejecuta la correccion sin imprimir en consola, sino para guardar los resultados
    for i, ruta in enumerate(rutas_examenes):
        res = corregir_examen(ruta, verbose=False)
        if res is None:
            continue
        img_bin = res['img_bin']
        filas   = res['filas']

        # localizamos el nombre del alumno del encabezado, imagen binarizada
        img = cv2.imread(ruta, cv2.IMREAD_GRAYSCALE)
        y_linea_relleno = filas[0]
        encabezado_bin = img_bin[:y_linea_relleno+1, :]
        h_enc, w_enc = encabezado_bin.shape
        lineas_largas = []
        for fila in range(h_enc):
            grupos_fila = encontrar_grupos(encabezado_bin[fila, :] > 0)
            for ini, fin in grupos_fila:
                if fin - ini >= w_enc * 0.05:
                    lineas_largas.append((fila, ini, fin))

        # realiza los recortes identificando las lineas largas del encabezado, 
        # obteniendo el inicio y fin de la primera
        if lineas_largas:
            fila_max = max(l[0] for l in lineas_largas)
            ranuras = sorted([l for l in lineas_largas if l[0] >= fila_max-2],
                             key=lambda l: l[1])
            if ranuras:
                x_ini = ranuras[0][1]
                x_fin = ranuras[0][2]
                crop_name = img[:y_linea_relleno+1, x_ini:x_fin]

        # si no detecta las lineas recortamos la mitad superior de la img para capturar igual
            else:
                crop_name = img[:y_linea_relleno+1, :w_enc//2]
        else:
            crop_name = img[:y_linea_relleno+1, :w_enc//2]

        # coloca un borde verde si aprobo, sino uno rojo.
        # coloca aprobado si aprobo sino desaprobado
        color = 'green' if res['aprobado'] else 'red'
        texto = "APROBADO" if res['aprobado'] else "DESAPROBADO"

        # estiliza un poco la interfaz con los recortes 
        axes[i].imshow(crop_name, cmap='gray')
        axes[i].set_ylabel(texto, color=color, fontweight='bold',
                           rotation=0, labelpad=50, va='center')
        axes[i].set_title(f"Archivo: {ruta}", loc='left', fontsize=9)
        axes[i].set_xticks([]), axes[i].set_yticks([])
        for spine in axes[i].spines.values():
            spine.set_edgecolor(color)
            spine.set_linewidth(2)

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    examenes = ['src/examen_1.png', 'src/examen_2.png', 'src/examen_3.png',
                'src/examen_4.png', 'src/examen_5.png']
    for ruta in examenes:
        corregir_examen(ruta, verbose=True)
    for ruta in examenes:
        validar_encabezado(ruta)
    generar_imagen_resultados(examenes)
