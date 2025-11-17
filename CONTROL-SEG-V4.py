import cv2
import time
from collections import deque
from pythonosc import udp_client
import numpy as np

# ---------------- CONFIGURACIÓN BASE ----------------

# Captura de cámara (ajusta el índice si tu cámara no es 0)
cap = cv2.VideoCapture(1)

# Configura la IP y el puerto del receptor OSC
ip = "127.0.0.1"
puerto = 16284

# Comandos OSC (OBSBOT)
mover_arriba = "/OBSBOT/WebCam/General/SetGimbalUp"
mover_abajo = "/OBSBOT/WebCam/General/SetGimbalDown"
mover_izquierda = "/OBSBOT/WebCam/General/SetGimbalLeft"
mover_derecha = "/OBSBOT/WebCam/General/SetGimbalRight"
reset_posicion = "/OBSBOT/WebCam/General/ResetGimbal"
zoom_set = "/OBSBOT/WebCam/General/SetZoom"

# Cliente OSC
cliente = udp_client.SimpleUDPClient(ip, puerto)

# Estados y control PTZ
movimientos_activos = {}
rostros_presentes = False
ultima_accion = None  # evita órdenes repetidas mientras siguen activas
zoom_actual = 0 # estado de zoom actual
reset_enviado = False
timer_reset = 0.0

# Verificación de cámara
if not cap.isOpened():
    print("No se pudo acceder a la cámara.")
    raise SystemExit

# Clasificador Haar Cascade (asegúrate de que el archivo exista en tu ruta)
cascade_path = "haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(cascade_path)
if face_cascade.empty():
    print("Error al cargar Haar Cascade")
    raise SystemExit

# ORB y matcher
orb = cv2.ORB_create(nfeatures=500)
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

# --- NUEVO: Estado del Tracker ---
# Ya no usamos 'rostros_previos' por ID inestable.
# Usamos una lista de diccionarios, donde cada dict es una persona seguida.
rostros_seguidos = []  # Ejemplo: [{"track_id": 1, "box": (x,y,x2,y2), "descriptors": des, "lost_frames": 0, "keypoints": kp}]
next_track_id = 0
MAX_FRAMES_PERDIDOS = 15  # Nº de frames que un tracker puede sobrevivir sin detección
MIN_MATCHES_PARA_SEGUIR = 10 # Nº mínimo de matches ORB para asociar una detección a un track

# Suavizado del centro promedio (buffer de últimos 10 frames)
buffer_centros = deque(maxlen=10)

# Referencia de frame para overlays
display_frame_ref = None


# ---------------- FUNCIONES ----------------

def iniciar_movimiento(cliente, accion, duracion=0.2, velocidad=60):
    """Inicia un movimiento PTZ evitando órdenes repetidas."""
    global ultima_accion

    # Si la orden es igual a la última y aún está activa, no enviar
    if accion == ultima_accion and accion in movimientos_activos:
        # Evita saturar con órdenes repetidas
        return

    cliente.send_message(accion, velocidad)
    movimientos_activos[accion] = time.time() + duracion
    ultima_accion = accion


def actualizar_movimientos(cliente):
    """Detiene movimientos activos cuando expira su duración."""
    global ultima_accion
    ahora = time.time()
    acciones_finalizadas = [a for a, fin in movimientos_activos.items() if ahora >= fin]
    for accion in acciones_finalizadas:
        # Enviar velocidad 0 para detener
        cliente.send_message(accion, 0)
        del movimientos_activos[accion]
        if accion == ultima_accion:
            ultima_accion = None  # liberar para permitir nueva orden igual más adelante


def detectar_rostros(frame, face_cascade):
    """Detecta rostros y devuelve Bounding Boxes (x, y, w, h).
       --- MODIFICADO: Ya no asigna IDs ni dibuja ---"""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
    )
    # Devuelve solo la lista de (x, y, w, h)
    return faces


def extraer_caracteristicas_orb(frame, faces_raw):
    """Extrae keypoints/descriptores ORB para los rostros detectados.
       --- MODIFICADO: Ya no dibuja, solo extrae datos ---"""
    detecciones_orb = []
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    for (x, y, w, h) in faces_raw:
        (x1, y1, x2, y2) = (x, y, x + w, y + h)

        # Sanitizar ROI por si está fuera de límites
        x1c, y1c = max(0, x1), max(0, y1)
        x2c, y2c = min(frame.shape[1], x2), min(frame.shape[0], y2)
        roi = gray[y1c:y2c, x1c:x2c]
        if roi.size == 0:
            continue

        keypoints, descriptors = orb.detectAndCompute(roi, None)

        # Guardamos keypoints con offset para dibujarlos en el frame principal luego
        keypoints_offset = []
        if keypoints:
            for kp in keypoints:
                kp_offset = cv2.KeyPoint(kp.pt[0] + x1c, kp.pt[1] + y1c, kp.size)
                keypoints_offset.append(kp_offset)

        if descriptors is not None:
            deteccion_info = {
                "box": (x1c, y1c, x2c, y2c), # Box en coords del frame completo
                "keypoints": keypoints_offset, # Kp en coords del frame completo
                "descriptors": descriptors
            }
            detecciones_orb.append(deteccion_info)

    return detecciones_orb


# --- NUEVA FUNCIÓN DE SEGUIMIENTO ---
def actualizar_seguimiento(detecciones_frame_actual, rostros_seguidos_prev, bf, next_track_id):
    """
    Función principal de tracking. Empareja detecciones nuevas con tracks existentes.
    """
    global MIN_MATCHES_PARA_SEGUIR, MAX_FRAMES_PERDIDOS

    rostros_seguidos_actual = []
    detecciones_usadas = [False] * len(detecciones_frame_actual)

    # 1. Intentar emparejar tracks existentes
    for track in rostros_seguidos_prev:
        
        # Si el track se perdió, no buscarlo.
        if track["lost_frames"] > 0:
            track["lost_frames"] += 1
            if track["lost_frames"] < MAX_FRAMES_PERDIDOS:
                rostros_seguidos_actual.append(track) # Mantenerlo vivo un poco más
            continue # Ir al siguiente track
            
        mejores_matches = 0
        mejor_idx_det = -1
        
        # Comprobar este track contra todas las detecciones *no usadas*
        for i, det in enumerate(detecciones_frame_actual):
            if detecciones_usadas[i]:
                continue
                
            if track["descriptors"] is None or det["descriptors"] is None:
                continue
            
            if len(track["descriptors"]) == 0 or len(det["descriptors"]) == 0:
                continue

            try:
                matches = bf.match(track["descriptors"], det["descriptors"])
                # Opcional: filtrar matches por distancia
                # good_matches = [m for m in matches if m.distance < 70]
                num_matches = len(matches)

                if num_matches > mejores_matches and num_matches > MIN_MATCHES_PARA_SEGUIR:
                    mejores_matches = num_matches
                    mejor_idx_det = i
                    
            except cv2.error as e:
                # Error común si un descriptor está vacío
                # print(f"Error de BFMatcher: {e}")
                continue

        # 2. Actualizar el track si se encontró un match
        if mejor_idx_det != -1:
            det_coincidente = detecciones_frame_actual[mejor_idx_det]
            
            # Actualizar el track con los nuevos datos
            track["box"] = det_coincidente["box"]
            track["keypoints"] = det_coincidente["keypoints"]
            # Suavizado de descriptores (opcional, por ahora solo actualizamos)
            track["descriptors"] = det_coincidente["descriptors"]
            track["lost_frames"] = 0 # Reseteamos el contador de frames perdidos
            
            rostros_seguidos_actual.append(track)
            detecciones_usadas[mejor_idx_det] = True
        else:
            # 3. Marcar el track como 'perdido' por este frame
            track["lost_frames"] += 1
            # Solo lo mantenemos si no ha superado el límite de frames perdidos
            if track["lost_frames"] < MAX_FRAMES_PERDIDOS:
                rostros_seguidos_actual.append(track)
                
    # 4. Añadir nuevas detecciones como nuevos tracks
    for i, det in enumerate(detecciones_frame_actual):
        if not detecciones_usadas[i]:
            nuevo_track = {
                "track_id": next_track_id,
                "box": det["box"],
                "keypoints": det["keypoints"],
                "descriptors": det["descriptors"],
                "lost_frames": 0
            }
            rostros_seguidos_actual.append(nuevo_track)
            next_track_id += 1

    return rostros_seguidos_actual, next_track_id


# --- NUEVA FUNCIÓN DE DIBUJO ---
def dibujar_seguimiento(frame, rostros_seguidos):
    """Dibuja los Bounding Boxes y los Track IDs estables."""
    for track in rostros_seguidos:
        (x1, y1, x2, y2) = track["box"]
        color = (0, 255, 0) if track["lost_frames"] == 0 else (0, 0, 255) # Verde si activo, Rojo si perdido
        
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f"ID: {track['track_id']}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Dibujar keypoints si el track está activo
        if track["lost_frames"] == 0 and track["keypoints"] is not None:
            cv2.drawKeypoints(frame, track["keypoints"], frame, color=(0, 255, 0), flags=0)


def ajustar_encuadre_ptz(face_data_seguida, w_frame, h_frame, cliente,
                         zoom_set, mover_arriba, mover_abajo, mover_izquierda, mover_derecha, reset_posicion):
    """
    MODIFICADO: Esta función ahora recibe 'face_data_seguida', que es la
    lista estable de 'rostros_seguidos' (solo los activos).
    La lógica interna de la zona muerta y el zoom no cambia.
    """
    global rostros_presentes, display_frame_ref, zoom_actual, reset_enviado, timer_reset

    # Solo considerar rostros que no estén "perdidos"
    rostros_activos = [f for f in face_data_seguida if f.get("lost_frames", 0) == 0]

    # Sin rostros activos: enviar reset siempre (sin limitante)
    if not rostros_activos and reset_enviado == False:
        cliente.send_message(reset_posicion, 100)
        print ("No hay rostros activos. Enviando reset de posición.")
        rostros_presentes = False
        buffer_centros.clear() # Limpiar buffer si no hay nadie
        reset_enviado = True
        timer_reset = time.time()
        return
    elif not rostros_activos and time.time() - timer_reset > 1.0:
        reset_enviado = False
        return
    elif not rostros_activos:
        return  # Si no hay rostros activos, no hacer nada más

    # Hay rostros
    rostros_presentes = True

    # --- Centro promedio de los rostros (primeros 4) ---
    centros_x = [(f["box"][0] + f["box"][2]) // 2 for f in rostros_activos[:4]]
    centros_y = [(f["box"][1] + f["box"][3]) // 2 for f in rostros_activos[:4]]
    centro_promedio = (sum(centros_x) // len(centros_x), sum(centros_y) // len(centros_y))

    # --- Suavizado con buffer (10 frames) ---
    buffer_centros.append(centro_promedio)
    centro_suavizado = (
        sum(c[0] for c in buffer_centros) // len(buffer_centros),
        sum(c[1] for c in buffer_centros) // len(buffer_centros)
    )

    # --- Zona muerta central ---
    zona_central_x = (int(w_frame * 0.35), int(w_frame * 0.65))
    zona_central_y = (int(h_frame * 0.35), int(h_frame * 0.65))

    # Decisión de movimiento por salida de la zona muerta
    mover = None
    if centro_suavizado[0] < zona_central_x[0]:
        mover = mover_izquierda
    elif centro_suavizado[0] > zona_central_x[1]:
        mover = mover_derecha
    elif centro_suavizado[1] < zona_central_y[0]:
        mover = mover_arriba
    elif centro_suavizado[1] > zona_central_y[1]:
        mover = mover_abajo
    print(f"Centro suavizado: {centro_suavizado}, Mover: {mover}")
    
    # --- Bounding box conjunto para ratio/zoom ---
    min_x = min(f["box"][0] for f in rostros_activos)
    min_y = min(f["box"][1] for f in rostros_activos)
    max_x = max(f["box"][2] for f in rostros_activos)
    max_y = max(f["box"][3] for f in rostros_activos)

    ancho_rostros = max_x - min_x
    alto_rostros = max_y - min_y
    ratio = max(ancho_rostros / w_frame, alto_rostros / h_frame)

    # --- Zoom progresivo base [0..30] (conservador para encuadrar a múltiples personas) ---
    zoom_value = int(30 - (ratio * 30))
    zoom_value = max(0, min(30, zoom_value))

    # --- Ajuste adicional: proximidad a bordes (zoom-out preventivo) ---
    margen = 80  # umbral de proximidad a bordes en píxeles
    dist_izq = min(f["box"][0] for f in rostros_activos)
    dist_der = min(w_frame - f["box"][2] for f in rostros_activos)
    dist_arr = min(f["box"][1] for f in rostros_activos)
    dist_aba = min(h_frame - f["box"][3] for f in rostros_activos)
    proximidad = min(dist_izq, dist_der, dist_arr, dist_aba)

    aplicar_zoom_out = True
    # Excepción: si solo hay un rostro y su centro está dentro de la zona muerta, no aplicar preventivo
    if len(rostros_activos) == 1:
        centro_unico = ((rostros_activos[0]["box"][0] + rostros_activos[0]["box"][2]) // 2,
                        (rostros_activos[0]["box"][1] + rostros_activos[0]["box"][3]) // 2)
        dentro_zona_muerta = (zona_central_x[0] <= centro_unico[0] <= zona_central_x[1] and
                              zona_central_y[0] <= centro_unico[1] <= zona_central_y[1])
        if dentro_zona_muerta:
            aplicar_zoom_out = False

    if aplicar_zoom_out and proximidad < margen:
        factor_out = max(0.0, (margen - proximidad) / margen)  # 0..1
        zoom_out_target = int(30 - 20 * (1 - factor_out))      # empuja hacia ~10..30 según proximidad
        zoom_value = min(zoom_value, max(10, zoom_out_target))

    
    # --- Velocidad adaptativa (mínimo = 10 para movimientos suaves con ratio bajo) ---
    velocidad = 20 if ratio < 0.20 else 70

    # Enviar movimiento respetando control de órdenes repetidas
    if mover:
        iniciar_movimiento(cliente, mover, duracion=0.15, velocidad=velocidad)

    # Enviar zoom
    if zoom_value < zoom_actual + 4 or zoom_value > zoom_actual - 4: 
        # Cambio mínimo de 4 para evitar saturación de órdenes
        cliente.send_message(zoom_set, zoom_actual)
    else:
        cliente.send_message(zoom_set, zoom_value)
        zoom_actual = zoom_value
    
    print(f"Zoom ajustado a: {zoom_value}")


    # Overlays de ayuda visual
    if display_frame_ref is not None:
        cv2.rectangle(
            display_frame_ref,
            (zona_central_x[0], zona_central_y[0]),
            (zona_central_x[1], zona_central_y[1]),
            (0, 255, 255), 1
        )
        cv2.circle(display_frame_ref, centro_suavizado, 4, (0, 255, 255), -1)


# ---------------- BUCLE PRINCIPAL (MODIFICADO) ----------------
try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Fin de captura.")
            break

        # Referencia para overlays
        display_frame_ref = frame
        h_frame, w_frame = frame.shape[:2]

        # 1) Detección de rostros (solo BBoxes)
        faces_raw = detectar_rostros(frame, face_cascade)

        # 2) ORB sobre los rostros detectados (devuelve dets con BBox, Kp, Desc)
        nuevas_detecciones = extraer_caracteristicas_orb(frame, faces_raw)

        # 3) Actualizar seguimiento: Emparejar detecciones con tracks existentes
        #    Esta es la función que realmente "integra" ORB en el seguimiento.
        rostros_seguidos, next_track_id = actualizar_seguimiento(
            nuevas_detecciones, 
            rostros_seguidos, 
            bf, 
            next_track_id
        )

        # 4) Dibujar los resultados del SEGUIMIENTO (IDs estables)
        dibujar_seguimiento(frame, rostros_seguidos)

        # 5) Ajuste PTZ basado en los rostros SEGUIDOS (no los detectados)
        #    La función ajustar_encuadre_ptz usará esta lista estable.
        ajustar_encuadre_ptz(
            rostros_seguidos, # <-- AHORA USAMOS LA LISTA ESTABLE
            w_frame, h_frame, cliente,
            zoom_set, mover_arriba, mover_abajo, mover_izquierda, mover_derecha, reset_posicion
        )

        # Actualizar movimientos activos (no bloqueante)
        actualizar_movimientos(cliente)

        # Mostrar ventana
        cv2.imshow('Cámara USB: Haar + ORB Tracker + PTZ adaptativo (zoom 0..30)', frame)

        # Salir con 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Salida solicitada por el usuario.")
            break

finally:
    cap.release()
    cv2.destroyAllWindows()