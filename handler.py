import runpod
import os
import websocket
import base64
import json
import uuid
import logging
import urllib.request
import urllib.parse
import binascii
import subprocess
import librosa
import shutil
import time
import boto3
from botocore.client import Config

# Configuración de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def truncate_base64_for_log(base64_str, max_length=50):
    """Trunca cadenas Base64 largas para visualización en logs."""
    if not base64_str:
        return "None"
    if len(base64_str) <= max_length:
        return base64_str
    return f"{base64_str[:max_length]}... (total {len(base64_str)} caracteres)"


server_address = os.getenv("SERVER_ADDRESS", "127.0.0.1")
client_id = str(uuid.uuid4())


def download_file_from_url(url, output_path):
    """Descarga un archivo desde una URL usando wget."""
    try:
        result = subprocess.run(
            ["wget", "-O", output_path, "--no-verbose", "--timeout=30", url],
            capture_output=True,
            text=True,
            timeout=60,
        )

        if result.returncode == 0:
            logger.info(
                f"✅ Archivo descargado exitosamente desde URL: {url} -> {output_path}"
            )
            return output_path
        else:
            logger.error(f"❌ Fallo en la descarga con wget: {result.stderr}")
            raise Exception(f"Fallo al descargar URL: {result.stderr}")
    except subprocess.TimeoutExpired:
        logger.error("❌ Tiempo de espera agotado al descargar archivo")
        raise Exception("Tiempo de descarga agotado")
    except Exception as e:
        logger.error(f"❌ Error durante la descarga: {e}")
        raise Exception(f"Error durante la descarga: {e}")


def save_base64_to_file(base64_data, temp_dir, output_filename):
    """Decodifica y guarda datos Base64 en un archivo local."""
    try:
        decoded_data = base64.b64decode(base64_data)
        os.makedirs(temp_dir, exist_ok=True)
        file_path = os.path.abspath(os.path.join(temp_dir, output_filename))
        with open(file_path, "wb") as f:
            f.write(decoded_data)

        logger.info(f"✅ Entrada Base64 guardada en '{file_path}'.")
        return file_path
    except (binascii.Error, ValueError) as e:
        logger.error(f"❌ Fallo al decodificar Base64: {e}")
        raise Exception(f"Fallo al decodificar Base64: {e}")


def process_input(input_data, temp_dir, output_filename, input_type):
    """Procesa los datos de entrada según su tipo y devuelve la ruta local del archivo."""
    if input_type == "path":
        logger.info(f"📁 Procesando entrada de ruta local: {input_data}")
        return input_data
    elif input_type == "url":
        logger.info(f"🌐 Procesando entrada desde URL: {input_data}")
        os.makedirs(temp_dir, exist_ok=True)
        # Preservar la extensión original del archivo desde la URL si existe
        parsed_path = urllib.parse.urlparse(input_data).path
        _, ext = os.path.splitext(parsed_path)
        if ext and not output_filename.endswith(ext.lower()):
            base_name, _ = os.path.splitext(output_filename)
            output_filename = f"{base_name}{ext}"
        file_path = os.path.abspath(os.path.join(temp_dir, output_filename))
        return download_file_from_url(input_data, file_path)
    elif input_type == "base64":
        logger.info("🔢 Procesando entrada en Base64")
        return save_base64_to_file(input_data, temp_dir, output_filename)
    else:
        raise Exception(f"Tipo de entrada no soportado: {input_type}")


def queue_prompt(prompt, input_type="image", person_count="single"):
    """Envía el flujo de trabajo (workflow) a la cola de ComfyUI."""
    url = f"http://{server_address}:8188/prompt"
    logger.info(f"Encolando prompt en: {url}")
    p = {"prompt": prompt, "client_id": client_id}
    data = json.dumps(p).encode("utf-8")

    # Logging para depuración del workflow
    logger.info(f"Total de nodos en el workflow: {len(prompt)}")
    if input_type == "image":
        logger.info(
            f"Configuración nodo de imagen (284): {prompt.get('284', {}).get('inputs', {}).get('image', 'NO_ENCONTRADO')}"
        )
    else:
        logger.info(
            f"Configuración nodo de video (228): {prompt.get('228', {}).get('inputs', {}).get('video', 'NO_ENCONTRADO')}"
        )
    logger.info(
        f"Configuración nodo de audio (125): {prompt.get('125', {}).get('inputs', {}).get('audio', 'NO_ENCONTRADO')}"
    )
    logger.info(
        f"Configuración nodo de texto (241): {prompt.get('241', {}).get('inputs', {}).get('positive_prompt', 'NO_ENCONTRADO')}"
    )
    if person_count == "multi":
        if "307" in prompt:
            logger.info(
                f"Configuración segundo nodo de audio (307): {prompt.get('307', {}).get('inputs', {}).get('audio', 'NO_ENCONTRADO')}"
            )
        elif "313" in prompt:
            logger.info(
                f"Configuración segundo nodo de audio (313): {prompt.get('313', {}).get('inputs', {}).get('audio', 'NO_ENCONTRADO')}"
            )

    req = urllib.request.Request(url, data=data)
    req.add_header("Content-Type", "application/json")

    try:
        response = urllib.request.urlopen(req)
        result = json.loads(response.read())
        logger.info(f"Prompt enviado exitosamente: {result}")
        return result
    except urllib.error.HTTPError as e:
        error_body = e.read().decode("utf-8")
        logger.error(f"Error HTTP ocurrido: {e.code} - {e.reason}")
        logger.error(f"Cuerpo de la respuesta: {error_body}")
        raise Exception(f"ComfyUI HTTP {e.code} Error ({e.reason}): {error_body}")
    except Exception as e:
        logger.error(f"Error al enviar el prompt: {e}")
        raise


def get_image(filename, subfolder, folder_type):
    """Obtiene una imagen generada desde el endpoint /view de ComfyUI."""
    url = f"http://{server_address}:8188/view"
    logger.info(f"Obteniendo imagen desde: {url}")
    data = {"filename": filename, "subfolder": subfolder, "type": folder_type}
    url_values = urllib.parse.urlencode(data)
    with urllib.request.urlopen(f"{url}?{url_values}") as response:
        return response.read()


def get_history(prompt_id):
    """Obtiene el historial de ejecución del prompt desde ComfyUI."""
    url = f"http://{server_address}:8188/history/{prompt_id}"
    logger.info(f"Consultando historial en: {url}")
    with urllib.request.urlopen(url) as response:
        return json.loads(response.read())


def get_videos(ws, prompt, input_type="image", person_count="single"):
    """Ejecuta el workflow mediante WebSocket y recopila las rutas de los videos generados."""
    prompt_id = queue_prompt(prompt, input_type, person_count)["prompt_id"]
    logger.info(f"Iniciando ejecución del workflow: prompt_id={prompt_id}")

    output_videos = {}
    while True:
        out = ws.recv()
        if isinstance(out, str):
            message = json.loads(out)
            if message["type"] == "executing":
                data = message["data"]
                if data["node"] is not None:
                    logger.info(f"Ejecutando nodo: {data['node']}")
                if data["node"] is None and data["prompt_id"] == prompt_id:
                    logger.info("Ejecución del workflow completada exitosamente")
                    break
        else:
            continue

    logger.info(f"Consultando historial para prompt_id={prompt_id}")
    history = get_history(prompt_id)[prompt_id]
    logger.info(f"Nodos de salida encontrados: {len(history['outputs'])}")

    for node_id in history["outputs"]:
        node_output = history["outputs"][node_id]
        videos_output = []
        if "gifs" in node_output:
            logger.info(
                f"Nodo {node_id}: encontrados {len(node_output['gifs'])} videos"
            )
            for idx, video in enumerate(node_output["gifs"]):
                video_path = video["fullpath"]
                logger.info(f"Ruta de archivo de video: {video_path}")

                if os.path.exists(video_path):
                    file_size = os.path.getsize(video_path)
                    logger.info(
                        f"Video {idx+1} encontrado: {video_path} (Tamaño: {file_size} bytes)"
                    )
                else:
                    logger.warning(f"El archivo de video no existe: {video_path}")

                videos_output.append(video_path)
        else:
            logger.info(f"Nodo {node_id} sin salida de videos")
        output_videos[node_id] = videos_output

    logger.info(f"Rutas de videos recopiladas de {len(output_videos)} nodos")
    return output_videos


def load_workflow(workflow_path):
    """Carga la definición del workflow desde un archivo JSON."""
    with open(workflow_path, "r", encoding="utf-8") as file:
        return json.load(file)


def get_workflow_path(input_type, person_count):
    """Devuelve la ruta del archivo de workflow correspondiente según input_type y person_count."""
    if input_type == "image":
        if person_count == "single":
            return "/I2V_single.json"
        else:  # multi
            return "/I2V_multi.json"
    else:  # video
        if person_count == "single":
            return "/V2V_single.json"
        else:  # multi
            return "/V2V_multi.json"


def get_audio_duration(audio_path):
    """Devuelve la duración de un archivo de audio en segundos."""
    try:
        duration = librosa.get_duration(path=audio_path)
        return duration
    except Exception as e:
        logger.warning(f"No se pudo calcular la duración del audio ({audio_path}): {e}")
        return None


def calculate_max_frames_from_audio(wav_path, wav_path_2=None, fps=25):
    """Calcula max_frames según la duración del archivo de audio más largo."""
    durations = []

    # Duración del primer audio
    duration1 = get_audio_duration(wav_path)
    if duration1 is not None:
        durations.append(duration1)
        logger.info(f"Duración del primer audio: {duration1:.2f}s")

    # Duración del segundo audio (en caso de multi-persona)
    if wav_path_2:
        duration2 = get_audio_duration(wav_path_2)
        if duration2 is not None:
            durations.append(duration2)
            logger.info(f"Duración del segundo audio: {duration2:.2f}s")

    if not durations:
        logger.warning("No se pudo determinar la duración del audio. Usando valor por defecto (81 frames).")
        return 81

    # Calcular los frames exactos necesarios para el audio
    import math
    max_duration = max(durations)
    needed_frames = int(math.ceil(max_duration * fps))

    # InfiniteTalk requiere al menos 81 frames para el bloque inicial
    if needed_frames <= 81:
        max_frames = 81
    else:
        # Wan2.1 requiere que (frames - 1) sea múltiplo de 4 para alineación temporal VAE
        remainder = (needed_frames - 1) % 4
        if remainder != 0:
            max_frames = needed_frames + (4 - remainder)
        else:
            max_frames = needed_frames

    logger.info(
        f"Duración máxima de audio: {max_duration:.2f}s -> frames exactos requeridos: {needed_frames}, "
        f"max_frames alineados con Wan2.1: {max_frames}"
    )
    return max_frames


def handler(job):
    job_input = job.get("input", {})

    # Logging del payload de entrada (truncando Base64 si existe)
    log_input = job_input.copy()
    for key in ["image_base64", "video_base64", "wav_base64", "wav_base64_2"]:
        if key in log_input:
            log_input[key] = truncate_base64_for_log(log_input[key])

    logger.info(f"Trabajo recibido con entrada: {log_input}")
    task_id = f"task_{uuid.uuid4()}"

    # Verificación de tipo de entrada y cantidad de personas
    input_type = job_input.get("input_type", "image")  # "image" o "video"
    person_count = job_input.get("person_count", "single")  # "single" o "multi"

    logger.info(f"Tipo de workflow: {input_type}, Modo de personas: {person_count}")

    # Determinar ruta del workflow
    workflow_path = get_workflow_path(input_type, person_count)
    logger.info(f"Archivo de workflow seleccionado: {workflow_path}")

    # Determinar directorio de trabajo temporal (preferir /ComfyUI/input para compatibilidad nativa)
    comfy_input_dir = "/ComfyUI/input"
    temp_dir = comfy_input_dir if os.path.isdir(comfy_input_dir) else os.path.abspath(task_id)
    os.makedirs(temp_dir, exist_ok=True)

    # Procesar archivo multimedia de entrada (imagen o video)
    media_path = None
    if input_type == "image":
        if "image_path" in job_input:
            media_path = process_input(
                job_input["image_path"], temp_dir, f"{task_id}_input_image.jpg", "path"
            )
        elif "image_url" in job_input:
            media_path = process_input(
                job_input["image_url"], temp_dir, f"{task_id}_input_image.jpg", "url"
            )
        elif "image_base64" in job_input:
            media_path = process_input(
                job_input["image_base64"], temp_dir, f"{task_id}_input_image.jpg", "base64"
            )
        else:
            media_path = "/examples/image.jpg"
            logger.info("Usando imagen predeterminada: /examples/image.jpg")
    else:  # video
        if "video_path" in job_input:
            media_path = process_input(
                job_input["video_path"], temp_dir, f"{task_id}_input_video.mp4", "path"
            )
        elif "video_url" in job_input:
            media_path = process_input(
                job_input["video_url"], temp_dir, f"{task_id}_input_video.mp4", "url"
            )
        elif "video_base64" in job_input:
            media_path = process_input(
                job_input["video_base64"], temp_dir, f"{task_id}_input_video.mp4", "base64"
            )
        else:
            media_path = "/examples/image.jpg"
            logger.info("Usando medio predeterminado: /examples/image.jpg")

    # Procesar archivo de audio principal
    wav_path = None
    wav_path_2 = None  # Segundo audio para multi-persona

    if "wav_path" in job_input:
        wav_path = process_input(
            job_input["wav_path"], temp_dir, f"{task_id}_input_audio.wav", "path"
        )
    elif "wav_url" in job_input:
        wav_path = process_input(
            job_input["wav_url"], temp_dir, f"{task_id}_input_audio.wav", "url"
        )
    elif "wav_base64" in job_input:
        wav_path = process_input(
            job_input["wav_base64"], temp_dir, f"{task_id}_input_audio.wav", "base64"
        )
    else:
        wav_path = "/examples/audio.mp3"
        logger.info("Usando audio predeterminado: /examples/audio.mp3")

    # Procesar segundo audio si es multi-persona
    if person_count == "multi":
        if "wav_path_2" in job_input:
            wav_path_2 = process_input(
                job_input["wav_path_2"], temp_dir, f"{task_id}_input_audio_2.wav", "path"
            )
        elif "wav_url_2" in job_input:
            wav_path_2 = process_input(
                job_input["wav_url_2"], temp_dir, f"{task_id}_input_audio_2.wav", "url"
            )
        elif "wav_base64_2" in job_input:
            wav_path_2 = process_input(
                job_input["wav_base64_2"], temp_dir, f"{task_id}_input_audio_2.wav", "base64"
            )
        else:
            wav_path_2 = wav_path
            logger.info("Segundo audio no especificado, reutilizando primer audio.")

    # Parámetros y valores predeterminados
    prompt_text = job_input.get("prompt", "A person talking naturally")
    width = job_input.get("width", 512)
    height = job_input.get("height", 512)
    fps = job_input.get("fps", 25)
    steps = job_input.get("steps")

    # Configuración de max_frame (calculado automáticamente si no se provee)
    max_frame = job_input.get("max_frame")
    if max_frame is None:
        logger.info(
            "max_frame no fue especificado. Calculando automáticamente en base al audio..."
        )
        max_frame = calculate_max_frames_from_audio(
            wav_path, wav_path_2 if person_count == "multi" else None, fps=fps
        )
    else:
        logger.info(f"max_frame especificado por el usuario: {max_frame}")

    logger.info(
        f"Parámetros del workflow: prompt='{prompt_text}', width={width}, height={height}, fps={fps}, max_frame={max_frame}"
    )
    logger.info(f"Ruta de medio de entrada: {media_path}")
    logger.info(f"Ruta de audio principal: {wav_path}")
    if person_count == "multi":
        logger.info(f"Ruta de segundo audio: {wav_path_2}")

    prompt = load_workflow(workflow_path)

    # Limpiar nodos huérfanos no conectados que fallan en la validación de ComfyUI
    for orphan_id in ["300", "306"]:
        if orphan_id in prompt and prompt[orphan_id].get("class_type") == "Wav2VecModelLoader":
            prompt.pop(orphan_id, None)

    # ------------------------------------------------------------------
    # Configuración dinámica de Force Offload y Steps ----
    # ------------------------------------------------------------------
    force_offload = job_input.get("force_offload", True)
    logger.info(f"🔧 Configuración: force_offload={force_offload}")

    sampler_node_id = None
    preferred_id = "128"

    if preferred_id in prompt and prompt[preferred_id].get("class_type") == "WanVideoSampler":
        sampler_node_id = preferred_id
    else:
        for node_id, node_data in prompt.items():
            if node_data.get("class_type") == "WanVideoSampler":
                sampler_node_id = node_id
                break

    if sampler_node_id:
        inputs = prompt[sampler_node_id].setdefault("inputs", {})
        inputs["force_offload"] = force_offload
        if steps is not None:
            inputs["steps"] = int(steps)
            logger.info(f"✅ Nodo {sampler_node_id} (WanVideoSampler) configurado con steps={steps}")
        logger.info(f"✅ Nodo {sampler_node_id} (WanVideoSampler) configurado con force_offload={force_offload}")
    else:
        logger.warning("⚠️ Advertencia: Nodo WanVideoSampler no encontrado. Se usarán valores predeterminados del workflow.")
    # ------------------------------------------------------------------

    # Validar existencia de archivos locales antes de encolar en ComfyUI
    if not os.path.exists(media_path):
        logger.error(f"El archivo multimedia no existe: {media_path}")
        return {"error": f"Archivo multimedia no encontrado: {media_path}"}

    if not os.path.exists(wav_path):
        logger.error(f"El archivo de audio no existe: {wav_path}")
        return {"error": f"Archivo de audio no encontrado: {wav_path}"}

    if person_count == "multi" and wav_path_2 and not os.path.exists(wav_path_2):
        logger.error(f"El segundo archivo de audio no existe: {wav_path_2}")
        return {"error": f"Segundo archivo de audio no encontrado: {wav_path_2}"}

    logger.info(f"Tamaño de archivo multimedia: {os.path.getsize(media_path)} bytes")
    logger.info(f"Tamaño de archivo de audio: {os.path.getsize(wav_path)} bytes")
    if person_count == "multi" and wav_path_2:
        logger.info(f"Tamaño de segundo archivo de audio: {os.path.getsize(wav_path_2)} bytes")

    # Inyección de parámetros en los nodos de ComfyUI
    if input_type == "image":
        prompt["284"]["inputs"]["image"] = media_path
    else:
        prompt["228"]["inputs"]["video"] = media_path

    prompt["125"]["inputs"]["audio"] = wav_path
    prompt["241"]["inputs"]["positive_prompt"] = prompt_text
    prompt["245"]["inputs"]["value"] = width
    prompt["246"]["inputs"]["value"] = height
    prompt["270"]["inputs"]["value"] = max_frame

    # Ajuste dinámico de FPS en los nodos correspondientes
    for node_id, node_data in prompt.items():
        c_type = node_data.get("class_type")
        if c_type == "MultiTalkWav2VecEmbeds" and "fps" in node_data.get("inputs", {}):
            node_data["inputs"]["fps"] = fps
        elif c_type == "VHS_VideoCombine" and "frame_rate" in node_data.get("inputs", {}):
            node_data["inputs"]["frame_rate"] = fps

    if person_count == "multi":
        if input_type == "image":
            if "307" in prompt:
                prompt["307"]["inputs"]["audio"] = wav_path_2
        else:
            if "313" in prompt:
                prompt["313"]["inputs"]["audio"] = wav_path_2

    ws_url = f"ws://{server_address}:8188/ws?clientId={client_id}"
    logger.info(f"Conectando a WebSocket: {ws_url}")

    # Verificar disponibilidad HTTP del servidor ComfyUI
    http_url = f"http://{server_address}:8188/"
    logger.info(f"Verificando conexión HTTP en: {http_url}")

    max_http_attempts = 180
    for http_attempt in range(max_http_attempts):
        try:
            response = urllib.request.urlopen(http_url, timeout=5)
            logger.info(f"Conexión HTTP establecida (intento {http_attempt+1})")
            break
        except Exception as e:
            logger.warning(
                f"Esperando respuesta HTTP (intento {http_attempt+1}/{max_http_attempts}): {e}"
            )
            if http_attempt == max_http_attempts - 1:
                raise Exception(
                    "No se pudo conectar al servidor ComfyUI. Verifique que el servicio esté corriendo."
                )
            time.sleep(1)

    ws = websocket.WebSocket()
    max_attempts = int(180 / 5)
    for attempt in range(max_attempts):
        try:
            ws.connect(ws_url)
            logger.info(f"Conexión WebSocket establecida (intento {attempt+1})")
            break
        except Exception as e:
            logger.warning(f"Esperando WebSocket (intento {attempt+1}/{max_attempts}): {e}")
            if attempt == max_attempts - 1:
                raise Exception("Tiempo de espera agotado al conectar por WebSocket (3 min)")
            time.sleep(5)

    videos = get_videos(ws, prompt, input_type, person_count)
    ws.close()
    logger.info("Conexión WebSocket cerrada")

    # =========================================================================
    # OBJETIVO 1: Selección inteligente del video final
    # =========================================================================
    logger.info("🔍 Recopilando y evaluando videos de salida de todos los nodos...")
    all_videos = []
    for node_id, file_list in videos.items():
        for file_path in file_list:
            if file_path and os.path.exists(file_path):
                all_videos.append(file_path)

    logger.info(f"Total de videos generados encontrados: {len(all_videos)}")
    for v in all_videos:
        logger.info(f"  - Candidato: {v} ({os.path.getsize(v)} bytes)")

    if not all_videos:
        logger.error("❌ No se encontraron archivos de video generados válidos.")
        return {"error": "No se encontraron videos de salida generados."}

    # Prioridad 1: Seleccionar el archivo que termine en -audio.mp4 (video ensamblado con audio)
    audio_videos = [v for v in all_videos if v.lower().endswith("-audio.mp4")]

    if audio_videos:
        selected_video_path = max(audio_videos, key=os.path.getsize)
        logger.info(
            f"🎯 Video seleccionado [Prioridad 1: Video ensamblado con audio]: {selected_video_path} "
            f"({os.path.getsize(selected_video_path)} bytes)"
        )
    else:
        # Prioridad 2: Seleccionar el archivo .mp4 con mayor tamaño en bytes
        mp4_videos = [v for v in all_videos if v.lower().endswith(".mp4")]
        if mp4_videos:
            selected_video_path = max(mp4_videos, key=os.path.getsize)
            logger.info(
                f"🎯 Video seleccionado [Prioridad 2: Mayor tamaño .mp4]: {selected_video_path} "
                f"({os.path.getsize(selected_video_path)} bytes)"
            )
        else:
            selected_video_path = max(all_videos, key=os.path.getsize)
            logger.info(
                f"🎯 Video seleccionado [Fallback: Mayor tamaño general]: {selected_video_path} "
                f"({os.path.getsize(selected_video_path)} bytes)"
            )

    # Verificación final de existencia del archivo seleccionado
    if not os.path.exists(selected_video_path):
        logger.error(f"❌ El archivo de video seleccionado no existe: {selected_video_path}")
        return {"error": f"El archivo de video seleccionado no existe: {selected_video_path}"}

    # =========================================================================
    # OBJETIVO 2: Subida directa a S3 / Cloudflare R2 con boto3
    # =========================================================================
    s3_endpoint = (
        job_input.get("s3_endpoint")
        or os.getenv("S3_ENDPOINT")
        or os.getenv("S3_ENDPOINT_URL")
    )
    s3_bucket = (
        job_input.get("s3_bucket")
        or os.getenv("S3_BUCKET")
        or os.getenv("S3_BUCKET_NAME")
    )
    s3_access_key = (
        job_input.get("s3_access_key")
        or os.getenv("S3_ACCESS_KEY")
        or os.getenv("S3_ACCESS_KEY_ID")
        or os.getenv("AWS_ACCESS_KEY_ID")
    )
    s3_secret_key = (
        job_input.get("s3_secret_key")
        or os.getenv("S3_SECRET_KEY")
        or os.getenv("S3_SECRET_ACCESS_KEY")
        or os.getenv("AWS_SECRET_ACCESS_KEY")
    )
    s3_public_domain = (
        job_input.get("s3_public_domain")
        or os.getenv("S3_PUBLIC_DOMAIN")
    )
    s3_region = (
        job_input.get("s3_region")
        or os.getenv("S3_REGION", "auto")
    )

    # Validar credenciales de S3
    if not all([s3_endpoint, s3_bucket, s3_access_key, s3_secret_key]):
        missing = []
        if not s3_endpoint:
            missing.append("s3_endpoint")
        if not s3_bucket:
            missing.append("s3_bucket")
        if not s3_access_key:
            missing.append("s3_access_key")
        if not s3_secret_key:
            missing.append("s3_secret_key")

        error_msg = f"Faltan credenciales requeridas de S3: {', '.join(missing)}"
        logger.error(f"❌ {error_msg}")
        return {"error": error_msg}

    try:
        s3_key = f"outputs/infinitetalk_{task_id}.mp4"
        file_size_mb = os.path.getsize(selected_video_path) / (1024 * 1024)
        logger.info(
            f"🚀 Subiendo video a S3: {selected_video_path} ({file_size_mb:.2f} MB) -> "
            f"s3://{s3_bucket}/{s3_key}"
        )

        s3_client = boto3.client(
            "s3",
            endpoint_url=s3_endpoint,
            aws_access_key_id=s3_access_key,
            aws_secret_access_key=s3_secret_key,
            region_name=s3_region if s3_region != "auto" else None,
            config=Config(signature_version="s3v4"),
        )

        # Content-Type para reproducción nativa en navegadores
        extra_args = {"ContentType": "video/mp4"}
        s3_client.upload_file(
            selected_video_path,
            s3_bucket,
            s3_key,
            ExtraArgs=extra_args,
        )
        logger.info(f"✅ Video subido exitosamente a S3: s3://{s3_bucket}/{s3_key}")

        # Construir la URL pública final
        if s3_public_domain:
            domain = s3_public_domain.strip().rstrip("/")
            if not domain.startswith(("http://", "https://")):
                domain = f"https://{domain}"
            video_url = f"{domain}/{s3_key.lstrip('/')}"
        else:
            endpoint = s3_endpoint.strip().rstrip("/")
            video_url = f"{endpoint}/{s3_bucket}/{s3_key.lstrip('/')}"
        # Limpieza de archivos temporales del trabajo para evitar saturar el disco del worker
        try:
            temp_dir_path = os.path.abspath(task_id)
            if os.path.exists(temp_dir_path):
                shutil.rmtree(temp_dir_path, ignore_errors=True)
                logger.info(f"🧹 Directorio temporal '{task_id}' limpiado con éxito.")
        except Exception as cleanup_err:
            logger.warning(f"No se pudo limpiar el directorio temporal: {cleanup_err}")

        logger.info(f"🎉 URL pública final: {video_url}")
        return {"video_url": video_url}

    except Exception as e:
        logger.error(f"❌ Error al subir el video a S3: {e}")
        return {"error": f"Error al subir el video a S3: {str(e)}"}


runpod.serverless.start({"handler": handler})
