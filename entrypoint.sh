#!/bin/bash

# Salir inmediatamente si algún comando falla
set -e

# Iniciar ComfyUI en segundo plano
echo "Iniciando ComfyUI en segundo plano..."
python /ComfyUI/main.py --listen --use-sage-attention &

# Esperar a que ComfyUI esté listo
echo "Esperando que ComfyUI esté listo para recibir peticiones..."
max_wait=120  # Espera máxima de 2 minutos
wait_count=0
while [ $wait_count -lt $max_wait ]; do
    if curl -s http://127.0.0.1:8188/ > /dev/null 2>&1; then
        echo "✅ ComfyUI está listo!"
        break
    fi
    echo "Esperando ComfyUI... ($wait_count/$max_wait)"
    sleep 2
    wait_count=$((wait_count + 2))
done

if [ $wait_count -ge $max_wait ]; then
    echo "❌ Error: ComfyUI no inició dentro de los $max_wait segundos"
    exit 1
fi

# Iniciar el handler en primer plano sin buffer para logs en tiempo real
echo "Iniciando el handler de RunPod Serverless..."
exec python -u handler.py