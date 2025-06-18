#!/usr/bin/env bash
set -e

echo "1. Удаляем устаревший пакет pinecone-client..."
pip uninstall -y pinecone-client || true

echo "2. Удаляем старый pinecone, если есть конфликтные версии..."
pip uninstall -y pinecone || true

echo "3. Устанавливаем свежую версию pinecone с gRPC..."
pip install "pinecone[grpc]" --upgrade

echo "📦 Установка завершена. Проверяем..."
pip show pinecone
