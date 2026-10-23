#!/bin/sh
# Однократная подготовка Debian/Ubuntu. Бот этот скрипт не вызывает.
set -eu

if ! command -v apt-get >/dev/null 2>&1; then
    echo "Нужна Debian/Ubuntu-подобная среда с apt-get." >&2
    exit 1
fi

if [ "$(id -u)" -ne 0 ]; then
    if command -v sudo >/dev/null 2>&1; then
        exec sudo sh "$0" "$@"
    fi
    echo "Запустите sh scripts/install_tesseract.sh от root (или через sudo)." >&2
    echo "Пакеты: tesseract-ocr tesseract-ocr-rus tesseract-ocr-eng." >&2
    exit 1
fi

apt-get update
apt-get install -y --no-install-recommends tesseract-ocr tesseract-ocr-rus tesseract-ocr-eng
tesseract --version
languages=$(tesseract --list-langs)
for language in rus eng; do
    if ! printf '%s\n' "$languages" | grep -qx "$language"; then
        echo "Язык $language недоступен. Проверьте TESSDATA_PREFIX." >&2
        exit 1
    fi
done
printf '%s\n' "$languages"
