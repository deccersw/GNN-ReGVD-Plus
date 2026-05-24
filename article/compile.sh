#!/bin/bash
# Компиляция статьи (требуется TeX Live или MacTeX)
# Установка на macOS: brew install --cask mactex
# Установка на Ubuntu: sudo apt-get install texlive-full

pdflatex -interaction=nonstopmode article.tex
pdflatex -interaction=nonstopmode article.tex  # второй проход для ссылок

echo ""
echo "=== Готово: article.pdf ==="
