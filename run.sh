#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"
mkdir -p out
javac -d out src/vectordb/*.java
exec java -cp out vectordb.Main
