#!/bin/bash
set -e

cd "$(dirname "$0")"
source venv/bin/activate

echo "🚀 Starting Stanford CoreNLP server..."
CORPUS_DIR="$HOME/.stanfordnlp_resources/stanford-corenlp-4.5.7"
if [ ! -d "$CORPUS_DIR" ]; then
  echo "Downloading CoreNLP..."
  mkdir -p "$HOME/.stanfordnlp_resources"
  wget -q https://nlp.stanford.edu/software/stanford-corenlp-4.5.7.zip -O /tmp/corenlp.zip
  unzip -q /tmp/corenlp.zip -d "$HOME/.stanfordnlp_resources"
  rm /tmp/corenlp.zip
fi

# start CoreNLP in background if not already running
if ! nc -z 127.0.0.1 9020; then
  java -Xmx4g -cp "$CORPUS_DIR/*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer \
      --port 9020 --timeout 500000 \
      --annotators tokenize,ssplit,pos,lemma,ner,parse,depparse,coref \
      --preload tokenize,ssplit,pos,lemma,ner,parse,depparse,coref \
      --coref.algorithm neural \
      > corenlp.log 2>&1 &
  echo "✓ CoreNLP started (port 9020)"
fi

echo "🧠 Running notebook..."
nohup papermill combined.ipynb combined_out.ipynb --log-output > run.log 2>&1 &

echo "📜 Logs will stream here:"
echo "   tail -f run.log"

