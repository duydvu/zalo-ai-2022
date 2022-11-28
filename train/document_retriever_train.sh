#!/bin/bash

python -m pyserini.index.lucene \
    --collection JsonCollection \
    --input data/pyserini2 \
    --language vi \
    --index indexes/paragraphs2 \
    --generator DefaultLuceneDocumentGenerator \
    --threads 20

python -m pyserini.index.lucene \
    --collection JsonCollection \
    --input data/pyserini_tokenized2 \
    --language vi \
    --index indexes/paragraphs_tokenized2 \
    --generator DefaultLuceneDocumentGenerator \
    --threads 20
