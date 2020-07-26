#!/usr/bin/env bash

MESSAGE="Create Treebank token, form and lemma FastText embedding vectors."

abspath() {
  echo "$(cd "$(dirname "$1")" || exit; pwd)/$(basename "$1")"
}

log() {
  >&2 echo "INFO: $*"
}

## Parameters:
FT_ROOT=
FT_MODEL_TYPE="word"
FT_MODEL_FILE_NAME="cc.he.300.bin"
TB_TRAIN_TOKENS_FILE_PATH=
TB_TRAIN_GOLD_LATTICES_FILE_PATH=
TB_DEV_TOKENS_FILE_PATH=
TB_DEV_GOLD_LATTICES_FILE_PATH=
TB_TEST_TOKENS_FILE_PATH=
TB_TEST_GOLD_LATTICES_FILE_PATH=
OUTPUT_FOLDER_PATH=

usage() {
    >&2 echo "Usage: $MESSAGE"
    >&2 echo ""
    >&2 echo "arguments:"
    >&2 echo "t     Path to FastText root folder"
    >&2 echo "a     Path to input train tokens file"
    >&2 echo "b     Path to input train gold lattices file"
    >&2 echo "c     Path to input dev tokens file"
    >&2 echo "d     Path to input dev gold lattices file"
    >&2 echo "e     Path to input test tokens file"
    >&2 echo "f     Path to input test gold lattices file"
    >&2 echo "o     Path to output folder"
}

while getopts "r:t:a:b:c:d:e:f:o:h" opt; do
    case $opt in
      r)
      FT_ROOT=$OPTARG
        ;;
      t)
      FT_MODEL_TYPE=$OPTARG
        ;;
      a)
      TB_TRAIN_TOKENS_FILE_PATH=$OPTARG
        ;;
      b)
      TB_TRAIN_GOLD_LATTICES_FILE_PATH=$OPTARG
        ;;
      c)
      TB_DEV_TOKENS_FILE_PATH=$OPTARG
        ;;
      d)
      TB_DEV_GOLD_LATTICES_FILE_PATH=$OPTARG
        ;;
      e)
      TB_TEST_TOKENS_FILE_PATH=$OPTARG
        ;;
      f)
      TB_TEST_GOLD_LATTICES_FILE_PATH=$OPTARG
        ;;
      o)
      OUTPUT_FOLDER_PATH=$OPTARG
        ;;
      h)
      usage
      exit 1
        ;;
      *)
      echo $MESSAGE
      usage
      exit 1
      ;;
    esac
done

log "FastText root folder is $FT_ROOT"
log "FastText model type is $FT_MODEL_TYPE"
log "input treebank train tokens file path is $TB_TRAIN_TOKENS_FILE_PATH"
log "input treebank train gold lattices file path is $TB_TRAIN_GOLD_LATTICES_FILE_PATH"
log "input treebank dev tokens file path is $TB_DEV_TOKENS_FILE_PATH"
log "input treebank dev gold lattices file path is $TB_DEV_GOLD_LATTICES_FILE_PATH"
log "input treebank test tokens file path is $TB_TEST_TOKENS_FILE_PATH"
log "input treebank test gold lattices file path is $TB_TEST_GOLD_LATTICES_FILE_PATH"
log "output folder path is $OUTPUT_FOLDER_PATH"

TOKEN_FILE_PATH="$(abspath $OUTPUT_FOLDER_PATH/$FT_MODEL_TYPE-token.txt)"
FORM_FILE_PATH="$(abspath $OUTPUT_FOLDER_PATH/$FT_MODEL_TYPE-form.txt)"
LEMMA_FILE_PATH="$(abspath $OUTPUT_FOLDER_PATH/$FT_MODEL_TYPE-lemma.txt)"
TOKEN_VEC_FILE_PATH="$(abspath $OUTPUT_FOLDER_PATH/$FT_MODEL_TYPE-token.vec)"
FORM_VEC_FILE_PATH="$(abspath $OUTPUT_FOLDER_PATH/$FT_MODEL_TYPE-form.vec)"
LEMMA_VEC_FILE_PATH="$(abspath $OUTPUT_FOLDER_PATH/$FT_MODEL_TYPE-lemma.vec)"
echo '<PAD>' > "$TOKEN_FILE_PATH"
echo '<SOS>' >> "$TOKEN_FILE_PATH"
echo '<ET>' >> "$TOKEN_FILE_PATH"
cat "$TB_TRAIN_TOKENS_FILE_PATH" "$TB_DEV_TOKENS_FILE_PATH" "$TB_TEST_TOKENS_FILE_PATH" | sort | uniq | sed  1d >> "$TOKEN_FILE_PATH"
echo '<PAD>' > "$FORM_FILE_PATH"
echo '<SOS>' >> "$FORM_FILE_PATH"
echo '<ET>' >> "$FORM_FILE_PATH"
cat "$TB_TRAIN_GOLD_LATTICES_FILE_PATH" "$TB_DEV_GOLD_LATTICES_FILE_PATH" "$TB_TEST_GOLD_LATTICES_FILE_PATH" | cut -f3 | sort | uniq | sed 1d >> "$FORM_FILE_PATH"
echo '<PAD>' > "$LEMMA_FILE_PATH"
echo '<SOS>' >> "$LEMMA_FILE_PATH"
echo '<ET>' >> "$LEMMA_FILE_PATH"
cat "$TB_TRAIN_GOLD_LATTICES_FILE_PATH" "$TB_DEV_GOLD_LATTICES_FILE_PATH" "$TB_TEST_GOLD_LATTICES_FILE_PATH" | cut -f4 | sort | uniq | sed  1d >> "$LEMMA_FILE_PATH"
PAD=$(python -c 'print("{} {}".format("<PAD>", " ".join(["0.0"] * 300)))')
pushd "$FT_ROOT" || exit 1

if [[ $FT_MODEL_TYPE == "morpheme" ]]
then
  FT_MODEL_FILE_NAME="wikipedia.alt_tok.yap_form.fasttext_skipgram.model.bin"
fi
./fasttext print-word-vectors models/$FT_MODEL_FILE_NAME < "$TOKEN_FILE_PATH" | sed "1 s/^.*$/$PAD/" > "$TOKEN_VEC_FILE_PATH"
./fasttext print-word-vectors models/$FT_MODEL_FILE_NAME < "$FORM_FILE_PATH" | sed "1 s/^.*$/$PAD/" > "$FORM_VEC_FILE_PATH"
./fasttext print-word-vectors models/$FT_MODEL_FILE_NAME < "$LEMMA_FILE_PATH" | sed "1 s/^.*$/$PAD/" > "$LEMMA_VEC_FILE_PATH"
