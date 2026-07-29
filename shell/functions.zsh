# transcribe-cli shell functions
# Source this from ~/.zshrc:
#   export TRANSCRIBE_CLI_HOME="$HOME/path/to/transcribe-cli"
#   source "$TRANSCRIBE_CLI_HOME/shell/functions.zsh"

# Python interpreter that has mlx-whisper + pyannote installed.
# Override in your shell if yours lives elsewhere.
WHISPERX_PYTHON="${WHISPERX_PYTHON:-$(command -v python3)}"
transcribe() {
  source ~/.whisperx_config
  # Strip any quotes or single trailing space that Mac drag-and-drop adds
  local file="${1//\'/}"
  file="${file% }"
  # Optional 2nd arg overrides LANGUAGE from the config for this run
  local lang="${2:-$LANGUAGE}"
  lang="${lang:-auto}"
  if [[ -z "$file" ]]; then
    echo "Usage: transcribe /path/to/audio/file [language]"
    echo "  language: en, zh, or auto (default: $LANGUAGE from ~/.whisperx_config)"
    echo "Tip: drag the file from Finder straight into this terminal window"
    return 1
  fi
  if [[ ! -f "$file" ]]; then
    echo "ERROR: File not found: $file"
    return 1
  fi
  # No .m4a conversion needed — the Python script decodes the audio itself,
  # which also avoids leaving huge .wav files behind in Downloads.
  if ! command -v ffmpeg &> /dev/null; then
    echo "ERROR: ffmpeg not installed. Run: brew install ffmpeg"
    return 1
  fi
  local LOG="$OUTPUT_DIR/whisperx_log.txt"
  echo "" > "$LOG"   # clear previous log
  echo "----------------------------------------"
  echo "File:     $file"
  echo "Language: $lang"
  echo "Speakers: $MIN_SPEAKERS-$MAX_SPEAKERS"
  echo "Output:   $OUTPUT_DIR"
  echo "Log:      $LOG"
  echo "----------------------------------------"
  echo "Running in background — you can close this window safely."
  echo "To watch live progress: logwatch"
  echo ""
  # The token is NOT passed here — the Python script reads it from the
  # Keychain itself. As an argument it would show up in the process list
  # and get written into the log in plain text.
  nohup "$WHISPERX_PYTHON" ${TRANSCRIBE_CLI_HOME:-$HOME/transcribe-cli}/transcribe_audio.py \
    "$file" \
    "$OUTPUT_DIR" \
    "$MIN_SPEAKERS" \
    "$MAX_SPEAKERS" \
    "$lang" \
    >> "$LOG" 2>&1 &
  local PID=$!
  echo "Started. PID: $PID"
  # Prevent Mac from sleeping until transcription finishes
  caffeinate -i &
  local CAFE_PID=$!
  echo "Caffeinate running (Mac won't sleep). PID: $CAFE_PID"
  # Watch for completion, stop caffeinate, and send Mac notification
  local filename=$(basename "$file")
  (
    while kill -0 $PID 2>/dev/null; do sleep 10; done
    kill $CAFE_PID 2>/dev/null
    if grep -q "Saved" "$LOG" 2>/dev/null; then
      osascript -e "display notification \"$filename is ready in Downloads\" with title \"Transcription complete ✓\" sound name \"Glass\""
    else
      osascript -e "display notification \"Check the log for details\" with title \"Transcription may have failed\" sound name \"Basso\""
    fi
  ) &
}
logwatch() {
  source ~/.whisperx_config 2>/dev/null
  local LOG="$OUTPUT_DIR/whisperx_log.txt"
  if [[ ! -f "$LOG" ]]; then
    echo "No log file found. Run transcribe first."
    return 1
  fi
  echo "Watching $LOG — press Ctrl+C to stop watching (transcription keeps running)"
  tail -f "$LOG"
}
_yt() {
  if [[ -z "$1" ]]; then
    echo "Usage: yt <youtube_url> [--workers N] [--output-dir DIR]"
    return 1
  fi
  python3 "${TRANSCRIBE_CLI_HOME:-$HOME/transcribe-cli}/transcribe_youtube.py" "$@"
}
alias yt='noglob _yt'
