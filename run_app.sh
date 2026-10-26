#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKEND_PORT="${BACKEND_PORT:-8000}"
FRONTEND_PORT="${FRONTEND_PORT:-5173}"
LOG_DIR="$ROOT_DIR/.run"
BACKEND_LOG="$LOG_DIR/backend.log"
FRONTEND_LOG="$LOG_DIR/frontend.log"

mkdir -p "$LOG_DIR"

BACKEND_PID=""
FRONTEND_PID=""

cleanup() {
  echo ""
  echo "Stopping app..."
  if [ -n "$BACKEND_PID" ]; then
    kill "$BACKEND_PID" 2>/dev/null || true
  fi
  if [ -n "$FRONTEND_PID" ]; then
    kill "$FRONTEND_PID" 2>/dev/null || true
  fi
}

trap cleanup EXIT INT TERM

kill_port() {
  local port="$1"
  local pids

  pids="$(lsof -ti tcp:"$port" 2>/dev/null || true)"
  if [ -z "$pids" ]; then
    return
  fi

  echo "Stopping existing process on port $port: $pids"
  kill $pids 2>/dev/null || true
  sleep 1

  pids="$(lsof -ti tcp:"$port" 2>/dev/null || true)"
  if [ -n "$pids" ]; then
    echo "Force stopping process on port $port: $pids"
    kill -9 $pids 2>/dev/null || true
  fi
}

find_npm() {
  if [ -x "$ROOT_DIR/frontend/.nodeenv-full/bin/npm" ]; then
    echo "$ROOT_DIR/frontend/.nodeenv-full/bin/npm"
    return
  fi

  if command -v npm >/dev/null 2>&1; then
    command -v npm
    return
  fi

  echo ""
}

wait_for_url() {
  local url="$1"
  local name="$2"
  local attempts=30

  for _ in $(seq 1 "$attempts"); do
    if curl -fsS "$url" >/dev/null 2>&1; then
      echo "$name is ready: $url"
      return 0
    fi
    sleep 1
  done

  echo "$name did not become ready at $url"
  return 1
}

NPM_BIN="$(find_npm)"
if [ -z "$NPM_BIN" ]; then
  echo "npm was not found."
  echo "Install Node.js/npm, or recreate the local runtime with:"
  echo "  python3 -m pip install nodeenv"
  echo "  python3 -m nodeenv --node=lts frontend/.nodeenv-full"
  exit 1
fi
NPM_DIR="$(dirname "$NPM_BIN")"
export PATH="$NPM_DIR:$PATH"

if [ ! -d "$ROOT_DIR/frontend/node_modules" ]; then
  echo "Frontend dependencies are missing."
  echo "Run this once before starting the app:"
  echo "  cd frontend"
  echo "  npm install"
  exit 1
fi

kill_port "$BACKEND_PORT"
kill_port "$FRONTEND_PORT"

echo "Starting backend on port $BACKEND_PORT..."
cd "$ROOT_DIR"
python3 -m uvicorn backend.main:app --host 0.0.0.0 --port "$BACKEND_PORT" >"$BACKEND_LOG" 2>&1 &
BACKEND_PID="$!"

echo "Starting frontend on port $FRONTEND_PORT..."
cd "$ROOT_DIR/frontend"
"$NPM_BIN" run dev >"$FRONTEND_LOG" 2>&1 &
FRONTEND_PID="$!"

echo "$BACKEND_PID" > "$LOG_DIR/backend.pid"
echo "$FRONTEND_PID" > "$LOG_DIR/frontend.pid"

wait_for_url "http://127.0.0.1:$BACKEND_PORT/api/health" "Backend"
wait_for_url "http://127.0.0.1:$FRONTEND_PORT" "Frontend"

LOCAL_URL="http://127.0.0.1:$FRONTEND_PORT"
NETWORK_IP="$(ipconfig getifaddr en0 2>/dev/null || true)"

echo ""
echo "Chest X-Ray Disease Detection is running."
echo "Open this URL on this computer:"
echo "  $LOCAL_URL"

if [ -n "$NETWORK_IP" ]; then
  echo ""
  echo "Open this URL on a phone on the same Wi-Fi:"
  echo "  http://$NETWORK_IP:$FRONTEND_PORT"
fi

echo ""
echo "Backend health:"
echo "  http://127.0.0.1:$BACKEND_PORT/api/health"
echo ""
echo "Logs:"
echo "  Backend:  $BACKEND_LOG"
echo "  Frontend: $FRONTEND_LOG"
echo ""
echo "Leave this terminal open. Press Ctrl+C to stop the app."
echo ""
tail -f "$BACKEND_LOG" "$FRONTEND_LOG"
