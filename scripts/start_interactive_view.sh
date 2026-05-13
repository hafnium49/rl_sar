#!/usr/bin/env bash
# Headless-but-interactive rl_sim_mujoco run for DGX Spark.
#
# Boots:
#   Xvfb :99      — virtual X server (large, so the MuJoCo viewer has room)
#   x11vnc        — mirrors Xvfb's framebuffer to RFB on :5900
#   websockify    — bridges WebSocket -> :5900 and serves noVNC on :6080
#   rl_sim_mujoco — the simulator, with DISPLAY=:99 and stdin from this terminal
#
# After it prints "noVNC ready", in another local terminal:
#   - VS Code Remote-SSH auto-forwards :6080. Click the URL in the Ports panel
#     OR open http://localhost:6080/vnc.html?host=localhost&port=6080
#   - Or via SSH tunnel: ssh -L 6080:localhost:6080 user@dgx-spark
#     then http://localhost:6080/vnc.html in your browser
#
# Keys (sent via the noVNC window's keyboard once connected):
#   0  Passive -> GetUp
#   1  GetUp -> RoboMimicLocomotion          (send AFTER "Getting up completed")
#   3  Locomotion -> WholeBodyTrackingDance102
#   9  back to GetDown / 0 back up
#   p  passive (kp=0, kd=8 damping)
#
# Ctrl+C this script when done; it cleans up Xvfb, x11vnc, websockify, binary.
set -u

REPO=/home/h_fujiwara/projects/rl_sar
DISPLAY_NUM=:99
W=1280
H=800
VNC_PORT=5900
WEB_PORT=6080

cleanup() {
    echo
    echo "[interactive] cleaning up..."
    [[ -n "${BIN_PID:-}" ]] && kill -INT "$BIN_PID" 2>/dev/null
    [[ -n "${WS_PID:-}"  ]] && kill -TERM "$WS_PID" 2>/dev/null
    [[ -n "${X11VNC_PID:-}" ]] && kill -TERM "$X11VNC_PID" 2>/dev/null
    [[ -n "${XVFB_PID:-}" ]] && kill -TERM "$XVFB_PID" 2>/dev/null
    sleep 1
    [[ -n "${BIN_PID:-}" ]] && kill -KILL "$BIN_PID" 2>/dev/null
    [[ -n "${XVFB_PID:-}" ]] && kill -KILL "$XVFB_PID" 2>/dev/null
    echo "[interactive] done."
}
trap cleanup EXIT INT TERM

# Sanity: required tools
for tool in Xvfb x11vnc websockify; do
    if ! command -v "$tool" >/dev/null; then
        echo "ERROR: $tool not installed. Run:"
        echo "  sudo apt-get install -y xvfb x11vnc novnc websockify"
        exit 1
    fi
done

if [[ ! -d /usr/share/novnc ]]; then
    echo "ERROR: /usr/share/novnc missing — install novnc:"
    echo "  sudo apt-get install -y novnc"
    exit 1
fi

# Free :99 if a stale Xvfb is still holding it
pkill -9 -x Xvfb 2>/dev/null && sleep 1 || true

echo "[interactive] Xvfb $DISPLAY_NUM at ${W}x${H}"
Xvfb $DISPLAY_NUM -screen 0 ${W}x${H}x24 >/dev/null 2>&1 &
XVFB_PID=$!
sleep 1

# Wait for it
for i in 1 2 3 4 5; do
    if xdpyinfo -display $DISPLAY_NUM >/dev/null 2>&1; then break; fi
    sleep 0.5
done

echo "[interactive] x11vnc -> :$VNC_PORT  (no password — local-only network)"
x11vnc -display $DISPLAY_NUM -forever -shared -nopw -quiet \
       -rfbport $VNC_PORT -bg \
       >/tmp/x11vnc.log 2>&1
X11VNC_PID=$(pgrep -f "x11vnc -display $DISPLAY_NUM" | head -1)

echo "[interactive] websockify+noVNC on :$WEB_PORT"
websockify --web=/usr/share/novnc $WEB_PORT localhost:$VNC_PORT \
           >/tmp/websockify.log 2>&1 &
WS_PID=$!
sleep 1

cat <<EOF

============================================================
 noVNC ready.
   Local browser:   http://localhost:$WEB_PORT/vnc.html?host=localhost&port=$WEB_PORT
   VS Code:         open http://localhost:$WEB_PORT/vnc.html in Simple Browser
                    (Remote-SSH auto-forwards :$WEB_PORT)
   SSH tunnel:      ssh -L $WEB_PORT:localhost:$WEB_PORT user@<this-host>

 Once connected, in the MuJoCo viewer:
   - click the X on each menu panel (File / Option / Joint / Control) to hide it
   - left-drag rotates the camera; right-drag pans; scroll zooms
   - then focus the viewer and press 0 then 1 then 3 to walk the FSM
   - Ctrl+C this script when done

============================================================

EOF

# Start the binary in this terminal (stdin = real PTY)
export DISPLAY=$DISPLAY_NUM
export LD_LIBRARY_PATH="$REPO/library/inference_runtime/libtorch/lib:$REPO/library/mujoco/lib:${LD_LIBRARY_PATH:-}"
echo "[interactive] launching rl_sim_mujoco g1 scene_29dof"
"$REPO/cmake_build/bin/rl_sim_mujoco" g1 scene_29dof &
BIN_PID=$!
wait "$BIN_PID"
