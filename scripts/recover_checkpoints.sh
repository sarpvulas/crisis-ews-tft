#!/bin/bash
# Recover RA-TFT X_combo checkpoints from exited RunPod pod
# Requires: credits added to RunPod account

POD_ID="476d35ccdw2jq5"
LOCAL_DIR="checkpoints/recovered"

# The key is never stored in this file. Take it from the environment, or fall
# back to the macOS Keychain.
API_KEY="${RUNPOD_API_KEY:-$(security find-generic-password \
  -s dev-command/crisis-ews-tft/RUNPOD_API_KEY -a "$USER" -w 2>/dev/null)}"

if [ -z "$API_KEY" ]; then
  echo "RUNPOD_API_KEY is not set and no Keychain entry was found." >&2
  echo "Store one with:" >&2
  echo "  security add-generic-password -U \\" >&2
  echo "    -s dev-command/crisis-ews-tft/RUNPOD_API_KEY -a \"\$USER\" -w" >&2
  exit 1
fi

set -e

echo "Step 1: Resuming pod $POD_ID..."
RESUME_RESULT=$(curl -s -H "Content-Type: application/json" \
  -H "Authorization: Bearer $API_KEY" \
  "https://api.runpod.io/graphql" \
  -d "{\"query\": \"mutation { podResume(input: {podId: \\\"$POD_ID\\\", gpuCount: 1}) { id desiredStatus } }\"}")
echo "$RESUME_RESULT"

if echo "$RESUME_RESULT" | grep -q "errors"; then
    echo "Resume failed. Check balance and try again."
    exit 1
fi

echo "Step 2: Waiting 90s for pod to boot..."
sleep 90

echo "Step 3: Getting connection info..."
POD_INFO=$(curl -s -H "Content-Type: application/json" \
  -H "Authorization: Bearer $API_KEY" \
  "https://api.runpod.io/graphql" \
  -d "{\"query\":\"{ pod(input: {podId: \\\"$POD_ID\\\"}) { runtime { ports { ip isIpPublic publicPort privatePort type } } } }\"}")
echo "$POD_INFO"

IP=$(echo "$POD_INFO" | python3 -c "import sys, json; data=json.load(sys.stdin); [print(p['ip']) for p in data['data']['pod']['runtime']['ports'] if p['privatePort']==22 and p['isIpPublic']]")
PORT=$(echo "$POD_INFO" | python3 -c "import sys, json; data=json.load(sys.stdin); [print(p['publicPort']) for p in data['data']['pod']['runtime']['ports'] if p['privatePort']==22 and p['isIpPublic']]")

echo "SSH: $IP:$PORT"

echo "Step 4: Waiting 20s more for SSH daemon..."
sleep 20

echo "Step 5: Downloading RA-TFT checkpoints..."
mkdir -p "$LOCAL_DIR"
scp -o StrictHostKeyChecking=no -P $PORT -r \
  root@$IP:/workspace/tft-project/checkpoints/validation/X_combo_best_seed* \
  "$LOCAL_DIR/"

echo "Step 6: Downloading validation log..."
scp -o StrictHostKeyChecking=no -P $PORT \
  root@$IP:/workspace/validation.log \
  "$LOCAL_DIR/validation.log"

echo "Step 7: Terminating pod..."
curl -s -H "Content-Type: application/json" \
  -H "Authorization: Bearer $API_KEY" \
  "https://api.runpod.io/graphql" \
  -d "{\"query\": \"mutation { podTerminate(input: {podId: \\\"$POD_ID\\\"}) }\"}"

echo "Done! Checkpoints recovered to $LOCAL_DIR/"
