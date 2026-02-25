#!/bin/bash

RENDER_URL="https://sentinelx-backend-rc6s.onrender.com/"
INTERVAL=600  # 10 minutes in seconds

echo "🔄 SentinelX Keep-Alive Started"
echo "📡 Pinging: $RENDER_URL every 10 minutes"
echo "Press CTRL+C to stop"
echo "─────────────────────────────────"

while true; do
    TIMESTAMP=$(date "+%H:%M:%S")
    STATUS=$(curl -s -o /dev/null -w "%{http_code}" --max-time 60 "$RENDER_URL")

    if [ "$STATUS" = "200" ]; then
        echo "✅ [$TIMESTAMP] Render is awake — HTTP $STATUS"
    else
        echo "⚠️  [$TIMESTAMP] Unexpected response — HTTP $STATUS"
    fi

    sleep $INTERVAL
done
