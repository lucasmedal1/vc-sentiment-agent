#!/bin/bash
# Installs a cron job to run the VC sentiment agent every 15 days at 8am.
# Usage: bash schedule.sh

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PYTHON="$(which python3)"
LOG="$SCRIPT_DIR/run.log"

# Every 15 days: runs on the 1st and 16th of each month at 08:00
CRON_LINE="0 8 1,16 * * cd \"$SCRIPT_DIR\" && $PYTHON main.py >> \"$LOG\" 2>&1"

# Check if already scheduled
if crontab -l 2>/dev/null | grep -qF "vc-sentiment-agent"; then
    echo "Cron job already exists. Current entry:"
    crontab -l | grep "vc-sentiment-agent"
    exit 0
fi

# Add to crontab
( crontab -l 2>/dev/null; echo "$CRON_LINE  # vc-sentiment-agent" ) | crontab -
echo "✅  Cron job installed:"
echo "    $CRON_LINE"
echo ""
echo "Reports will be saved to: $SCRIPT_DIR"
echo "Logs will be written to:  $LOG"
echo ""
echo "To remove: crontab -e  (delete the vc-sentiment-agent line)"
