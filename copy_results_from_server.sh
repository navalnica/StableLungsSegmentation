# should be executed with `source` or `.` command:
# `source copy_results_from_server.sh`

RESULTS_DIR='results'
CHECKPONTS_DIR="model_checkpoints"
OUT_DIR='results_server'

# download results excluding all model checkponts but the best.
# order of filters matters.
# add `--dry-run` option to only simulate copying without any actions performed.
rsync -v -h -I -r \
  -e "ssh -p ${LUNGS_SERVER_PORT}" \
  --filter="+ ${CHECKPONTS_DIR}/*best.pth" \
  --filter="- ${CHECKPONTS_DIR}/*" \
  "${LUNGS_SERVER_ADDRESS}:~/dev/lungs/${RESULTS_DIR}/*" \
  "${OUT_DIR}"/
