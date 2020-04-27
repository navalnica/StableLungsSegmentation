# should be executed with `source` or `.` command:
# `source copy_project_to_server.sh`

# `--exclude` option excludes folders and SUBFOLDERS that have corresponding names
# `--dry-run` option does not copy anything but simulates the process
rsync -v -r \
  -e "ssh -p ${LUNGS_SERVER_PORT}" \
  --exclude={'venv','.git','.idea','notebooks','results','results_server','__pycache__','.gitignore'} \
  . \
  "${LUNGS_SERVER_ADDRESS}:~/dev/lungs/" \
#  --dry-run
