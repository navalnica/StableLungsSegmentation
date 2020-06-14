# should be executed with `source` or `.` command:
# `source copy_project_to_server.sh`

# `--exclude` option excludes folders and SUBFOLDERS that have corresponding names
# add `--dry-run` option to only simulate copying without any actions performed.
rsync -v -r \
  -e "ssh -p ${LUNGS_SERVER_PORT}" \
  --exclude={'venv','.git','.idea','notebooks','results','segmented','img','results_server','__pycache__','.gitignore'} \
  . \
  "${LUNGS_SERVER_ADDRESS}:~/dev/lungs/"
