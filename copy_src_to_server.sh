shopt -s extglob
scp -P ${LUNGS_SERVER_PORT} -r !(venv|notebooks) ${LUNGS_SERVER_ADDRESS}:~/dev/lungs/