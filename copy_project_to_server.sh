shopt -s extglob
scp -P ${LUNGS_SERVER_PORT} -r !(venv|notebooks|results) ${LUNGS_SERVER_ADDRESS}:~/dev/lungs/