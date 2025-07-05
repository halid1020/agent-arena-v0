. ./setup.sh

ray start --head --node-ip-address=$(hostname --ip-address) \
          --port=6379 --block