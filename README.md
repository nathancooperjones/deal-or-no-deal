# deal-or-no-deal
Let's play Deal or No Deal better than a human.

![](https://media.firstcoastnews.com/assets/WTLV/images/552259058/552259058_750x422.jpg)

### Development
Begin by installing [Docker](https://docs.docker.com/install/) if you have not already. Once Docker is running, run development from within the Docker container:

```bash
# build the image to extract features
docker build -t deal_or_no_deal .

# run the container in interactive mode on the CPU...
docker run \
    -it \
    --rm \
    -v "${PWD}:/deal_or_no_deal" \
    -p 8888:8888 \
    deal_or_no_deal /bin/bash -c "pip install -r requirements-dev.txt && bash"
```

To run the Docker container on the GPU, use:
```bash
# ... or run the container in interactive mode on the GPU...
docker run \
    -it \
    --rm \
    --gpus all \
    -v "${PWD}:/deal_or_no_deal" \
    -p 8888:8888 \
    deal_or_no_deal /bin/bash -c "pip install -r requirements-dev.txt && bash"
```

### Start JupyterLab
To run JupyterLab, start the container and execute the following:
```bash
jupyter lab --ip 0.0.0.0 --no-browser --allow-root --NotebookApp.token='' --NotebookApp.password=''
```
Connect to JupyterLab here: [http://localhost:8888/tree?](http://localhost:8888/tree?)
