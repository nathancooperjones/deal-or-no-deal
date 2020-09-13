# deal-or-no-deal
Let's play Deal or No Deal better than a human.

![](https://media.firstcoastnews.com/assets/WTLV/images/552259058/552259058_750x422.jpg)

### What is this?
Deal or No Deal is the greatest TV game show of all time, but for some unknown reason, it does not get a lot of love in the ML community...

... until now!

### Why (in audiovisual form)?
See my talk [here](https://youtu.be/7jSvFHGyEeE) or just my slides [here](https://drive.google.com/file/d/1Dd-AYJv7QoEukR8kW8E1ny-VduZGuCT8/view?usp=sharing)!

### Why (in code form)?
Despite it's simplicity on the surface, Deal or No Deal is a great environment to build a simple supervised model to predict the Banker's model and train a reinforcement learning model in a simple (but tricky) game environment without a clear strategy.

This repo not only contains the data of all decisions made in over 60 episodes of Deal or No Deal, but also:
- A [`Docker`](https://github.com/nathancooperjones/deal-or-no-deal/blob/master/Dockerfile) container that installs all necessary requirements to run all the code
- Code to preprocess the TV data into a ML-ready format ([`deal_or_no_deal/preprocess.py`](https://github.com/nathancooperjones/deal-or-no-deal/blob/master/deal_or_no_deal/preprocess.py))
- A "random ensemble" supervised machine learning model to input a game state and predict a shockingly close Banker offer ([`deal_or_no_deal/Banker.py`](https://github.com/nathancooperjones/deal-or-no-deal/blob/master/deal_or_no_deal/Banker.py))
- An OpenAI Gym Deal or No Deal environment ([`deal_or_no_deal/envs/deal_or_no_deal.py`](https://github.com/nathancooperjones/deal-or-no-deal/blob/master/deal_or_no_deal/envs/deal_or_no_deal.py))
- Code to train a DQN reinforcement learning agent to explore the Deal or No Deal Environment ([`deal_or_no_deal/dqn.py`](https://github.com/nathancooperjones/deal-or-no-deal/blob/master/deal_or_no_deal/dqn.py))
- Code for an agent to input a game state and play *n* games in the future, revealing the percentage of future games that end with winnings higher than the current Banker's offer ([`deal_or_no_deal/fast_play.py`](https://github.com/nathancooperjones/deal-or-no-deal/blob/master/deal_or_no_deal/fast_play.py))
- A slew of notebooks showing how to interact with all the code ([`nbs`](https://github.com/nathancooperjones/deal-or-no-deal/tree/master/nbs))

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
