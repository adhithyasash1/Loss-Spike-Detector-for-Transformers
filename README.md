# loss spike detector & autopsy tool

this is a sandbox to help understand, artificially trigger, and automatically detect training instabilities (aka "loss spikes") in llms. 

it uses a tiny gpt-2 model (~10m parameters) so you can run rapid experiments locally and see exactly what happens when a model breaks down during training, and how to catch it mathematically before it ruins a massive run.

## first principles: what even is a loss spike?

when you train a neural network, it learns by taking small math steps down a hill toward a lower error rate. it figures out how steep the hill is using gradients. 

usually these steps are smooth. but sometimes, things break:
1. it reads a batch of complete garbage data
2. the gradients explode and the steepness math goes crazy
3. learning rate acts up and it takes a step that is way too big

when this happens, the loss suddenly shoots up. if you don't catch it, the model gets super confused and forgets everything it learned. you could lose months of training.

## how we catch them: statistical process control (spc)

instead of waiting for the loss to explode and the run to crash, we can treat model training like a factory assembly line. we use statistical process control tools from manufacturing to monitor the "quality" of our training steps in real time.

we use two main math detectors here:

### 1. shewhart control charts (the sudden explosion detector)
this watches out for sudden, massive outliers. it keeps a rolling average of what a normal gradient or loss looks like. if a single step jumps way out of bounds (like 3 standard deviations out), it sounds the alarm immediately. this catches things like instant gradient explosions.

### 2. cusum (the slow drift detector)
cumulative sum (cusum) keeps a running tally of tiny deviations from normal. if the loss is consistently just a little bit higher than it should be, the sum builds up over time until it crosses a threshold and triggers an alert. this is great for catching slow burns, like when bad data slowly destabilizes the model over time.

## what's in the repo

everything is built in plain pytorch without annoying bloated dependencies.

### 1. `model.py`
a minimal 10 million parameter causal language model built exactly like gpt-2. small enough to iterate on quickly without an expensive gpu.

### 2. `train.py`
the training loop. normally you want this to run perfectly, but this one is deliberately sabotaged. during the 600 training steps, it injects intentional instabilities:
* steps 150-151: swaps text out for complete random noise
* steps 300-302: spikes the learning rate up by 10x
* steps 400-402: injects garbage all-the-same-token batches
* step 500: manually multiplies all gradients by 50x

### 3. `spike_detector/`
this is the actual tooling you could drop into a real production environment.
* `detectors.py`: the math for the shewhart and cusum detectors
* `monitor.py`: hooks into the pytorch loop right before the optimizer takes a step. if an alert triggers, it takes a "forensic snapshot" right at that millisecond, saving gradients, optimizer state, and the batch data.
* `report.py`: after a crash or finish, this reads the snapshots and generates visual charts, heatmaps of broken layers, and a text autopsy report.

## running it

1. install the requirements:
```bash
pip install -r requirements.txt
```

2. run the experiment:
```bash
python train.py
```
(if you're on an apple silicon mac, this automatically uses the mps backend so it's snappy)

3. read the post-mortem:
you'll see alerts pop up in the terminal live as the script injects bad data and gradients. when it finishes, check the newly created `reports/` folder. it will have png charts visualizing the training process, control limits, heatmaps of broken layers, and a full text autopsy of the run.
