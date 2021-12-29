# PyTorch_Adam_vs_LBFGS
Curve fitting comparison between Adam and L-BFGS optimizer

## Motivation
I'm studying the NN tools for theoretical chemistry simulation, especially potential energy surface (PES) fitting.

At first, I chose TensorFlow for NN simulation. 
I have successfully constructed a Diabatic PES in TensorFlow 2.4 with adam optimizer, and the result has been published in [J. Chem. Phys. 155, 214102 (2021)](https://aip.scitation.org/doi/10.1063/5.0072004).
However, some of the reviews said the second optimizer, like Levenberg-Marquardt, can provide better convergence results efficiently.
In my previous simulation, it usually takes almost 10^7 epochs in a week for convergency, which can be 10^3 level in L-M optimizer in review.
So, I want to test the performance of the second optimizer in the regression problem.

## previous comparison
- [Fabio Di Marco](https://github.com/fabiodimarco/tf-levenberg-marquardt) has compared Levenberg-Marquardt and Adam with TensorFlow. The target function is sinc function.
- [Soham Pal](https://soham.dev/posts/linear-regression-pytorch/) has compared L-BFGS and Adam with PyTorch in linear regression problem.
- [NN-PES review](https://pubs.acs.org/doi/10.1021/acs.chemrev.0c00665) has compared some optimizers but it lacks details. And matlab has more study costs (in my point of view). 

## L-BFGS in PyTorch
Since TensorFlow does not have an official second optimizer, I will use pyTorch L-BFGS optimizer in this test.

You can find some information about L-BFGS algorithms on many websites, and I will not discuss this.
However, when you use L-BFGS in PyTorch, you need to define a 'closure' function for gradient evaluation.
I'm not so familiar with optimization algorithms, and simply follow the code written by [Soham Pal](https://soham.dev/posts/linear-regression-pytorch/). 
The 'train' function will be:
```python
def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    lm_lbfgs=model.to(device)
    #spacial function for LBFGS
    for batch, (X, y) in enumerate(dataloader):
    ¦   x_ = Variable(X, requires_grad=True)
    ¦   y_ = Variable(y)
    ¦   def closure():
    ¦   ¦   # Zero gradients
    ¦   ¦   optimizer.zero_grad()
    ¦   ¦   # Forward pass
    ¦   ¦   y_pred = lm_lbfgs(x_)
    ¦   ¦   # Compute loss
    ¦   ¦   loss = loss_fn(y_pred, y_)
    ¦   ¦   # Backward pass
    ¦   ¦   loss.backward()
    ¦   ¦   return loss

    ¦   optimizer.step(closure)
    ¦   loss=closure()

    ¦   if batch % train_size == 0:
    ¦   ¦   loss, current = loss.item(), batch * len(X)
    ¦   ¦   print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
    return loss_train
```

Also, I use strong_wolfe option. Otherwise, the loss function will become very large (I don't know the reason).

```python
optimizer_lbfgs= torch.optim.LBFGS(model.parameters(), lr=1,
    ¦   history_size=100, max_iter=20,
    ¦   line_search_fn="strong_wolfe"
    ¦   )
```

The code for this simulation can be found in src/lbfgs_simple.py

## Fitting detail:

### NN structure
I compared two NN structures:

1. One hidden layer with 20 neurons. Linear output. I use t20 to denote this situation while "t" means using tanh for activation function.
2. Two hidden layers with 20 neurons each. Linear output. I use t20-t20 to denote this.


### Train data
I use 20000 sampled points from Sinc function:
x in [-1.1], y=sinc(x)=( 1 if x=0 or sin(x)/x if else )

And 80% of data was randomly chosen for training.

## Result
### prediction plot
<img src="https://github.com/youli-jlu/PyTorch_Adam_vs_LBFGS/blob/main/line_plot.png" width="600"/>
It is not surprise that adam t20 perform worst, and adam t20-t20 seems has the same performance with l-bfgs
 
However, if we zoom into boundary:

<img src="https://github.com/youli-jlu/PyTorch_Adam_vs_LBFGS/blob/main/line_plot_boundary.png" width="600"/>

The green line adam t20-t20 derivate a lot from the target.

### Training loss function
The loss decay curve in log() can illustrate the fitting error better.

Adam t20-t20 is still worse than lbfgs t20 in several orders of magnitude. 

<img src="https://github.com/youli-jlu/PyTorch_Adam_vs_LBFGS/blob/main/adam_vs_lbfgs_in_log.png" width="600"/>

### computational cost
Almost the same due to parallel.

## Conclusion
Please try second-order optimizer in regression problems if possible, especially for small networks.
