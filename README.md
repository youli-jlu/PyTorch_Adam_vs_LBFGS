# PyTorch_Adam_vs_LBFGS
Curve fitting comparison between Adam and L-BFGS optimizer

## Motivation
I'm studying the NN tools for theoritical chemistry simulation, espacially potential energy surface (PES) fitting.

At first, I chose tensorflow for NN simulation. 
I have succesfully constructed a Diabatic PES in tensorflow 2.4 with adam optimizer, and the result has been published in [J. Chem. Phys. 155, 214102 (2021)](https://aip.scitation.org/doi/10.1063/5.0072004).
However, some of reviews said the second optimizer, like Levenberg-Marquardt, can provied better convergence result efficiently.
In my previous simulation, it usually takes almost 10^7 epochs in a week for convergency, which can be 10^3 level in L-M optimizer in review.
So, I really want ot know the performance of second optimizer in regression problem.

## previous comparison
- [Fabio Di Marco](https://github.com/fabiodimarco/tf-levenberg-marquardt) has compared Levenberg-Marquardt and Adam with tensorflow. Target function is sinc function.
- [Soham Pal](https://soham.dev/posts/linear-regression-pytorch/) has compared L-BFGS and Adam with PyTorch in linear regression problem.
- [NN-PES review](https://pubs.acs.org/doi/10.1021/acs.chemrev.0c00665) has compared some optimizer but is lack of details. And matlab has more study cost(in my point of view). 

## L-BFGS in PyTorch
Since tensorflow do not have official second optimizer, I will use pyTorch L-BFGS optimizer in this test.

You can find some information about L-BFGS algorithms in many website, and I will not disccus this.
However, when you use L-BFGS in PyTorch, you need to define a 'closure' function for gradiant evaluation.
I'm not so familliar for optimization algorithms, and simply follow the code written by [Soham Pal](https://soham.dev/posts/linear-regression-pytorch/). 
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
The code for this simulation can be found in src/

## Fitting result
I compared two NN structure:

1. One hidden layer with 20 neuron. Linear output. I use 1-t20-1 to detnote this situation while "t" means using tanh for activation function.
2. Two hidden layer with 20 neuron each. Linear output. I use 1-t20-t20-1 to detnote this.


### Sinc function
x in [-1.1], y=sinc(x)=( 1 if x=0 or sin(x)/x if else )

### prediction plot
