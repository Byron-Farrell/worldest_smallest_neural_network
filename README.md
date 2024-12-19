# worlds_smallest_neural_network

### Weighted Sum

- $x$ = input
- $w$ = weight
- $b$ = bias

$F(x) = w \dot x + b$

### Cost 

- $y$ = predicted output
- $\hat y$ = expected output

$C(y) = (y - \hat y)^2$

### Cost derivative

$C'(y) = 2(y - \hat y)$

### Weighted Sum Partial derivative with respect to w

$\frac{\partial F}{\partial w}(W \dot X + B) = x$

$\frac{\partial F}{\partial w} = x$

### Weighted Sum Partial derivative with respect to b

$\frac{\partial F}{\partial b}(W \dot X + B) = 1$

$\frac{\partial F}{\partial b} = 1$

### weight gradient

$\nabla w = \frac{\partial C}{\partial F} \cdot \frac{\partial F}{\partial w}$

### bias gradient

$\nabla b = \frac{\partial C}{\partial F} \cdot \frac{\partial F}{\partial b} $

### adjusting weight and bias
- $\sigma$ = learning rate
- $\nabla w$ = weight gradient
- $\nabla b$ = bias gradient

$\Delta w = \sigma \cdot \nabla w$

$w = w - \Delta w$

$\Delta b = \sigma \cdot \nabla b$

$b = b - \Delta b$
