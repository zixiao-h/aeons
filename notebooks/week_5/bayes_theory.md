#### **Derive marginalised probability**
Ultimately want to maximise $P(\theta | \{\log L_i\})$. Bayes' rule to invert in terms of known quantities 


$$P(\{\log L_i\}\mid\theta) = \frac{ P(\{\log L_i\}\mid\theta) \times P(\theta)}{ P(\{\log L_i\}) } $$
Assume an uninformative constant prior $P(\theta)$, and disregard the total evidence $P(\{\log L_i\})$ which is independent of $\theta$. The remaining quantity is the evidence for $\theta$, which is calculated by marginalising over $\bm{X}$

$$ P(\{\log L_i\}\mid\theta) \propto P(\{\log L_i\}\mid\theta) = \int P(\{\log L_i\}\mid\theta, \bm{X}) P(\bm{X})\ d\bm{X} $$

$$ P(\bm{X}) = \frac{1}{2\pi |\Sigma|} \exp\left[- \frac{1}{2} (\bm{X} - \bm{\mu})^T \Sigma^{-1} (\bm{X} - \bm{\mu})\right] $$ 

$$ P(\{\log L_i\} \mid \bm{X}, \theta) = \prod_{i} \delta(\log L_i - f(X_i, \theta)) $$

$$\therefore
P(\theta \mid \{\log L_i\}) = \int \frac{1}{2\pi |\Sigma|} \exp\left[- \frac{1}{2} (\bm{X} - \bm{\mu})^\intercal \Sigma^{-1} (\bm{X} - \bm{\mu})\right] \left(\prod_{i} \delta(\log L_i - f(X_i, \theta))\right)\ dX_1 \cdots \ dX_n
$$

$$ f(X_i, \theta) = \log L_\text{max} - \frac{X_i^{2/d}}{2\sigma^2} $$


#### **Evaluate integral**

Consider the identity

$$ \delta(g(x)) = \sum_{\text{roots}\ j} \frac{\delta(x - x_j)}{|g'(x_j)|}$$ 

For the above case we expect only one root for each $X_i$, namely at $\log L_i = f(X_i^*, \theta)$ (assuming that $f$ is one-to-one, which it is for the form we are using). The product reduces to 

$$\prod_{i} \frac{\delta(X_i - X_i^*)}{|f'(X_i^*, \theta)|} $$

Putting this together, the integral evaluates to 

$$ P(\theta \mid \log \bm{L}) = \frac{1}{2\pi |\Sigma|} \left(\prod_i \frac{1}{|f'(X_i^*, \theta)|}\right) \exp[- \frac{1}{2} (\bm{X^*} - \bm{\mu})^\intercal \Sigma^{-1} (\bm{X^*} - \bm{\mu})]$$

where the $\bm{X^*}$ are given by the inverse of $f$. Discard normalising term:

$$ \log P(\theta \mid \log \bm{L}) = - \left(\sum_i \log |f'(X_i^*, \theta)|\right) - \frac{1}{2} (\bm{X^*} - \bm{\mu})^\intercal \Sigma^{-1} (\bm{X^*} - \bm{\mu})$$


#### **Substitute specific form of $f$**

Invert $f$ to find $X_i^*$ and $|f'(X_i^*)|$

$$ f(X_i, \theta) = f(X_i, d) = - X_i^{2/d} $$
$$ X_i^* = (-\log L_i)^{d/2} $$

$$ f'(X_i, d) = - \frac{2}{d} X_i^{2/d - 1} $$

$$ \therefore \log P(\theta \mid \log \bm{L}) = - \left(\sum_i \log \left|\frac{2}{d} X_i^*{^{2/d - 1}}\right|\right) - \frac{1}{2} (\bm{X^*} - \bm{\mu})^\intercal \Sigma^{-1} (\bm{X^*} - \bm{\mu})$$

Maximisation of final expression implemented below