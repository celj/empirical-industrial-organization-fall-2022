# Midterm exam

## Empirical Industrial Organization

###### Instituto Tecnológico Autónomo de México

###### Carlos Lezama

This exam is inspired by Luco (2019).

Individuals in their working age make savings decisions for retirement. We approach the worker's decision problem in a static manner, i.e., the worker makes a one-time decision.

### One Level Decision

The worker must choose one of $J$ pension fund administrators (PFA) to manage her retirement savings. The worker is mandated to save $10\%$ of her salary, $y_i$. Each PFA charges a percentage fee, $p_j$, over the worker's salary. PFAs differ in their return on investment, $R_j$. Finally, $\varepsilon_{ij}$ is an i.i.d. Type I Extreme Value preference shock. With all this in hand, we can write down the indirect utility that worker $i$ obtains from enrolling in PFA $j$ as

$$
u_{ij} = \alpha_i\left( y_i - 0.1y_i - p_jy_i \right) + \beta_iR_j + \varepsilon_{ij} ,
$$

where $\alpha_i$ and $\beta_i$ denote random coefficients. Moreover, let $\gamma_i = \left( \alpha_{i}, \beta_{i} \right)$, and $\gamma = \left( \alpha,\beta \right)$. We assume,

$$
\gamma_i = \gamma + \Gamma D_i + \nu_i ,
$$

where $D_i$ is a $d\times1$ vector of demographic variables, $\Gamma$ is a $2\times d$ matrix of coefficients that measure how taste varies with demographics, and $\nu_i$ is a $2\times1$ vector of unobserved individual characteristics determining taste, where $\nu_i \sim F_\nu(\cdot)$, with $F_\nu(\cdot)$ a distribution function.

Following Train (2009, 3.1), we can write the probability that worker $i$ chooses PFA $j$, $s_{ij}$, as follows:

$$
\begin{align}
s_{ij} &= P(u_{ij} \geq u_{ik},\ \forall j \neq k) \\
&= P(\varepsilon_{ij} - \varepsilon_{ik} \geq \omega_{ik} - \omega_{ij}) \\
&= \frac{\exp(\omega_{ij})}{\displaystyle \sum_{j \in J} \exp(\omega_{ik})}
\end{align}
$$

where $\omega_{ij} = \beta_i R_j - \alpha_i p_j y_i$.

Additionally, we can express the price elasticity of the demand for PFA $j$, $\eta_{j, p_j}$, and the cross return elasticity of the demand for PFA $j$ with respect to the return of PFA $k$ (with $j \neq k$), $\eta_{j, R_k}$, like so

$$
\begin{align}
\eta_{j, p_j} &= \frac{p_j}{s_{ij}} \cdot \frac{\partial s_{ij}}{\partial p_j} \\
&= p_j (1 - s_{ij}) \cdot \frac{\partial \omega_{ij}}{\partial p_j} \\
&= - \alpha_i p_j y_i (1 - s_{ij}), \\ \\
\eta_{j, R_k} &= \frac{R_k}{s_{ij}} \cdot \frac{\partial s_{ij}}{\partial R_k} \\
&= - R_k s_{ik} \cdot \frac{\partial \omega_{ik}}{\partial R_k} \\
&= - \beta_i R_k s_{ik}.
\end{align}
$$

as derived in Train (2009, 3.6).

Subsequently, with this setup, we can describe the log-likelihood of the model, $\mathcal{L} (\cdot)$, such that

$$
\begin{align}
\mathcal{L} (\cdot) &= \sum_i \sum_j \mathbf{1}_{ij} \log (s_{ij}) \\
&= \sum_i \sum_j \left( \frac{\exp(\omega_{ij})}{\displaystyle \sum_{j \in J} \exp(\omega_{ik})} \right).
\end{align}
$$

### Two Level Decision

Now, suppose that each PFA offers two portfolios that workers can choose where to invest their savings in: a high-return, high-risk portfolio, $h$; and low-return, low-risk portfolio, $l$.

We model the worker's choice of PFA and portfolio as a sequential decision, where individuals first choose the PFA, and then, conditional on the PFA, they choose the portfolio. The following diagram depicts the sequential two-level decision of the worker,

<img src="img/1.png" style="width:6cm;" />

In this context, we model the indirect utility of choosing PFA $j=1,\dots,J$ and portfolio $g=h,l$ as,

$$
u_{ijg} = \alpha\left( y_i - 0.1y_i - p_jy_i \right) + \beta R_{jg} + \theta 
C_{jg} + \varepsilon_{ijg} ,
$$

where now, for simplicity, $\alpha$ and $\beta$ are not random but homogeneous coefficients across individuals. The variable $R_{jg}$ denotes the return of portfolio $g$ in PFA $j$ and, similarly, $C_{jg}$ denotes the risk of portfolio $g$ in PFA $j$. Finally, $\varepsilon_{ijg}$ is an i.i.d. GEV preference shock such that the model is a two-level nested logit.

Note that the indirect utility can be rewritten as $u_{ijg} = u_{ij} + u_{ig \mid j} + \varepsilon_{ijg}$, where

$$
u_{ij} = \alpha (y_i - 0.1 y_i - p_i y_i)
$$

is the indirect utility associated to choosing a PFA (i.e., first level), and

$$
u_{ig \mid j} = \beta R_{jg} + \theta C_{jg}
$$

is the indirect utility associated to choosing a portfolio conditional on a particular PFA (i.e., second level). Note that $u_{ig \mid j} = u_{g \mid j}$, $\forall i$.

Furthermore, as shown in Train (2009, 4.2.3), the probability of choosing PFA $j$ and portfolio $g$, $s_{ijg}$, can be expressed as the product of a marginal probability of choosing PFA $j$ and a conditional probability of choosing portfolio $g$ conditional on PFA $j$. That is, $s_{ijg} = s_{ij} \cdot s_{ig|j}$, where

$$
s_{ij} = \frac{\exp \left( u_{ij} + \lambda_j I_{ij} \right)}{\displaystyle \sum_j \exp \left( u_{ij} + \lambda_j I_{ij} \right)} \qquad \qquad \text{(upper level model)}
$$

and

$$
s_{ig|j} = \frac{ \exp \left( \displaystyle \frac{u_{ig \mid j}}{\lambda_j} \right) }{ \displaystyle \sum_g \exp \left( \displaystyle \frac{u_{ig \mid j}}{\lambda_j} \right) }  \qquad \qquad \text{(lower level model)}
$$

with

$$
I_{ij} = \log \sum_g \exp \left( \displaystyle \frac{u_{ig \mid j}}{\lambda_j} \right) \qquad \qquad \text{(inclusive value)}
$$
Bear in mind that $\lambda_j$ must be in the $[0, 1]$ interval as a sufficient condition that characterizes the correlation of utilities that a consumer experiences among the products in the same group $j$.

Finally, we can express the price elasticity of the demand for PFA $j$, $\eta_{j, p_j}$, and the cross price elasticity of the demand for PFA $j$ with respect to the price of PFA $k$ (with $j \neq k$), $\eta_{j, p_k}$, like so

$$
\begin{align}
\eta_{j, p_j} &= \frac{p_j}{s_{ijg}} \cdot \frac{\partial s_{ijg}}{\partial p_j} \\
&= \frac{p_j}{s_{ijg}} \left( s_{ig \mid j} \cdot \frac{\partial s_{ij}}{\partial p_j} + s_{ij} \cdot \frac{\partial s_{ig \mid j}}{\partial p_j} \right) \\
&= p_j \left( \frac{s_{ig \mid j}}{s_{ijg}} \right) \frac{\partial s_{ij}}{\partial p_j} \\
&= p_j \left( \frac{s_{ig \mid j} (1 - s_{ij}) s_{ij}}{s_{ijg}} \right) \frac{\partial u_{ij}}{\partial p_j} \\
&= - \alpha p_j y_i (1 - s_{ij}), \\ \\
\eta_{j, p_k} &= \frac{p_k}{s_{ijg}} \cdot \frac{\partial s_{ijg}}{\partial p_k} \\
&= \frac{p_k}{s_{ijg}} \left( s_{ig \mid j} \cdot \frac{\partial s_{ij}}{\partial p_k} + s_{ij} \cdot \frac{\partial s_{ig \mid j}}{\partial p_k} \right) \\
&= p_k \left( \frac{s_{ig \mid j}}{s_{ijg}} \right) \frac{\partial s_{ij}}{\partial p_k} \\
&= - p_k \left( \frac{s_{ig \mid j} \cdot s_{ij} \cdot s_{ik}}{s_{ijg}} \right) \frac{\partial u_{ik}}{\partial p_k} \\
&= \alpha p_k y_i s_{ik}.
\end{align}
$$