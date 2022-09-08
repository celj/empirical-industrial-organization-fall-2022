# Problem Set 1

## Empirical Industrial Organisation

###### Instituto Tecnológico Autónomo de México

###### Carlos Lezama

### The Data

The dataset contains (fictitious) individual level purchase data from an online trading site for memory chips. There are 48 possible combinations for a chip, and one seller for each type who may change its price over time. The data show the characteristics of each type of chip when the customer visits, including a variable that indicates if the chip was available.

The following is the description of the variables in the dataset.

- `visitid`: an ID number that indicates the visit of a particular customer (e.g. Horca Andy visiting at 11:01pm) — for each visit there are a series of rows, one for each of the product types available when the visit occurred
- `size`: chip memory storage (values from 1 to 6 representing categorical values for sizes such as 512 MB, 16 GB)
- `speed`: chip speed rated from 1 to 4
- `branded`: branded or generic dummy variable
- `price`: sale price
- `outofstock`: availability dummy variable
- `wholecost`: seller's wholesale cost
- `choice`: indicator of whether the visitor bought the product (suppose he can only buy one unit) — if no product is purchased, then the consumer chooses the outside good

### Aggregate Data

We are interested in the following preferences model:

$$
U_{ij} = \beta_1 p_j + \beta'_2 X\_j + \xi_j + \varepsilon_{ij}
$$

where the subscripts $i$ and $j$ refer to customer $i$ and product $j$ The variable $p_j$ denotes the product price logarithm. The vector $X_j$ groups exogenous characteristics that include `size`, `speed`, `branded`. As usual, $\xi_j$ captures unobserved characteristics of the product $j$ that are relevant for the customer’s decision. Finally, $\varepsilon_{ij}$ is an idiosyncratic Type 1 Extreme Value Error term. The utility of buying the outside good is $\varepsilon_{i0}$.

### Individual Level Data

We are interested in the following preferences model:

$$
U_{ij} = \beta_1 p_j + \beta'_2 X_j + \varepsilon_{ij}.
$$

Note that we have abstracted from the existence of any unobserved product characteristic, $\xi_{j}$. Additionally, observe that, with individual level data, $X_{jt}$ is 1 everywhere, except for the outside good, where it is 0.

We will estimate this model by maximum likelihood using multiple optimisation routines. Furthermore, we want to estimate each estimator's performance.