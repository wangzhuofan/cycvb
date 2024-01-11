
# cycvb

<!-- badges: start -->
<!-- badges: end -->

The goal of cycvb is to provide a variational inference algorithm for cyclic structural equation models

## Installation

You can install the development version of cycvb from [GitHub](https://github.com/) with:

``` r
# install.packages("devtools")
devtools::install_github("wangzhuofan/cycvb")
```

## Example

This is a basic example which shows you how to solve a common problem:

``` r
library(cycvb)
## basic example code
cycvb(matrix(rt(500*10,3),500,10),"t")
```

