# Scratch_Machine_Learning_Algorithms
Writing Object-Oriented Python Programs for ML Algorithms instead of using Libraries

# Linear Regression with Gradient Descent Derivation

## Linear Algebra background:

\nabla_xb^T\ =b

(A + B)^T = A^T + B^T

(AB)^T = B^TA^T

y^TX\beta=(X\beta)^Ty=\ \beta^TX^Ty

\frac{\partial\beta^TX\beta}{\partial\beta}=(X+\ X^T)\beta

If the matrix A is symmetric

\nabla_xx^TAx\ =2Ax





RSS(\beta) = \sum_{i=1}^{N}{(y_i}-\ \beta_0-\ \sum_{j=1}^{p}x_{ij}\beta_j)^2 = (y - X\mathbit{\beta})^T(y - X\mathbit{\beta})

(y – X\mathbit{\beta})^T(y - X\mathbit{\beta}) = \mathbit{y}^T\mathbit{y}-(\mathbit{X\beta})^T\mathbit{y}-\ \mathbit{y}^T\mathbit{X\beta}+(\mathbit{X\beta})^T\mathbit{X\beta}

\frac{\partial RSS}{\partial\beta}=\ 0\ =0-\ \mathbit{y}^T\mathbit{X}-\ \mathbit{y}^T\mathbit{X}-\left(\mathbit{X}^\mathbit{T}\mathbit{X}+\mathbit{X}\mathbit{X}^\mathbit{T}\right)\mathbit{\beta}=\ -2\mathbit{X}^T\mathbit{y}+2\mathbit{X}^\mathbit{T}\mathbit{X\beta}

2\mathbit{X}^T\mathbit{y}=2\mathbit{X}^\mathbit{T}\mathbit{X\beta}

\mathbit{X}^T\mathbit{y}=\mathbit{X}^\mathbit{T}\mathbit{X\beta}

\hat{\mathbit{\beta}}=(\mathbit{X}^T\mathbit{X})^{-\mathbf{1}}\mathbit{X}^T\mathbit{y}

\hat{\mathbit{y}}=\mathbit{X}\hat{\mathbit{\beta}}=\mathbit{X}(\mathbit{X}^T\mathbit{X})^{-\mathbf{1}}\mathbit{X}^T\mathbit{y}
