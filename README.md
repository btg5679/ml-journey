# Journey Notebook Summaries

## Phase1_Day1_Setup.ipynb – Environment & Dependency Setup

This notebook prepares the development environment for the rest of the Journey project.

- **Installs core ML & scientific libraries** in one `pip install` command: `torch`, `numpy`, `pandas`, `matplotlib`, `sympy`.
- **Adds helpful utilities** that will be used later in the course: `streamlit` (web apps), `huggingface_hub` (model sharing), `faiss-cpu` (vector search), and `scipy`.
- **Verifies the setup** by printing the detected PyTorch version (expect ≥ 2.4) after installation.
- **Runs a quick sanity-check** computation: creates random tensors `A` (3×4) and `B` (4×5) and multiplies them with `torch.mm`, printing the resulting 3×5 tensor.
- Configured to run on a **GPU-backed Colab runtime** (T4) for accelerated operations.

This lightweight check confirms that all dependencies are installed correctly and that GPU acceleration is available before moving on to more complex exercises.

## Day2_SymPy_Gradients.ipynb – Symbolic Gradients with SymPy

This notebook dives into analytic differentiation using the SymPy library and connects the results to practical ML/AI workflows.

- **SymPy refresher & import** – sets up the symbolic math environment.

- **1-D gradient demos**

  - _Polynomial + sine_: f(x) = x² + sin(x). Derivative (2x + cos x) is computed, printed, and numerically checked at x = 0 and x = π/2 via `lambdify`.
  - _Gaussian-cosine mix_: f(x) = e^(−x²) + cos(2x). The derivative −2x e^(−x²) − 2 sin(2x) is confirmed symbolically and evaluated at sample points with both SymPy and NumPy back-ends.

- **Multi-variable calculus example**

  - Defines g(x, y) = x² y + sin x + cos y.
  - Computes partial derivatives ∂g/∂x, ∂g/∂y, assembles the full gradient vector, and then the Hessian matrix.
  - Numerically evaluates the gradient at (x=1, y=0) to validate formulas.

- **Vector distance sanity-check** – Generates two random 3-D NumPy vectors and prints their Euclidean distance to illustrate links between gradients and embedding geometry.

Overall, Day 2 reinforces core calculus operations (derivatives, gradients, Hessians) in SymPy and shows how to bridge symbolic results to numerical workflows common in modern ML pipelines.

## Day3_Symbolic_Gradients_Embeddings.ipynb – Gradients, Embeddings & Bias Checks

Day 3 builds on the previous notebooks by marrying symbolic calculus with vector-space operations that are common in modern ML systems.

- **Warm-up recap** – revisits univariate differentiation (`f(x)=x²+sin x`) and prints the derivative using SymPy to reinforce Day 2 concepts.

- **Multivariable extension**

  - Defines `g(x,y)=x²y+cos(xy)` and symbolically derives ∂g/∂x, ∂g/∂y, and the full gradient vector.

- **Embedding-style loss**

  - Symbolically differentiates a mean-squared-error-like loss `L=(v₁−v₂)²`, yielding the familiar `2 (v₁−v₂)` gradient—highlighting links between calculus and embedding optimizers.

- **Embedding simulator**

  - Generates 100 random 5-D NumPy vectors.
  - Computes Euclidean and cosine distances pair-wise, plus a quick scatter plot (2-D projection) for intuition about cluster structure.

- **Responsible-AI quick check**
  - Splits embeddings into two label groups (0/1) and computes average intra-group cosine distance as a lightweight bias flag. (Placeholder for Fairlearn-style metrics later.)

This notebook bridges symbolic math with practical embedding workflows and introduces an initial fairness metric, laying groundwork for downstream ethical AI discussions.
