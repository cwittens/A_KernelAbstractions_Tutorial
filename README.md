# Writing GPU Kernels with KernelAbstractions.jl

A tutorial on writing portable GPU kernels in Julia using KernelAbstractions.jl.

An Website version of the tutorial is available at [cwittens.github.io/A_KernelAbstractions_Tutorial](https://cwittens.github.io/A_KernelAbstractions_Tutorial).

## Files

- [`KA_Tutorial_Literate_Source.jl`](KA_Tutorial_Literate_Source.jl) — Literate.jl source file
- [`KA_Tutorial_Markdown.md`](KA_Tutorial_Markdown.md) — tutorial (readable on GitHub)
- [`KA_Tutorial_JupyterNotebook.ipynb`](KA_Tutorial_JupyterNotebook.ipynb) — tutorial (Jupyter notebook)

The Markdown and Notebook files were generated from the Literate source using [Literate.jl](https://github.com/fredrikekre/Literate.jl).

<!--
with the julia commands
Literate.markdown("KA_Tutorial_Literate_Source.jl", "."; name="KA_Tutorial_Markdown", execute=true, flavor = Literate.CommonMarkFlavor())
and
Literate.notebook("KA_Tutorial_Literate_Source.jl", "."; name="KA_Tutorial_JupyterNotebook", execute=true)
-->

## Author

[Collin Wittenstein](https://cwittens.github.io/) — cwittens@mit.edu