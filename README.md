# Gaste Package

Welcome to the Gaste package! This package provides a set of tools and utilities for analyzing stratified 2x2 contingency table. Gives the exact or approximate p-value of the overall association between features and outcomes under 2x2 stratified contingency table.

## Installation

To install the Gaste package, simply run the following command:

```
pip install gaste-test
```

## Usage

Once installed, you can import the Gaste package or the main function in your Python code using the following line:

```python
import gaste-test
from gaste-test import get_pval_comb, StratifiedTable2x2
```

## Features

The Gaste package offers the following features:

- Exact calcul of p-value combination of one tail Fisher's exact test
- Approximation of the law of combination by Gamma approximation distribution
- Incorporating truncation into the p-value combination enhances statistical power in scenarios featuring few effects or contradictory effects between strata
- Visualization: Forest plot for data analysis.

## Documentation

For detailed information on how to use the Gaste package, please refer to the sphinx documentation

## Contributing

We welcome contributions from the community! If you would like to contribute to the Gaste package, feel free to contact the autor by mail.

## License

The Gaste package is licensed under the [MIT License](https://github.com/your-username/gaste/LICENSE).
