# Heston Model Option Pricing

This repository contains a Python script to simulate stock price dynamics under the Heston stochastic volatility model. The script prices a 3-month American put option using the Longstaff-Schwartz Monte Carlo (LSM) method with antithetic variates for variance reduction. It also calculates the European put option price using the closed-form formula under the Heston model, and provides statistical analysis of the log-terminal stock price and terminal variance.

## Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Usage](#usage)
- [Parameters](#parameters)
- [Outputs](#outputs)
- [Dependencies](#dependencies)

## Overview

This script performs the following tasks:
1. Simulates stock price (`S`) and variance (`v`) paths using the Heston model parameters.
2. Analyzes the statistical properties (mean, median, skewness, kurtosis) of log-terminal stock price `log(ST)` and terminal variance `vT`.
3. Prices a 3-month American put option with strike price `K = 130` using the Longstaff-Schwartz Monte Carlo (LSM) method with different basis sets.
4. Calculates the European put option price using the Heston closed-form formula for comparison.

## Installation

1. Clone this repository:
    ```bash
    git clone https://github.com/yourusername/heston-option-pricing.git
    cd heston-option-pricing
    ```
2. Install required packages:
    ```bash
    pip install numpy matplotlib scipy
    ```

## Usage

Run the script using:
```bash
python heston_option_pricing.py
