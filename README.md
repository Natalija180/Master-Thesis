# Master-Thesis

## ESG Portfolio Analysis Thesis Code

This repository contains the Python code used for the empirical analysis in my master thesis on ESG-sorted equity portfolios.

## Scope
The code covers:
- data preparation and cleaning
- investable-universe construction
- ESG sorting into Top-10 and Bottom-10 portfolios
- portfolio construction using market-cap weighting, equal weighting, Risk-Parity, and Black-Litterman
- annual rebalancing and buy-and-hold implementation
- performance and risk measurement
- Fama-French three-factor regressions
- export of result tables and charts

## Input files
The analysis requires the following input files in the `Data` folder:
- `ESG_Ratings_SP100.csv`
- `ESG_Ratings_DAX.csv`
- `Prices.csv`
- `FamaFrench_US_Monthly.csv`
- `FamaFrench_Europe_Monthly.csv`

## Output
The script exports:
- portfolio return series
- summary tables
- performance and risk metrics
- Fama-French regression results
- charts

Outputs are saved in the `Output` folder.

## Main file
Run the main Python script to reproduce the empirical results.

## Note
The repository contains the full code implementation. Selected excerpts are included in the appendix of the thesis.