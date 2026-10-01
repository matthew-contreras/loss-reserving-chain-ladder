# Loss Reserving & Chain Ladder Analysis

An interactive Streamlit application that estimates ultimate losses and unpaid loss reserves from a P&C paid claims triangle using the chain ladder method, with adjustable tail factors and an inflation scenario for sensitivity testing.

**Live demo:** [ADD YOUR .streamlit.app LINK HERE]

## Features

- **Data input:** Generate a realistic synthetic P&C incremental claims dataset (accident years 2015-2024, development periods 1-10), or upload your own CSV.
- **Triangles:** Converts incremental payments into a cumulative paid loss triangle, displayed as tables and a heatmap.
- **Development factors:** Calculates age-to-age (link) ratios, with volume-weighted and simple-average development factors. Link ratios that deviate from the selected factor by more than a user-set threshold are highlighted.
- **Tail factor:** User-adjustable tail factor (1.000-1.200) applied beyond the last development period.
- **Reserving:** Projects ultimate losses and the total unpaid reserve (ultimate less paid to date) by accident year.
- **Inflation scenario:** A sidebar control (-5% to +15%) applies additional annual inflation to projected future payments, with a sensitivity table across scenarios.
- **Visualizations (Plotly):** Cumulative triangle heatmap, loss development curves by accident year, and paid losses vs. estimated IBNR by accident year.
- **Export:** Download the reserve summary as a CSV.

## Methodology

1. **Cumulative triangle:** Cum(AY, d) = sum of incremental paid for development periods 1 through d.
2. **Link ratios:** f(AY, j) = Cum(AY, j+1) / Cum(AY, j).
3. **Volume-weighted factor:** f_j = sum of Cum(AY, j+1) / sum of Cum(AY, j), over accident years observed at both ages.
4. **Projection:** Future cumulative losses are projected by multiplying the latest paid amount by the remaining selected factors; ultimate = projected cumulative at the final period x tail factor.
5. **Inflation scenario:** Each projected future incremental payment is scaled by (1 + i)^t, where t is the number of calendar years beyond the latest diagonal in which it is paid. Historical factors already reflect past inflation, so this is an additional shift; 0% reproduces the standard chain ladder result.
6. **Reserve:** Unpaid reserve (labelled IBNR in the app) = ultimate - paid to date.

## Run locally

Requires Python 3.10 or newer.

```bash
git clone https://github.com/YOUR-USERNAME/loss-reserving-chain-ladder.git
cd loss-reserving-chain-ladder
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

The app opens at `http://localhost:8501`.

## Uploading your own data

Upload a CSV with these columns (one row per accident year and development period):

| AccidentYear | DevelopmentPeriod | IncrementalPaid |
|---|---|---|
| 2015 | 1 | 2250000 |
| 2015 | 2 | 1800000 |

Each accident year needs consecutive development periods starting at 1. A sample file can be downloaded from the app sidebar.

## Limitations

- **The sample data is synthetic.** It is generated to resemble a P&C paid triangle and is for demonstration only; the results are not a real reserve estimate.
- The model uses a **paid** triangle only. "IBNR" here means total unpaid reserve (ultimate less paid), which includes case reserves; separating true IBNR requires reported-loss data.
- Chain ladder assumes past development patterns persist, and it can be unstable for immature accident years.
- The tail factor is a user input, not fitted from the data.
- The output is a point estimate with no measure of reserve uncertainty.

## Possible extensions

- Mack model for reserve standard errors
- Bornhuetter-Ferguson and Cape Cod methods
- Reported (incurred) triangles and case reserve analysis
- Fitted tail factors (e.g. exponential decay)

## Built with

Python, pandas, NumPy, Plotly, Streamlit
