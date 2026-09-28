# Retail Customer Segmentation — Streamlit App

This project converts the original RFM + K-Means customer segmentation program into an interactive Streamlit application.

## Features

- Upload a retail CSV dataset
- Data cleaning and validation
- RFM (Recency, Frequency, Monetary) feature engineering
- Log transformation and standardization
- K-Means clustering
- Elbow method and silhouette score
- Automatic K selection using the highest tested silhouette score
- Customer segment interpretation
- Marketing strategy suggestions
- PCA visualization
- New-customer segment prediction
- Download customer-level CSV
- Download segment-profile CSV
- Download Excel report

## Required CSV columns

The uploaded CSV should contain:

- `CustomerID`
- `InvoiceNo`
- `InvoiceDate`
- `Quantity`
- `UnitPrice`

Column names are normalized automatically for spaces and capitalization.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

Then open the local Streamlit URL shown in the terminal.

## Deploy on Streamlit Community Cloud

1. Create a GitHub repository.
2. Upload:
   - `app.py`
   - `requirements.txt`
   - `README.md`
   - `.gitignore`
3. Go to Streamlit Community Cloud.
4. Create a new app and select your repository.
5. Set the main file to `app.py`.
6. Deploy.
7. Upload the retail CSV using the app's sidebar.

The CSV does not need to be committed to GitHub because the application provides a file uploader.

## Project logic

The application follows the original project logic:

- Required transaction columns are checked.
- Invalid dates/numbers, duplicate rows, zero quantities, non-positive prices, and non-positive transaction values are removed.
- Returned/cancelled transactions are removed by retaining positive quantities.
- RFM features are calculated per customer.
- RFM values are transformed using `log1p`.
- Features are standardized with `StandardScaler`.
- K-Means models are evaluated for K=2 through K=10, subject to dataset size.
- The K with the highest tested silhouette score is selected.
- Cluster profiles are mapped to descriptive customer segments.
