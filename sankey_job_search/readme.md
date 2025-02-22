# Dynamic Sankey Chart from Google Sheets

This repository contains a **Streamlit** application that dynamically loads data from a **Google Sheet** and visualizes it as a **Sankey diagram** using **Plotly**.

## Features
- **Real-time Data Sync**: Automatically updates the Sankey chart each time the app is opened by fetching data from a Google Sheet.
- **Custom Styling**: Uses a custom color scheme and improved text contrast for better readability.
- **Google Sheets Integration**: Connects seamlessly using a Google Service Account for authentication.
- **Streamlit UI**: A simple, interactive web app for data visualization.

## Installation
### 1. Clone the Repository
```bash
git clone https://github.com/your-username/sankey-google-sheets.git
cd sankey-google-sheets
```

### 2. Install Dependencies
Make sure you have Python installed, then install the required libraries:
```bash
pip install streamlit gspread pandas plotly oauth2client
```

### 3. Set Up Google Sheets API
1. **Create a Google Service Account**: Follow [Google's guide](https://cloud.google.com/iam/docs/creating-managing-service-accounts) to create a service account.
2. **Enable Google Sheets & Drive APIs** in [Google Cloud Console](https://console.cloud.google.com/).
3. **Download the JSON Key**: Save the service account credentials as `service_account.json` in the project root.
4. **Share the Google Sheet**: Give **Editor** access to the service account email (found in `service_account.json`).

### 4. Run the Application
```bash
streamlit run sankey_app.py
```

## Deployment to Streamlit Cloud
1. **Push your code to GitHub**.
2. **Go to [Streamlit Cloud](https://share.streamlit.io/)** and deploy the app.
3. **Add `service_account.json` to Streamlit Secrets**.

## Usage
- The app fetches **source**, **target**, and **value** columns from the specified Google Sheet.
- The **Sankey chart** represents the flow between different categories.
- Refresh the app to pull in the latest data.

## Author
**Ben Bowman**  
[www.benbowman.io](https://www.benbowman.io)

