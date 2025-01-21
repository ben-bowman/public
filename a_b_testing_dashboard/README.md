
# A/B Testing Dashboard

This is a containerized Streamlit application for performing A/B testing. It allows users to upload their data, select control and treatment groups, and analyze the results using statistical tests. The tool also provides visualizations of the data and actionable conclusions based on statistical significance.

## Features
- Upload CSV data files.
- Select columns for control and treatment groups.
- Perform independent t-tests for comparing groups.
- Visualize data distributions with interactive charts.
- Generate insights and conclusions about statistical significance.

## Prerequisites
- Docker installed on your machine.

## How to Use

1. **Clone the Repository**:
   ```bash
   git clone <repository-url>
   cd <repository-folder>
   ```

   **Alternatively**, download the repository directly from the GitHub UI:
   - Go to the repository's GitHub page.
   - Click the green **Code** button.
   - Select **Download ZIP** and extract it to your desired folder.
   - Navigate to the extracted folder using your terminal or command prompt.

2. **Install Docker**:
   Follow the official instructions to install Docker on your system:
   - **Windows/Mac**: Download and install Docker Desktop from [https://www.docker.com/products/docker-desktop/](https://www.docker.com/products/docker-desktop/).
   - **Linux**: Follow the Linux-specific instructions at [https://docs.docker.com/engine/install/](https://docs.docker.com/engine/install/).

   Once installed, verify Docker is running by executing:
   ```bash
   docker --version
   ```

3. **Build the Docker Image**:
   ```bash
   docker build -t ab-testing-tool .
   ```

4. **Run the Docker Container**:
   ```bash
   docker run -p 8501:8501 ab-testing-tool
   ```

5. **Access the Application**:
   Open your web browser and go to: [http://localhost:8501](http://localhost:8501)

## Example Usage
- **Scenario**: You have a dataset containing conversion rates for a website's control and treatment groups.
  - Upload the CSV file via the sidebar.
  - Select the appropriate columns for control and treatment groups.
  - View the statistical analysis results and data visualizations.

- **Sample Data**: You can use the provided sample data for testing purposes. [Download sample_data.csv](./sample_data.csv)

## Requirements
This application uses the following Python libraries:
- `streamlit` - For creating the web app.
- `pandas` - For data manipulation.
- `scipy` - For statistical testing.
- `plotly` - For creating interactive visualizations.

## Commands Recap
- Build the image:
  ```bash
  docker build -t ab-testing-tool .
  ```
- Run the container:
  ```bash
  docker run -p 8501:8501 ab-testing-tool
  ```

## Additional Notes
- Ensure your CSV file includes the columns for control and treatment groups with numerical data.
- This tool is ideal for initial analyses of A/B tests in business or product management scenarios.

Feel free to contribute or suggest improvements!