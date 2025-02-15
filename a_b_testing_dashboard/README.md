
# A/B Testing Dashboard

This is a containerized Streamlit application for performing A/B testing. It allows users to upload their own data, select control and treatment groups, and analyze the results using t test. The tool also provides visualizations of the data and actionable conclusions based on statistical significance (P-value). There is default sample data included in the repo. You can also see the dashboard on [Streamlit Cloud](https://https://a-b-test.streamlit.app/).

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
   git clone https://github.com/ben-bowman/public.git
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

   ### For Windows Users:
   - Open **PowerShell** or **Command Prompt**.
   - Navigate to the directory containing the `Dockerfile` and other project files using:
     ```bash
     cd /path/to/project-directory
     ```
   - To confirm you're in the correct directory, use:
     - In **Command Prompt**: 
       ```bash
       dir
       ```
     - In **PowerShell**: 
       ```bash
       Get-ChildItem
       ```
   - Build the Docker image using:
     ```bash
     docker build -t ab-testing-tool .
     ```
   - If you encounter an error like "`docker buildx build` requires exactly 1 argument," use the following command instead:
     ```bash
     docker buildx build --tag ab-testing-tool .
     ```

   ### For Mac/Linux Users:
   - Open a terminal.
   - Navigate to the directory containing the `Dockerfile` and other project files using:
     ```bash
     cd /path/to/project-directory
     ```
   - To confirm you're in the correct directory, use:
     ```bash
     ls
     ```
   - Build the Docker image:
     ```bash
     docker build -t ab-testing-tool .
     ```
   - If you're not in the same directory, provide the full path to the project directory:
     ```bash
     docker build -t ab-testing-tool /full/path/to/project-directory
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

- **Sample Data**: You can use the provided sample data for testing purposes.

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
