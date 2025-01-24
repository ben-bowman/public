# dbt_northwind

This is a dbt project using the Northwind database. The project demonstrates dbt functionality with a SQLite backend. It includes the data and profiles needed. There are some prerequisites if you wish to run the dbt models. However, if you only want to view it, feel free to explore the repo.

## Project Setup

Follow these instructions to set up and run the dbt_northwind project.

### Prerequisites

1. **Install Python:**
   - Ensure you have Python 3.7 or higher installed.
   - You can download it from [python.org](https://www.python.org/).

2. **Install dbt-core:**
   - Install dbt using pip:
     ```bash
     pip install dbt-core
     ```

3. **Install the SQLite adapter for dbt:**
   - Install the `dbt-sqlite` adapter:
     ```bash
     pip install dbt-sqlite
     ```

4. **Clone the repository:**
   - Clone this project to your local machine:
     ```bash
     git clone <repository_url>
     cd dbt_northwind
     ```

### Configuration

1. **Use the provided `profiles.yml`:**
   - This project includes a pre-configured `profiles.yml` file in the repository under the `config/` directory.
   - Instead of modifying your global dbt `profiles.yml`, set the `DBT_PROFILES_DIR` environment variable to point to the `config/` directory:
     ```bash
     export DBT_PROFILES_DIR=$(pwd)/config
     ```

2. **Install project dependencies:**
   - This project includes a dependency dbt_utils (specified in `packages.yml`), ensure you install it using:
     ```bash
     dbt deps
     ```

3. **Initialize the dbt project:**
   - Run the following command to ensure dbt recognizes the project and everything is set up correctly:
     ```bash
     dbt debug
     ```

### Reviewing the SQL

1. **Compile the models:**
   - To generate the raw SQL files from the dbt models without executing them, use:
     ```bash
     dbt compile
     ```

2. **Locate the compiled SQL files:**
   - Compiled SQL files will be located in the `public\dbt_northwind\target\compiled\dbt_northwind\models\` directory within the project folder.

### Running the Project

1. **Run dbt models:**
   - To execute the dbt models and generate results:
     ```bash
     dbt run
     ```

2. **Run dbt tests:**
   - To validate the models and ensure data integrity, run the tests included in the project:
     ```bash
     dbt test
     ```

4. **View dbt documentation:**
   - To generate and view the project documentation:
     ```bash
     dbt docs generate
     dbt docs serve
     ```
