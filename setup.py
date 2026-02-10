"""Setup script to create project structure."""

import os
from config import DATA_DIR, OUTPUT_DIR, LOG_DIR

def create_project_structure():
    """Create the project directory structure."""
    directories = [
        DATA_DIR,
        OUTPUT_DIR,
        LOG_DIR,
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"Created/verified directory: {directory}")
    
    # Create a README in the data directory
    readme_path = os.path.join(DATA_DIR, "README.md")
    if not os.path.exists(readme_path):
        with open(readme_path, 'w') as f:
            f.write("# Data Directory\n\n")
            f.write("Place your data files here:\n")
            f.write("- `all_features_df.csv`\n")
            f.write("- `thinned_snapshots_60s.csv`\n")
            f.write("- `rationale_from_llm.xlsx`\n")
            f.write("- `final_labels.csv` (for evaluation)\n")
    
    print("\nProject structure created successfully!")
    print(f"Data directory: {DATA_DIR}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Log directory: {LOG_DIR}")

if __name__ == "__main__":
    create_project_structure()