#import numpy as np
import pandas as pd
from data_profiling import ProfileReport
from pathlib import Path

DATA_DIR = Path("hockey_development_data")
#REPORT_DIR = Path("reports")
HTML_DIR = Path("reports/html")
JSON_DIR = Path("reports/json")

HTML_DIR.mkdir(parents=True, exist_ok=True)
JSON_DIR.mkdir(parents=True, exist_ok=True)

#REPORT_DIR.mkdir(exist_ok=True)


for file in DATA_DIR.glob("*.csv"):
    
    print(f"Profiling: {file.name}")

    try:
        df = pd.read_csv(file, low_memory=False)
        
        profile = ProfileReport(
            df,
            title=f"Profiling Report - {file.stem}",
            explorative=True
        )

        #html 
        html_file = HTML_DIR / f"{file.stem}.html"
        profile.to_file(html_file)

        #json 
        json_file = JSON_DIR / f"{file.stem}.json"
        with open(json_file, "w", encoding="utf-8") as f:
            f.write(profile.to_json())


        print(f"Completed the report for {file.name}")

    except Exception as e:
        print(f"Error - Could not process {file.name} : {e}")

print("Completed all reports")


