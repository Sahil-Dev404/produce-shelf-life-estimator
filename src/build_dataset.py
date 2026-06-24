import os
import pandas as pd
import random

#paths as per the folder structure
RAW_DATA_DIR="./data/raw"
PROCESSED_DATA_DIR="./data/processed"

#ensure the processed folder exists
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)

#the baseline mapping and variance 
# variance is added so the model learns in an cutinous cureve not in static no's
LABEL_HEURISTIC={
    "unripe":(7.0,1.0), #between 6.0 to 8.0
    "ripe":(3.0,1.0), #between 2.0 to 4.0
    "overripe":(1.0,0.5), #between 0.5 to 1.5
    "rotten":(0.0,0.0) # strictly 0.0 
}

splits=["train" ,"valid" ,"test"]

for split in splits:
    split_dir=os.path.join(RAW_DATA_DIR, split)
    records=[]

    if not os.path.exists(split_dir):
        print(f"Warning missing folder {split_dir}. skipping.")
        continue

    print(f"Processing '{split}' dataset....")

    #iterating through the 4 category folders inside the current split
    for category, (base_days, variance) in LABEL_HEURISTIC.items():
        category_path=os.path.join(split_dir, category)

        if os.path.exists(category_path):
            for file_name in os.listdir(category_path):
                #only processing the images
                if file_name.lower().endswith(('.png','.jpg','.jpeg')):
                    #use relative path so that the script works on an machine
                    file_path=os.path.join(category_path, file_name)

                    #calculate the days remaing with variance
                    random_shift=random.uniform(-variance, variance)
                    assigned_days=max(0.0, round(base_days+random_shift,2))

                    records.append({
                        "image_path":file_path,
                        "days_remaining":assigned_days
                    })


    # conerting the records into a dataframe
    df=pd.DataFrame(records)

    #shuffle the dataset so the netwrok doesnt memorize the file order
    df=df.sample(frac=1).reset_index(drop=True)

    #export the mapped data to a csv
    output_file=os.path.join(PROCESSED_DATA_DIR, f"{split}_mapped.csv")
    df.to_csv(output_file, index=False)

    print(f"-> successfully saved {len(df)} records to {output_file}")

print("\n data eng completed .")