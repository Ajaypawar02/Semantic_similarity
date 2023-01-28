import pandas as pd
from sklearn import model_selection
from args import args
# from args import args

# Training data is in a csv file called train.csv 
df = pd.read_csv(args.train_path)
# we create a new column called kfold and fill it with -1
df["kfold"] = -1

# fetch targets
y = df.label.values

# initiate the kfold class from model_selection module
kf = model_selection.StratifiedKFold(n_splits=args.splits, shuffle=True, random_state=42)

# fill the new kfold column
for f, (t_, v_) in enumerate(kf.split(X=df, y=y)):
    df.loc[v_, 'kfold'] = f

# save the new csv with kfold column
df.to_csv("train_folds.csv", index=False)