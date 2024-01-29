import pandas as pd
import tqdm

our_training_path = "../Data/MELD_train_efr.json"
our_testing_path = "../Data/MELD_test_efr.json"
our_dev_path = "../Data/MELD_dev_efr.json"

df_train = pd.read_json(our_training_path)
df_test = pd.read_json(our_testing_path)
df_dev = pd.read_json(our_dev_path)


def convert_df(df):
    df_new = pd.DataFrame()
    for i, row in tqdm.tqdm(df.iterrows()):
        episode_num = int(row["episode"][10:])
        episodes = ([episode_num] * len(row["utterances"]))
        rows_df = pd.DataFrame({'Dialogue_Id': episodes, 'Speaker': row["speakers"], 'Emotion_name': row["emotions"],
                 'Utterance': row["utterances"], 'Annotate(0/1)': row["triggers"]}, dtype=object)
        df_new = pd.concat([df_new, rows_df])
        df_new.loc[len(df_new)] = pd.Series()
    return df_new


convert_df(df_train).to_csv("../Data/MELD_train_efr_singles.csv", index=False)
convert_df(df_test).to_csv("../Data/MELD_test_efr_singles.csv", index=False)
convert_df(df_dev).to_csv("../Data/MELD_dev_efr_singles.csv", index=False)
