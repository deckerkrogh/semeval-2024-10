import pandas as pd

our_training_path = "../Data/MELD_train_efr.json"
our_testing_path = "../Data/MELD_test_efr.json"
our_dev_path = "../Data/MELD_dev_efr.json"

df_train = pd.read_json(our_training_path)
df_test = pd.read_json(our_testing_path)
df_dev = pd.read_json(our_dev_path)


def convert_df(df):
    #df_new = pd.DataFrame()
    speakers = []
    emotions = []
    utterances = []
    episodes = []
    triggers = []
    for i, row in df.iterrows():
        episode_num = int(row["episode"][10:])
        episodes = episodes + ([episode_num] * len(row["utterances"]))
        speakers = speakers + row["speakers"]
        emotions = emotions + row["emotions"]
        utterances = utterances + row["utterances"]
        triggers = triggers + row["triggers"]
    df_new = pd.DataFrame({'Dialogue_Id': episodes, 'Speaker': speakers, 'Emotion_name': emotions,
                           'Utterance': utterances, 'Annotate(0/1)': triggers})
    return df_new


convert_df(df_train).to_csv("../Data/MELD_train_efr_singles.csv")
convert_df(df_test).to_csv("../Data/MELD_test_efr_singles.csv")
convert_df(df_dev).to_csv("../Data/MELD_dev_efr_singles.csv")
