# Read the dataset description
from pandas import read_csv

tagged = dict([(p,w) for _,p,w in read_csv('C:/Users/bikas/PycharmProjects/Humpback_Whale_Identification-/train.csv').to_records()])
submit = [p for _,p,_ in read_csv('../input/whale-categorization-playground/sample_submission.csv').to_records()]
join   = list(tagged.keys()) + submit
len(tagged),len(submit),len(join),list(tagged.items())[:5],submit[:5]