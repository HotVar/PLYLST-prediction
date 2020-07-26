import numpy as np
import pandas as pd
import json

train = pd.read_json('train.json', encoding='utf-8')
test = pd.read_json('test.json', encoding='utf-8')
# 추천 모델의 data column
unique_tags = np.array((pd.Series(np.concatenate(train['tags'].values)).value_counts()[:500].index))
unique_songs = np.array((pd.Series(np.concatenate(train['songs'].values)).value_counts()[:500].index))

# songs 예측모델 학습결과 load
res_songs = pd.DataFrame(np.load('test_songs_pred.npy'), columns=np.concatenate([unique_songs, [f'_{i}' for i in range(449)]]))
res_songs = res_songs[3000:] # 5000행부터 test 데이터

# tags 예측모델 학습결과 load
res_tags = pd.DataFrame(np.load('test_tags_pred.npy'), columns=np.concatenate([unique_tags, [f'_{i}' for i in range(749)]]))
res_tags = res_tags[5000:] # 5000행부터 test 데이터

# 최종제출 json포맷 생성
submission = []
for i, r in test.iterrows():
    item = dict()
    item['id'] = r['id']

    item['songs'] = []
    item['tags'] = []

    # 결과의 500열 까지가 tags 정보.
    pred = res_tags.iloc[i, :500]
    _sorted = pred.sort_values(ascending=False)

    # test셋에 이미 주어진 tag일 경우 예외처리
    max_pred = 10
    j = 0
    while j < max_pred:
        j += 1
        if _sorted.index[j] in r['tags']:
            max_pred += 1
        else:
            item['tags'].append(_sorted.index[j])

    # 결과의 500열 까지가 songs 정보.
    pred = res_songs.iloc[i, :500]
    _sorted = pred.sort_values(ascending=False)

    # test셋에 이미 주어진 tag일 경우 예외처리
    max_pred = 100
    j = 0
    while j < max_pred:
        j += 1
        if _sorted.index[j] in r['tags']:
            max_pred += 1
        else:
            item['songs'].append(_sorted.index[j])

    submission.append(item)

with open('results.json', 'w') as f:
    json.dump(submission, f)
