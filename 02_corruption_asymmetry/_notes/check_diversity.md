python -c "
import json
from collections import Counter
contexts = [json.loads(l)['context'] for l in open('eidetic_training_data_train.jsonl')]
print(f'Unique contexts: {len(set(contexts))}')
print(f'Total samples: {len(contexts)}')
print('Top 10:', Counter(contexts).most_common(10))
"
