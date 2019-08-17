different_set = set()
for i in range(1, val.num_items):
    different_set.add(np.argmax(model.predict(val.sequences[i])))
print(val.sequences.shape[0], len(different_set))

# %%

s = [1000]
problem_count = val.sequences.shape[1]
for _ in range(100):
    item_ids = [i for i in range(problem_count) if i not in s]

    s.append(item_ids[np.argmax(model.predict(np.array(s),
                                              item_ids=np.expand_dims(np.array(item_ids), 0).T))])
print(s)

# %%


np.expand_dims(np.array(item_ids), 0).T