import os

max_files = 65
output_path = "finetuning_test"

with open("test.txt", "r") as f:
    lines = [line.rstrip('\n') for line in f]

finetuning_test_files = []
category_count_dict = {}
for line in lines:
    category, _ = line.split('/')
    if category == "13gliedrig":
        continue
    if category in category_count_dict.keys():
        if category_count_dict[category] != max_files:
            category_count_dict[category] +=1
            finetuning_test_files.append(line)
        else:
            continue
    else:
        category_count_dict[category] = 1
        finetuning_test_files.append(line)

with open(output_path, "w") as f:
    for line in finetuning_test_files:
        f.write(f"{line}\n")