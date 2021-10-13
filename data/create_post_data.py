def read(path):
    with open(path) as f:
        lines = []
        for line in f.readlines():
            line = line.strip().split('\t')
            label, utterances = line[0], line[1:]
            if label == '1':
                if 'ubuntu' not in path:
                    utterances = [''.join(u.split()) for u in utterances]
                lines.append(utterances)
    print(f'[!] collect {len(lines)} samples from {path}')
    return lines

def write(path, datasets):
    with open(path, 'w') as f:
        for item in datasets:
            for u in item:
                f.write(f'{u}\n')
            f.write('\n')


if __name__ == "__main__":
    datasets = ['ubuntu', 'douban', 'ecommerce']
    for d in datasets:
        read_train_path = f'{d}/train.txt'
        read_test_path = f'{d}/test.txt'
        write_train_path = f'{d}/train_post.txt'
        write_test_path = f'{d}/test_post.txt'
        train_dataset = read(read_train_path)
        test_dataset = read(read_test_path)
        write(write_train_path, train_dataset)
        write(write_test_path, test_dataset)
