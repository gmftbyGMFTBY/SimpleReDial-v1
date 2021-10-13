import torch
from tqdm import tqdm

class RandomAccessReader(object):

    def __init__(self, filepath, endline_character='\n'):
        """
        :param filepath:  Absolute path to file
        :param endline_character: Delimiter for lines. Defaults to newline character (\n)
        """
        self._filepath = filepath
        self._endline = endline_character

    @property
    def size(self):
        return len(self._lines)

    def init(self):
        lines, has_more, start_idx = [], True, 0
        line_counter = 0
        with open(self._filepath) as f:
            while has_more:
                current = f.read(1)
                if current == self._endline:
                    now_idx = f.tell()
                    lines.append({'position': start_idx, 'length': line_counter})
                    start_idx = now_idx
                    line_counter = 0
                elif current == '':
                    break
                else:
                    line_counter += 1
                if len(lines) % 100000 == 0:
                    print(f'[!] loaded {len(lines)} lines', end='\r')
        self._lines = lines

    def init_file_handler(self):
        self.file_handler = open(self._filepath)

    def get_line(self, line_number):
        line_data = self._lines[line_number]
        self.file_handler.seek(line_data['position'])
        string = self.file_handler.read(line_data['length'])
        return string

if __name__ == "__main__":
    # reader = RandomAccessReader('train.txt')
    # reader.init()
    reader = torch.load('test.rar')
    print(f'[!] reader lines: {reader.size}')
    # torch.save(reader, 'test.rar')

    # test error
    reader.init_file_handler()
    error = 0
    for i in tqdm(range(1000000)):
        try:
            reader.get_line(i)
        except:
            error += 1
    print(f'[!] error num: {error}')
    if error == 0:
        print(f'[!] test perfectly with no errors')
