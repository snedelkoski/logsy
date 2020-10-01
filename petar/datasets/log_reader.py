import re
import pandas as pd
import os.path


class LogReader:
    def __init__(self, log_format, log_path, rex=None, every_n=10, max_lines=False):

        self.log_path = log_path
        self.df_log = None
        self.log_format = log_format
        self.rex = rex or []
        self.every_n = every_n
        self.max_lines = max_lines

    def log_to_dataframe(self, log_file, regex, headers):
        """ Function to transform log file to dataframe
        """
        log_messages = []
        l_count = 0

        with open(log_file, 'r', encoding="latin-1") as fin:
            for i, line in enumerate(fin):
                if i % self.every_n == 0:
                    try:
                        match = regex.search(line.strip())
                        message = [match.group(header) for header in headers]
                        log_messages.append(message)
                        l_count += 1
                    except Exception as e:
                        pass
                if self.max_lines:
                    if i == self.max_lines:
                        break
        log_df = pd.DataFrame(log_messages, columns=headers)
        log_df['LineId'] = [i + 1 for i in range(l_count)]
        return log_df

    @staticmethod
    def generate_log_format_regex(log_format):
        """
        Function to generate regular expression to split log messages
        """
        headers = []
        splitters = re.split(r'(<[^<>]+>)', log_format)
        regex = ''
        for k in range(len(splitters)):
            if k % 2 == 0:
                splitter = re.sub(' +', '\\\s+', splitters[k])
                regex += splitter
            else:
                header = splitters[k].strip('<').strip('>')
                regex += '(?P<%s>.*?)' % header
                headers.append(header)
        regex = re.compile('^' + regex + '$')
        return headers, regex

    def load_data(self):
        headers, regex = self.generate_log_format_regex(self.log_format)
        return self.log_to_dataframe(self.log_path, regex, headers)
