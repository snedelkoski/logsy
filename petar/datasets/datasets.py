from .log_reader import LogReader
import os
from os.path import join, abspath, dirname, exists
import wget
import shutil
import sys
import gzip
import tarfile
import numpy as np


def gunzip_shutil(source_filepath, dest_filepath, block_size=65536):
    with gzip.open(source_filepath, 'rb') as s_file, \
            open(dest_filepath, 'wb') as d_file:
        shutil.copyfileobj(s_file, d_file, block_size)


def unpack_tar(source_filepath, dest_filepath):
    file = tarfile.open(source_filepath, 'r')
    member = file.getmembers()[0]
    file.extract(member, path=dest_filepath)
    file.close()


def bar_progress(current, total, width=80):
    progress_message = "Downloading: %d%% [%d / %d] bytes" % (current / total * 100, current, total)
    # Don't use print() as it will print in new line every time.
    if current % 10000 == 0:
        sys.stdout.write("\r" + progress_message)
        sys.stdout.flush()


class Dataset:
    def __init__(self):
        self.log_reader = self.aux = None
        self.filename = ""
        self.dl_file = ""
        self.dl_link = ""

    def _initialize_dataset(self):
        self.dir = join(dirname(abspath(__file__)), "data")
        self.ensure_dir_exists()
        self.filepath = join(self.dir, self.filename)
        self.ensure_file_exists()

    def ensure_dir_exists(self):
        if not exists(self.dir):
            os.makedirs(self.dir)

    def ensure_file_exists(self):
        if not exists(self.filepath):
            if not exists(join(self.dir, self.dl_file)):
                wget.download(self.dl_link, out=self.dir, bar=bar_progress)
            if ".tar" in self.dl_file:
                unpack_tar(join(self.dir, self.dl_file), self.dir)
            else:
                gunzip_shutil(join(self.dir, self.dl_file), self.filepath)
            os.remove(join(self.dir, self.dl_file))

    def get_data(self):
        data = self.log_reader.load_data()
        print(data.shape)
        log_payload, true_labels = data.Content, np.where(data.t.values == '-', 0, 1)
        if self.aux != 0:
            df_anomalies = log_payload.iloc[true_labels.flatten() == 1].sample(n=self.aux).values
            df_normal = log_payload.iloc[true_labels.flatten() == 0].sample(n=self.aux).values
            return df_normal, df_anomalies
        else:
            return log_payload, true_labels

    def get_data_a(self):
        data = self.log_reader.load_data()
        log_payload, true_labels = data.Content, np.where(data.t.values == 'FATAL', 0, 1)
        if self.aux != 0:
            df_anomalies = log_payload.iloc[true_labels.flatten() == 1].sample(n=self.aux).values
            df_normal = log_payload.iloc[true_labels.flatten() == 0].sample(n=self.aux).values
            return df_normal, df_anomalies
        else:
            return log_payload, true_labels


class BGLDataset(Dataset):
    def __init__(self, log_format, regex=None, every_n=100, aux_size=0, max_lines=False):
        super().__init__()
        self.aux = aux_size
        self.regex = regex or []
        self.filename = "BGL.log"
        self.dl_file = "bgl2.gz"
        self.dl_link = "http://0b4af6cdc2f0c5998459-c0245c5c937c5dedcca3f1764ecc9b2f.r43.cf2.rackcdn.com/hpc4/bgl2.gz"
        self._initialize_dataset()
        self.log_reader = LogReader(log_format, self.filepath, rex=regex, every_n=every_n,
                                    max_lines=max_lines)


class SpiritDataset(Dataset):
    def __init__(self, log_format, regex=None, every_n=100, aux_size=0, max_lines=False):
        super().__init__()
        self.aux = aux_size
        self.regex = regex or []
        self.filename = "spirit2s"
        self.dl_file = "spirit2.gz"
        self.dl_link = "http://0b4af6cdc2f0c5998459-c0245c5c937c5dedcca3f1764ecc9b2f.r43.cf2.rackcdn.com/hpc4/spirit2.gz"
        self._initialize_dataset()
        self.log_reader = LogReader(log_format, self.filepath, rex=regex, every_n=every_n,
                                    max_lines=max_lines)


class ThunderbirdDataset(Dataset):
    def __init__(self, log_format, regex=None, every_n=100, aux_size=0, max_lines=False):
        super().__init__()
        self.aux = aux_size
        self.regex = regex or []
        self.filename = "tbird2"
        self.dl_file = "tbird2.gz"
        self.dl_link = "http://0b4af6cdc2f0c5998459-c0245c5c937c5dedcca3f1764ecc9b2f.r43.cf2.rackcdn.com/hpc4/tbird2.gz"
        self._initialize_dataset()
        self.log_reader = LogReader(log_format, self.filepath, rex=regex, every_n=every_n,
                                    max_lines=max_lines)


class InterpidScrubbedDataset(Dataset):
    def __init__(self, log_format, regex=None, every_n=100, aux_size=0, max_lines=False):
        super().__init__()
        self.aux = aux_size
        self.regex = regex or []
        self.filename = "Intrepid_RAS_0901_0908_scrubbed"
        self.dl_file = "4372-intrepid_ras_0901_0908_scrubbed.zip.tar"
        self.dl_link = "https://www.usenix.org/sites/default/files/4372-intrepid_ras_0901_0908_scrubbed.zip.tar"
        self._initialize_dataset()
        self.log_reader = LogReader(log_format, self.filepath, rex=regex, every_n=every_n,
                                    max_lines=max_lines)

