from __future__ import annotations

import json
from abc import ABC
from copy import deepcopy
from pathlib import Path
from typing import TYPE_CHECKING, Any

from symusic import Score
from torch import LongTensor, randint
from torch.utils.data import Dataset
from tqdm import tqdm

from miditok.constants import MIDI_FILES_EXTENSIONS
from math import ceil

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

    from miditok import MIDITokenizer

#TODO - bar4 단위로 split하는 코드로 수정할 것
def split_seq_in_subsequences(
    seq: Sequence[any], min_seq_len: int, max_seq_len: int
) -> list[Sequence[Any]]:
    r"""
    Split a sequence of tokens into subsequences.

    The subsequences will have lengths comprised between ``min_seq_len`` and
    ``max_seq_len``: ``min_seq_len <= len(sub_seq) <= max_seq_len``.

    :param seq: sequence to split.
    :param min_seq_len: minimum sequence length.
    :param max_seq_len: maximum sequence length.
    :return: list of subsequences.
    """
    sub_seq = []
    i = 0
    while i < len(seq):
        if i >= len(seq) - min_seq_len:
            break  # last sample is too short
        sub_seq.append(LongTensor(seq[i : i + max_seq_len]))
        i += len(sub_seq[-1])  # could be replaced with max_seq_len

    return sub_seq

class _DatasetABC(Dataset, ABC):
    r"""
    Abstract ``Dataset`` class.

    It holds samples (and optionally labels) and implements the basic magic methods.

    :param samples: sequence of input samples. It can directly be data, or a paths to
        files to be loaded.
    :param labels: sequence of labels associated with the samples. (default: ``None``)
    :param sample_key_name: name of the dictionary key containing the sample data when
        iterating the dataset. (default: ``"input_ids"``)
    :param labels_key_name: name of the dictionary key containing the labels data when
        iterating the dataset. (default: ``"labels"``)
    """

    def __init__(
        self,
        samples: Sequence[Any] | None = None,
        labels: Sequence[Any] | None = None,
        sample_key_name: str = "input_ids",
        labels_key_name: str = "labels",
    ) -> None:
        if samples is not None and labels is not None and len(samples) != len(labels):
            msg = "The number of samples must be the same as the number of labels"
            raise ValueError(msg)
        self.samples = samples if samples is not None else []
        self.labels = labels
        self.sample_key_name = sample_key_name
        self.labels_key_name = labels_key_name
        self.__iter_count = 0

    def reduce_num_samples(self, num_samples: int) -> None:
        pass
        # r"""
        # Reduce the size of the dataset, by keeping `num_samples` samples.

        # :param num_samples: number of samples to keep. They will be randomly picked.
        # """
        # idx = randint(0, len(self), (num_samples,))
        # self.samples = [self.samples[id_] for id_ in idx.tolist()]
        # if self.labels is not None:
        #     self.labels = [self.labels[id_] for id_ in idx.tolist()]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Mapping[str, Any]:
        item = {self.sample_key_name: self.samples[idx]}
        if self.labels is not None:
            item[self.labels_key_name] = self.labels[idx]

        return item

    def __iter__(self) -> _DatasetABC:
        return self

    def __next__(self) -> Mapping[str, Any]:
        if self.__iter_count >= len(self):
            self.__iter_count = 0
            raise StopIteration

        self.__iter_count += 1
        return self[self.__iter_count - 1]

    def __repr__(self) -> str:
        return self.__str__()

    def __str__(self) -> str:
        return "No data loaded" if len(self) == 0 else f"{len(self.samples)} samples"

# class DatasetTok(_DatasetABC): ...


class CodeplayDataset(_DatasetABC):
    r"""
    Basic ``Dataset`` loading and tokenizing MIDIs or JSON token files.

    The token ids will be stored in RAM. It outputs token sequences that can be used to
    train models.

    The tokens sequences being loaded will then be split into subsequences, of length
    comprise between ``min_seq_len`` and ``max_seq_len``.
    For example, with ``min_seq_len = 50`` and ``max_seq_len = 100``:
    * a sequence of 650 tokens will be split into 6 subsequences of 100 tokens plus one
    subsequence of 50 tokens;
    * a sequence of 620 tokens will be split into 6 subsequences of 100 tokens, the
    last 20 tokens will be discarded;
    * a sequence of 670 tokens will be split into 6 subsequences of 100 tokens plus one
    subsequence of 50 tokens, and the last 20 tokens will be discarded.

    This `Dataset` class is well suited if you have enough RAM to store all the data,
    as it does not require you to prior split the dataset into subsequences of the
    length you desire. Note that if you directly load MIDI files, the loading can take
    some time as they will need to be tokenized. You might want to tokenize them before
    once with the ``tokenizer.tokenize_midi_dataset()`` method.

    Additionally, you can use the `func_to_get_labels` argument to provide a method
    allowing to use labels (one label per file).

    :param files_paths: list of paths to files to load.
    :param min_seq_len: minimum sequence length (in num of tokens)
    :param max_seq_len: maximum sequence length (in num of tokens)
    :param tokenizer: tokenizer object, to use to load MIDIs instead of tokens.
        (default: ``None``)
    :param one_token_stream: give False if the token files contains multiple tracks,
        i.e. the first dimension of the value of the "ids" entry corresponds to
        several tracks. Otherwise, leave False. (default: ``True``)
    :param func_to_get_labels: a function to retrieve the label of a file. The method
        must take two positional arguments: the first is either a MidiFile or the
        tokens loaded from the json file, the second is the path to the file just
        loaded. The method must return an integer which correspond to the label id
        (and not the absolute value, e.g. if you are classifying 10 musicians, return
        the id from 0 to 9 included corresponding to the musician). (default: ``None``)
    :param sample_key_name: name of the dictionary key containing the sample data when
        iterating the dataset. (default: ``"input_ids"``)
    :param labels_key_name: name of the dictionary key containing the labels data when
        iterating the dataset. (default: ``"labels"``)
    """

    def __init__(
        self,
        min_seq_len: int,
        max_seq_len: int,
        midis: list | None = None,
        tokenizer: MIDITokenizer = None,
        one_token_stream: bool = True,
        func_to_get_labels: Callable[[Score | Sequence, Path], int] | None = None,
        sample_key_name: str = "input_ids",
        labels_key_name: str = "labels",
    ) -> None:
        labels = None
        samples = []
        # 기존 Miditok Dataset에선 필요한 코드이나, 우리 코드에선 없어도 동일하게 작동하므로 주석처리하였습니다.
        # if tokenizer is not None:
        #     one_token_stream = tokenizer.one_token_stream

        for i in tqdm(
            range(len(midis)),
            # desc=f"Loading data: {midis[0].parent}",
            miniters=int(len(midis) / 20),
            maxinterval=480,
        ):
            label = None
            midi, genre, emotion, tempo = midis[i]
            tokens = tokenizer(midi).tokens
            
            nnn_tokens = []
            i = 0
            while i < len(tokens):
                tk = tokens[i]
                if tk.startswith('Pitch_'):
                    new_tk = f'{tokens[i]}+{tokens[i+1]}+{tokens[i+2]}'
                    nnn_tokens.append(new_tk)
                    i += 3
                else:
                    nnn_tokens.append(tk)
                    i += 1
            tokens_ids = [tokenizer[tk] for tk in nnn_tokens]
                
            # Concat meta tokens
            meta_ids = []
            meta_ids.append(tokenizer[f'Genre_{genre}'])
            meta_ids.append(tokenizer[f'Emotion_{emotion}'])
            # meta_ids.append(tokenizer[f'Tempo_{tempo}'])
            tokens_ids = meta_ids + tokens_ids

            # Cut tokens in samples of appropriate length
            subseqs = split_seq_in_subsequences(tokens_ids, min_seq_len, max_seq_len)
            samples += subseqs
            if label is not None:
                labels += [label] * len(subseqs)
            # for loop ended
            
        if labels is not None:
            labels = LongTensor(labels)
        super().__init__(
            samples,
            labels,
            sample_key_name=sample_key_name,
            labels_key_name=labels_key_name,
        )
        
#TODO - Dataset Collator
# BOS, EOS, PAD, MASK

# url type is str of list of str
def load_midi_paths(url: str|list[str]) -> list[Path]:
    r"""
    Load the paths to the MIDI files from a file or a list of paths.

    :param url: path to a file containing the paths to the MIDI files, or a list of
        paths to the MIDI files.
    :return: list of paths to the MIDI files.
    """
    if isinstance(url, str):
        paths = [url]
    elif isinstance(url, list):
        paths = url
    else:
        raise ValueError("url must be a string or a list of strings")

    midi_paths = []
    for p in tqdm(paths):
        midi_paths += list(Path(p).glob("**/*.mid"))
    
    return midi_paths

def chunk_midi(midi_paths: list[Path], chunk_bar_num=4):
    r"""
    Chunk the midi files into chunks of 4 bars.
    
    :param midi_paths: list of paths to the MIDI files.
    :param chunk_bar_num: number of bars in a chunk.
    :return: list of the MIDI files.
    """
    chunks = []
    err_cnt = 0
    print(f'Chunking {len(midi_paths)} midi files')
    for midi_path in tqdm(midi_paths):
        try:
            midi = Score(midi_path)
        except Exception as e:
            err_cnt += 1
            continue
        
        ticks_per_chunk = midi.tpq * 4 * chunk_bar_num
        nb_chunk = midi.end() // ticks_per_chunk
        if nb_chunk < 2:
            chunks.append(midi)
            continue
        elif nb_chunk > 48:
            continue
        
        for i in range(nb_chunk):
            chunk = midi.copy()
            chunk = chunk.clip(i*ticks_per_chunk, (i+1)*ticks_per_chunk)
            chunk = chunk.shift_time(-i*ticks_per_chunk)
            chunks.append([chunk, i+1])
    
    print(f'Chunked {len(chunks)} chunks')
    print(f'Failed to chunk {err_cnt} midi files')
    return chunks


def overlap_chunk_midi(midi_paths:list[Path], chunk_bar_num=4, overlap=2):
    chunks = []
    err_cnt = 0
    print(f'Chunking {len(midi_paths)} midi files')
    for midi_path in tqdm(midi_paths):
        try:
            midi = Score(midi_path)
        except Exception as e:
            err_cnt += 1
            continue
        
        ticks_per_chunk = midi.tpq * 4 * chunk_bar_num
        ticks_per_overlap = midi.tpq * 4 * overlap
        nb_chunk = ceil(midi.end() / (ticks_per_chunk - ticks_per_overlap))
        if nb_chunk < 2:
            chunks.append(midi)
            continue
        elif nb_chunk > 60:
            continue
        
        for i in range(0, midi.end(), ticks_per_chunk - ticks_per_overlap):
            chunk = midi.copy()
            chunk = chunk.clip(i, i+ticks_per_chunk)
            chunk = chunk.shift_time(-i)
            chunks.append([chunk, i+1])
            
    print(f'Chunked {len(chunks)} chunks')
    print(f'Failed to chunk {err_cnt} midi files')
    return chunks

def meta_chunk_midi(midi_paths:list, chunk_bar_num=4, overlap=2):
    print(f'Chunking {len(midi_paths)} midi files')
    
    chunks = []
    err_cnt = 0
    # [path, genre, emotion, tempo]
    for path, genre, emotion, tempo in midi_paths:
        try:
            midi = Score(path)
        except Exception as e:
            err_cnt += 1
            continue
        
        ticks_per_chunk = midi.tpq * 4 * chunk_bar_num
        ticks_per_overlap = midi.tpq * 4 * overlap
        nb_chunk = ceil(midi.end() / (ticks_per_chunk - ticks_per_overlap))
        if nb_chunk < 2:
            chunks.append([midi, genre, emotion, tempo])
            continue
        elif nb_chunk > 60:
            continue
        
        for i in range(0, midi.end(), ticks_per_chunk - ticks_per_overlap):
            chunk = midi.copy()
            chunk = chunk.clip(i, i+ticks_per_chunk)
            chunk = chunk.shift_time(-i)
            chunks.append([chunk, genre, emotion, tempo])
        
    print(f'Chunked {len(chunks)} chunks')
    print(f'Failed to chunk {err_cnt} midi files')
    
    return chunks