from miditok import MMM, TokenizerConfig
from typing import Union, Optional
from pathlib import Path
import numpy as np
from symusic import Score, TimeSignature
from miditok.constants import TIME_SIGNATURE
from miditok.utils import get_midi_ticks_per_beat

class CodeplayTokenizer(MMM):
    def __init__(self, config: TokenizerConfig):
        super().__init__(config)
    
    def save_pretrained(
        self,
        save_directory: Union[str, Path],
        *,
        config: Optional[Union[dict, "DataclassInstance"]] = None,
        repo_id: Optional[str] = None,
        push_to_hub: bool = False,
        **push_to_hub_kwargs,
    ) -> Optional[str]:
        """
        Save weights in local directory.

        Args:
            save_directory (`str` or `Path`):
                Path to directory in which the model weights and configuration will be saved.
            config (`dict` or `DataclassInstance`, *optional*):
                Model configuration specified as a key/value dictionary or a dataclass instance.
            push_to_hub (`bool`, *optional*, defaults to `False`):
                Whether or not to push your model to the Huggingface Hub after saving it.
            repo_id (`str`, *optional*):
                ID of your repository on the Hub. Used only if `push_to_hub=True`. Will default to the folder name if
                not provided.
            kwargs:
                Additional key word arguments passed along to the [`~ModelHubMixin.push_to_hub`] method.
        """
        save_directory = Path(save_directory)
        save_directory.mkdir(parents=True, exist_ok=True)

        # save model weights/files (framework-specific)
        self._save_pretrained(save_directory)

        # push to the Hub if required
        if push_to_hub:
            kwargs = push_to_hub_kwargs.copy()  # soft-copy to avoid mutating input
            if config is not None:  # kwarg for `push_to_hub`
                kwargs["config"] = config
            if repo_id is None:
                repo_id = save_directory.name  # Defaults to `save_directory` name
            return self.push_to_hub(repo_id=repo_id, **kwargs)
        return None

    def preprocess_midi(self, midi: Score) -> Score:
        r"""
        Pre-process a MIDI file to resample its time and events values.

        This method is called before parsing a MIDI's contents for tokenization.
        Its notes attributes (times, pitches, velocities) will be downsampled and
        sorted, duplicated notes removed, as well as tempos. Empty tracks (with no
        note) will be removed from the MIDI object. Notes with pitches
        outside ``self.config.pitch_range`` will be deleted.

        :param midi: MIDI object to preprocess.
        """
        # Filter time signatures.
        # We need to do this first to determine the MIDI's new time division.
        if self.config.use_time_signatures:
            self._filter_unsupported_time_signatures(midi.time_signatures)
            # We mock the first with 0, even if there are already time signatures. This
            # is required as if the MIDI only had */2 time signatures, we must make
            # sure the resampling tpq is calculated according to a maximum denom of 4
            # if the beginning of the MIDI is mocked at 4/4.
            if len(midi.time_signatures) == 0 or midi.time_signatures[0].time != 0:
                midi.time_signatures.insert(0, TimeSignature(0, *TIME_SIGNATURE))
            # The new time division is chosen depending on its highest time signature
            # denominator, and is equivalent to the highest possible tick/beat ratio.
            max_midi_denom = max(ts.denominator for ts in midi.time_signatures)
            new_tpq = int(self.config.max_num_pos_per_beat * max_midi_denom / 4)
        else:
            new_tpq = self.config.max_num_pos_per_beat

        # Resample time (not inplace)
        if midi.ticks_per_quarter != new_tpq:
            midi = midi.resample(new_tpq, min_dur=1)

        # Merge instruments of the same program / inst before preprocessing them.
        # This allows to avoid potential duplicated notes in some multitrack settings
        # This can however mess up chord detections.
        # if self.config.use_programs and self.one_token_stream:
        #     merge_same_program_tracks(midi.tracks)

        # Process time signature changes
        # We need to do it before computing the ticks_per_beat sections
        if self.config.use_time_signatures and len(midi.time_signatures) > 0:
            self._preprocess_time_signatures(
                midi.time_signatures, midi.ticks_per_quarter
            )

        # Compute resampling ratios to update times of events when several time sig,
        # and ticks per beat ratios.
        # Resampling factors are used to resample times of events when the MIDI has
        # several different time signature denominators.
        # ticks_per_beat ratios are used to adjust durations values according to the
        # the tokenizer's vocabulary, i.e. *Duration* tokens.
        if not self._note_on_off or (
            self.config.use_sustain_pedals and self.config.sustain_pedal_duration
        ):
            if self.config.use_time_signatures:
                ticks_per_beat = get_midi_ticks_per_beat(midi)
            else:
                ticks_per_beat = np.array([[midi.end(), midi.ticks_per_quarter]])
        else:
            ticks_per_beat = None
        if (
            self.config.use_time_signatures
            and len({ts.denominator for ts in midi.time_signatures}) > 1
        ):
            tpq_resampling_factors = self._get_midi_resampling_factor(midi)
        else:
            tpq_resampling_factors = None

        # Preprocess track events
        for t in range(len(midi.tracks) - 1, -1, -1):
            if len(midi.tracks[t].notes) == 0:
                del midi.tracks[t]
                continue
            # Preprocesses notes
            midi.tracks[t].notes = self._preprocess_notes(
                midi.tracks[t].notes,
                midi.tracks[t].is_drum,
                tpq_resampling_factors,
                ticks_per_beat,
            )

            if len(midi.tracks[t].notes) == 0:
                del midi.tracks[t]
                continue

            # Resample pitch bend values
            if self.config.use_pitch_bends and len(midi.tracks[t].pitch_bends) > 0:
                midi.tracks[t].pitch_bends = self._preprocess_pitch_bends(
                    midi.tracks[t].pitch_bends, tpq_resampling_factors
                )

            # Resample pedals durations
            if self.config.use_sustain_pedals and len(midi.tracks[t].pedals) > 0:
                midi.tracks[t].pedals = self._preprocess_pedals(
                    midi.tracks[t].pedals, tpq_resampling_factors, ticks_per_beat
                )

        # Process tempo changes
        if self.config.use_tempos:
            midi.tempos = self._preprocess_tempos(midi.tempos, tpq_resampling_factors)

        # We do not change key signature changes, markers and lyrics here as they are
        # not used by MidiTok (yet)

        return midi
        
GENRE_TOKEN_LIST = ['Rock', 'Pop', 'Jazz']
GENRE_TOKEN_LIST = ['Genre_Unk'] + ['Genre_'+genre for genre in GENRE_TOKEN_LIST]
GENRE_TOKEN_LIST += ['Genre_'+str(i+1) for i in range(40-len(GENRE_TOKEN_LIST))] #40
BAR2_TOKEN_LIST = ['Bar2_Unk'] + ['Bar2_'+str(i+1) for i in range(127)] # 128

def get_custom_tokenizer():
    TOKENIZER_NAME = CodeplayTokenizer
    config = TokenizerConfig(
        num_velocities=16,
        use_chord=True,
        use_pitch_intervals=True,
        use_programs=True,)
    tokenizer = TOKENIZER_NAME(config)
    
    # MMM tokenizer
    mmm = len(tokenizer)-1
    print(f'MMM Tokenizer bandwith : 0 ~ {mmm}, ({mmm+1} tokens)')
    
    # Add genre token
    for genre_tk in GENRE_TOKEN_LIST:
        tokenizer.add_to_vocab(genre_tk)
    genre = len(tokenizer)-1
    print(f'Genre Tokenizer bandwith : {mmm+1} ~ {genre}, ({genre-mmm} tokens)')
    
    # Add cut(bar4) token
    for cut_tk in BAR2_TOKEN_LIST:
        tokenizer.add_to_vocab(cut_tk)
    # Add cut Unused token
    cut = len(tokenizer)-1
    print(f'Bar2 Cut Tokenizer bandwith : {genre+1} ~ {cut}, ({cut-genre} tokens)')
    
    print(f'Total Tokenizer bandwith : 0 ~ {cut}, ({len(tokenizer)} tokens)')
    return tokenizer

def get_nnn_tokenizer(num_velocities=8):
    NNN = CodeplayTokenizer
    config = TokenizerConfig(
        num_velocities=num_velocities,
        use_programs=True
    )
    tokenizer = NNN(config)
    prev_len = len(tokenizer)
    vocabs = list(tokenizer.vocab.keys())
    
    pitches = [v for v in vocabs if v.startswith('Pitch_') ]
    velocities = [v for v in vocabs if v.startswith('Velocity_') ]
    durations = [v for v in vocabs if v.startswith('Duration_') ]
    
    for p in pitches:
        for v in velocities:
            for d in durations:
                new_tk = f'{p}+{v}+{d}'
                tokenizer.add_to_vocab(new_tk)
    
    print(f'MMM Tokenizer bandwith : 0 ~ {prev_len}, ({prev_len} tokens)')
    print(f'NNN Tokenizer bandwith : {prev_len} ~ {len(tokenizer)}, ({len(tokenizer)-prev_len} tokens)')
    return tokenizer
    
lakh_genres = ['Unk', 'Rock', 'Pop', 'Dance/Electronic', 'Jazz', 'R&B', 'Groove', 'Folk', 'Classical', 'World', 'Metal', "Children", "Trot", "Hiphop", 'Ballade']
lakh_genres += [str(i+1) for i in range(60-len(lakh_genres))] #60
lakh_emotions = ['Unk', 'nostalgia', 'excitement', 'love', 'anger', 'happiness', 'sadness','calmness', 'gratitude', 'loneliness', 'anticipation']
lakh_emotions += [str(i+1) for i in range(40-len(lakh_emotions))] #40
lakh_tempos = ['Unk', 'Moderato', 'Allegro', 'Presto', 'Andante']
lakh_tempos += [str(i+1) for i in range(40-len(lakh_tempos))] #40
def get_nnn_meta_tokenizer(num_velocities=4):
    NNN = CodeplayTokenizer
    config = TokenizerConfig(
        num_velocities=num_velocities,
        use_programs=True
    )
    tokenizer = NNN(config)
    mmm_len = len(tokenizer)
    vocabs = list(tokenizer.vocab.keys())
    
    pitches = [v for v in vocabs if v.startswith('Pitch_') ]
    velocities = [v for v in vocabs if v.startswith('Velocity_') ]
    durations = [v for v in vocabs if v.startswith('Duration_') ]
    
    for p in pitches:
        for v in velocities:
            for d in durations:
                new_tk = f'{p}+{v}+{d}'
                tokenizer.add_to_vocab(new_tk)
    nnn_len = len(tokenizer)
    
    # genre tokens
    for genre in lakh_genres:
        tokenizer.add_to_vocab(f'Genre_{genre}')
    genre_len = len(tokenizer)
    
    # emotion tokens
    for emotion in lakh_emotions:
        tokenizer.add_to_vocab(f'Emotion_{emotion}')
    emotion_len = len(tokenizer)
    
    for tempo in lakh_tempos:
        tokenizer.add_to_vocab(f'Tempo_{tempo}')
    tempo_len = len(tokenizer)
    
    print(f'MMM Tokenizer bandwith : 0 ~ {mmm_len}, ({mmm_len} tokens)')
    print(f'NNN Tokenizer bandwith : {mmm_len} ~ {nnn_len}, ({nnn_len-mmm_len} tokens)')
    print(f'Genre Tokenizer bandwith : {nnn_len} ~ {genre_len}, ({genre_len-nnn_len} tokens)')
    print(f'Emotion Tokenizer bandwith : {genre_len} ~ {emotion_len}, ({emotion_len-genre_len} tokens)')
    print(f'Tempo Tokenizer bandwith : {emotion_len} ~ {tempo_len}, ({tempo_len-emotion_len} tokens)')

    return tokenizer
    
