from miditok import MMM, TokenizerConfig
from typing import Union, Optional
from pathlib import Path

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