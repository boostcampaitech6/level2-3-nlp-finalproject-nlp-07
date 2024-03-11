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

#NOTE - maximus 64 genre tokens
GENRE_TOKEN_LIST = ['Rock', 'Pop', 'Jazz', 'Unk']
GENRE_TOKEN_LIST = ['Genre_'+genre for genre in GENRE_TOKEN_LIST]

#NOTE - maximus 64 cut tokens
CUT_TOKEN_LIST = ['Cut_'+str(i) for i in range(64)] + ['Cut_Unk']

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
    print(f'MMM Tokenizer bandwith : 0 ~ {mmm}')
    
    # Add genre token
    for genre_tk in GENRE_TOKEN_LIST:
        tokenizer.add_to_vocab(genre_tk)
    # Add genre Unused token
    for i in range(40-len(GENRE_TOKEN_LIST)):
        tokenizer.add_to_vocab(f'Genre_{i}')
    genre = len(tokenizer)-1
    print(f'Genre Tokenizer bandwith : {mmm+1} ~ {genre}')
    
    # Add cut(bar4) token
    for cut_tk in CUT_TOKEN_LIST:
        tokenizer.add_to_vocab(cut_tk)
    # Add cut Unused token
    for i in range(64-len(CUT_TOKEN_LIST)):
        tokenizer.add_to_vocab(f'Cut_{i}')
    cut = len(tokenizer)-1
    print(f'Cut Tokenizer bandwith : {genre+1} ~ {cut}')
    
    print(f'Total Tokenizer bandwith : 0 ~ {cut}, ({len(tokenizer)} tokens)')
    return tokenizer