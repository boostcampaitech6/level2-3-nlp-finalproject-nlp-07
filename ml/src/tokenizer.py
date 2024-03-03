from miditok import MMM, TokenizerConfig
from typing import Union, Optional
from pathlib import Path

GENRE_TOKEN_LIST = [
    "Genre_Rock",
    "Genre_Pop",
]
BAR4_TOKEN_LIST = ['Bar_'+str(i) for i in range(24)]

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

    

def get_custom_tokenizer(genre=False, bar4=False):
    additonal_toknes = []
    if genre:
        additonal_toknes += GENRE_TOKEN_LIST
        print("Adding genre tokens to tokenizer...")
    if bar4:
        additonal_toknes += BAR4_TOKEN_LIST
        print("Adding bar4 tokens to tokenizer...")
    
    TOKENIZER_NAME = CodeplayTokenizer
    config = TokenizerConfig(
        num_velocities=16,
        use_chord=True,
        use_pitch_intervals=True,
        use_programs=True,
        special_tokens=["PAD", "BOS", "EOS", "MASK"]+additonal_toknes
    )

    tokenizer = TOKENIZER_NAME(config)
    return tokenizer


