from miditok import MMM, TokenizerConfig

GENRE_TOKEN_LIST = [
    "Genre_Rock",
    "Genre_Pop",
]
BAR4_TOKEN_LIST = ['Bar_'+str(i) for i in range(24)]

def get_custom_tokenizer(genre=False, bar4=False):
    additonal_toknes = []
    if genre:
        additonal_toknes += GENRE_TOKEN_LIST
        print("Adding genre tokens to tokenizer...")
    if bar4:
        additonal_toknes += BAR4_TOKEN_LIST
        print("Adding bar4 tokens to tokenizer...")
    
    TOKENIZER_NAME = MMM
    config = TokenizerConfig(
        num_velocities=16,
        use_chord=True,
        use_pitch_intervals=True,
        use_programs=True,
        special_tokens=["PAD", "BOS", "EOS", "MASK"]+additonal_toknes
    )

    tokenizer = TOKENIZER_NAME(config)
    return tokenizer