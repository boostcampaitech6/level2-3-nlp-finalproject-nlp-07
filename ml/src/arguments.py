import argparse


def set_arguments(args_name='default'):
    if args_name == 'default':
        parser = default_arguments()
    
    return parser

def default_arguments(args_name='default'):
    parser = argparse.ArgumentParser(description=f'Codeplay-{args_name}-Arguments')
    
    parser.add_argument('--seed', type=int, default=2024, 
                        help='random seed (default: 2024)')
    parser.add_argument('--deterministic', type=bool, default=False, 
                        help='deterministic (default: False)')
    
    # data params
    parser.add_argument('--dataset', type=str, default='lakh-clean',
                        help='dataset (default: lakh-clean)')
    parser.add_argument('--max_seq_len', type=int, default=1024,
                        help='max sequence length (default: 1024)')
    parser.add_argument('--batch_size', type=int, default=16, 
                        help='input batch size for training (default: 16)')
    parser.add_argument('--tokenizer', type=str, default='NNN-vel4',
                        help='tokenizer (default: NNN-vel4)')
    parser.add_argument('--chunk_bar_num', type=int, default=4,
                        help='chunk bar number (default: 4)')
    parser.add_argument('--overlap', type=int, default=0,
                        help='overlap (default: 0)')
    
    # model params
    parser.add_argument('--model', type=str, default='gpt2',
                        help='model (default: gpt2)')
    parser.add_argument('--n_layer', type=int, default=6,
                        help='number of layers (default: 6)')
    parser.add_argument('--n_head', type=int, default=6,
                        help='number of heads (default: 6)')
    parser.add_argument('--n_emb', type=int, default=768,
                        help='embedding dimension (default: 768)')
    
    # trainer
    parser.add_argument('--steps', type=int, default=400,
                        help='number of steps (default: 400)')
    
    
    return parser