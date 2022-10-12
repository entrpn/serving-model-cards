import re
import math
import torch

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

re_attention = re.compile(r"""
\\\(|
\\\)|
\\\[|
\\]|
\\\\|
\\|
\(|
\[|
:([+-]?[.\d]+)\)|
\)|
]|
[^\\()\[\]:]+|
:
""", re.X)

def get_target_prompt_token_count(token_count):
    return math.ceil(max(token_count, 1) / 75) * 75

def parse_prompt_attention(text):
    """
    Parses a string with attention tokens and returns a list of pairs: text and its assoicated weight.
    Accepted tokens are:
      (abc) - increases attention to abc by a multiplier of 1.1
      (abc:3.12) - increases attention to abc by a multiplier of 3.12
      [abc] - decreases attention to abc by a multiplier of 1.1
      \( - literal character '('
      \[ - literal character '['
      \) - literal character ')'
      \] - literal character ']'
      \\ - literal character '\'
      anything else - just text
    >>> parse_prompt_attention('normal text')
    [['normal text', 1.0]]
    >>> parse_prompt_attention('an (important) word')
    [['an ', 1.0], ['important', 1.1], [' word', 1.0]]
    >>> parse_prompt_attention('(unbalanced')
    [['unbalanced', 1.1]]
    >>> parse_prompt_attention('\(literal\]')
    [['(literal]', 1.0]]
    >>> parse_prompt_attention('(unnecessary)(parens)')
    [['unnecessaryparens', 1.1]]
    >>> parse_prompt_attention('a (((house:1.3)) [on] a (hill:0.5), sun, (((sky))).')
    [['a ', 1.0],
     ['house', 1.5730000000000004],
     [' ', 1.1],
     ['on', 1.0],
     [' a ', 1.1],
     ['hill', 0.55],
     [', sun, ', 1.1],
     ['sky', 1.4641000000000006],
     ['.', 1.1]]
    """
    res = []
    round_brackets = []
    square_brackets = []

    round_bracket_multiplier = 1.1
    square_bracket_multiplier = 1 / 1.1

    def multiply_range(start_position, multiplier):
        for p in range(start_position, len(res)):
            res[p][1] *= multiplier

    for m in re_attention.finditer(text):
        text = m.group(0)
        weight = m.group(1)

        if text.startswith('\\'):
            res.append([text[1:], 1.0])
        elif text == '(':
            round_brackets.append(len(res))
        elif text == '[':
            square_brackets.append(len(res))
        elif weight is not None and len(round_brackets) > 0:
            multiply_range(round_brackets.pop(), float(weight))
        elif text == ')' and len(round_brackets) > 0:
            multiply_range(round_brackets.pop(), round_bracket_multiplier)
        elif text == ']' and len(square_brackets) > 0:
            multiply_range(square_brackets.pop(), square_bracket_multiplier)
        else:
            res.append([text, 1.0])

    for pos in round_brackets:
        multiply_range(pos, round_bracket_multiplier)

    for pos in square_brackets:
        multiply_range(pos, square_bracket_multiplier)

    if len(res) == 0:
        res = [["", 1.0]]

    # merge runs of identical weights
    i = 0
    while i + 1 < len(res):
        if res[i][1] == res[i + 1][1]:
            res[i][0] += res[i + 1][0]
            res.pop(i + 1)
        else:
            i += 1

    return res

class FrozenCLIPEmbedderWithCustomWords(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.token_mults = {}

        self.comma_token = [v for k, v in self.model.tokenizer.get_vocab().items() if k == ',</w>'][0]
        
        tokens_with_parens = [(k, v) for k, v in self.model.tokenizer.get_vocab().items() if '(' in k or ')' in k or '[' in k or ']' in k]
        for text, ident in tokens_with_parens:
            mult = 1.0
            for c in text:
                if c == '[':
                    mult /= 1.1
                if c == ']':
                    mult *= 1.1
                if c == '(':
                    mult *= 1.1
                if c == ')':
                    mult /= 1.1

            if mult != 1.0:
                self.token_mults[ident] = mult
    
    def tokenize_line(self, line):
        id_end = self.model.tokenizer.eos_token_id

        parsed = parse_prompt_attention(line)

        tokenized = self.model.tokenizer([text for text, _ in parsed], truncation=False, add_special_tokens=False)["input_ids"]

        remade_tokens = []
        multipliers = []
        last_comma = -1

        for tokens, (_, weight) in zip(tokenized, parsed):
            i = 0
            while i < len(tokens):
                token = tokens[i]

                if token == self.comma_token:
                    last_comma = len(remade_tokens)
                elif max(len(remade_tokens), 1) % 75 == 0 and last_comma != -1:
                    last_comma +=1
                    reloc_tokens = remade_tokens[last_comma:]
                    reloc_mults = multipliers[last_comma:]

                    remade_tokens = remade_tokens[:last_comma]
                    length = len(remade_tokens)

                    rem = int(math.ceil(length/75)) * 75 - length
                    remade_tokens += [id_end] * rem + reloc_tokens
                    multipliers = multipliers[:last_comma] + [1.0] * rem + reloc_mults
                i += 1
                
                remade_tokens.append(token)
                multipliers.append(weight)
        
        token_count = len(remade_tokens)
        prompt_target_length = get_target_prompt_token_count(token_count)
        tokens_to_add = prompt_target_length - len(remade_tokens)

        remade_tokens = remade_tokens + [id_end] * tokens_to_add
        multipliers = multipliers + [1.0] * tokens_to_add

        return remade_tokens, multipliers, token_count

    def process_text(self, prompts):
        remade_batch_tokens = []
        token_count = 0

        batch_multipliers = []
        for prompt in prompts:
            remade_tokens, multipliers, current_token_count = self.tokenize_line(prompt)
            token_count = max(current_token_count, token_count)
        
            remade_batch_tokens.append(remade_tokens)
            batch_multipliers.append(multipliers)
        
        return batch_multipliers, remade_batch_tokens, token_count

    def process_tokens(self, remade_batch_tokens, batch_multipliers):
        remade_batch_tokens = [[self.model.tokenizer.bos_token_id] + x[:75] + [self.model.tokenizer.eos_token_id] for x in remade_batch_tokens]
        batch_multipliers = [[1.0] + x[:75] + [1.0] for x in batch_multipliers]
            
        tokens = torch.asarray(remade_batch_tokens).to(device)
        outputs = self.model.transformer(input_ids=tokens, output_hidden_states=-1)

        z = outputs.last_hidden_state

        # restoring original mean is likely not correct, but it seems to work well to prevent artifacts that happen otherwise
        batch_multipliers_of_same_length = [x + [1.0] * (75 - len(x)) for x in batch_multipliers]
        batch_multipliers = torch.asarray(batch_multipliers_of_same_length).to(device)
        original_mean = z.mean()
        z *= batch_multipliers.reshape(batch_multipliers.shape + (1,)).expand(z.shape)
        new_mean = z.mean()
        z *= original_mean / new_mean

        return z

    def forward(self, text):
        batch_multipliers, remade_batch_tokens, _ = self.process_text(text)

        z = None
        i = 0
        while max(map(len, remade_batch_tokens)) != 0:
            rem_tokens = [x[75:] for x in remade_batch_tokens]
            rem_multipliers = [x[75:] for x in batch_multipliers]

            z1 = self.process_tokens([x[:75] for x in remade_batch_tokens], [x[:75] for x in batch_multipliers])
            z = z1 if z is None else torch.cat((z,z1), axis=-2)
            remade_batch_tokens = rem_tokens
            batch_multipliers = rem_multipliers
            i += 1
        return z



