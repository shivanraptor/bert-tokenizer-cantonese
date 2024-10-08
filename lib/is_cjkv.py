import regex as re

#pattern = re.compile(r'[\p{Unified_Ideograph}\u3006\u3007]')
pattern = re.compile(r'[\u4e00-\u9fff]+')

def is_cjkv(token: str) -> bool:
    return bool(pattern.fullmatch(token))
