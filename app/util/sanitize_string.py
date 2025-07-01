import re

def clean_markdown_code_blocks(code: str) -> str:
    """
    LLM이 생성한 코드에서 마크다운 코드 블록 구문을 제거합니다.

    Args:
        code (str): 마크다운 코드 블록이 포함된 코드 문자열

    Returns:
        str: 정리된 순수 코드 문자열
    """
    # 코드 블록 패턴들 정의
    patterns = [
        r'^```[\w]*\n(.*?)```$',  # ```python ... ``` 또는 ```javascript ... ``` 등
        r'^```\n(.*?)```$',  # ``` ... ```
        r'^`(.*?)`$',  # `single line code`
    ]

    cleaned_code = code.strip()

    # 각 패턴에 대해 순차적으로 정리
    for pattern in patterns:
        match = re.match(pattern, cleaned_code, re.DOTALL)
        if match:
            cleaned_code = match.group(1).strip()
            break

    # 추가적인 정리: 앞뒤 공백 및 불필요한 줄바꿈 제거
    cleaned_code = cleaned_code.strip()

    return cleaned_code