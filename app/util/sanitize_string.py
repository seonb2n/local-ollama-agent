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


def extract_code_only(llm_response: str) -> str:
    """
    LLM 응답에서 순수 코드만 추출합니다.
    설명, 주석, 마크다운 등을 모두 제거하고 실행 가능한 코드만 반환합니다.
    """
    response = llm_response.strip()
    
    # 1. 마크다운 코드 블록 추출
    code_block_patterns = [
        r'```(?:python|py|javascript|js|java|go|rust|cpp|c|csharp)?\s*\n(.*?)```',
        r'```\s*\n(.*?)```'
    ]
    
    for pattern in code_block_patterns:
        matches = re.findall(pattern, response, re.DOTALL)
        if matches:
            # 가장 긴 코드 블록을 선택 (메인 코드일 가능성이 높음)
            code = max(matches, key=len).strip()
            if code:
                return _clean_extracted_code(code)
    
    # 2. 코드 블록이 없으면 전체 응답에서 코드 추출 시도
    lines = response.split('\n')
    code_lines = []
    in_code_section = False
    
    for line in lines:
        # 설명 문장 패턴 (한국어/영어)
        if re.match(r'^(다음|아래|위|이|이것|Here|This|The|Below|Above)', line.strip()):
            continue
        if re.match(r'.*[입니다|습니다|해주세요|됩니다|합니다]\s*:?\s*$', line.strip()):
            continue
        if re.match(r'^\*\*.*\*\*$', line.strip()):  # **제목** 형태
            continue
        if line.strip().startswith('#'):  # 마크다운 헤더
            continue
            
        # 코드 라인 판별
        stripped = line.strip()
        if (stripped and 
            not stripped.startswith('//') and  # 주석 제외
            not stripped.startswith('#') and   # 파이썬 주석 제외 (import 등은 포함)
            (stripped.startswith(('import ', 'from ', 'def ', 'class ', 'if ', 'for ', 'while ', 'try:', 'except', 'with ', 'return', '@')) or
             '=' in stripped or
             stripped.endswith((':',)) or
             in_code_section)):
            code_lines.append(line)
            in_code_section = True
        elif in_code_section and (line.startswith('    ') or line.startswith('\t') or not stripped):
            # 들여쓰기된 라인이거나 빈 라인 (코드 블록 내부)
            code_lines.append(line)
        elif stripped and in_code_section:
            # 코드 섹션이 끝남
            break
    
    if code_lines:
        return _clean_extracted_code('\n'.join(code_lines))
    
    # 3. 마지막 시도: 전체 응답 반환 (이미 코드일 수 있음)
    return _clean_extracted_code(response)


def _clean_extracted_code(code: str) -> str:
    """
    추출된 코드를 최종 정리합니다.
    """
    lines = code.split('\n')
    cleaned_lines = []
    
    for line in lines:
        # 순수 설명 라인 제거
        stripped = line.strip()
        if (stripped and 
            not re.match(r'^(이|이것은|다음은|아래는|위는|Here|This|The|Below|Above)', stripped) and
            not re.match(r'.*[입니다|습니다|해주세요|됩니다|합니다]\s*:?\s*$', stripped) and
            not stripped.startswith('**') and
            not (stripped.startswith('#') and not stripped.startswith('#!'))):
            cleaned_lines.append(line)
    
    return '\n'.join(cleaned_lines).strip()