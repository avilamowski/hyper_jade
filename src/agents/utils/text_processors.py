import re

def clean_llm_response(self, raw_content: str) -> str:
    """
    Clean LLM response by removing think tags, explanations, and other non-requirement content
    
    Args:
        raw_content: The raw response from the LLM
        
    Returns:
        Cleaned content containing only the requirements
    """
    content = raw_content
    
    # Remove think tags and their content
    
    # Remove <think>...</think> tags and their content
    content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL)
    
    # Remove <thinking>...</thinking> tags and their content
    content = re.sub(r'<thinking>.*?</thinking>', '', content, flags=re.DOTALL)
    
    # Remove <reasoning>...</reasoning> tags and their content
    content = re.sub(r'<reasoning>.*?</reasoning>', '', content, flags=re.DOTALL)
    
    # Remove <analysis>...</analysis> tags and their content
    content = re.sub(r'<analysis>.*?</analysis>', '', content, flags=re.DOTALL)
    
    # Find the first occurrence of a dash (-) and cut everything before it
    dash_index = content.find('-')
    if dash_index != -1:
        content = content[dash_index:]
    
    # If no dash found, try to find numbered requirements
    if dash_index == -1:
        # Look for "Requirement 1:" pattern
        req_match = re.search(r'Requirement\s+\d+:', content)
        if req_match:
            content = content[req_match.start():]
        else:
            # Look for numbered list (1., 2., etc.)
            num_match = re.search(r'\d+\.', content)
            if num_match:
                content = content[num_match.start():]
    
    # Clean up any leading/trailing whitespace
    content = content.strip()
    
    return content


def clean_llm_response_tags(raw_content: str) -> str:
    """
    Clean LLM response by removing think tags, explanations, and other non-requirement content
    Specifically designed for XML tags format with numbered containers

    Args:
        raw_content: The raw response from the LLM

    Returns:
        Cleaned content containing only the requirements in XML format
    """
    content = raw_content

    # Remove think tags and their content
    # Remove <think>...</think> tags and their content
    content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL)

    # Remove <thinking>...</thinking> tags and their content
    content = re.sub(r"<thinking>.*?</thinking>", "", content, flags=re.DOTALL)

    # Remove <reasoning>...</reasoning> tags and their content
    content = re.sub(r"<reasoning>.*?</reasoning>", "", content, flags=re.DOTALL)

    # Remove <analysis>...</analysis> tags and their content
    content = re.sub(r"<analysis>.*?</analysis>", "", content, flags=re.DOTALL)

    # Find the first occurrence of a numbered XML tag (e.g., <1>) and cut everything before it
    xml_match = re.search(r"<\d+>", content)
    if xml_match:
        content = content[xml_match.start() :]
    else:
        # Fallback: Find the first occurrence of a dash (-) and cut everything before it
        dash_index = content.find("-")
        if dash_index != -1:
            content = content[dash_index:]
        else:
            # If no dash found, try to find numbered requirements
            # Look for "Requirement 1:" pattern
            req_match = re.search(r"Requirement\s+\d+:", content)
            if req_match:
                content = content[req_match.start() :]
            else:
                # Look for numbered list (1., 2., etc.)
                num_match = re.search(r"\d+\.", content)
                if num_match:
                    content = content[num_match.start() :]

    # Clean up any leading/trailing whitespace
    content = content.strip()

    return content