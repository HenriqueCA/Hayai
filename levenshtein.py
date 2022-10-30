def levenshtein(string1 : str, string2 : str) -> int:
    """Calculates edit distance between two arbitrary strings.

    Args:
        string1 (str): A string to be compared
        string2 (str): A string to be compared

    Returns:
        int: The minimum edit distance between the two strings
    """
    
    dp = [[0]*(len(string1)+1) for _ in range(2)]

    dp[0] = list(range(len(string1) + 1))

    for pos_string2 in range(1,len(string2)):
        for pos_string1 in range(len(string1)):
            current_row = not pos_string2&1
            if not pos_string1:
                dp[current_row][pos_string1] = pos_string2

            elif string1[pos_string1 - 1] == string2[pos_string2 - 1]:
                dp[current_row][pos_string1] = dp[not current_row][pos_string1-1]
            
            else:
                dp[current_row] = 1 + min(dp[not current_row][pos_string1],
                                  dp[current_row][pos_string1-1],
                                  dp[not current_row][pos_string1-1])
    
    return dp[not len(string2) & 1][len(string1)]