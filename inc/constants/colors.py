class Colors:
    """
    A class to represent ANSI escape sequences for terminal text colors and styles.

    Attributes:
        RESET (str): The ANSI escape sequence to reset the text style.
        BOLD (str): The ANSI escape sequence to make the text bold.
        BLACK (str): The ANSI escape sequence to set the text color to black.
        RED (str): The ANSI escape sequence to set the text color to red.
        GREEN (str): The ANSI escape sequence to set the text color to green.
        YELLOW (str): The ANSI escape sequence to set the text color to yellow.
        BLUE (str): The ANSI escape sequence to set the text color to blue.
        MAGENTA (str): The ANSI escape sequence to set the text color to magenta.
        CYAN (str): The ANSI escape sequence to set the text color to cyan.
        WHITE (str): The ANSI escape sequence to set the text color to white.

    """

    RESET   = "\033[0m"
    BOLD    = "\033[1m"
    BLACK   = "\033[30m"
    RED     = "\033[31m"
    GREEN   = "\033[32m"
    YELLOW  = "\033[33m"
    BLUE    = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN    = "\033[36m"
    WHITE   = "\033[37m"
