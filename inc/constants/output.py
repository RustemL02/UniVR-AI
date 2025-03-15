from string import Template

from inc.constants.colors import Colors


# Messages style.
TITLE   = Template(Colors.CYAN   + "$msg" + Colors.RESET)
INFO    = Template(Colors.BLUE   + "$msg" + Colors.RESET)
ERROR   = Template(Colors.RED    + "$msg" + Colors.RESET)
WARNING = Template(Colors.YELLOW + "$msg" + Colors.RESET)
SUCCESS = Template(Colors.GREEN  + "$msg" + Colors.RESET)


class GeneralMessages:
    """
    A class to hold general message templates.

    Attributes:
        CORRECT (str): A template for a correct message.

    """

    CORRECT = SUCCESS.substitute(msg = "Correct!")


class SearchMessages(GeneralMessages):
    """
    A class contains specific error messages related to search algorithms.

    Attributes:
        NOT_CORRECT_SOLUTION (str): A template message indicating that the solution is not correct.
        NOT_CORRECT_TIME_COST (str): A template message indicating that the amount nodes explored is not correct.
        NOT_CORRECT_SPACE_COST (str): A template message indicating that the max amount nodes in memory is not correct.
        NOT_CORRECT_ITERATIONS (str): A template message indicating that the amount iterations is not correct.

    """

    NOT_CORRECT_SOLUTION   = ERROR.substitute(msg = "The solution is not correct, should be: {}")
    NOT_CORRECT_TIME_COST  = ERROR.substitute(msg = "The number of nodes explored is not correct, should be: {}")
    NOT_CORRECT_SPACE_COST = ERROR.substitute(msg = "The max number of nodes in memory is not correct, should be: {}")
    NOT_CORRECT_ITERATIONS = ERROR.substitute(msg = "The number of iterations is not correct, should be: {}")


class PolicyMessages(GeneralMessages):
    """
    A class contains specific error messages related to MDP and RL algorithms.

    Attributes:
        NOT_CORRECT_POLICY (str): A template for a message indicating that the policy is not optimal.

    """

    NOT_CORRECT_POLICY = ERROR.substitute(msg = "The policy is not optimal, should be: \n{}")
