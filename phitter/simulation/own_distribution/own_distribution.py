import random
import numpy as np


class OwnDistributions:
    def __init__(self, parameters: dict):
        """Creates the "OwnDistributions" Class

        Args:
            parameters (dict): Parameters of that distribution. all keys should be numbers greater or equals than zero. All values must sum up to 1.
        """
        self.__parameters = parameters

        self.__first_verification()

        self.__acummulative_parameters = dict()
        acum = 0
        for key in self.__parameters.keys():
            acum += self.__parameters[key]
            self.__acummulative_parameters[key] = acum

        self.__error_detections()

    def __first_verification(self) -> None:
        """Verify if the keys are integers greater than zero, and verify if all values are floats."""
        # Quick verification
        if isinstance(self.__parameters, dict) == False:
            raise ValueError("You must pass a dictionary")

        for key in self.__parameters.keys():
            if isinstance(key, int) == False:
                raise ValueError(
                    f"""All keys must be integers greater or equal than 0."""
                )
            elif key < 0:
                raise ValueError(
                    f"""All keys must be integers greater or equal than 0."""
                )

            if isinstance(self.__parameters[key], float) == False:
                raise ValueError(f"""All keys must be floats.""")

    def __error_detections(self) -> None:
        """Identify the values that are greater than 1 or less than 0. Verify if accumulative probabilities are less or greater than 1. Must sum 1"""

        for key in self.__parameters.keys():
            if self.__parameters[key] <= 0 or self.__parameters[key] >= 1:
                raise ValueError(
                    f"""All probabilities must be greater than 0 and less than 1. You have a value of {self.__parameters[key]} for key {key}"""
                )

            if (
                self.__acummulative_parameters[key] > 1
                or self.__acummulative_parameters[key] <= 0
            ):
                raise ValueError(
                    f"""All probabilities must be add up to 1 and must be greater than 0. You have a acummulative value of {self.__acummulative_parameters[key]}"""
                )
            else:
                last = self.__acummulative_parameters[key]

        if last != 1:
            raise ValueError(
                f"""All probabilities must be add up to 1, your probabilities sum a total of {last}"""
            )

    def ppf(self, probability: int) -> int:
        """Assign a label according to a probability given by the created distribution

        Args:
            probability (int): Number between 0 and 1

        Returns:
            int: Returns label according to probability
        """
        for label in self.__acummulative_parameters.keys():
            if probability <= self.__acummulative_parameters[label]:
                return label
