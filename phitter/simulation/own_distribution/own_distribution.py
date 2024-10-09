import random
import numpy as np


class OwnDistributions:
    def __init__(self, parameters):
        self.__parameters = parameters

        self.__first_verification()

        self.__acummulative_parameters = dict()
        acum = 0
        for key in self.__parameters.keys():
            acum += self.__parameters[key]
            self.__acummulative_parameters[key] = acum

        self.__error_detections()

    def __first_verification(self):
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

    def __error_detections(self):

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

    def ppf(self, probability: int):
        for label in self.__acummulative_parameters.keys():
            if probability <= self.__acummulative_parameters[label]:
                return label
