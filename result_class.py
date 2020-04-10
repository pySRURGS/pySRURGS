import numpy as np 


def str_e(my_number):
    '''
        return a number in consistent scientific notation
    '''
    return "{:.2E}".format(my_number)


class Result(object):
    """
    Stores the result of a single run of pySRURGS

    Parameters
    ----------
    simple_equation: string
        The simplified equation string for this run of pySRURGS

    equation: string
        The pySRURGS equation string

    MSE: float like
        The mean squared error of this proposed model

    R2: float like
        The coefficient of determination of this proposed model

    params: lmfit.Parameters
        The fitting parameters of this model. Will be saved as a numpy.array
        and not lmfit.Parameters

    Returns
    -------
    self: Result
    """

    def __init__(self, simple_equation, equation, MSE, R2, params):
        self._simple_equation = simple_equation
        self._equation = equation
        self._MSE = MSE
        self._R2 = R2
        self._params = np.array(params)

    def print(self):
        print(str_e(self._MSE), str_e(self._R2), self._simple_equation)

    def summarize(self, mode='succinct'):
        if mode == 'succinct':
            summary = [self._MSE, self._R2, self._simple_equation]
        elif mode == 'detailed':
            summary = [
                self._MSE,
                self._R2,
                self._simple_equation,
                self._equation]
        parameters = []
        if self._params.shape != ():  # avoid cases with no fit params
            for param in self._params:
                parameters.append(str_e(param))
        parameters_str = ','.join(parameters)
        summary.append(parameters_str)
        return summary
'''
    def predict(self, dataset):
        """
        Calculates the predicted value of the dependent variable given a new
        pySRURGS.Dataset object. Can be used to test models against a test or
        validation dataset.

        Parameters
        ----------
        dataset: pySRURGS.Dataset

        Returns
        -------
        y_calc: array like
            The predicted values of the dependent variable
        """
        n_params = dataset._int_max_params
        parameters = create_fitting_parameters(n_params, self._params)
        eqn_original_cleaned = clean_funcstring(self._equation)
        y_calc = eval_equation(parameters, eqn_original_cleaned, dataset,
                               mode='y_calc')
        return y_calc
'''
